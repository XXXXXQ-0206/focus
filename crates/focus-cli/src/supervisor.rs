//! Stable process supervisor and TUI handoff protocol.

use std::{
    env, fs,
    io::{self, BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    thread,
    time::Duration,
};

use focus_runtime::update::{UpdateArtifact, VersionedArtifactStore, default_update_root};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub(crate) const APP_CHILD_ENV: &str = "FOCUS_APP_CHILD";
pub(crate) const CONTROL_ADDR_ENV: &str = "FOCUS_SUPERVISOR_ADDR";
pub(crate) const CONTROL_TOKEN_ENV: &str = "FOCUS_SUPERVISOR_TOKEN";
pub(crate) const HANDOFF_PATH_ENV: &str = "FOCUS_HANDOFF_PATH";
pub(crate) const UPDATE_ROOT_ENV: &str = "FOCUS_UPDATE_ROOT";
pub(crate) const RESTART_EXIT_CODE: i32 = 75;
pub(crate) const RESTART_REQUESTED: &str = "focus supervisor handoff requested";
const HANDOFF_SCHEMA_VERSION: u32 = 1;
const MAX_CONTROL_FRAME_BYTES: usize = 64 * 1024;
const MAX_HANDOFF_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RestartRecovery {
    RelaunchCurrent,
    Rollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HandoffAttempt {
    Replace,
    RecoverCurrent,
}

fn restart_recovery(
    status_code: Option<i32>,
    has_handoff: bool,
    pending_handoff: Option<HandoffAttempt>,
) -> Option<RestartRecovery> {
    if status_code != Some(RESTART_EXIT_CODE) || has_handoff {
        return None;
    }
    Some(if pending_handoff == Some(HandoffAttempt::Replace) {
        RestartRecovery::Rollback
    } else {
        RestartRecovery::RelaunchCurrent
    })
}

/// One authenticated command sent from an app-child to its supervisor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum SupervisorCommand {
    /// Request a versioned app restart while retaining the current terminal.
    Handoff {
        /// Per-supervisor artifact to launch next.
        token: String,
        /// Hash-verified staged artifact.
        artifact: UpdateArtifact,
        /// Serialized TUI state consumed by the next app-child.
        handoff_path: PathBuf,
    },
    /// Health check for diagnostics and tests.
    Ping {
        /// Per-supervisor authentication token.
        token: String,
    },
    /// Confirm that a replacement TUI has entered and owns the terminal.
    Ready {
        /// Per-supervisor authentication token.
        token: String,
    },
}

impl SupervisorCommand {
    fn token(&self) -> &str {
        match self {
            Self::Handoff { token, .. } | Self::Ping { token } | Self::Ready { token } => token,
        }
    }
}

/// Bounded state transferred between TUI app-child processes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct HandoffState {
    /// Handoff record schema version.
    pub schema_version: u32,
    /// Runtime session being projected.
    pub session_id: Option<Uuid>,
    /// Last canonical event observed by the old child.
    pub event_cursor: Option<Uuid>,
    /// Composer text preserved across the restart.
    pub composer: String,
    /// UTF-8 byte cursor inside `composer`.
    pub composer_cursor: usize,
    /// Transcript rows scrolled away from the tail.
    pub scroll_from_tail: usize,
    /// Whether tool details were open.
    pub details_visible: bool,
    /// Persisted language code.
    pub language: String,
}

impl HandoffState {
    /// Write a bounded handoff record through a temporary file and rename.
    pub(crate) fn write_atomic(&self, path: &Path) -> Result<(), String> {
        let encoded = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        if encoded.len() > MAX_HANDOFF_BYTES {
            return Err("handoff state exceeds the 1 MiB limit".into());
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        let temporary = path.with_extension(format!("tmp-{}", Uuid::new_v4()));
        fs::write(&temporary, encoded).map_err(|error| error.to_string())?;
        if let Err(error) = replace_file(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(error.to_string());
        }
        Ok(())
    }

    /// Read and validate one handoff record.
    pub(crate) fn read(path: &Path) -> Result<Self, String> {
        let metadata = fs::metadata(path).map_err(|error| error.to_string())?;
        if metadata.len() > MAX_HANDOFF_BYTES as u64 {
            return Err("handoff state exceeds the 1 MiB limit".into());
        }
        let encoded = fs::read(path).map_err(|error| error.to_string())?;
        let state: Self = serde_json::from_slice(&encoded).map_err(|error| error.to_string())?;
        if state.schema_version != HANDOFF_SCHEMA_VERSION {
            return Err(format!(
                "unsupported handoff schema {}",
                state.schema_version
            ));
        }
        if state.composer.len() > 256 * 1024
            || !state.composer.is_char_boundary(state.composer_cursor)
        {
            return Err("handoff composer is invalid or exceeds the 256 KiB limit".into());
        }
        Ok(state)
    }
}

/// One supervisor-owned authenticated loopback endpoint.
pub(crate) struct ControlEndpoint {
    listener: TcpListener,
    token: String,
}

impl ControlEndpoint {
    /// Bind an ephemeral loopback port and generate an unguessable token.
    pub(crate) fn bind() -> io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            token: Uuid::new_v4().to_string(),
        })
    }

    /// Return the endpoint address passed to the app-child.
    pub(crate) fn address(&self) -> SocketAddr {
        self.listener
            .local_addr()
            .expect("bound listener has an address")
    }

    /// Return the authentication token passed to the app-child.
    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    /// Poll one bounded JSONL command from the endpoint.
    pub(crate) fn try_receive(&self) -> io::Result<Option<SupervisorCommand>> {
        let (stream, _) = match self.listener.accept() {
            Ok(pair) => pair,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
            Err(error) => return Err(error),
        };
        stream.set_read_timeout(Some(Duration::from_millis(250)))?;
        let mut line = String::new();
        BufReader::new(stream)
            .take(MAX_CONTROL_FRAME_BYTES as u64)
            .read_line(&mut line)?;
        let command: SupervisorCommand = serde_json::from_str(line.trim())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if command.token() != self.token {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "invalid supervisor token",
            ));
        }
        Ok(Some(command))
    }
}

/// Run the stable supervisor around the current executable.
pub(crate) fn supervise(args: Vec<String>) -> Result<ExitCode, String> {
    let current_exe = env::current_exe().map_err(|error| error.to_string())?;
    let endpoint = ControlEndpoint::bind().map_err(|error| error.to_string())?;
    let update_root = default_update_root();
    let update_root_text = update_root.to_string_lossy().into_owned();
    let store = VersionedArtifactStore::new(&update_root);
    let mut executable = startup_executable(&store, &update_root, &current_exe);
    let mut handoff_path = None;
    let mut pending_handoff = None;

    loop {
        let mut command = Command::new(&executable);
        command
            .args(&args)
            .env(APP_CHILD_ENV, "1")
            .env(CONTROL_ADDR_ENV, endpoint.address().to_string())
            .env(CONTROL_TOKEN_ENV, endpoint.token())
            .env(UPDATE_ROOT_ENV, &update_root_text);
        let child_handoff_path = handoff_path.take();
        if let Some(path) = child_handoff_path.as_ref() {
            command.env(HANDOFF_PATH_ENV, path);
        } else {
            command.env_remove(HANDOFF_PATH_ENV);
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(_) if pending_handoff == Some(HandoffAttempt::Replace) => {
                handoff_path = child_handoff_path;
                executable = rollback_executable(&store, &update_root, &current_exe);
                pending_handoff = None;
                continue;
            }
            Err(error) => return Err(error.to_string()),
        };
        let mut request = None;
        let status = loop {
            match endpoint.try_receive() {
                Ok(Some(SupervisorCommand::Ready { .. })) if pending_handoff.is_some() => {
                    pending_handoff = None;
                    if let Some(path) = child_handoff_path.as_ref() {
                        let _ = fs::remove_file(path);
                    }
                }
                Ok(Some(command @ SupervisorCommand::Handoff { .. })) if request.is_none() => {
                    request = Some(command);
                }
                Ok(Some(_)) | Ok(None) | Err(_) => {}
            }
            if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                break status;
            }
            thread::sleep(Duration::from_millis(10));
        };
        if status.code() == Some(RESTART_EXIT_CODE)
            && request.is_none()
            && let Ok(Some(command @ SupervisorCommand::Handoff { .. })) = endpoint.try_receive()
        {
            request = Some(command);
        }
        let has_handoff = request.is_some();
        if status.code() == Some(RESTART_EXIT_CODE)
            && let Some(SupervisorCommand::Handoff {
                artifact,
                handoff_path: next_handoff,
                ..
            }) = request
        {
            if let Err(error) = validate_handoff_artifact(&store, &artifact) {
                eprintln!(
                    "supervisor handoff validation failed; retrying current artifact: {error}"
                );
                handoff_path = Some(next_handoff);
                pending_handoff = Some(HandoffAttempt::RecoverCurrent);
                continue;
            }
            executable = update_root.join(artifact.executable);
            handoff_path = Some(next_handoff);
            pending_handoff = Some(HandoffAttempt::Replace);
            continue;
        }
        if let Some(recovery) = restart_recovery(status.code(), has_handoff, pending_handoff) {
            match recovery {
                RestartRecovery::RelaunchCurrent => {
                    handoff_path = None;
                    pending_handoff = None;
                }
                RestartRecovery::Rollback => {
                    handoff_path = child_handoff_path;
                    executable = rollback_executable(&store, &update_root, &current_exe);
                    pending_handoff = None;
                }
            }
            continue;
        }
        return Ok(exit_code(status.code()));
    }
}

fn startup_executable(
    store: &VersionedArtifactStore,
    update_root: &Path,
    current_exe: &Path,
) -> PathBuf {
    store
        .load_manifest()
        .ok()
        .flatten()
        .map(|manifest| update_root.join(manifest.active.executable))
        .unwrap_or_else(|| current_exe.to_owned())
}

fn rollback_executable(
    store: &VersionedArtifactStore,
    update_root: &Path,
    current_exe: &Path,
) -> PathBuf {
    match store.rollback() {
        Ok(manifest) => update_root.join(manifest.active.executable),
        Err(_) => current_exe.to_owned(),
    }
}

fn validate_handoff_artifact(
    store: &VersionedArtifactStore,
    artifact: &UpdateArtifact,
) -> Result<(), String> {
    store.verify(artifact).map_err(|error| error.to_string())?;
    let active = store
        .load_manifest()
        .map_err(|error| error.to_string())?
        .map(|manifest| manifest.active)
        .ok_or_else(|| "no active update manifest for handoff".to_owned())?;
    if active != *artifact {
        return Err("handoff artifact does not match the active manifest".into());
    }
    Ok(())
}

/// Send a handoff command to the current supervisor.
pub(crate) fn request_handoff(
    artifact: UpdateArtifact,
    state: HandoffState,
    handoff_path: &Path,
) -> Result<(), String> {
    state.write_atomic(handoff_path)?;
    let address = env::var(CONTROL_ADDR_ENV).map_err(|_| "Focus is not supervised".to_owned())?;
    let token = env::var(CONTROL_TOKEN_ENV).map_err(|_| "Focus is not supervised".to_owned())?;
    let address: SocketAddr = address
        .parse::<SocketAddr>()
        .map_err(|error| error.to_string())?;
    let command = SupervisorCommand::Handoff {
        token,
        artifact,
        handoff_path: handoff_path.to_owned(),
    };
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(500))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_millis(500)))
        .map_err(|error| error.to_string())?;
    serde_json::to_writer(&mut stream, &command).map_err(|error| error.to_string())?;
    stream.write_all(b"\n").map_err(|error| error.to_string())?;
    stream.flush().map_err(|error| error.to_string())
}

/// Confirm that an app-child replacement has attached its full-screen TUI.
pub(crate) fn confirm_tui_ready() -> Result<(), String> {
    let address = env::var(CONTROL_ADDR_ENV).map_err(|_| "Focus is not supervised".to_owned())?;
    let token = env::var(CONTROL_TOKEN_ENV).map_err(|_| "Focus is not supervised".to_owned())?;
    let address: SocketAddr = address
        .parse::<SocketAddr>()
        .map_err(|error| error.to_string())?;
    let command = SupervisorCommand::Ready { token };
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(500))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_millis(500)))
        .map_err(|error| error.to_string())?;
    serde_json::to_writer(&mut stream, &command).map_err(|error| error.to_string())?;
    stream.write_all(b"\n").map_err(|error| error.to_string())?;
    stream.flush().map_err(|error| error.to_string())
}

/// Load one app-child handoff record from the supervisor-provided environment.
pub(crate) fn load_handoff_from_env() -> Result<Option<HandoffState>, String> {
    let Some(path) = env::var_os(HANDOFF_PATH_ENV) else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let state = HandoffState::read(&path)?;
    Ok(Some(state))
}

fn replace_file(temporary: &Path, target: &Path) -> io::Result<()> {
    if target.exists() {
        let backup = target.with_extension(format!("bak-{}", Uuid::new_v4()));
        fs::rename(target, &backup)?;
        if let Err(error) = fs::rename(temporary, target) {
            let _ = fs::rename(&backup, target);
            return Err(error);
        }
        let _ = fs::remove_file(backup);
        return Ok(());
    }
    fs::rename(temporary, target)
}

fn exit_code(code: Option<i32>) -> ExitCode {
    ExitCode::from(code.unwrap_or(1).clamp(0, 255) as u8)
}

#[cfg(test)]
mod tests {
    use super::{
        ControlEndpoint, HandoffAttempt, HandoffState, RESTART_EXIT_CODE, RestartRecovery,
        SupervisorCommand, restart_recovery, rollback_executable, startup_executable,
        validate_handoff_artifact,
    };
    use std::{fs, io::Write, path::PathBuf};
    use uuid::Uuid;

    #[test]
    fn handoff_state_round_trips_with_session_and_event_cursor() {
        let root = std::env::temp_dir().join(format!("focus-handoff-{}", Uuid::new_v4()));
        let path = root.join("state.json");
        let state = HandoffState {
            schema_version: 1,
            session_id: Some(Uuid::new_v4()),
            event_cursor: Some(Uuid::new_v4()),
            composer: "继续编辑中文".into(),
            composer_cursor: 6,
            scroll_from_tail: 4,
            details_visible: true,
            language: "zh-CN".into(),
        };
        state.write_atomic(&path).unwrap();
        assert_eq!(HandoffState::read(&path).unwrap(), state);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn control_endpoint_accepts_only_its_token() {
        let endpoint = ControlEndpoint::bind().unwrap();
        let command = SupervisorCommand::Ping {
            token: endpoint.token().into(),
        };
        send(endpoint.address(), &command);
        assert!(endpoint.try_receive().unwrap().is_some());

        let rejected = SupervisorCommand::Ping {
            token: "wrong-token".into(),
        };
        send(endpoint.address(), &rejected);
        assert!(endpoint.try_receive().is_err());
    }

    #[test]
    fn handoff_command_keeps_artifact_and_state_path() {
        let command = SupervisorCommand::Handoff {
            token: "token".into(),
            artifact: focus_runtime::update::UpdateArtifact {
                version: "v1".into(),
                commit: "abc".into(),
                executable: PathBuf::from("versions/v1/focus.exe"),
                sha256: "0".repeat(64),
                size: 10,
            },
            handoff_path: PathBuf::from("handoff.json"),
        };
        let encoded = serde_json::to_string(&command).unwrap();
        let decoded: SupervisorCommand = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, command);
    }

    #[test]
    fn restart_exit_code_is_reserved_for_supervisor_handoff() {
        assert_eq!(RESTART_EXIT_CODE, 75);
    }

    #[test]
    fn restart_without_handoff_recovers_the_running_child() {
        assert_eq!(
            restart_recovery(Some(RESTART_EXIT_CODE), false, None),
            Some(RestartRecovery::RelaunchCurrent)
        );
        assert_eq!(
            restart_recovery(
                Some(RESTART_EXIT_CODE),
                false,
                Some(HandoffAttempt::Replace),
            ),
            Some(RestartRecovery::Rollback)
        );
        assert_eq!(restart_recovery(Some(0), false, None), None);
        assert_eq!(restart_recovery(Some(RESTART_EXIT_CODE), true, None), None);
    }

    #[test]
    fn invalid_handoff_retries_current_without_manifest_rollback() {
        assert_eq!(
            restart_recovery(
                Some(RESTART_EXIT_CODE),
                false,
                Some(HandoffAttempt::RecoverCurrent),
            ),
            Some(RestartRecovery::RelaunchCurrent)
        );
        assert_eq!(
            restart_recovery(
                Some(RESTART_EXIT_CODE),
                false,
                Some(HandoffAttempt::Replace),
            ),
            Some(RestartRecovery::Rollback)
        );
    }

    #[test]
    fn invalid_handoff_artifact_is_rejected_without_touching_the_manifest() {
        let root =
            std::env::temp_dir().join(format!("focus-handoff-validation-{}", Uuid::new_v4()));
        let update_root = root.join("updates");
        let source = root.join("focus.exe");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"v1").unwrap();
        let store = focus_runtime::update::VersionedArtifactStore::new(&update_root);
        let active = store.stage_file(&source, "v1", "one").unwrap();
        store.activate(&active).unwrap();
        let mut invalid = active.clone();
        invalid.sha256 = "0".repeat(64);

        let error = validate_handoff_artifact(&store, &invalid).unwrap_err();
        assert!(error.contains("metadata does not match"));
        assert_eq!(store.load_manifest().unwrap().unwrap().active, active);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn startup_prefers_verified_active_and_rollback_returns_previous() {
        let root = std::env::temp_dir().join(format!("focus-supervisor-{}", Uuid::new_v4()));
        let update_root = root.join("updates");
        let source = root.join("focus.exe");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"v1").unwrap();
        let store = focus_runtime::update::VersionedArtifactStore::new(&update_root);
        let first = store.stage_file(&source, "v1", "one").unwrap();
        store.activate(&first).unwrap();
        fs::write(&source, b"v2").unwrap();
        let second = store.stage_file(&source, "v2", "two").unwrap();
        store.activate(&second).unwrap();

        assert_eq!(
            startup_executable(&store, &update_root, &source),
            update_root.join(&second.executable)
        );
        assert_eq!(
            rollback_executable(&store, &update_root, &source),
            update_root.join(&first.executable)
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn send(address: std::net::SocketAddr, command: &SupervisorCommand) {
        let mut stream = std::net::TcpStream::connect(address).unwrap();
        serde_json::to_writer(&mut stream, command).unwrap();
        stream.write_all(b"\n").unwrap();
    }
}
