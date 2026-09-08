//! Runtime-owned session metadata, resume, fork, and replay.

use std::{
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use focus_kernel::{AgentState, Event, EventKind, EventStore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    RuntimeError,
    memory::{RecordLock, acquire_lock_file, acquire_record_lock, atomic_json_write},
};

/// Long-lived lifecycle status stored in session metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionPhase {
    /// Created but not yet executing.
    Created,
    /// An agent loop is active.
    Running,
    /// The latest agent loop completed normally.
    Complete,
    /// The latest agent loop failed.
    Failed,
    /// The session was cancelled.
    Cancelled,
}

/// Whether a child session sees its ancestor transcript during resume and replay.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptInheritance {
    /// Normal user fork inherits the ancestor transcript by reference.
    #[default]
    Full,
    /// Subagent child retains ancestry metadata but starts with a bounded context only.
    None,
}

/// Non-event session metadata for indexing and ancestry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session ID used by event and memory stores.
    pub id: Uuid,
    /// A short description suitable for a CLI/TUI session list.
    pub title: String,
    /// Parent session for a fork, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Uuid>,
    /// Transcript inheritance policy for this ancestry edge.
    #[serde(default)]
    pub transcript_inheritance: TranscriptInheritance,
    /// Creation timestamp.
    pub created_at_ms: u128,
    /// Last update timestamp.
    pub updated_at_ms: u128,
    /// Runtime lifecycle phase.
    pub phase: SessionPhase,
}

/// Runtime session coordination around the kernel's canonical event stream.
#[derive(Clone)]
pub struct SessionManager {
    root: PathBuf,
    events: std::sync::Arc<dyn EventStore>,
}

/// Process and cross-process lease held for the duration of one active run.
#[derive(Debug)]
pub struct SessionLease {
    _lock: RecordLock,
}

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionManager")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl SessionManager {
    /// Create a session manager. The event store remains the sole event persistence implementation.
    pub fn new(
        root: impl Into<PathBuf>,
        events: std::sync::Arc<dyn EventStore>,
    ) -> Result<Self, RuntimeError> {
        let root = root.into();
        std::fs::create_dir_all(root.join("metadata"))
            .map_err(|error| RuntimeError::Session(error.to_string()))?;
        Ok(Self { root, events })
    }

    /// Start an independent session.
    pub fn create(&self, title: impl Into<String>) -> Result<SessionMetadata, RuntimeError> {
        self.create_with_id_and_phase(Uuid::new_v4(), title, SessionPhase::Created)
    }

    /// Persist a session whose identifier was allocated by an external host.
    pub fn create_with_id(
        &self,
        id: Uuid,
        title: impl Into<String>,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.create_with_id_and_phase(id, title, SessionPhase::Created)
    }

    /// Start an independent session in a known lifecycle phase with one metadata write.
    pub fn create_with_phase(
        &self,
        title: impl Into<String>,
        phase: SessionPhase,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.create_with_id_and_phase(Uuid::new_v4(), title, phase)
    }

    fn create_with_id_and_phase(
        &self,
        id: Uuid,
        title: impl Into<String>,
        phase: SessionPhase,
    ) -> Result<SessionMetadata, RuntimeError> {
        let now = now_ms();
        let metadata = SessionMetadata {
            id,
            title: title.into(),
            parent_id: None,
            transcript_inheritance: TranscriptInheritance::Full,
            created_at_ms: now,
            updated_at_ms: now,
            phase,
        };
        self.save_new(&metadata)?;
        Ok(metadata)
    }

    /// Fork without duplicating events: child state inherits its ancestor stream during resume.
    pub fn fork(
        &self,
        parent_id: Uuid,
        title: impl Into<String>,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.fork_with_inheritance(parent_id, title, TranscriptInheritance::Full)
    }

    /// Fork with an explicit ancestry/transcript boundary.
    pub fn fork_with_inheritance(
        &self,
        parent_id: Uuid,
        title: impl Into<String>,
        transcript_inheritance: TranscriptInheritance,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.fork_with_inheritance_and_phase(
            parent_id,
            title,
            transcript_inheritance,
            SessionPhase::Created,
        )
    }

    /// Fork a child directly into a known lifecycle phase with one metadata write.
    pub fn fork_with_inheritance_and_phase(
        &self,
        parent_id: Uuid,
        title: impl Into<String>,
        transcript_inheritance: TranscriptInheritance,
        phase: SessionPhase,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.load(parent_id)?;
        let now = now_ms();
        let metadata = SessionMetadata {
            id: Uuid::new_v4(),
            title: title.into(),
            parent_id: Some(parent_id),
            transcript_inheritance,
            created_at_ms: now,
            updated_at_ms: now,
            phase,
        };
        self.save(&metadata)?;
        Ok(metadata)
    }

    /// Load one session's metadata.
    pub fn load(&self, id: Uuid) -> Result<SessionMetadata, RuntimeError> {
        let content = std::fs::read_to_string(self.path(id))
            .map_err(|error| RuntimeError::Session(error.to_string()))?;
        serde_json::from_str(&content).map_err(|error| RuntimeError::Session(error.to_string()))
    }

    /// List metadata in newest-first order.
    pub fn list(&self) -> Result<Vec<SessionMetadata>, RuntimeError> {
        let directory = self.root.join("metadata");
        let mut sessions = std::fs::read_dir(directory)
            .map_err(|error| RuntimeError::Session(error.to_string()))?
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "json")
            })
            .map(|entry| {
                let content = std::fs::read_to_string(entry.path())
                    .map_err(|error| RuntimeError::Session(error.to_string()))?;
                serde_json::from_str::<SessionMetadata>(&content)
                    .map_err(|error| RuntimeError::Session(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        sessions.sort_by_key(|session| std::cmp::Reverse(session.updated_at_ms));
        Ok(sessions)
    }

    /// Mark a lifecycle phase using the same metadata record.
    pub fn set_phase(&self, id: Uuid, phase: SessionPhase) -> Result<(), RuntimeError> {
        let _lock = acquire_record_lock(&self.path(id))
            .map_err(|error| RuntimeError::Session(error.to_string()))?;
        let mut metadata = self.load(id)?;
        metadata.phase = phase;
        metadata.updated_at_ms = now_ms();
        self.save(&metadata)
    }

    /// Acquire the exclusive active-run lease for a session.
    pub fn acquire_lease(&self, id: Uuid) -> Result<SessionLease, RuntimeError> {
        let directory = self.root.join("leases");
        std::fs::create_dir_all(&directory)
            .map_err(|error| RuntimeError::Session(error.to_string()))?;
        let path = directory.join(format!("{id}.lock"));
        let lock = match acquire_lock_file(&path, std::time::Duration::ZERO) {
            Ok(lock) => lock,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(RuntimeError::Session(format!(
                    "session {id} already has an active run"
                )));
            }
            Err(error) => return Err(RuntimeError::Session(error.to_string())),
        };
        Ok(SessionLease { _lock: lock })
    }

    /// Rebuild model state from a session and its parent chain.
    pub fn resume(&self, id: Uuid) -> Result<AgentState, RuntimeError> {
        let metadata = self.load(id)?;
        let mut state = if let Some(parent_id) = inherited_parent_id(&metadata) {
            self.resume(parent_id)?
        } else {
            AgentState::new(id)
        };
        state.session_id = id;
        for event in self.events.load(id).map_err(RuntimeError::Kernel)? {
            apply_event(&mut state, event);
        }
        Ok(state)
    }

    /// Return inherited events followed by local events for deterministic replay.
    pub fn replay(&self, id: Uuid) -> Result<Vec<Event>, RuntimeError> {
        let metadata = self.load(id)?;
        let mut events = match inherited_parent_id(&metadata) {
            Some(parent_id) => self.replay(parent_id)?,
            None => Vec::new(),
        };
        events.extend(self.events.load(id).map_err(RuntimeError::Kernel)?);
        Ok(events)
    }

    fn save(&self, metadata: &SessionMetadata) -> Result<(), RuntimeError> {
        atomic_json_write(&self.path(metadata.id), metadata)
            .map_err(|error| RuntimeError::Session(error.to_string()))
    }

    fn save_new(&self, metadata: &SessionMetadata) -> Result<(), RuntimeError> {
        let path = self.path(metadata.id);
        let content = serde_json::to_vec_pretty(metadata)
            .map_err(|error| RuntimeError::Session(error.to_string()))?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| RuntimeError::Session(error.to_string()))?;
        if let Err(error) = file.write_all(&content).and_then(|()| file.sync_all()) {
            drop(file);
            let _ = std::fs::remove_file(&path);
            return Err(RuntimeError::Session(error.to_string()));
        }
        Ok(())
    }

    fn path(&self, id: Uuid) -> PathBuf {
        self.root.join("metadata").join(format!("{id}.json"))
    }
}

fn inherited_parent_id(metadata: &SessionMetadata) -> Option<Uuid> {
    (metadata.transcript_inheritance == TranscriptInheritance::Full)
        .then_some(metadata.parent_id)
        .flatten()
}

fn apply_event(state: &mut AgentState, event: Event) {
    match event.kind {
        EventKind::StateChanged { status, turn } => {
            state.status = status;
            state.turn = turn;
        }
        EventKind::MessageAdded { message } => state.transcript.push(message),
        EventKind::ToolCallRequested { .. }
        | EventKind::ToolResultReceived { .. }
        | EventKind::Model { .. }
        | EventKind::Runtime { .. } => {}
    }
}

pub(crate) fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use focus_kernel::{EventKind, JsonlEventStore, Message, Role};

    use super::*;

    #[test]
    fn fork_inherits_parent_transcript_without_copying_it() {
        let directory =
            std::env::temp_dir().join(format!("runtime-session-test-{}", Uuid::new_v4()));
        let events = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let sessions = SessionManager::new(directory.join("sessions"), events.clone()).unwrap();
        let parent = sessions.create("parent").unwrap();
        events
            .append(&Event::now(
                parent.id,
                EventKind::MessageAdded {
                    message: Message::text(Role::User, "parent request"),
                },
            ))
            .unwrap();
        let child = sessions.fork(parent.id, "child").unwrap();

        let state = sessions.resume(child.id).unwrap();

        assert_eq!(state.session_id, child.id);
        assert_eq!(state.transcript[0].content, "parent request");
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn isolated_child_keeps_ancestry_without_parent_transcript() {
        let directory =
            std::env::temp_dir().join(format!("runtime-isolated-session-{}", Uuid::new_v4()));
        let events = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let sessions = SessionManager::new(directory.join("sessions"), events.clone()).unwrap();
        let parent = sessions.create("parent").unwrap();
        events
            .append(&Event::now(
                parent.id,
                EventKind::MessageAdded {
                    message: Message::text(Role::User, "private parent transcript"),
                },
            ))
            .unwrap();

        let child = sessions
            .fork_with_inheritance(parent.id, "child", TranscriptInheritance::None)
            .unwrap();
        let state = sessions.resume(child.id).unwrap();

        assert_eq!(child.parent_id, Some(parent.id));
        assert!(state.transcript.is_empty());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn creates_a_session_directly_in_the_requested_phase() {
        let directory =
            std::env::temp_dir().join(format!("runtime-session-phase-{}", Uuid::new_v4()));
        let events = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let sessions = SessionManager::new(directory.join("sessions"), events).unwrap();

        let session = sessions
            .create_with_phase("running", SessionPhase::Running)
            .unwrap();

        assert_eq!(session.phase, SessionPhase::Running);
        assert_eq!(
            sessions.load(session.id).unwrap().phase,
            SessionPhase::Running
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn creates_a_session_using_the_host_allocated_id_without_overwriting() {
        let directory =
            std::env::temp_dir().join(format!("runtime-session-host-id-{}", Uuid::new_v4()));
        let events = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let sessions = SessionManager::new(directory.join("sessions"), events).unwrap();
        let id = Uuid::new_v4();

        let session = sessions.create_with_id(id, "host allocated").unwrap();

        assert_eq!(session.id, id);
        assert_eq!(sessions.load(id).unwrap().title, "host allocated");
        assert!(sessions.create_with_id(id, "duplicate").is_err());
        assert_eq!(sessions.load(id).unwrap().title, "host allocated");
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn set_phase_replaces_existing_session_metadata() {
        let directory =
            std::env::temp_dir().join(format!("runtime-session-replace-{}", Uuid::new_v4()));
        let events = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let sessions = SessionManager::new(directory.join("sessions"), events).unwrap();
        let session = sessions.create("replace").unwrap();

        sessions
            .set_phase(session.id, SessionPhase::Running)
            .unwrap();
        sessions
            .set_phase(session.id, SessionPhase::Complete)
            .unwrap();

        assert_eq!(
            sessions.load(session.id).unwrap().phase,
            SessionPhase::Complete
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn acquires_a_lease_when_a_stale_lock_path_remains() {
        let directory =
            std::env::temp_dir().join(format!("runtime-session-stale-lock-{}", Uuid::new_v4()));
        let events = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let sessions = SessionManager::new(directory.join("sessions"), events).unwrap();
        let session = sessions.create("recover lease").unwrap();
        let lock = directory
            .join("sessions")
            .join("leases")
            .join(format!("{}.lock", session.id));
        std::fs::create_dir_all(lock.parent().unwrap()).unwrap();
        std::fs::write(&lock, "stale path contents").unwrap();

        let lease = sessions.acquire_lease(session.id).unwrap();

        drop(lease);
        assert!(lock.exists());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn forks_a_child_directly_in_the_requested_phase() {
        let directory =
            std::env::temp_dir().join(format!("runtime-child-phase-{}", Uuid::new_v4()));
        let events = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let sessions = SessionManager::new(directory.join("sessions"), events).unwrap();
        let parent = sessions.create("parent").unwrap();

        let child = sessions
            .fork_with_inheritance_and_phase(
                parent.id,
                "child",
                TranscriptInheritance::None,
                SessionPhase::Running,
            )
            .unwrap();

        assert_eq!(child.parent_id, Some(parent.id));
        assert_eq!(child.phase, SessionPhase::Running);
        assert_eq!(
            sessions.load(child.id).unwrap().phase,
            SessionPhase::Running
        );
        let _ = std::fs::remove_dir_all(directory);
    }
}
