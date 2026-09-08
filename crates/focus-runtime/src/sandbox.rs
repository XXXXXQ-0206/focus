//! Workspace-bound filesystem and command execution primitives.

use std::{
    ffi::OsString,
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use command_group::{CommandGroup, GroupChild};
use focus_kernel::{CancellationSignal, NoCancellation};
use serde::{Deserialize, Serialize};

use crate::RuntimeError;

#[cfg(test)]
pub(crate) fn process_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// The enforced boundary for a Runtime instance.
#[derive(Clone)]
pub struct WorkspaceSandbox {
    root: PathBuf,
    max_read_bytes: usize,
    search_limits: SearchLimits,
    max_command_output_bytes: usize,
    executor: Arc<dyn CommandExecutor>,
}

impl std::fmt::Debug for WorkspaceSandbox {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkspaceSandbox")
            .field("root", &self.root)
            .field("max_read_bytes", &self.max_read_bytes)
            .field("search_limits", &self.search_limits)
            .field("max_command_output_bytes", &self.max_command_output_bytes)
            .finish_non_exhaustive()
    }
}

/// Aggregate limits for a single workspace search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchLimits {
    /// Maximum matching lines returned to the caller.
    pub max_results: usize,
    /// Maximum regular files opened during one traversal.
    pub max_files_scanned: usize,
    /// Maximum file bytes examined during one traversal.
    pub max_bytes_scanned: usize,
    /// Maximum wall-clock time spent traversing and reading.
    pub max_elapsed: Duration,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            max_results: 100,
            max_files_scanned: 10_000,
            max_bytes_scanned: 32 * 1024 * 1024,
            max_elapsed: Duration::from_secs(5),
        }
    }
}

/// Bounded workspace-search output with evidence when scanning stops early.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Matching lines discovered before the search completed or reached a limit.
    pub matches: Vec<String>,
    /// Whether one of the configured limits stopped the traversal.
    pub truncated: bool,
    /// Number of regular files opened during the search.
    pub files_scanned: usize,
    /// Number of file bytes examined during the search.
    pub bytes_scanned: usize,
}

/// Resource limits shared by native and container command adapters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum container memory in bytes when supported.
    pub memory_bytes: Option<u64>,
    /// Maximum logical CPUs when supported.
    pub cpus: Option<f64>,
    /// Maximum process count when supported.
    pub pids: Option<u32>,
}

impl ResourceLimits {
    const MAX_MEMORY_BYTES: u64 = 1024 * 1024 * 1024 * 1024;
    const MAX_CPUS: f64 = 128.0;
    const MAX_PIDS: u32 = 32_768;

    /// Validate executor-independent resource bounds.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if let Some(memory_bytes) = self.memory_bytes
            && !(1..=Self::MAX_MEMORY_BYTES).contains(&memory_bytes)
        {
            return Err(RuntimeError::Configuration(format!(
                "memory limit must be between 1 and {} bytes",
                Self::MAX_MEMORY_BYTES
            )));
        }
        if let Some(cpus) = self.cpus
            && (!cpus.is_finite() || !(f64::MIN_POSITIVE..=Self::MAX_CPUS).contains(&cpus))
        {
            return Err(RuntimeError::Configuration(format!(
                "CPU limit must be finite and between {} and {}",
                f64::MIN_POSITIVE,
                Self::MAX_CPUS
            )));
        }
        if let Some(pids) = self.pids
            && !(1..=Self::MAX_PIDS).contains(&pids)
        {
            return Err(RuntimeError::Configuration(format!(
                "PID limit must be between 1 and {}",
                Self::MAX_PIDS
            )));
        }
        Ok(())
    }
}

/// One bounded command execution request.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandRequest {
    /// Shell command executed in the workspace.
    pub command: String,
    /// Wall-clock limit before process-tree termination.
    pub timeout: Duration,
    /// Maximum retained bytes for each output stream.
    pub max_output_bytes: usize,
    /// Optional resource controls consumed by capable executors.
    pub resources: ResourceLimits,
}

impl CommandRequest {
    /// Create a request with production defaults.
    #[must_use]
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            timeout: Duration::from_secs(120),
            max_output_bytes: 128_000,
            resources: ResourceLimits::default(),
        }
    }

    /// Replace the command timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Replace the retained output limit per stream.
    #[must_use]
    pub fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    /// Attach executor-specific resource controls.
    #[must_use]
    pub fn with_resources(mut self, resources: ResourceLimits) -> Self {
        self.resources = resources;
        self
    }

    /// Validate process limits before the executor is allowed to start a child.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.command.trim().is_empty() {
            return Err(RuntimeError::Sandbox("command must not be empty".into()));
        }
        if self.timeout.is_zero() || self.timeout > Duration::from_secs(60 * 60) {
            return Err(RuntimeError::Configuration(
                "command timeout must be between 1ms and 3600s".into(),
            ));
        }
        if self.max_output_bytes > 8 * 1024 * 1024 {
            return Err(RuntimeError::Configuration(
                "command output limit must be at most 8388608 bytes".into(),
            ));
        }
        self.resources.validate()
    }
}

/// Observed reason a command stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandTermination {
    /// Process exited with an operating-system status code.
    Exited(i32),
    /// Process ended without a portable exit code.
    Signaled,
    /// Runtime terminated the process tree after its deadline.
    TimedOut,
    /// Runtime terminated the process tree after cancellation.
    Cancelled,
}

/// Result captured from a command launched by [`WorkspaceSandbox`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutput {
    /// Process exit status.
    pub exit_code: Option<i32>,
    /// Canonical termination reason.
    pub termination: CommandTermination,
    /// Captured standard output, UTF-8-lossy and bounded.
    pub stdout: String,
    /// Captured standard error, UTF-8-lossy and bounded.
    pub stderr: String,
    /// Whether retained stdout omitted drained bytes.
    pub stdout_truncated: bool,
    /// Whether retained stderr omitted drained bytes.
    pub stderr_truncated: bool,
}

/// Replaceable command execution backend shared by every Runtime interface.
pub trait CommandExecutor: Send + Sync {
    /// Execute a request below `root` while observing cancellation.
    fn execute(
        &self,
        root: &Path,
        request: &CommandRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<CommandOutput, RuntimeError>;
}

/// Native workspace process executor with bounded capture and process-tree termination.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeCommandExecutor;

/// Docker or Podman executor adapter with network and resource isolation flags.
#[derive(Debug, Clone)]
pub struct ContainerCommandExecutor {
    program: PathBuf,
    image: String,
}

/// Runtime-selected command isolation backend.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SandboxBackend {
    /// Native workspace process with timeout, cancellation, and bounded capture.
    #[default]
    Native,
    /// Docker container with disabled networking and configured resource flags.
    Docker {
        /// Container image containing the project's toolchain.
        image: String,
    },
    /// Podman container with disabled networking and configured resource flags.
    Podman {
        /// Container image containing the project's toolchain.
        image: String,
    },
}

/// Host-side readiness observed for one supported container runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerCapability {
    /// Executable name used by the adapter.
    pub runtime: String,
    /// Stable machine-readable readiness state.
    pub status: ContainerCapabilityStatus,
    /// Resolved executable path when present on `PATH`.
    pub executable: Option<PathBuf>,
    /// Server version returned by a live daemon.
    pub server_version: Option<String>,
    /// Bounded diagnostic detail for unavailable runtimes.
    pub detail: Option<String>,
    /// Direct operator action for an unavailable runtime.
    pub remediation: Option<String>,
}

/// Stable container capability states exposed through `doctor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainerCapabilityStatus {
    /// Executable and daemon both answered the bounded probe.
    Available,
    /// No matching executable was found on `PATH`.
    Missing,
    /// Executable exists but its daemon is unavailable or rejected the probe.
    DaemonUnavailable,
    /// The daemon probe exceeded its deadline and its process tree was terminated.
    ProbeTimedOut,
}

/// Capability snapshot for all container adapters supported by the Runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerCapabilities {
    /// Docker host capability.
    pub docker: ContainerCapability,
    /// Podman host capability.
    pub podman: ContainerCapability,
}

/// Evidence produced by an explicitly requested live container acceptance run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveContainerAcceptance {
    /// Adapter used for this run.
    pub runtime: String,
    /// Image used for the disposable container.
    pub image: String,
    /// Whether `/workspace` was observable through the host bind mount.
    pub workspace_mount: bool,
    /// Whether only the loopback interface was visible in the container.
    pub network_isolated: bool,
    /// Whether the requested memory, CPU, and PID cgroup limits were observed.
    pub resource_limits: bool,
    /// Whether the named disposable container was absent after execution.
    pub cleanup: bool,
    /// Canonical command termination.
    pub termination: CommandTermination,
}

impl SandboxBackend {
    /// Build the executor shared by every Runtime tool path.
    #[must_use]
    pub fn executor(&self) -> Arc<dyn CommandExecutor> {
        match self {
            Self::Native => Arc::new(NativeCommandExecutor),
            Self::Docker { image } => Arc::new(ContainerCommandExecutor::docker(image.clone())),
            Self::Podman { image } => Arc::new(ContainerCommandExecutor::podman(image.clone())),
        }
    }

    /// Run a disposable Linux-container acceptance check for the selected backend.
    pub fn live_acceptance(&self, root: &Path) -> Result<LiveContainerAcceptance, RuntimeError> {
        let (runtime, image) = match self {
            Self::Docker { image } => ("docker", image),
            Self::Podman { image } => ("podman", image),
            Self::Native => {
                return Err(RuntimeError::Sandbox(
                    "live container acceptance requires a docker or podman sandbox backend".into(),
                ));
            }
        };
        run_live_container_acceptance(root, runtime, image)
    }
}

/// Probe Docker and Podman without starting user workloads or pulling images.
#[must_use]
pub fn probe_container_capabilities() -> ContainerCapabilities {
    ContainerCapabilities {
        docker: probe_container_runtime("docker"),
        podman: probe_container_runtime("podman"),
    }
}

impl WorkspaceSandbox {
    /// Bind a sandbox to a pre-existing project root.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
        if !root.is_dir() {
            return Err(RuntimeError::Sandbox(format!(
                "workspace root is not a directory: {}",
                root.display()
            )));
        }
        Ok(Self {
            root,
            max_read_bytes: 1_000_000,
            search_limits: SearchLimits::default(),
            max_command_output_bytes: 128_000,
            executor: Arc::new(NativeCommandExecutor),
        })
    }

    /// Return the canonical workspace root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Restrict the largest individual file returned by a read tool.
    #[must_use]
    pub fn with_max_read_bytes(mut self, max_read_bytes: usize) -> Self {
        self.max_read_bytes = max_read_bytes;
        self
    }

    /// Replace the aggregate limits used by workspace search operations.
    #[must_use]
    pub fn with_search_limits(mut self, search_limits: SearchLimits) -> Self {
        self.search_limits = search_limits;
        self
    }

    /// Restrict process output retained in the session context.
    #[must_use]
    pub fn with_max_command_output_bytes(mut self, max_command_output_bytes: usize) -> Self {
        self.max_command_output_bytes = max_command_output_bytes;
        self
    }

    /// Replace the command execution backend while preserving all tool and policy paths.
    #[must_use]
    pub fn with_executor(mut self, executor: Arc<dyn CommandExecutor>) -> Self {
        self.executor = executor;
        self
    }

    /// Resolve an existing relative path and prove it remains below the root.
    pub fn resolve_existing(&self, requested: impl AsRef<Path>) -> Result<PathBuf, RuntimeError> {
        let requested = requested.as_ref();
        validate_relative_path(requested)?;
        let path = self
            .root
            .join(requested)
            .canonicalize()
            .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
        self.ensure_inside(&path)?;
        Ok(path)
    }

    /// Resolve a path intended for creation or replacement below the root.
    pub fn resolve_write_path(&self, requested: impl AsRef<Path>) -> Result<PathBuf, RuntimeError> {
        let requested = requested.as_ref();
        validate_relative_path(requested)?;
        let joined = self.root.join(requested);
        let parent = joined
            .parent()
            .ok_or_else(|| RuntimeError::Sandbox("write path has no parent".into()))?;
        let canonical_parent = parent
            .canonicalize()
            .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
        self.ensure_inside(&canonical_parent)?;
        Ok(joined)
    }

    /// Create nested project-relative directories without traversing symlink components.
    pub fn create_dir_all(&self, requested: impl AsRef<Path>) -> Result<(), RuntimeError> {
        let requested = requested.as_ref();
        validate_relative_path(requested)?;
        let mut current = self.root.clone();
        for component in requested.components() {
            let Component::Normal(component) = component else {
                continue;
            };
            current.push(component);
            match std::fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(RuntimeError::Sandbox(format!(
                        "directory component is a symlink: {}",
                        current.display()
                    )));
                }
                Ok(metadata) if !metadata.is_dir() => {
                    return Err(RuntimeError::Sandbox(format!(
                        "directory component is not a directory: {}",
                        current.display()
                    )));
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    std::fs::create_dir(&current)
                        .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
                }
                Err(error) => return Err(RuntimeError::Sandbox(error.to_string())),
            }
            let canonical = current
                .canonicalize()
                .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
            self.ensure_inside(&canonical)?;
        }
        Ok(())
    }

    /// Read a bounded UTF-8-lossy file.
    pub fn read_to_string(&self, requested: impl AsRef<Path>) -> Result<String, RuntimeError> {
        let path = self.resolve_existing(requested)?;
        let metadata =
            std::fs::metadata(&path).map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
        if metadata.len() > self.max_read_bytes as u64 {
            return Err(RuntimeError::Sandbox(format!(
                "file exceeds read limit of {} bytes",
                self.max_read_bytes
            )));
        }
        let bytes =
            std::fs::read(path).map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
        if bytes.len() > self.max_read_bytes {
            return Err(RuntimeError::Sandbox(format!(
                "file exceeds read limit of {} bytes",
                self.max_read_bytes
            )));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Atomically replace a file below the workspace boundary.
    pub fn write_string(
        &self,
        requested: impl AsRef<Path>,
        content: &str,
    ) -> Result<(), RuntimeError> {
        let path = self.resolve_write_path(requested)?;
        let parent = path
            .parent()
            .ok_or_else(|| RuntimeError::Sandbox("write path has no parent".into()))?;
        let temp = parent.join(format!(".{}.focus-harness.tmp", uuid::Uuid::new_v4()));
        std::fs::write(&temp, content).map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
        std::fs::rename(&temp, &path).map_err(|error| RuntimeError::Sandbox(error.to_string()))
    }

    /// Find UTF-8 text matches below the workspace without following symlinks.
    pub fn search(&self, query: &str, max_results: usize) -> Result<Vec<String>, RuntimeError> {
        self.search_with_cancellation(query, max_results, &NoCancellation)
    }

    /// Find UTF-8 text matches while allowing a caller to stop a long traversal.
    pub fn search_with_cancellation(
        &self,
        query: &str,
        max_results: usize,
        cancellation: &dyn CancellationSignal,
    ) -> Result<Vec<String>, RuntimeError> {
        Ok(self
            .search_with_cancellation_report(query, max_results, cancellation)?
            .matches)
    }

    /// Search with the sandbox defaults and retain whether the traversal was truncated.
    pub fn search_with_cancellation_report(
        &self,
        query: &str,
        max_results: usize,
        cancellation: &dyn CancellationSignal,
    ) -> Result<SearchResult, RuntimeError> {
        self.search_with_limits(
            query,
            SearchLimits {
                max_results,
                ..self.search_limits
            },
            cancellation,
        )
    }

    /// Search with explicit aggregate resource limits.
    pub fn search_with_limits(
        &self,
        query: &str,
        limits: SearchLimits,
        cancellation: &dyn CancellationSignal,
    ) -> Result<SearchResult, RuntimeError> {
        if query.is_empty() {
            return Err(RuntimeError::Sandbox(
                "search query must not be empty".into(),
            ));
        }
        if limits.max_results == 0
            || limits.max_files_scanned == 0
            || limits.max_bytes_scanned == 0
            || limits.max_elapsed.is_zero()
        {
            return Err(RuntimeError::Sandbox(
                "search limits must all be positive".into(),
            ));
        }
        let started = Instant::now();
        let mut result = SearchResult {
            matches: Vec::new(),
            truncated: false,
            files_scanned: 0,
            bytes_scanned: 0,
        };
        let mut pending = vec![self.root.clone()];
        while let Some(directory) = pending.pop() {
            if cancellation.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            if started.elapsed() >= limits.max_elapsed {
                result.truncated = true;
                return Ok(result);
            }
            let entries = std::fs::read_dir(directory)
                .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
            for entry in entries {
                if cancellation.is_cancelled() {
                    return Err(RuntimeError::Cancelled);
                }
                if started.elapsed() >= limits.max_elapsed {
                    result.truncated = true;
                    return Ok(result);
                }
                let entry = entry.map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
                let file_type = entry
                    .file_type()
                    .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
                if file_type.is_symlink() {
                    continue;
                }
                if file_type.is_dir() {
                    if !matches!(
                        entry.file_name().to_str(),
                        Some(".git" | "target" | "node_modules")
                    ) {
                        pending.push(entry.path());
                    }
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }
                let metadata = entry
                    .metadata()
                    .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
                if metadata.len() > self.max_read_bytes as u64 {
                    continue;
                }
                let file_bytes = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
                if result.files_scanned >= limits.max_files_scanned
                    || file_bytes
                        > limits
                            .max_bytes_scanned
                            .saturating_sub(result.bytes_scanned)
                {
                    result.truncated = true;
                    return Ok(result);
                }
                let entry_path = entry.path();
                let contents = match std::fs::read_to_string(&entry_path) {
                    Ok(contents) => contents,
                    Err(_) => continue,
                };
                result.files_scanned += 1;
                result.bytes_scanned = result.bytes_scanned.saturating_add(file_bytes);
                for (number, line) in contents.lines().enumerate() {
                    if cancellation.is_cancelled() {
                        return Err(RuntimeError::Cancelled);
                    }
                    if line.contains(query) {
                        let relative = entry_path.strip_prefix(&self.root).unwrap_or(&entry_path);
                        result.matches.push(format!(
                            "{}:{}:{}",
                            relative.display(),
                            number + 1,
                            line.trim()
                        ));
                        if result.matches.len() >= limits.max_results {
                            result.truncated = true;
                            return Ok(result);
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    /// Execute a fully bounded command request through the configured backend.
    pub fn run_command(
        &self,
        request: CommandRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<CommandOutput, RuntimeError> {
        request.validate()?;
        self.executor.execute(&self.root, &request, cancellation)
    }

    fn ensure_inside(&self, path: &Path) -> Result<(), RuntimeError> {
        if path.starts_with(&self.root) {
            Ok(())
        } else {
            Err(RuntimeError::Sandbox(format!(
                "path escapes workspace: {}",
                path.display()
            )))
        }
    }
}

impl CommandExecutor for NativeCommandExecutor {
    fn execute(
        &self,
        root: &Path,
        request: &CommandRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<CommandOutput, RuntimeError> {
        #[cfg(windows)]
        let mut command = {
            let mut command = Command::new("cmd.exe");
            command.args(["/D", "/S", "/C", &request.command]);
            command
        };
        #[cfg(not(windows))]
        let mut command = {
            let mut command = Command::new("sh");
            command.args(["-lc", &request.command]);
            command
        };
        command.current_dir(root);
        execute_command(command, request, cancellation)
    }
}

impl ContainerCommandExecutor {
    /// Create a Docker-backed executor for one image.
    #[must_use]
    pub fn docker(image: impl Into<String>) -> Self {
        Self {
            program: PathBuf::from("docker"),
            image: image.into(),
        }
    }

    /// Create a Podman-backed executor for one image.
    #[must_use]
    pub fn podman(image: impl Into<String>) -> Self {
        Self {
            program: PathBuf::from("podman"),
            image: image.into(),
        }
    }

    /// Build argument-safe container launcher arguments for inspection and execution.
    #[must_use]
    pub fn arguments_for(
        &self,
        root: &Path,
        request: &CommandRequest,
        name: &str,
    ) -> Vec<OsString> {
        let mut arguments = vec![
            OsString::from("run"),
            OsString::from("--rm"),
            OsString::from("--name"),
            OsString::from(name),
            OsString::from("--network"),
            OsString::from("none"),
        ];
        if let Some(memory) = request.resources.memory_bytes {
            arguments.extend([
                OsString::from("--memory"),
                OsString::from(memory.to_string()),
            ]);
        }
        if let Some(cpus) = request.resources.cpus {
            arguments.extend([OsString::from("--cpus"), OsString::from(cpus.to_string())]);
        }
        if let Some(pids) = request.resources.pids {
            arguments.extend([
                OsString::from("--pids-limit"),
                OsString::from(pids.to_string()),
            ]);
        }
        arguments.extend([
            OsString::from("--volume"),
            OsString::from(format!("{}:/workspace", root.display())),
            OsString::from("--workdir"),
            OsString::from("/workspace"),
            OsString::from(&self.image),
            OsString::from("sh"),
            OsString::from("-lc"),
            OsString::from(&request.command),
        ]);
        arguments
    }
}

impl CommandExecutor for ContainerCommandExecutor {
    fn execute(
        &self,
        root: &Path,
        request: &CommandRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<CommandOutput, RuntimeError> {
        if self.image.trim().is_empty() {
            return Err(RuntimeError::Sandbox(
                "container image must not be empty".into(),
            ));
        }
        let name = format!("focus-harness-{}", uuid::Uuid::new_v4().simple());
        let mut command = Command::new(&self.program);
        command.args(self.arguments_for(root, request, &name));
        let output = execute_command(command, request, cancellation)?;
        if matches!(
            output.termination,
            CommandTermination::TimedOut | CommandTermination::Cancelled
        ) {
            let _ = Command::new(&self.program)
                .args(["rm", "--force", &name])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        Ok(output)
    }
}

fn probe_container_runtime(runtime: &str) -> ContainerCapability {
    let executable = find_executable(runtime);
    let Some(executable_path) = executable.clone() else {
        return ContainerCapability {
            runtime: runtime.into(),
            status: ContainerCapabilityStatus::Missing,
            executable: None,
            server_version: None,
            detail: Some("executable not found on PATH".into()),
            remediation: Some(format!(
                "Install {runtime} or select the native sandbox backend."
            )),
        };
    };
    let request = CommandRequest::new("version --format {{.Server.Version}}")
        .with_timeout(Duration::from_secs(3))
        .with_max_output_bytes(256);
    let mut command = Command::new(&executable_path);
    command.args(["version", "--format", "{{.Server.Version}}"]);
    classify_container_probe(
        runtime,
        executable_path,
        execute_command(command, &request, &NoCancellation),
    )
}

fn classify_container_probe(
    runtime: &str,
    executable_path: PathBuf,
    probe: Result<CommandOutput, RuntimeError>,
) -> ContainerCapability {
    match probe {
        Ok(output) if matches!(output.termination, CommandTermination::Exited(0)) => {
            let version = output.stdout.trim().to_owned();
            ContainerCapability {
                runtime: runtime.into(),
                status: ContainerCapabilityStatus::Available,
                executable: Some(executable_path),
                server_version: (!version.is_empty()).then_some(version),
                detail: None,
                remediation: None,
            }
        }
        Ok(output) if output.termination == CommandTermination::TimedOut => ContainerCapability {
            runtime: runtime.into(),
            status: ContainerCapabilityStatus::ProbeTimedOut,
            executable: Some(executable_path),
            server_version: None,
            detail: Some("daemon probe exceeded 3 seconds".into()),
            remediation: Some(format!(
                "Inspect the {runtime} daemon and rerun doctor; the bounded probe timed out."
            )),
        },
        Ok(output) => ContainerCapability {
            runtime: runtime.into(),
            status: ContainerCapabilityStatus::DaemonUnavailable,
            executable: Some(executable_path),
            server_version: None,
            detail: bounded_detail(&output.stderr),
            remediation: Some(format!("Start the {runtime} daemon and rerun doctor.")),
        },
        Err(error) => ContainerCapability {
            runtime: runtime.into(),
            status: ContainerCapabilityStatus::DaemonUnavailable,
            executable: Some(executable_path),
            server_version: None,
            detail: Some(
                bounded_detail(&error.to_string()).unwrap_or_else(|| "probe failed".into()),
            ),
            remediation: Some(format!("Start the {runtime} daemon and rerun doctor.")),
        },
    }
}

fn bounded_detail(text: &str) -> Option<String> {
    let detail = text.trim();
    if detail.is_empty() {
        None
    } else {
        Some(detail.chars().take(256).collect())
    }
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let candidate = Path::new(name);
    if candidate.components().count() > 1 && candidate.is_file() {
        return Some(candidate.to_path_buf());
    }
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let direct = directory.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        #[cfg(windows)]
        for extension in [".exe", ".cmd", ".bat"] {
            let with_extension = directory.join(format!("{name}{extension}"));
            if with_extension.is_file() {
                return Some(with_extension);
            }
        }
    }
    None
}

fn run_live_container_acceptance(
    root: &Path,
    runtime: &str,
    image: &str,
) -> Result<LiveContainerAcceptance, RuntimeError> {
    if image.trim().is_empty() {
        return Err(RuntimeError::Sandbox(
            "container image must not be empty".into(),
        ));
    }
    let executable = find_executable(runtime).ok_or_else(|| {
        RuntimeError::Sandbox(format!(
            "{runtime} executable is missing; live acceptance not run"
        ))
    })?;
    let name = format!("focus-harness-live-{}", uuid::Uuid::new_v4().simple());
    let resources = ResourceLimits {
        memory_bytes: Some(128 * 1024 * 1024),
        cpus: Some(0.5),
        pids: Some(64),
    };
    let request = CommandRequest::new(
        "set -eu; test -d /workspace; set -- /sys/class/net/*; test \"$#\" -eq 1; test -e /sys/class/net/lo; mem=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes); test \"$mem\" -le 134217728; pids=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max); test \"$pids\" = 64; if test -f /sys/fs/cgroup/cpu.max; then set -- $(cat /sys/fs/cgroup/cpu.max); test \"$1\" != max; test $(( $1 * 100 / $2 )) -le 50; else quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); test $(( quota * 100 / period )) -le 50; fi; printf live-ok > /workspace/.focus-harness-live-marker",
    )
    .with_timeout(Duration::from_secs(30))
    .with_max_output_bytes(4 * 1024)
    .with_resources(resources);
    let executor = ContainerCommandExecutor {
        program: executable.clone(),
        image: image.into(),
    };
    let mut command = Command::new(executable);
    command.args(executor.arguments_for(root, &request, &name));
    let output = execute_command(command, &request, &NoCancellation)?;
    let marker = root.join(".focus-harness-live-marker");
    let workspace_mount = marker.is_file();
    let _ = std::fs::remove_file(&marker);
    let residual = Command::new(runtime)
        .args([
            "ps",
            "-a",
            "--filter",
            &format!("name=^{name}$"),
            "--format",
            "{{.Names}}",
        ])
        .output()
        .map(|result| !String::from_utf8_lossy(&result.stdout).trim().is_empty())
        .unwrap_or(true);
    if residual {
        let _ = Command::new(runtime)
            .args(["rm", "--force", &name])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    Ok(LiveContainerAcceptance {
        runtime: runtime.into(),
        image: image.into(),
        workspace_mount,
        network_isolated: output.termination == CommandTermination::Exited(0),
        resource_limits: output.termination == CommandTermination::Exited(0),
        cleanup: !residual,
        termination: output.termination,
    })
}

fn execute_command(
    mut command: Command,
    request: &CommandRequest,
    cancellation: &dyn CancellationSignal,
) -> Result<CommandOutput, RuntimeError> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .group_spawn()
        .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
    let stdout = child
        .inner()
        .stdout
        .take()
        .ok_or_else(|| RuntimeError::Sandbox("command stdout was not piped".into()))?;
    let stderr = child
        .inner()
        .stderr
        .take()
        .ok_or_else(|| RuntimeError::Sandbox("command stderr was not piped".into()))?;
    let stdout_reader = spawn_capture(stdout, request.max_output_bytes);
    let stderr_reader = spawn_capture(stderr, request.max_output_bytes);
    let started = Instant::now();
    let termination = loop {
        if cancellation.is_cancelled() {
            terminate_process_group(&mut child);
            break CommandTermination::Cancelled;
        }
        if started.elapsed() >= request.timeout {
            terminate_process_group(&mut child);
            break CommandTermination::TimedOut;
        }
        if stdout_reader.is_finished() && stderr_reader.is_finished() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    break status
                        .code()
                        .map_or(CommandTermination::Signaled, CommandTermination::Exited);
                }
                Ok(None) => {}
                Err(error) => {
                    terminate_process_group(&mut child);
                    return Err(RuntimeError::Sandbox(error.to_string()));
                }
            }
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| RuntimeError::Sandbox("stdout reader panicked".into()))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| RuntimeError::Sandbox("stderr reader panicked".into()))??;
    let exit_code = match termination {
        CommandTermination::Exited(code) => Some(code),
        _ => None,
    };
    Ok(CommandOutput {
        exit_code,
        termination,
        stdout: stdout.text,
        stderr: stderr.text,
        stdout_truncated: stdout.truncated,
        stderr_truncated: stderr.truncated,
    })
}

struct CapturedOutput {
    text: String,
    truncated: bool,
}

fn spawn_capture(
    reader: impl Read + Send + 'static,
    limit: usize,
) -> thread::JoinHandle<Result<CapturedOutput, RuntimeError>> {
    thread::spawn(move || {
        let captured = crate::provider::read_bounded(reader, limit)
            .map_err(|error| RuntimeError::Sandbox(error.to_string()))?;
        let mut text = String::from_utf8_lossy(&captured.bytes).into_owned();
        if captured.truncated {
            text.push_str("\n[output truncated]");
        }
        Ok(CapturedOutput {
            text,
            truncated: captured.truncated,
        })
    })
}

fn terminate_process_group(child: &mut GroupChild) {
    let _ = child.kill();
    let _ = child.wait();
}

fn validate_relative_path(path: &Path) -> Result<(), RuntimeError> {
    if path.is_absolute() {
        return Err(RuntimeError::Sandbox(
            "absolute paths are not accepted".into(),
        ));
    }
    let mut normal_components = 0;
    for component in path.components() {
        match component {
            Component::Normal(_) => normal_components += 1,
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(RuntimeError::Sandbox(
                    "path traversal components are not accepted".into(),
                ));
            }
        }
    }
    if normal_components == 0 {
        return Err(RuntimeError::Sandbox("path must not be empty".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::Arc,
        time::{Duration, Instant},
    };

    use focus_kernel::NoCancellation;

    use super::{
        CommandOutput, CommandRequest, CommandTermination, ContainerCapabilityStatus,
        ContainerCommandExecutor, NativeCommandExecutor, ResourceLimits, SandboxBackend,
        SearchLimits, WorkspaceSandbox, classify_container_probe, probe_container_runtime,
    };

    #[test]
    fn creates_nested_directories_without_allowing_traversal() {
        let directory =
            std::env::temp_dir().join(format!("sandbox-mkdir-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();

        sandbox.create_dir_all("nested/source").unwrap();
        sandbox
            .write_string("nested/source/lib.rs", "pub fn ready() {}")
            .unwrap();

        assert!(directory.join("nested/source/lib.rs").is_file());
        assert!(sandbox.create_dir_all("../outside").is_err());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn write_string_replaces_an_existing_file() {
        let directory =
            std::env::temp_dir().join(format!("sandbox-replace-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();

        sandbox.write_string("state.txt", "first").unwrap();
        sandbox.write_string("state.txt", "second").unwrap();

        assert_eq!(
            std::fs::read_to_string(directory.join("state.txt")).unwrap(),
            "second"
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn command_rejects_non_finite_cpu_limits_before_execution() {
        let directory =
            std::env::temp_dir().join(format!("sandbox-resources-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();
        let command = if cfg!(windows) { "exit 0" } else { "true" };

        let error = sandbox
            .run_command(
                CommandRequest::new(command).with_resources(ResourceLimits {
                    memory_bytes: None,
                    cpus: Some(f64::NAN),
                    pids: None,
                }),
                &NoCancellation,
            )
            .unwrap_err();

        assert!(error.to_string().contains("CPU limit"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn search_reports_truncation_when_the_file_budget_is_exhausted() {
        let directory =
            std::env::temp_dir().join(format!("sandbox-search-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("first.txt"), "needle\n").unwrap();
        std::fs::write(directory.join("second.txt"), "needle\n").unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();

        let result = sandbox
            .search_with_limits(
                "needle",
                SearchLimits {
                    max_results: 10,
                    max_files_scanned: 1,
                    max_bytes_scanned: 1_024,
                    max_elapsed: Duration::from_secs(1),
                },
                &NoCancellation,
            )
            .unwrap();

        assert_eq!(result.matches.len(), 1);
        assert!(result.truncated);
        assert_eq!(result.files_scanned, 1);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn native_executor_times_out_and_reaps_the_process_tree() {
        let _process_guard = super::process_test_lock().lock().unwrap();
        let directory =
            std::env::temp_dir().join(format!("sandbox-timeout-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory)
            .unwrap()
            .with_executor(Arc::new(NativeCommandExecutor));
        let command = if cfg!(windows) {
            "powershell -NoProfile -Command \"Start-Sleep -Seconds 10\""
        } else {
            "sleep 10"
        };
        let started = Instant::now();

        let output = sandbox
            .run_command(
                CommandRequest::new(command).with_timeout(Duration::from_millis(150)),
                &NoCancellation,
            )
            .unwrap();

        assert_eq!(output.termination, CommandTermination::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn native_executor_timeout_includes_inherited_pipe_drain() {
        let _process_guard = super::process_test_lock().lock().unwrap();
        let directory =
            std::env::temp_dir().join(format!("sandbox-pipe-timeout-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let marker = directory.join("background-finished");
        let sandbox = WorkspaceSandbox::new(&directory)
            .unwrap()
            .with_executor(Arc::new(NativeCommandExecutor));
        let command = if cfg!(windows) {
            format!(
                "start \"\" /B powershell -NoProfile -Command \"Start-Sleep -Seconds 10; Set-Content -LiteralPath '{}' -Value done\" & exit /B 0",
                marker.display()
            )
        } else {
            format!("(sleep 10; printf done > '{}') &", marker.display())
        };
        let started = Instant::now();

        let output = sandbox
            .run_command(
                CommandRequest::new(command).with_timeout(Duration::from_millis(150)),
                &NoCancellation,
            )
            .unwrap();

        assert_eq!(
            output.termination,
            CommandTermination::TimedOut,
            "stdout: {:?}, stderr: {:?}",
            output.stdout,
            output.stderr
        );
        // Keep the assertion tolerant of concurrent Windows test scheduling while
        // still proving the 150 ms deadline does not wait for a ten-second inherited pipe.
        assert!(started.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(1_300));
        assert!(!marker.exists());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn native_executor_observes_in_flight_cancellation() {
        let _process_guard = super::process_test_lock().lock().unwrap();
        let directory =
            std::env::temp_dir().join(format!("sandbox-cancel-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();
        let cancellation = crate::subagent::CancellationToken::default();
        let trigger = cancellation.clone();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            trigger.cancel();
        });
        let command = if cfg!(windows) {
            "powershell -NoProfile -Command \"Start-Sleep -Seconds 5\""
        } else {
            "sleep 5"
        };

        let output = sandbox
            .run_command(CommandRequest::new(command), &cancellation)
            .unwrap();
        worker.join().unwrap();

        assert_eq!(output.termination, CommandTermination::Cancelled);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn native_executor_drains_but_bounds_stdout_and_stderr() {
        let directory =
            std::env::temp_dir().join(format!("sandbox-output-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory)
            .unwrap()
            .with_executor(Arc::new(NativeCommandExecutor));
        let command = if cfg!(windows) {
            "(for /L %i in (1,1,2000) do @echo 0123456789) & (for /L %i in (1,1,2000) do @echo 0123456789 1>&2)"
        } else {
            "head -c 20000 /dev/zero; head -c 20000 /dev/zero >&2"
        };

        let output = sandbox
            .run_command(
                CommandRequest::new(command).with_max_output_bytes(1_024),
                &NoCancellation,
            )
            .unwrap();

        assert_eq!(output.termination, CommandTermination::Exited(0));
        assert!(output.stdout.len() < 1_100);
        assert!(output.stderr.len() < 1_100);
        assert!(output.stdout_truncated);
        assert!(output.stderr_truncated);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn container_adapter_applies_network_and_resource_limits() {
        let root = if cfg!(windows) {
            std::path::PathBuf::from(r"C:\workspace")
        } else {
            std::path::PathBuf::from("/workspace")
        };
        let request = CommandRequest::new("cargo test").with_resources(ResourceLimits {
            memory_bytes: Some(512 * 1024 * 1024),
            cpus: Some(2.0),
            pids: Some(128),
        });
        let executor = ContainerCommandExecutor::docker("rust:latest");

        let arguments = executor.arguments_for(&root, &request, "focus-agent-test");
        let rendered = arguments
            .iter()
            .map(|value| value.to_string_lossy())
            .collect::<Vec<_>>();

        assert!(
            rendered
                .windows(2)
                .any(|pair| pair == ["--network", "none"])
        );
        assert!(
            rendered
                .windows(2)
                .any(|pair| pair == ["--pids-limit", "128"])
        );
        assert!(
            rendered
                .windows(2)
                .any(|pair| pair == ["--memory", "536870912"])
        );
        assert!(rendered.windows(2).any(|pair| pair == ["--cpus", "2"]));
        assert!(rendered.contains(&std::borrow::Cow::Borrowed("rust:latest")));
    }

    #[test]
    fn container_probe_reports_a_missing_runtime_without_spawning() {
        let capability = probe_container_runtime("focus-harness-missing-container-runtime");

        assert_eq!(capability.status, ContainerCapabilityStatus::Missing);
        assert!(capability.executable.is_none());
        assert!(capability.server_version.is_none());
    }

    #[test]
    fn container_probe_reports_an_actionable_daemon_failure() {
        let capability = classify_container_probe(
            "docker",
            std::path::PathBuf::from("C:/tools/docker.exe"),
            Ok(CommandOutput {
                exit_code: Some(1),
                termination: CommandTermination::Exited(1),
                stdout: String::new(),
                stderr: "error during connect: daemon unavailable".into(),
                stdout_truncated: false,
                stderr_truncated: false,
            }),
        );

        assert_eq!(
            capability.status,
            ContainerCapabilityStatus::DaemonUnavailable
        );
        assert_eq!(
            capability.executable,
            Some(std::path::PathBuf::from("C:/tools/docker.exe"))
        );
        assert!(
            capability
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("daemon unavailable"))
        );
        assert!(
            capability
                .remediation
                .as_deref()
                .is_some_and(|action| action.contains("Start"))
        );
    }

    #[test]
    fn native_backend_rejects_live_container_acceptance() {
        let error = SandboxBackend::Native
            .live_acceptance(Path::new("."))
            .unwrap_err();

        assert!(error.to_string().contains("requires a docker or podman"));
    }

    #[test]
    #[ignore = "set a locally available Linux image and run through the doctor CLI"]
    fn live_container_acceptance_checks_isolation_mount_limits_and_cleanup() {
        let image = std::env::var("PI_CODE_AGENT_LIVE_CONTAINER_IMAGE")
            .expect("PI_CODE_AGENT_LIVE_CONTAINER_IMAGE is required");
        let backend = SandboxBackend::Docker { image };
        let evidence = backend.live_acceptance(Path::new(".")).unwrap();

        assert_eq!(evidence.termination, CommandTermination::Exited(0));
        assert!(evidence.workspace_mount);
        assert!(evidence.network_isolated);
        assert!(evidence.resource_limits);
        assert!(evidence.cleanup);
    }
}
