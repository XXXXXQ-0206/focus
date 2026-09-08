//! Persistent stdio MCP transport and Runtime tool registration.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    io::{self, BufRead, BufReader, Read, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use focus_kernel::{CancellationSignal, NoCancellation, ToolDefinition};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    RuntimeError,
    policy::ToolOperation,
    tools::{RuntimeToolSpec, ToolHandler, ToolRegistry},
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_STDERR_LIMIT: usize = 16 * 1024;
const DEFAULT_RESPONSE_LIMIT: usize = 4 * 1024 * 1024;
const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const MAX_STDERR_LIMIT: usize = 1024 * 1024;
const MAX_RESPONSE_LIMIT: usize = 8 * 1024 * 1024;
const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const SUPPORTED_MCP_PROTOCOL_VERSIONS: &[&str] =
    &[MCP_PROTOCOL_VERSION, "2025-03-26", "2024-11-05"];
const MAX_TOOL_PAGES: usize = 128;
const MAX_DISCOVERED_TOOLS: usize = 4_096;
const CANCELLATION_POLL: Duration = Duration::from_millis(25);

/// Configuration for one persistent stdio MCP server process.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Stable Runtime-local server name used in canonical tool names.
    pub name: String,
    /// Executable to launch directly without an intermediate shell.
    pub command: PathBuf,
    /// Arguments passed to the executable.
    pub args: Vec<String>,
    /// Environment additions or overrides for the child.
    pub env: BTreeMap<String, String>,
    /// Optional child working directory.
    pub current_dir: Option<PathBuf>,
    /// Maximum duration for each JSON-RPC response.
    pub request_timeout: Duration,
    /// Maximum retained stderr tail in bytes.
    pub stderr_limit: usize,
    /// Maximum bytes accepted for one JSON-RPC response line.
    pub response_limit: usize,
    /// Canonical policy capability applied to every tool from this server.
    pub operation: ToolOperation,
}

impl McpServerConfig {
    /// Create a stdio server configuration with bounded production defaults.
    #[must_use]
    pub fn new(name: impl Into<String>, command: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            current_dir: None,
            request_timeout: DEFAULT_TIMEOUT,
            stderr_limit: DEFAULT_STDERR_LIMIT,
            response_limit: DEFAULT_RESPONSE_LIMIT,
            operation: ToolOperation::Other,
        }
    }

    /// Replace the child argument list.
    #[must_use]
    pub fn with_args(mut self, args: impl IntoIterator<Item = String>) -> Self {
        self.args = args.into_iter().collect();
        self
    }

    /// Replace the child environment additions.
    #[must_use]
    pub fn with_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Set the child working directory.
    #[must_use]
    pub fn with_current_dir(mut self, current_dir: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(current_dir.into());
        self
    }

    /// Set the response timeout applied independently to each request.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Set the canonical policy capability applied to this server's tools.
    #[must_use]
    pub fn with_operation(mut self, operation: ToolOperation) -> Self {
        self.operation = operation;
        self
    }

    /// Validate process and transport bounds before a Runtime starts the server.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.name.trim().is_empty() {
            return Err(RuntimeError::Mcp(
                "MCP server name must not be empty".into(),
            ));
        }
        if self.command.as_os_str().is_empty() {
            return Err(RuntimeError::Mcp(
                "MCP server command must not be empty".into(),
            ));
        }
        if self.request_timeout.is_zero() || self.request_timeout > MAX_REQUEST_TIMEOUT {
            return Err(RuntimeError::Mcp(format!(
                "MCP request timeout must be between 1ms and {}s",
                MAX_REQUEST_TIMEOUT.as_secs()
            )));
        }
        if !(1..=MAX_STDERR_LIMIT).contains(&self.stderr_limit) {
            return Err(RuntimeError::Mcp(format!(
                "MCP stderr limit must be between 1 and {MAX_STDERR_LIMIT} bytes"
            )));
        }
        if !(1..=MAX_RESPONSE_LIMIT).contains(&self.response_limit) {
            return Err(RuntimeError::Mcp(format!(
                "MCP response limit must be between 1 and {MAX_RESPONSE_LIMIT} bytes"
            )));
        }
        Ok(())
    }
}

/// A discovered MCP tool before its remote name is canonicalized for providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpTool {
    /// Server hosting the tool.
    pub server: String,
    /// Provider-neutral schema supplied by that server.
    pub definition: ToolDefinition,
}

/// Persistent client for newline-delimited JSON-RPC over child stdio.
pub struct McpClient {
    server_name: String,
    connection: Arc<McpConnection>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpClient")
            .field("server_name", &self.server_name)
            .finish_non_exhaustive()
    }
}

impl McpClient {
    /// Start a server, complete the MCP handshake, and discover every tool page.
    pub fn connect(config: &McpServerConfig) -> Result<(Arc<Self>, Vec<McpTool>), RuntimeError> {
        config.validate()?;

        let connection = Arc::new(McpConnection::spawn(config)?);
        let client = Arc::new(Self {
            server_name: config.name.clone(),
            connection,
        });
        client.initialize()?;
        let tools = client.list_all_tools()?;
        Ok((client, tools))
    }

    fn initialize(&self) -> Result<(), RuntimeError> {
        let result = self.connection.request(
            "initialize",
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "focus-harness", "version": env!("CARGO_PKG_VERSION")}
            }),
        )?;
        let result: InitializeResult = serde_json::from_value(result).map_err(|error| {
            self.connection
                .fatal_protocol_error(format!("invalid initialize result: {error}"))
        })?;
        if !SUPPORTED_MCP_PROTOCOL_VERSIONS.contains(&result.protocol_version.as_str()) {
            return Err(self.connection.fatal_protocol_error(format!(
                "unsupported MCP protocol version `{}`; supported versions: {}",
                result.protocol_version,
                SUPPORTED_MCP_PROTOCOL_VERSIONS.join(", ")
            )));
        }
        if !result.capabilities.is_object() {
            return Err(self
                .connection
                .fatal_protocol_error("initialize capabilities must be an object"));
        }
        if result.server_info.name.trim().is_empty() || result.server_info.version.trim().is_empty()
        {
            return Err(self
                .connection
                .fatal_protocol_error("initialize serverInfo must contain name and version"));
        }
        self.connection
            .notify("notifications/initialized", json!({}))
    }

    fn list_all_tools(&self) -> Result<Vec<McpTool>, RuntimeError> {
        let mut cursor = None;
        let mut seen_cursors = BTreeSet::new();
        let mut tools = Vec::new();
        for page_number in 0..MAX_TOOL_PAGES {
            let params = cursor
                .as_ref()
                .map_or_else(|| json!({}), |cursor| json!({"cursor": cursor}));
            let result = self.connection.request("tools/list", params)?;
            let page: ToolsPage = serde_json::from_value(result).map_err(|error| {
                self.connection
                    .protocol_error(format!("invalid tools/list result: {error}"))
            })?;
            tools.extend(page.tools.into_iter().map(|tool| McpTool {
                server: self.server_name.clone(),
                definition: ToolDefinition {
                    name: tool.name,
                    description: tool.description.unwrap_or_default(),
                    input_schema: tool.input_schema,
                },
            }));
            if tools.len() > MAX_DISCOVERED_TOOLS {
                return Err(self.connection.fatal_protocol_error(format!(
                    "tools/list exceeded the {MAX_DISCOVERED_TOOLS} tool limit"
                )));
            }
            match page.next_cursor.filter(|value| !value.is_empty()) {
                Some(next) if seen_cursors.insert(next.clone()) => cursor = Some(next),
                Some(next) => {
                    return Err(self
                        .connection
                        .protocol_error(format!("tools/list cursor loop at `{next}`")));
                }
                None => return Ok(tools),
            }
            if page_number + 1 == MAX_TOOL_PAGES {
                return Err(self.connection.fatal_protocol_error(format!(
                    "tools/list exceeded the {MAX_TOOL_PAGES} page limit"
                )));
            }
        }
        unreachable!("bounded MCP pagination loop always returns")
    }

    /// Invoke one remote MCP tool on the caller's Runtime executor.
    pub async fn call_tool_async(
        self: &Arc<Self>,
        remote_name: &str,
        arguments: Value,
        cancellation: &dyn CancellationSignal,
    ) -> Result<String, RuntimeError> {
        let client = self.clone();
        let remote_name = remote_name.to_owned();
        let cancellation = focus_kernel::CancellationBridge::from_signal(cancellation);
        tokio::task::spawn_blocking(move || {
            client.call_tool_blocking(&remote_name, arguments, &cancellation)
        })
        .await
        .map_err(|error| RuntimeError::Mcp(format!("MCP worker failed: {error}")))?
    }

    fn call_tool_blocking(
        &self,
        remote_name: &str,
        arguments: Value,
        cancellation: &dyn CancellationSignal,
    ) -> Result<String, RuntimeError> {
        let result = self.connection.request_with_cancellation(
            "tools/call",
            json!({"name": remote_name, "arguments": arguments}),
            cancellation,
        )?;
        let result: ToolCallResult = serde_json::from_value(result).map_err(|error| {
            self.connection
                .protocol_error(format!("invalid tools/call result: {error}"))
        })?;
        let content = render_tool_content(&result.content, result.structured_content.as_ref());
        if result.is_error {
            Err(RuntimeError::Mcp(if content.is_empty() {
                format!("MCP tool `{remote_name}` reported an error")
            } else {
                content
            }))
        } else {
            Ok(content)
        }
    }
}

/// Register all tools from one server into the Runtime's sole tool registry.
pub fn register_mcp_server(
    registry: &mut ToolRegistry,
    config: &McpServerConfig,
) -> Result<Arc<McpClient>, RuntimeError> {
    let (client, discovered) = McpClient::connect(config)?;
    let existing = registry
        .definitions()
        .into_iter()
        .map(|definition| definition.name)
        .collect::<BTreeSet<_>>();
    let mut pending = Vec::with_capacity(discovered.len());
    let mut new_names = BTreeSet::new();
    for tool in discovered {
        let remote_name = tool.definition.name.clone();
        let canonical = canonical_tool_name(&tool.server, &remote_name);
        if existing.contains(&canonical) || !new_names.insert(canonical.clone()) {
            return Err(RuntimeError::Mcp(format!(
                "duplicate canonical MCP tool: {canonical}"
            )));
        }
        let mut definition = tool.definition;
        definition.name = canonical;
        pending.push(
            RuntimeToolSpec::new(
                definition,
                config.operation,
                format!("Call MCP tool `{remote_name}` on server `{}`.", tool.server),
                Arc::new(McpToolHandler {
                    client: client.clone(),
                    remote_name,
                }),
            )
            .with_execution_class(focus_kernel::ToolExecutionClass::Exclusive),
        );
    }
    for spec in pending {
        registry.register(spec)?;
    }
    Ok(client)
}

/// Produce a deterministic OpenAI-safe Runtime name no longer than 64 bytes.
#[must_use]
pub fn canonical_tool_name(server: &str, remote_name: &str) -> String {
    let source = format!("{server}__{remote_name}");
    let normalized = source
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' {
                byte as char
            } else {
                '_'
            }
        })
        .collect::<String>();
    if !normalized.is_empty() && normalized == source && normalized.len() <= 64 {
        return normalized;
    }
    let hash = stable_hash(source.as_bytes());
    let prefix_len = 64 - 1 - 16;
    let mut prefix = normalized.chars().take(prefix_len).collect::<String>();
    if prefix.is_empty() {
        prefix.push_str("mcp");
    }
    format!("{prefix}_{hash:016x}")
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct McpToolHandler {
    client: Arc<McpClient>,
    remote_name: String,
}

impl ToolHandler for McpToolHandler {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        Box::pin(async move {
            self.client
                .call_tool_async(&self.remote_name, arguments, cancellation)
                .await
        })
    }
}

#[derive(Debug, Deserialize)]
struct ToolsPage {
    tools: Vec<RemoteTool>,
    #[serde(default, rename = "nextCursor")]
    next_cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    protocol_version: String,
    capabilities: Value,
    #[serde(rename = "serverInfo")]
    server_info: ServerInfo,
}

#[derive(Debug, Deserialize)]
struct ServerInfo {
    name: String,
    version: String,
}

#[derive(Debug, Deserialize)]
struct RemoteTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default = "empty_object", rename = "inputSchema")]
    input_schema: Value,
}

#[derive(Debug, Deserialize)]
struct ToolCallResult {
    #[serde(default)]
    content: Vec<Value>,
    #[serde(default, rename = "structuredContent")]
    structured_content: Option<Value>,
    #[serde(default, rename = "isError")]
    is_error: bool,
}

fn empty_object() -> Value {
    json!({"type": "object"})
}

fn render_tool_content(content: &[Value], structured: Option<&Value>) -> String {
    let mut rendered = String::new();
    for item in content {
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        if let Some(text) = item.get("text").and_then(Value::as_str) {
            rendered.push_str(text);
        } else {
            rendered.push_str(&item.to_string());
        }
    }
    if let Some(structured) = structured {
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str(&structured.to_string());
    }
    rendered
}

enum StdoutEvent {
    Line(String),
    Eof,
    Error(String),
}

struct McpConnection {
    stdin: Mutex<Option<ChildStdin>>,
    responses: Mutex<Receiver<StdoutEvent>>,
    child: Mutex<Option<Child>>,
    stderr: Arc<Mutex<BoundedBuffer>>,
    stdout_thread: Mutex<Option<JoinHandle<()>>>,
    stderr_thread: Mutex<Option<JoinHandle<()>>>,
    request_lock: Mutex<()>,
    next_id: AtomicU64,
    timeout: Duration,
}

impl McpConnection {
    fn spawn(config: &McpServerConfig) -> Result<Self, RuntimeError> {
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .envs(&config.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(current_dir) = &config.current_dir {
            command.current_dir(current_dir);
        }
        let mut child = command.spawn().map_err(|error| {
            RuntimeError::Mcp(format!(
                "failed to start MCP server `{}` using `{}`: {error}",
                config.name,
                config.command.display()
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| RuntimeError::Mcp("MCP child stdin was not piped".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RuntimeError::Mcp("MCP child stdout was not piped".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| RuntimeError::Mcp("MCP child stderr was not piped".into()))?;

        let (sender, responses) = mpsc::channel();
        let response_limit = config.response_limit;
        let stdout_thread = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                match read_bounded_line(&mut reader, response_limit) {
                    Ok(None) => {
                        let _ = sender.send(StdoutEvent::Eof);
                        break;
                    }
                    Ok(Some(line)) => {
                        if sender.send(StdoutEvent::Line(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(StdoutEvent::Error(error.to_string()));
                        break;
                    }
                }
            }
        });
        let stderr_buffer = Arc::new(Mutex::new(BoundedBuffer::new(config.stderr_limit)));
        let stderr_target = stderr_buffer.clone();
        let stderr_thread = thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut chunk = [0_u8; 1024];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => {
                        if let Ok(mut buffer) = stderr_target.lock() {
                            buffer.extend(&chunk[..read]);
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self {
            stdin: Mutex::new(Some(stdin)),
            responses: Mutex::new(responses),
            child: Mutex::new(Some(child)),
            stderr: stderr_buffer,
            stdout_thread: Mutex::new(Some(stdout_thread)),
            stderr_thread: Mutex::new(Some(stderr_thread)),
            request_lock: Mutex::new(()),
            next_id: AtomicU64::new(1),
            timeout: config.request_timeout,
        })
    }

    fn request(&self, method: &str, params: Value) -> Result<Value, RuntimeError> {
        self.request_with_cancellation(method, params, &NoCancellation)
    }

    fn request_with_cancellation(
        &self,
        method: &str,
        params: Value,
        cancellation: &dyn CancellationSignal,
    ) -> Result<Value, RuntimeError> {
        let _guard = loop {
            if cancellation.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            match self.request_lock.try_lock() {
                Ok(guard) => break guard,
                Err(TryLockError::WouldBlock) => thread::sleep(CANCELLATION_POLL),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(RuntimeError::Mcp("MCP request lock was poisoned".into()));
                }
            }
        };
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        }))?;
        self.receive_response(id, cancellation)
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), RuntimeError> {
        let _guard = self
            .request_lock
            .lock()
            .map_err(|_| RuntimeError::Mcp("MCP request lock was poisoned".into()))?;
        self.write_message(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }))
    }

    fn write_message(&self, message: &Value) -> Result<(), RuntimeError> {
        let encoded = serde_json::to_vec(message)
            .map_err(|error| self.protocol_error(format!("could not encode request: {error}")))?;
        let mut stdin = self
            .stdin
            .lock()
            .map_err(|_| RuntimeError::Mcp("MCP stdin lock was poisoned".into()))?;
        let stdin = stdin
            .as_mut()
            .ok_or_else(|| self.protocol_error("MCP stdin is closed"))?;
        stdin
            .write_all(&encoded)
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .map_err(|error| self.protocol_error(format!("failed to write request: {error}")))
    }

    fn receive_response(
        &self,
        expected_id: u64,
        cancellation: &dyn CancellationSignal,
    ) -> Result<Value, RuntimeError> {
        let deadline = Instant::now() + self.timeout;
        loop {
            if cancellation.is_cancelled() {
                return Err(self.cancelled());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(self.fatal_protocol_error(format!(
                    "request {expected_id} timed out after {:?}",
                    self.timeout
                )));
            }
            let event = self
                .responses
                .lock()
                .map_err(|_| RuntimeError::Mcp("MCP response lock was poisoned".into()))?
                .recv_timeout(remaining.min(CANCELLATION_POLL));
            let line = match event {
                Ok(StdoutEvent::Line(line)) => line,
                Ok(StdoutEvent::Eof) | Err(RecvTimeoutError::Disconnected) => {
                    return Err(self.protocol_error("unexpected EOF while awaiting response"));
                }
                Ok(StdoutEvent::Error(error)) => {
                    return Err(self.protocol_error(format!("stdout read failed: {error}")));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if Instant::now() >= deadline {
                        return Err(self.fatal_protocol_error(format!(
                            "request {expected_id} timed out after {:?}",
                            self.timeout
                        )));
                    }
                    continue;
                }
            };
            let response: Value = serde_json::from_str(&line).map_err(|error| {
                self.fatal_protocol_error(format!("malformed JSON response: {error}"))
            })?;
            if response.get("jsonrpc") != Some(&Value::String("2.0".into())) {
                return Err(self.fatal_protocol_error(format!(
                    "invalid JSON-RPC version: {}",
                    response.get("jsonrpc").unwrap_or(&Value::Null)
                )));
            }
            if response.get("method").is_some() {
                if let Some(id) = response.get("id").cloned() {
                    self.write_message(&json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32601, "message": "client method not supported"}
                    }))?;
                }
                continue;
            }
            if response.get("id") != Some(&json!(expected_id)) {
                return Err(self.fatal_protocol_error(format!(
                    "response id did not match request {expected_id}: {}",
                    response.get("id").unwrap_or(&Value::Null)
                )));
            }
            let result = response.get("result");
            let error = response.get("error");
            if result.is_some() == error.is_some() {
                return Err(self.fatal_protocol_error(
                    "JSON-RPC response must contain exactly one of result or error",
                ));
            }
            if let Some(error) = error {
                let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
                let message = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown JSON-RPC error");
                return Err(self.protocol_error(format!("RPC error {code}: {message}")));
            }
            return Ok(result.cloned().unwrap_or(Value::Null));
        }
    }

    fn protocol_error(&self, message: impl Into<String>) -> RuntimeError {
        let mut message = message.into();
        let stderr = self
            .stderr
            .lock()
            .map(|buffer| buffer.snapshot())
            .unwrap_or_default();
        if !stderr.trim().is_empty() {
            message.push_str("; stderr tail: ");
            message.push_str(stderr.trim());
        }
        RuntimeError::Mcp(message)
    }

    fn fatal_protocol_error(&self, message: impl Into<String>) -> RuntimeError {
        let error = self.protocol_error(message);
        self.terminate();
        error
    }

    fn cancelled(&self) -> RuntimeError {
        self.terminate();
        RuntimeError::Cancelled
    }

    fn terminate(&self) {
        if let Ok(mut stdin) = self.stdin.lock() {
            stdin.take();
        }
        if let Ok(mut child) = self.child.lock()
            && let Some(mut child) = child.take()
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn read_bounded_line(reader: &mut impl BufRead, limit: usize) -> io::Result<Option<String>> {
    let mut bytes = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if bytes.is_empty() {
                Ok(None)
            } else {
                Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
            };
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |index| index + 1);
        if bytes.len().saturating_add(consumed) > limit {
            reader.consume(consumed);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("MCP response exceeded the {limit} byte limit"),
            ));
        }
        bytes.extend_from_slice(&available[..consumed]);
        reader.consume(consumed);
        if newline.is_some() {
            while bytes
                .last()
                .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
            {
                bytes.pop();
            }
            return Ok(Some(String::from_utf8_lossy(&bytes).into_owned()));
        }
    }
}

impl Drop for McpConnection {
    fn drop(&mut self) {
        self.terminate();
        if let Ok(thread) = self.stdout_thread.get_mut()
            && let Some(thread) = thread.take()
        {
            let _ = thread.join();
        }
        if let Ok(thread) = self.stderr_thread.get_mut()
            && let Some(thread) = thread.take()
        {
            let _ = thread.join();
        }
    }
}

struct BoundedBuffer {
    bytes: VecDeque<u8>,
    limit: usize,
}

impl BoundedBuffer {
    fn new(limit: usize) -> Self {
        Self {
            bytes: VecDeque::with_capacity(limit),
            limit,
        }
    }

    fn extend(&mut self, bytes: &[u8]) {
        if self.limit == 0 {
            return;
        }
        self.bytes.extend(bytes.iter().copied());
        while self.bytes.len() > self.limit {
            self.bytes.pop_front();
        }
    }

    fn snapshot(&self) -> String {
        let bytes = self.bytes.iter().copied().collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
        process::Command,
        thread,
        time::{Duration, Instant},
    };

    use uuid::Uuid;

    use super::McpServerConfig;

    pub(crate) struct FakeMcpServer {
        pub(crate) root: PathBuf,
        pub(crate) log: PathBuf,
        pid: PathBuf,
        command: String,
        args: Vec<String>,
    }

    impl FakeMcpServer {
        pub(crate) fn new(scenario: &str) -> Self {
            let root = std::env::temp_dir().join(format!("fake-mcp-{}", Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            let log = root.join("requests.log");
            let pid = root.join("server.pid");

            #[cfg(windows)]
            let (command, args) = {
                let script = root.join("server.ps1");
                std::fs::write(&script, WINDOWS_SERVER).unwrap();
                (
                    "powershell.exe".to_owned(),
                    vec![
                        "-NoLogo".into(),
                        "-NoProfile".into(),
                        "-NonInteractive".into(),
                        "-ExecutionPolicy".into(),
                        "Bypass".into(),
                        "-File".into(),
                        script.display().to_string(),
                    ],
                )
            };

            #[cfg(not(windows))]
            let (command, args) = {
                use std::os::unix::fs::PermissionsExt;
                let script = root.join("server.sh");
                std::fs::write(&script, UNIX_SERVER).unwrap();
                let mut permissions = std::fs::metadata(&script).unwrap().permissions();
                permissions.set_mode(0o755);
                std::fs::set_permissions(&script, permissions).unwrap();
                ("/bin/sh".to_owned(), vec![script.display().to_string()])
            };

            let server = Self {
                root,
                log,
                pid,
                command,
                args,
            };
            std::fs::write(server.root.join("scenario"), scenario).unwrap();
            server
        }

        pub(crate) fn config(&self) -> McpServerConfig {
            let mut env = BTreeMap::new();
            let scenario = std::fs::read_to_string(self.root.join("scenario")).unwrap();
            env.insert("FAKE_MCP_LOG".into(), self.log.display().to_string());
            env.insert("FAKE_MCP_PID".into(), self.pid.display().to_string());
            env.insert("FAKE_MCP_SCENARIO".into(), scenario.clone());
            let timeout = if scenario == "timeout" {
                Duration::from_secs(5)
            } else {
                // Align the test fixture with the production default request
                // timeout so slow CI hosts (PowerShell cold start, spawn
                // contention) do not spuriously fail the handshake.
                Duration::from_secs(30)
            };
            let mut config = McpServerConfig::new("Fixture Server", &self.command)
                .with_args(self.args.clone())
                .with_env(env)
                .with_timeout(timeout);
            config.stderr_limit = 4_096;
            config
        }

        pub(crate) fn process_id(&self) -> u32 {
            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                if let Ok(value) = std::fs::read_to_string(&self.pid)
                    && let Ok(process_id) = value.trim().parse()
                {
                    return process_id;
                }
                assert!(
                    Instant::now() < deadline,
                    "fake MCP server did not publish its PID"
                );
                thread::sleep(Duration::from_millis(10));
            }
        }

        pub(crate) fn methods(&self) -> Vec<String> {
            std::fs::read_to_string(&self.log)
                .unwrap_or_default()
                .lines()
                .map(str::to_owned)
                .collect()
        }
    }

    #[cfg(windows)]
    pub(crate) fn process_exists(process_id: u32) -> bool {
        Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                &format!(
                    "if (Get-Process -Id {process_id} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}"
                ),
            ])
            .status()
            .is_ok_and(|status| status.success())
    }

    #[cfg(not(windows))]
    pub(crate) fn process_exists(process_id: u32) -> bool {
        Command::new("kill")
            .args(["-0", &process_id.to_string()])
            .status()
            .is_ok_and(|status| status.success())
    }

    pub(crate) fn wait_for_process_exit(process_id: u32, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while process_exists(process_id) {
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
        true
    }

    pub(crate) fn wait_for_logged_method(log: &Path, method: &str, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if std::fs::read_to_string(log)
                .unwrap_or_default()
                .lines()
                .any(|logged| logged == method)
            {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    impl Drop for FakeMcpServer {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[cfg(windows)]
    const WINDOWS_SERVER: &str = r#"
$scenario = $env:FAKE_MCP_SCENARIO
$log = $env:FAKE_MCP_LOG
$PID | Set-Content -LiteralPath $env:FAKE_MCP_PID
while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) {
        if ($scenario -eq 'ignore_eof') { Start-Sleep -Seconds 30 }
        break
    }
    $request = $line | ConvertFrom-Json
    Add-Content -LiteralPath $log -Value $request.method
    if ($request.method -eq 'notifications/initialized') { continue }
    if ($request.method -eq 'initialize') {
        if ($scenario -eq 'timeout') { Start-Sleep -Seconds 10 }
        if ($scenario -eq 'eof') { exit 0 }
        if ($scenario -eq 'malformed') { [Console]::Out.WriteLine('{not-json'); continue }
        if ($scenario -eq 'oversized') { [Console]::Out.WriteLine(('x' * 2048)); continue }
        if ($scenario -eq 'stderr') {
            [Console]::Error.WriteLine(('x' * 10000) + 'TAIL_MARKER')
            [Console]::Error.Flush()
            Start-Sleep -Milliseconds 100
            [Console]::Out.WriteLine('{not-json')
            continue
        }
        $id = $request.id
        if ($scenario -eq 'wrong_id') { $id = 999 }
        if ($scenario -eq 'rpc_error') {
            $response = @{jsonrpc='2.0'; id=$id; error=@{code=-32001; message='fixture rpc error'}}
        } else {
            $response = @{jsonrpc='2.0'; id=$id; result=@{protocolVersion='2025-06-18'; capabilities=@{}; serverInfo=@{name='fixture'; version='1'}}}
        }
        if ($scenario -eq 'protocol_mismatch') { $response.result.protocolVersion = '1900-01-01' }
        if ($scenario -eq 'legacy_protocol') { $response.result.protocolVersion = '2025-03-26' }
        if ($scenario -eq 'wrong_jsonrpc') { $response.jsonrpc = '1.0' }
        [Console]::Out.WriteLine(($response | ConvertTo-Json -Compress -Depth 10))
        continue
    }
    if ($request.method -eq 'tools/list') {
        $cursor = $request.params.cursor
        if ($null -eq $cursor) {
            $response = @{jsonrpc='2.0'; id=$request.id; result=@{tools=@(@{name='Echo Tool'; description='echo'; inputSchema=@{type='object'}}); nextCursor='page-2'}}
        } elseif ($scenario -eq 'cursor_loop') {
            $response = @{jsonrpc='2.0'; id=$request.id; result=@{tools=@(); nextCursor='page-2'}}
        } else {
            $response = @{jsonrpc='2.0'; id=$request.id; result=@{tools=@(@{name='Danger/Tool'; description='second'; inputSchema=@{type='object'}})}}
        }
        [Console]::Out.WriteLine(($response | ConvertTo-Json -Compress -Depth 10))
        continue
    }
    if ($request.method -eq 'tools/call') {
        if ($scenario -eq 'slow_call') { Start-Sleep -Seconds 10 }
        $response = @{jsonrpc='2.0'; id=$request.id; result=@{content=@(@{type='text'; text='tool-result'}); isError=$false}}
        [Console]::Out.WriteLine(($response | ConvertTo-Json -Compress -Depth 10))
    }
}
"#;

    #[cfg(not(windows))]
    const UNIX_SERVER: &str = r#"#!/bin/sh
scenario="$FAKE_MCP_SCENARIO"
printf '%s\n' "$$" > "$FAKE_MCP_PID"
while IFS= read -r line; do
  method=$(printf '%s' "$line" | sed -n 's/.*"method":"\([^"]*\)".*/\1/p')
  printf '%s\n' "$method" >> "$FAKE_MCP_LOG"
  [ "$method" = "notifications/initialized" ] && continue
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  if [ "$method" = "initialize" ]; then
    [ "$scenario" = "timeout" ] && sleep 10
    [ "$scenario" = "eof" ] && exit 0
    [ "$scenario" = "malformed" ] && { printf '{not-json\n'; continue; }
    [ "$scenario" = "oversized" ] && { head -c 2048 /dev/zero | tr '\0' x; printf '\n'; continue; }
    [ "$scenario" = "stderr" ] && { head -c 10000 /dev/zero | tr '\0' x >&2; printf 'TAIL_MARKER\n' >&2; sleep 0.1; printf '{not-json\n'; continue; }
    [ "$scenario" = "wrong_id" ] && id=999
    [ "$scenario" = "rpc_error" ] && { printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32001,"message":"fixture rpc error"}}\n' "$id"; continue; }
    [ "$scenario" = "protocol_mismatch" ] && { printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"1900-01-01","capabilities":{},"serverInfo":{"name":"fixture","version":"1"}}}\n' "$id"; continue; }
    [ "$scenario" = "legacy_protocol" ] && { printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"fixture","version":"1"}}}\n' "$id"; continue; }
    [ "$scenario" = "wrong_jsonrpc" ] && { printf '{"jsonrpc":"1.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fixture","version":"1"}}}\n' "$id"; continue; }
    printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fixture","version":"1"}}}\n' "$id"
  elif [ "$method" = "tools/list" ]; then
    if printf '%s' "$line" | grep -q 'page-2'; then
      if [ "$scenario" = "cursor_loop" ]; then
        printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[],"nextCursor":"page-2"}}\n' "$id"
      else
        printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"Danger/Tool","description":"second","inputSchema":{"type":"object"}}]}}\n' "$id"
      fi
    else
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"Echo Tool","description":"echo","inputSchema":{"type":"object"}}],"nextCursor":"page-2"}}\n' "$id"
    fi
  elif [ "$method" = "tools/call" ]; then
    [ "$scenario" = "slow_call" ] && sleep 10
    printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"tool-result"}],"isError":false}}\n' "$id"
  fi
done
[ "$scenario" = "ignore_eof" ] && sleep 30
"#;
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    use focus_kernel::CancellationSignal;
    use serde_json::json;

    use crate::policy::ToolOperation;

    use super::{
        McpClient, McpServerConfig, canonical_tool_name, render_tool_content,
        test_support::{FakeMcpServer, process_exists, wait_for_process_exit},
    };

    struct CancelNow;

    impl CancellationSignal for CancelNow {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    #[derive(Clone, Default)]
    struct ObservedCancellation {
        cancelled: Arc<AtomicBool>,
        checks: Arc<AtomicUsize>,
    }

    impl ObservedCancellation {
        fn cancel(&self) {
            self.cancelled.store(true, Ordering::Release);
        }
    }

    impl CancellationSignal for ObservedCancellation {
        fn is_cancelled(&self) -> bool {
            self.checks.fetch_add(1, Ordering::AcqRel);
            self.cancelled.load(Ordering::Acquire)
        }
    }

    #[test]
    fn server_operation_defaults_to_other_and_can_be_overridden() {
        let default = McpServerConfig::new("fixture", "fixture-command");
        let network = default.clone().with_operation(ToolOperation::Network);

        assert_eq!(default.operation, ToolOperation::Other);
        assert_eq!(network.operation, ToolOperation::Network);
    }

    #[test]
    fn cancellation_before_send_preserves_the_shared_server() {
        let fixture = FakeMcpServer::new("ok");
        let (client, _) = McpClient::connect(&fixture.config()).unwrap();
        let process_id = fixture.process_id();

        let error = client
            .call_tool_blocking("Echo Tool", json!({}), &CancelNow)
            .unwrap_err();

        assert!(matches!(error, crate::RuntimeError::Cancelled));
        assert!(process_exists(process_id));
        assert_eq!(
            client
                .call_tool_blocking("Echo Tool", json!({}), &focus_kernel::NoCancellation)
                .unwrap(),
            "tool-result"
        );
    }

    #[test]
    fn cancellation_while_waiting_for_the_request_lock_preserves_the_shared_server() {
        let fixture = FakeMcpServer::new("ok");
        let (client, _) = McpClient::connect(&fixture.config()).unwrap();
        let process_id = fixture.process_id();
        let request_guard = client.connection.request_lock.lock().unwrap();
        let cancellation = ObservedCancellation::default();
        let worker_cancellation = cancellation.clone();
        let worker_client = client.clone();
        let worker = thread::spawn(move || {
            worker_client.call_tool_blocking("Echo Tool", json!({}), &worker_cancellation)
        });
        let deadline = Instant::now() + Duration::from_secs(1);
        while cancellation.checks.load(Ordering::Acquire) < 2 {
            assert!(
                Instant::now() < deadline,
                "request did not enter lock polling"
            );
            thread::sleep(Duration::from_millis(5));
        }
        cancellation.cancel();

        let error = worker.join().unwrap().unwrap_err();

        assert!(matches!(error, crate::RuntimeError::Cancelled));
        assert!(process_exists(process_id));
        drop(request_guard);
        assert_eq!(
            client
                .call_tool_blocking("Echo Tool", json!({}), &focus_kernel::NoCancellation)
                .unwrap(),
            "tool-result"
        );
    }

    #[test]
    fn canonical_names_are_stable_openai_safe_and_bounded() {
        let name = canonical_tool_name(
            "Server with spaces and punctuation!!!",
            "A very long remote/tool name that would otherwise exceed the OpenAI limit by a lot",
        );

        assert_eq!(
            name,
            canonical_tool_name(
                "Server with spaces and punctuation!!!",
                "A very long remote/tool name that would otherwise exceed the OpenAI limit by a lot"
            )
        );
        assert!(name.len() <= 64);
        assert!(
            name.bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        );
        assert_ne!(
            canonical_tool_name("a b", "tool"),
            canonical_tool_name("a@b", "tool")
        );
    }

    #[tokio::test]
    async fn persistent_transport_initializes_pages_and_calls_in_order() {
        let fixture = FakeMcpServer::new("ok");
        let (client, tools) = McpClient::connect(&fixture.config()).unwrap();

        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Echo Tool", "Danger/Tool"]
        );
        assert_eq!(
            client
                .call_tool_async(
                    "Echo Tool",
                    json!({"value": 7}),
                    &focus_kernel::NoCancellation
                )
                .await
                .unwrap(),
            "tool-result"
        );
        assert_eq!(
            fixture.methods(),
            vec![
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/list",
                "tools/call"
            ]
        );
    }

    #[test]
    fn tool_call_timeout_terminates_server_and_closes_the_connection() {
        let fixture = FakeMcpServer::new("slow_call");
        let (mut client, _) = McpClient::connect(&fixture.config()).unwrap();
        let connection = &mut Arc::get_mut(&mut client).unwrap().connection;
        Arc::get_mut(connection).unwrap().timeout = Duration::from_secs(1);
        let process_id = fixture.process_id();

        let timeout_error = client
            .call_tool_blocking("Echo Tool", json!({}), &focus_kernel::NoCancellation)
            .unwrap_err()
            .to_string();

        assert!(timeout_error.contains("timed out"), "{timeout_error}");
        assert!(timeout_error.len() < 512, "{timeout_error}");
        assert!(wait_for_process_exit(process_id, Duration::from_secs(1)));

        let retry_started = Instant::now();
        let retry_error = client
            .call_tool_blocking("Echo Tool", json!({}), &focus_kernel::NoCancellation)
            .unwrap_err()
            .to_string();

        assert!(retry_started.elapsed() < Duration::from_millis(500));
        assert!(retry_error.contains("stdin is closed"), "{retry_error}");
        assert!(retry_error.len() < 512, "{retry_error}");
    }

    #[test]
    fn transport_reports_timeout_eof_malformed_wrong_id_rpc_error_and_cursor_loop() {
        let cases = [
            ("timeout", "timed out"),
            ("eof", "EOF"),
            ("malformed", "malformed JSON"),
            ("wrong_id", "response id"),
            ("rpc_error", "fixture rpc error"),
            ("cursor_loop", "cursor loop"),
            ("protocol_mismatch", "protocol version"),
            ("wrong_jsonrpc", "JSON-RPC version"),
        ];
        for (scenario, expected) in cases {
            let fixture = FakeMcpServer::new(scenario);
            let error = McpClient::connect(&fixture.config()).unwrap_err();
            assert!(error.to_string().contains(expected), "{scenario}: {error}");
        }
    }

    #[test]
    fn negotiates_a_supported_legacy_protocol_version() {
        let fixture = FakeMcpServer::new("legacy_protocol");

        let (_client, tools) = McpClient::connect(&fixture.config()).unwrap();

        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn structured_content_is_retained_alongside_text_blocks() {
        let rendered = render_tool_content(
            &[json!({"type":"text","text":"summary"})],
            Some(&json!({"count": 3})),
        );

        assert_eq!(rendered, "summary\n{\"count\":3}");
    }

    #[test]
    fn oversized_response_lines_are_rejected_before_json_parsing() {
        let fixture = FakeMcpServer::new("oversized");
        let mut config = fixture.config();
        config.response_limit = 1_024;

        let error = McpClient::connect(&config).unwrap_err().to_string();

        assert!(error.contains("exceeded the 1024 byte limit"), "{error}");
    }

    #[test]
    fn stderr_diagnostics_are_tail_bounded() {
        let fixture = FakeMcpServer::new("stderr");
        let error = McpClient::connect(&fixture.config())
            .unwrap_err()
            .to_string();

        assert!(error.contains("TAIL_MARKER"));
        assert!(error.len() < 5_000, "diagnostic was {} bytes", error.len());
    }

    #[test]
    fn dropping_the_last_client_kills_and_waits_for_the_server() {
        let fixture = FakeMcpServer::new("ignore_eof");
        let (client, _) = McpClient::connect(&fixture.config()).unwrap();
        let process_id = client
            .connection
            .child
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .id();

        drop(client);

        assert!(wait_for_process_exit(process_id, Duration::from_secs(1)));
    }
}
