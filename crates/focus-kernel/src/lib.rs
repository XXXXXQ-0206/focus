//! Small, provider-agnostic primitives for an agent turn loop.
//!
//! This crate intentionally contains no filesystem policy, coding workflow,
//! project memory, subagent orchestration, or user interface concerns. Those
//! live in `focus-runtime` and consume these stable primitives.

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

thread_local! {
    static SYNC_RUNTIME: RefCell<Option<tokio::runtime::Runtime>> = const { RefCell::new(None) };
}

use async_trait::async_trait;
use futures::{Stream, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

/// Error returned by kernel abstractions.
#[derive(Debug, Error)]
pub enum KernelError {
    /// The embedding Runtime requested cooperative cancellation.
    #[error("agent loop was cancelled")]
    Cancelled,
    /// A model provider failed to return a valid response.
    #[error("model provider failed: {0}")]
    Model(String),
    /// A requested tool is not registered.
    #[error("tool `{0}` is not registered")]
    UnknownTool(String),
    /// A tool could not complete.
    #[error("tool `{name}` failed: {message}")]
    Tool {
        /// Tool name.
        name: String,
        /// Error detail.
        message: String,
    },
    /// An event store could not be read or written.
    #[error("event store error: {0}")]
    Store(String),
    /// The loop exceeded its configured number of turns.
    #[error("agent loop reached its turn limit ({0})")]
    TurnLimit(usize),
}

/// A role in the model-visible transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Runtime instructions.
    System,
    /// User request.
    User,
    /// Model response.
    Assistant,
    /// Tool output.
    Tool,
}

/// One normalized transcript item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Sender role.
    pub role: Role,
    /// Plain text content. Provider-specific rich blocks belong in provider adapters.
    pub content: String,
    /// Present for a result matching a model tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Assistant tool calls retained across session resume.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Tool name for a result message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Whether a tool result represents a failed execution.
    #[serde(default)]
    pub is_error: bool,
}

impl Message {
    /// Create a normal text message.
    #[must_use]
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
            tool_name: None,
            is_error: false,
        }
    }

    /// Create a tool result transcript message.
    #[must_use]
    pub fn tool_result(
        id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_call_id: Some(id.into()),
            tool_calls: Vec::new(),
            tool_name: Some(name.into()),
            is_error: false,
        }
    }

    /// Create a failed tool result transcript message.
    #[must_use]
    pub fn tool_error(
        id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        let mut message = Self::tool_result(id, name, content);
        message.is_error = true;
        message
    }

    /// Create an assistant message containing text and structured tool calls.
    #[must_use]
    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_call_id: None,
            tool_calls,
            tool_name: None,
            is_error: false,
        }
    }
}

/// A model-initiated invocation of a named tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Stable identifier supplied by the model adapter.
    pub id: String,
    /// Registered tool name.
    pub name: String,
    /// JSON object supplied to the tool.
    pub arguments: Value,
}

impl ToolCall {
    /// Return the representation that may be persisted, replayed, or sent back
    /// to a provider on a later turn.
    ///
    /// The executor receives the original call. A `web_fetch` URL can carry a
    /// signed query or OAuth callback code, neither of which belongs in the
    /// durable transcript or the next model request.
    #[must_use]
    pub fn redacted_for_persistence(&self) -> Self {
        let mut call = self.clone();
        redact_json_value(&mut call.arguments);
        call
    }
}

/// The normalized result of a tool invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID from the matching [`ToolCall`].
    pub tool_call_id: String,
    /// Tool name.
    pub name: String,
    /// Content returned to the model.
    pub content: String,
    /// Whether the tool execution completed successfully.
    pub is_error: bool,
}

/// Static tool metadata supplied to a model provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Canonical tool name.
    pub name: String,
    /// Human-readable contract.
    pub description: String,
    /// JSON Schema object for tool arguments.
    pub input_schema: Value,
}

/// Provider-neutral model request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Model-visible messages.
    pub messages: Vec<Message>,
    /// Tools available for this turn.
    pub tools: Vec<ToolDefinition>,
}

/// Provider-neutral model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    /// Assistant text, including its final answer when no calls remain.
    pub content: String,
    /// Tool calls selected by the model.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

impl ModelResponse {
    /// Return the response shape suitable for persistence and event streaming.
    #[must_use]
    pub fn redacted_for_persistence(&self) -> Self {
        Self {
            content: redact_sensitive_text(&self.content),
            tool_calls: self
                .tool_calls
                .iter()
                .map(ToolCall::redacted_for_persistence)
                .collect(),
        }
    }
}

fn redact_url(url: &str) -> String {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    let mut rendered = url[..end].to_owned();
    if let Some(scheme_end) = rendered.find("://") {
        let authority_start = scheme_end + 3;
        let path_start = rendered[authority_start..]
            .find('/')
            .map_or(rendered.len(), |offset| authority_start + offset);
        if let Some(at_offset) = rendered[authority_start..path_start].rfind('@') {
            let at = authority_start + at_offset;
            rendered.replace_range(authority_start..=at, "<redacted>@");
        }
    }
    if end < url.len() {
        rendered.push(url.as_bytes()[end] as char);
        rendered.push_str("<redacted>");
    }
    rendered
}

fn redact_urls_in_text(text: &str) -> String {
    let mut rendered = String::with_capacity(text.len());
    let mut remaining = text;
    loop {
        let http = remaining.find("http://");
        let https = remaining.find("https://");
        let start = match (http, https) {
            (Some(http), Some(https)) => http.min(https),
            (Some(start), None) | (None, Some(start)) => start,
            (None, None) => {
                rendered.push_str(remaining);
                return rendered;
            }
        };
        rendered.push_str(&remaining[..start]);
        let url = &remaining[start..];
        let end = url
            .find(|character: char| {
                character.is_whitespace()
                    || matches!(character, '"' | '\'' | '<' | '>' | ')' | ']' | '}')
            })
            .unwrap_or(url.len());
        rendered.push_str(&redact_url(&url[..end]));
        remaining = &url[end..];
    }
}

fn redacted_tool_result(mut result: ToolResult, web_fetch_url: Option<&str>) -> ToolResult {
    if result.name == "web_fetch" {
        result.content = redact_urls_in_text(&result.content);
        if let Some(url) = web_fetch_url {
            result.content = redact_echoed_query(&result.content, url);
        }
    }
    result.content = redact_sensitive_text(&result.content);
    result
}

/// Redact common credentials before text crosses a durable or visible boundary.
#[must_use]
pub fn redact_sensitive_text(text: &str) -> String {
    let text = redact_private_key_blocks(text);
    let text = redact_urls_in_text(&text);
    let text = redact_authorization_values(&text);
    redact_key_value_secrets(&text)
}

fn redact_json_value(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(redact_json_value),
        Value::Object(values) => {
            for (key, value) in values {
                if is_secret_name(key) {
                    *value = Value::String("<redacted>".into());
                } else {
                    redact_json_value(value);
                }
            }
        }
        Value::String(text) => *text = redact_sensitive_text(text),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn is_secret_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase().replace('-', "_");
    matches!(
        name.as_str(),
        "api_key"
            | "apikey"
            | "access_key"
            | "authorization"
            | "credential"
            | "credentials"
            | "cookie"
            | "password"
            | "private_key"
            | "secret"
            | "token"
    ) || name.ends_with("_api_key")
        || name.ends_with("_password")
        || name.ends_with("_secret")
        || name.ends_with("_token")
}

fn redact_private_key_blocks(text: &str) -> String {
    const BEGIN: &str = "-----BEGIN ";
    const END: &str = "-----END ";
    let mut rendered = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find(BEGIN) {
        rendered.push_str(&remaining[..start]);
        let block = &remaining[start..];
        let Some(header_end) = block[BEGIN.len()..]
            .find("-----")
            .map(|offset| BEGIN.len() + offset)
        else {
            rendered.push_str(block);
            return rendered;
        };
        if !block[..header_end].contains("PRIVATE KEY") {
            let next = header_end + "-----".len();
            rendered.push_str(&block[..next]);
            remaining = &block[next..];
            continue;
        }
        let end = block
            .find(END)
            .and_then(|offset| block[offset..].find("-----").map(|end| offset + end + 5))
            .unwrap_or(block.len());
        rendered.push_str("<redacted private key>");
        remaining = &block[end..];
    }
    rendered.push_str(remaining);
    rendered
}

fn redact_authorization_values(text: &str) -> String {
    const MARKER: &str = "authorization:";
    let mut rendered = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = find_ascii_case_insensitive(remaining, MARKER) {
        rendered.push_str(&remaining[..start]);
        let value = &remaining[start + MARKER.len()..];
        let end = value.find(['\r', '\n', '\'', '"']).unwrap_or(value.len());
        rendered.push_str("Authorization: <redacted>");
        remaining = &value[end..];
    }
    rendered.push_str(remaining);
    rendered
}

fn redact_key_value_secrets(text: &str) -> String {
    const NAMES: &[&str] = &["access_token", "api_key", "password", "secret", "token"];
    let mut rendered = text.to_owned();
    for name in NAMES {
        let mut search_from = 0;
        while let Some(relative) = find_ascii_case_insensitive(&rendered[search_from..], name) {
            let start = search_from + relative;
            let end_of_name = start + name.len();
            let before_is_identifier = start > 0
                && rendered.as_bytes()[start - 1].is_ascii_alphanumeric()
                || (start > 0 && rendered.as_bytes()[start - 1] == b'_');
            if before_is_identifier {
                search_from = end_of_name;
                continue;
            }
            let mut separator = end_of_name;
            while rendered
                .as_bytes()
                .get(separator)
                .is_some_and(u8::is_ascii_whitespace)
            {
                separator += 1;
            }
            if !matches!(rendered.as_bytes().get(separator), Some(b'=' | b':')) {
                search_from = end_of_name;
                continue;
            }
            let mut value_start = separator + 1;
            while rendered
                .as_bytes()
                .get(value_start)
                .is_some_and(u8::is_ascii_whitespace)
            {
                value_start += 1;
            }
            let value_end = rendered[value_start..]
                .find(|character: char| {
                    character.is_whitespace()
                        || matches!(character, '&' | ',' | ';' | '\'' | '"' | ')' | ']' | '}')
                })
                .map_or(rendered.len(), |offset| value_start + offset);
            if value_start == value_end {
                search_from = value_start;
                continue;
            }
            rendered.replace_range(value_start..value_end, "<redacted>");
            search_from = value_start + "<redacted>".len();
        }
    }
    rendered
}

fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    haystack.to_ascii_lowercase().find(needle)
}

fn redact_echoed_query(content: &str, url: &str) -> String {
    let Some((_, query_and_fragment)) = url.split_once('?') else {
        return content.to_owned();
    };
    let query = query_and_fragment
        .split_once('#')
        .map_or(query_and_fragment, |(query, _)| query);
    if query.is_empty() {
        content.to_owned()
    } else {
        content.replace(query, "<redacted web_fetch query>")
    }
}

fn redact_model_event(event: ModelEvent, tool_names: &mut HashMap<String, String>) -> ModelEvent {
    match event {
        ModelEvent::TextDelta { text } => ModelEvent::TextDelta {
            text: redact_sensitive_text(&text),
        },
        ModelEvent::ReasoningSummaryDelta { text } => ModelEvent::ReasoningSummaryDelta {
            text: redact_sensitive_text(&text),
        },
        ModelEvent::ToolCallStarted { id, name } => {
            tool_names.insert(id.clone(), name.clone());
            ModelEvent::ToolCallStarted { id, name }
        }
        ModelEvent::ToolArgumentsDelta {
            id,
            json_fragment: _,
        } => {
            if tool_names.get(&id).is_none_or(|name| name == "web_fetch") {
                ModelEvent::ToolArgumentsDelta {
                    id,
                    json_fragment: "<redacted web_fetch arguments>".into(),
                }
            } else {
                ModelEvent::ToolArgumentsDelta {
                    id,
                    json_fragment: "<redacted tool arguments>".into(),
                }
            }
        }
        ModelEvent::ToolCallReady { call } => {
            tool_names.insert(call.id.clone(), call.name.clone());
            ModelEvent::ToolCallReady {
                call: call.redacted_for_persistence(),
            }
        }
        ModelEvent::Completed { response } => ModelEvent::Completed {
            response: response.redacted_for_persistence(),
        },
        ModelEvent::RetryScheduled {
            attempt,
            delay_ms,
            reason,
        } => ModelEvent::RetryScheduled {
            attempt,
            delay_ms,
            reason: redact_sensitive_text(&reason),
        },
        ModelEvent::Failed { error } => ModelEvent::Failed {
            error: redact_sensitive_text(&error),
        },
        event => event,
    }
}

/// A provider-neutral lifecycle fact emitted while a model request is in progress.
///
/// Provider-private reasoning is intentionally excluded. A provider may expose a separately
/// labeled, user-visible reasoning summary for hosts that support it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelEvent {
    /// The provider accepted a request and began producing a response.
    RequestStarted {
        /// Provider adapter name.
        provider: String,
        /// Requested model identifier.
        model: String,
        /// Redacted request endpoint.
        endpoint: String,
    },
    /// Visible assistant text produced incrementally.
    TextDelta {
        /// Newly available visible text.
        text: String,
    },
    /// A provider-supplied, user-visible reasoning summary increment.
    ///
    /// This never represents hidden chain-of-thought content. Hosts may display it as a
    /// transient Thinking status and retain the completed summary in their local transcript.
    ReasoningSummaryDelta {
        /// Newly available summary text.
        text: String,
    },
    /// A model tool call became identifiable.
    ToolCallStarted {
        /// Provider-assigned call identifier.
        id: String,
        /// Registered Focus tool name.
        name: String,
    },
    /// An incremental JSON fragment for a visible tool call.
    ToolArgumentsDelta {
        /// Provider-assigned call identifier.
        id: String,
        /// Newly available JSON source fragment.
        json_fragment: String,
    },
    /// A tool call has complete, validated arguments.
    ToolCallReady {
        /// Canonical model tool call.
        call: ToolCall,
    },
    /// Token usage reported by the provider.
    Usage {
        /// Input token count, if reported by the provider.
        input_tokens: u64,
        /// Output token count, if reported by the provider.
        output_tokens: u64,
    },
    /// The adapter will retry after a transient provider failure.
    RetryScheduled {
        /// One-based retry attempt number.
        attempt: u32,
        /// Backoff before retrying.
        delay_ms: u64,
        /// Redacted error reason.
        reason: String,
    },
    /// The provider completed with the canonical final response.
    Completed {
        /// Final normalized response used by Pi.
        response: ModelResponse,
    },
    /// The provider stopped with a redacted error.
    Failed {
        /// Diagnostic safe for event persistence.
        error: String,
    },
    /// The shared Runtime cancellation source stopped the request.
    Cancelled,
}

/// A cancellable provider stream of normalized lifecycle facts.
pub type ModelEventStream = Pin<Box<dyn Stream<Item = Result<ModelEvent, KernelError>> + Send>>;

/// Abstraction over a model endpoint or local model adapter.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Start one request and yield lifecycle facts as soon as they are available.
    async fn stream(
        &self,
        request: ModelRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ModelEventStream, KernelError>;
}

/// Owned cancellation flag used when work must cross into a blocking worker.
#[derive(Debug, Clone, Default)]
pub struct CancellationBridge(Arc<AtomicBool>);

impl CancellationBridge {
    /// Create a bridge, reusing a source flag when the source exposes one.
    #[must_use]
    pub fn from_signal(source: &dyn CancellationSignal) -> Self {
        Self(
            source
                .shared_flag()
                .unwrap_or_else(|| Arc::new(AtomicBool::new(source.is_cancelled()))),
        )
    }

    /// Request cancellation on the owned bridge.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

impl CancellationSignal for CancellationBridge {
    fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }

    fn shared_flag(&self) -> Option<Arc<AtomicBool>> {
        Some(self.0.clone())
    }
}

/// Kernel-level tool interface. Runtime decorators add policy and sandboxing.
pub trait Tool: Send + Sync {
    /// Machine-readable tool definition.
    fn definition(&self) -> ToolDefinition;
    /// Scheduling class for a model-selected call.
    fn execution_class(&self) -> ToolExecutionClass {
        ToolExecutionClass::Parallel
    }
    /// Execute on the caller's Runtime executor while observing cancellation.
    fn call_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, KernelError>>;
}

/// Static scheduler class for one tool invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionClass {
    /// The call may overlap independent calls up to the configured bound.
    Parallel,
    /// The call waits for prior work and blocks later submission until it finishes.
    Exclusive,
}

/// Lifecycle state of an active agent loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// The state has not entered the loop.
    Idle,
    /// Waiting for the provider.
    AwaitingModel,
    /// Executing model-selected tools.
    ExecutingTools,
    /// The model returned a tool-free answer.
    Complete,
    /// The loop exited due to an error.
    Failed,
    /// The runtime requested cancellation.
    Cancelled,
}

/// Mutable state owned by one agent conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Globally unique session identity.
    pub session_id: Uuid,
    /// Monotonic model turn index.
    pub turn: usize,
    /// Current lifecycle status.
    pub status: AgentStatus,
    /// Provider-neutral conversation history.
    pub transcript: Vec<Message>,
}

impl AgentState {
    /// Create an empty conversation state.
    #[must_use]
    pub fn new(session_id: Uuid) -> Self {
        Self {
            session_id,
            turn: 0,
            status: AgentStatus::Idle,
            transcript: Vec::new(),
        }
    }
}

/// A timestamped event emitted by the kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event ID, unique across sessions.
    pub id: Uuid,
    /// Session receiving the event.
    pub session_id: Uuid,
    /// Unix epoch milliseconds.
    pub timestamp_ms: u128,
    /// Event payload.
    pub kind: EventKind,
}

impl Event {
    /// Build an event with the current wall-clock timestamp.
    #[must_use]
    pub fn now(session_id: Uuid, kind: EventKind) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        Self {
            id: Uuid::new_v4(),
            session_id,
            timestamp_ms,
            kind,
        }
    }
}

/// Stable event payloads shared by the runtime and interfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    /// A state transition occurred.
    StateChanged {
        /// New agent-loop status.
        status: AgentStatus,
        /// Monotonic model turn number at the transition.
        turn: usize,
    },
    /// A transcript message was appended.
    MessageAdded {
        /// Message appended to the transcript.
        message: Message,
    },
    /// A tool call was requested by the model.
    ToolCallRequested {
        /// Requested tool invocation.
        call: ToolCall,
    },
    /// A tool call completed.
    ToolResultReceived {
        /// Normalized completion result.
        result: ToolResult,
    },
    /// A provider-neutral model lifecycle fact observed during the active request.
    Model {
        /// The normalized provider event.
        event: ModelEvent,
    },
    /// A runtime extension emitted structured telemetry.
    Runtime {
        /// Stable Runtime event name.
        name: String,
        /// Runtime-specific structured payload.
        data: Value,
    },
}

/// Event consumer used for streaming UIs and persistence.
pub trait EventSink: Send + Sync {
    /// Consume one ordered event.
    fn emit(&self, event: Event) -> Result<(), KernelError>;
}

/// Cooperative cancellation source observed between provider and tool operations.
pub trait CancellationSignal: Send + Sync {
    /// Return true when the loop should stop before starting more work.
    fn is_cancelled(&self) -> bool;

    /// Expose a shared flag for adapters that support in-flight cancellation.
    fn shared_flag(&self) -> Option<Arc<AtomicBool>> {
        None
    }
}

/// A cancellation source for normal foreground runs.
#[derive(Debug, Default)]
pub struct NoCancellation;

impl CancellationSignal for NoCancellation {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Append-only storage for a session event stream.
pub trait EventStore: Send + Sync {
    /// Persist an event atomically with respect to a single append.
    fn append(&self, event: &Event) -> Result<(), KernelError>;
    /// Load the ordered event history for a session.
    fn load(&self, session_id: Uuid) -> Result<Vec<Event>, KernelError>;
}

/// Required persistence strength for one event append.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// Flush bytes and force the file contents to stable storage before returning.
    Immediate,
    /// Flush userspace buffers without a per-event disk cache flush.
    Batch,
}

impl Durability {
    /// Select the minimum persistence strength required by an event payload.
    #[must_use]
    pub fn for_event(event: &Event) -> Self {
        match &event.kind {
            EventKind::Model {
                event:
                    ModelEvent::TextDelta { .. }
                    | ModelEvent::ReasoningSummaryDelta { .. }
                    | ModelEvent::ToolArgumentsDelta { .. },
            } => Self::Batch,
            _ => Self::Immediate,
        }
    }
}

/// A compact JSONL event store, one file per session.
#[derive(Debug, Clone)]
pub struct JsonlEventStore {
    root: PathBuf,
    append_lock: Arc<Mutex<()>>,
}

impl JsonlEventStore {
    /// Create a store rooted at `root`, creating its directory when needed.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, KernelError> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|error| KernelError::Store(error.to_string()))?;
        Ok(Self {
            root,
            append_lock: Arc::new(Mutex::new(())),
        })
    }

    fn session_path(&self, session_id: Uuid) -> PathBuf {
        self.root.join(format!("{session_id}.jsonl"))
    }

    fn session_lock_path(&self, session_id: Uuid) -> PathBuf {
        self.root.join(format!("{session_id}.jsonl.lock"))
    }

    fn lock_session_append(&self, session_id: Uuid) -> Result<File, KernelError> {
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.session_lock_path(session_id))
            .map_err(|error| KernelError::Store(error.to_string()))?;
        lock.lock()
            .map_err(|error| KernelError::Store(error.to_string()))?;
        Ok(lock)
    }
}

impl EventStore for JsonlEventStore {
    fn append(&self, event: &Event) -> Result<(), KernelError> {
        let _guard = self
            .append_lock
            .lock()
            .map_err(|_| KernelError::Store("event append lock was poisoned".into()))?;
        let _session_lock = self.lock_session_append(event.session_id)?;
        let encoded =
            serde_json::to_string(event).map_err(|error| KernelError::Store(error.to_string()))?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.session_path(event.session_id))
            .map_err(|error| KernelError::Store(error.to_string()))?;
        file.write_all(encoded.as_bytes())
            .and_then(|()| file.write_all(b"\n"))
            .map_err(|error| KernelError::Store(error.to_string()))?;
        match Durability::for_event(event) {
            Durability::Immediate => file.sync_all(),
            Durability::Batch => file.flush(),
        }
        .map_err(|error| KernelError::Store(error.to_string()))
    }

    fn load(&self, session_id: Uuid) -> Result<Vec<Event>, KernelError> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let contents =
            std::fs::read_to_string(path).map_err(|error| KernelError::Store(error.to_string()))?;
        let lines = contents
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>();
        let mut events = Vec::with_capacity(lines.len());
        for (index, line) in lines.iter().enumerate() {
            match serde_json::from_str(line) {
                Ok(event) => events.push(event),
                // A process can be interrupted after an append begins. Preserve the durable prefix.
                Err(_) if index + 1 == lines.len() => break,
                Err(error) => return Err(KernelError::Store(error.to_string())),
            }
        }
        Ok(events)
    }
}

/// Configuration for one call to [`AgentLoop::run`].
#[derive(Debug, Clone, Copy)]
pub struct LoopConfig {
    /// Maximum provider calls before terminating with an error.
    pub max_turns: usize,
    /// Maximum simultaneous tool calls. A zero value is treated as one.
    pub max_parallel_tool_calls: usize,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_turns: 64,
            max_parallel_tool_calls: 4,
        }
    }
}

/// Minimal agent loop ported from the pinned MIT Pi agent-loop semantics.
pub struct AgentLoop {
    provider: Arc<dyn ModelProvider>,
    tool_definitions: Vec<ToolDefinition>,
    tools_by_name: HashMap<String, Arc<dyn Tool>>,
    config: LoopConfig,
}

impl AgentLoop {
    /// Create a loop with normalized Runtime provider and tool contracts.
    #[must_use]
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        tools: impl IntoIterator<Item = Arc<dyn Tool>>,
    ) -> Self {
        let mut tool_definitions = Vec::new();
        let mut tools_by_name = HashMap::new();
        for tool in tools {
            let definition = tool.definition();
            tools_by_name
                .entry(definition.name.clone())
                .or_insert_with(|| tool.clone());
            tool_definitions.push(definition);
        }
        Self {
            provider,
            tool_definitions,
            tools_by_name,
            config: LoopConfig::default(),
        }
    }

    /// Set provider-turn limits.
    #[must_use]
    pub fn with_config(mut self, config: LoopConfig) -> Self {
        self.config = config;
        self
    }

    /// Run until the provider returns a tool-free assistant response.
    pub fn run(&self, state: &mut AgentState, sink: Arc<dyn EventSink>) -> Result<(), KernelError> {
        self.run_with_cancellation(state, sink, &NoCancellation)
    }

    /// Run from a non-async embedding boundary.
    pub fn run_with_cancellation(
        &self,
        state: &mut AgentState,
        sink: Arc<dyn EventSink>,
        cancellation: &dyn CancellationSignal,
    ) -> Result<(), KernelError> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(KernelError::Model(
                "AgentLoop::run_with_cancellation cannot run inside Tokio; use run_with_cancellation_async"
                    .into(),
            ));
        }
        SYNC_RUNTIME.with(|slot| {
            let mut runtime = slot.borrow_mut();
            if runtime.is_none() {
                *runtime = Some(
                    tokio::runtime::Builder::new_current_thread()
                        .enable_time()
                        .build()
                        .map_err(|error| KernelError::Model(error.to_string()))?,
                );
            }
            runtime
                .as_mut()
                .expect("synchronous runtime was initialized")
                .block_on(self.run_with_cancellation_async(state, sink, cancellation))
        })
    }

    /// Run while bridging Runtime cancellation and normalized events.
    pub async fn run_async(
        &self,
        state: &mut AgentState,
        sink: Arc<dyn EventSink>,
    ) -> Result<(), KernelError> {
        self.run_with_cancellation_async(state, sink, &NoCancellation)
            .await
    }

    /// Run on the caller's Tokio runtime.
    pub async fn run_with_cancellation_async(
        &self,
        state: &mut AgentState,
        sink: Arc<dyn EventSink>,
        cancellation: &dyn CancellationSignal,
    ) -> Result<(), KernelError> {
        self.run_with_cancellation_inner_async(state, sink, cancellation, true)
            .await
    }

    /// Run one complete agent loop while leaving successful terminal ownership to a caller gate.
    pub async fn run_with_deferred_completion_async(
        &self,
        state: &mut AgentState,
        sink: Arc<dyn EventSink>,
        cancellation: &dyn CancellationSignal,
    ) -> Result<(), KernelError> {
        self.run_with_cancellation_inner_async(state, sink, cancellation, false)
            .await
    }

    /// Commit the single successful terminal state after an outer completion gate passes.
    pub fn complete_run(
        &self,
        state: &mut AgentState,
        sink: &dyn EventSink,
    ) -> Result<(), KernelError> {
        self.transition(state, AgentStatus::Complete, sink)
    }

    async fn run_with_cancellation_inner_async(
        &self,
        state: &mut AgentState,
        sink: Arc<dyn EventSink>,
        cancellation: &dyn CancellationSignal,
        complete_on_success: bool,
    ) -> Result<(), KernelError> {
        // Public cancellation sources need not expose a shareable flag. Keep the
        // original source at this boundary and relay it into one owned flag for
        // providers and blocking tools.
        let cancellation_bridge = CancellationBridge::from_signal(cancellation);
        if cancellation.is_cancelled() {
            cancellation_bridge.cancel();
            return self.cancel(state, sink.as_ref());
        }
        for _ in 0..self.config.max_turns {
            if cancellation.is_cancelled() {
                cancellation_bridge.cancel();
                return self.cancel(state, sink.as_ref());
            }
            self.transition(state, AgentStatus::AwaitingModel, sink.as_ref())?;
            let response = match self
                .stream_response(
                    ModelRequest {
                        messages: state.transcript.clone(),
                        tools: self.tool_definitions.clone(),
                    },
                    sink.as_ref(),
                    state.session_id,
                    cancellation,
                    &cancellation_bridge,
                )
                .await
            {
                Ok(response) => response,
                Err(KernelError::Cancelled) => return self.cancel(state, sink.as_ref()),
                Err(error) => {
                    self.transition(state, AgentStatus::Failed, sink.as_ref())?;
                    return Err(error);
                }
            };

            state.turn = state.turn.saturating_add(1);
            let ModelResponse {
                content,
                tool_calls,
            } = response;
            let empty_tool_free_response = tool_calls.is_empty() && content.trim().is_empty();
            let persisted_tool_calls = tool_calls
                .iter()
                .map(ToolCall::redacted_for_persistence)
                .collect();
            self.push_message(
                state,
                Message::assistant(content, persisted_tool_calls),
                sink.as_ref(),
            )?;

            if tool_calls.is_empty() {
                if complete_on_success && empty_tool_free_response {
                    self.transition(state, AgentStatus::Failed, sink.as_ref())?;
                    return Err(KernelError::Model(
                        "model returned an empty assistant response".into(),
                    ));
                }
                if complete_on_success {
                    self.complete_run(state, sink.as_ref())?;
                }
                return Ok(());
            }

            self.transition(state, AgentStatus::ExecutingTools, sink.as_ref())?;
            let results = match self
                .execute_tool_calls(
                    tool_calls,
                    sink.clone(),
                    state.session_id,
                    cancellation,
                    &cancellation_bridge,
                )
                .await
            {
                Ok(results) => results,
                Err(KernelError::Cancelled) => return self.cancel(state, sink.as_ref()),
                Err(error) => {
                    self.transition(state, AgentStatus::Failed, sink.as_ref())?;
                    return Err(error);
                }
            };
            for result in results {
                let message = if result.is_error {
                    Message::tool_error(result.tool_call_id, result.name, result.content)
                } else {
                    Message::tool_result(result.tool_call_id, result.name, result.content)
                };
                self.push_message(state, message, sink.as_ref())?;
            }
        }

        self.transition(state, AgentStatus::Failed, sink.as_ref())?;
        Err(KernelError::TurnLimit(self.config.max_turns))
    }

    async fn stream_response(
        &self,
        request: ModelRequest,
        sink: &dyn EventSink,
        session_id: Uuid,
        cancellation: &dyn CancellationSignal,
        cancellation_bridge: &CancellationBridge,
    ) -> Result<ModelResponse, KernelError> {
        let mut stream = self.provider.stream(request, cancellation_bridge).await?;
        let mut tool_names = HashMap::new();
        while let Some(event) = tokio::select! {
            () = Self::wait_for_cancellation(cancellation) => {
                cancellation_bridge.cancel();
                sink.emit(Event::now(
                    session_id,
                    EventKind::Model {
                        event: ModelEvent::Cancelled,
                    },
                ))?;
                return Err(KernelError::Cancelled);
            }
            event = stream.next() => event,
        } {
            let event = event?;
            let terminal = match &event {
                ModelEvent::Completed { response } => Some(Ok(ModelResponse {
                    content: redact_sensitive_text(&response.content),
                    tool_calls: response.tool_calls.clone(),
                })),
                ModelEvent::Failed { error } => Some(Err(KernelError::Model(error.clone()))),
                ModelEvent::Cancelled => Some(Err(KernelError::Cancelled)),
                ModelEvent::RequestStarted { .. }
                | ModelEvent::TextDelta { .. }
                | ModelEvent::ReasoningSummaryDelta { .. }
                | ModelEvent::ToolCallStarted { .. }
                | ModelEvent::ToolArgumentsDelta { .. }
                | ModelEvent::ToolCallReady { .. }
                | ModelEvent::Usage { .. }
                | ModelEvent::RetryScheduled { .. } => None,
            };
            let event = redact_model_event(event, &mut tool_names);
            sink.emit(Event::now(session_id, EventKind::Model { event }))?;
            if let Some(terminal) = terminal {
                return terminal;
            }
        }
        Err(KernelError::Model(
            "model provider stream ended without a terminal event".into(),
        ))
    }

    async fn wait_for_cancellation(cancellation: &dyn CancellationSignal) {
        while !cancellation.is_cancelled() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn execute_tool_calls(
        &self,
        calls: Vec<ToolCall>,
        sink: Arc<dyn EventSink>,
        session_id: Uuid,
        cancellation: &dyn CancellationSignal,
        cancellation_bridge: &CancellationBridge,
    ) -> Result<Vec<ToolResult>, KernelError> {
        let max_parallel = self.config.max_parallel_tool_calls.max(1);
        let total = calls.len();
        let mut pending = calls.into_iter().enumerate().peekable();
        let mut running = FuturesUnordered::new();
        let mut results = std::iter::repeat_with(|| None)
            .take(total)
            .collect::<Vec<Option<ToolResult>>>();
        let mut first_error = None;
        let mut exclusive_in_flight = false;
        let mut next_result_to_commit = 0;

        while pending.peek().is_some() || !running.is_empty() {
            if cancellation.is_cancelled() {
                cancellation_bridge.cancel();
                while running.next().await.is_some() {}
                return Err(KernelError::Cancelled);
            }

            while !exclusive_in_flight && running.len() < max_parallel {
                let Some((_, call)) = pending.peek() else {
                    break;
                };
                let execution_class = self
                    .tools_by_name
                    .get(&call.name)
                    .map_or(ToolExecutionClass::Parallel, |tool| tool.execution_class());
                if execution_class == ToolExecutionClass::Exclusive && !running.is_empty() {
                    break;
                }
                let (index, call) = pending.next().expect("peeked call exists");
                let sink = sink.clone();
                running.push(Box::pin(async move {
                    let result = self
                        .execute_tool_call(
                            call,
                            sink,
                            session_id,
                            cancellation,
                            cancellation_bridge,
                        )
                        .await;
                    (index, execution_class, result)
                }));
                if execution_class == ToolExecutionClass::Exclusive {
                    exclusive_in_flight = true;
                    break;
                }
            }

            let completed = tokio::select! {
                () = Self::wait_for_cancellation(cancellation) => {
                    cancellation_bridge.cancel();
                    while running.next().await.is_some() {}
                    return Err(KernelError::Cancelled);
                }
                completed = running.next() => completed,
            };
            let Some(completed) = completed else {
                continue;
            };
            let (index, execution_class, outcome) = completed;
            if execution_class == ToolExecutionClass::Exclusive {
                exclusive_in_flight = false;
            }
            match outcome {
                Ok(result) => {
                    results[index] = Some(result);
                    while let Some(result) =
                        results.get(next_result_to_commit).and_then(Option::as_ref)
                    {
                        if let Err(error) = sink.emit(Event::now(
                            session_id,
                            EventKind::ToolResultReceived {
                                result: result.clone(),
                            },
                        )) {
                            first_error.get_or_insert(error);
                        }
                        next_result_to_commit = next_result_to_commit.saturating_add(1);
                    }
                }
                Err(KernelError::Cancelled) => {
                    cancellation_bridge.cancel();
                    while running.next().await.is_some() {}
                    return Err(KernelError::Cancelled);
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }

        if let Some(error) = first_error {
            return Err(error);
        }
        results
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| KernelError::Model("tool scheduler lost a completed result".into()))
    }

    async fn execute_tool_call(
        &self,
        call: ToolCall,
        sink: Arc<dyn EventSink>,
        session_id: Uuid,
        cancellation: &dyn CancellationSignal,
        cancellation_bridge: &CancellationBridge,
    ) -> Result<ToolResult, KernelError> {
        sink.emit(Event::now(
            session_id,
            EventKind::ToolCallRequested {
                call: call.redacted_for_persistence(),
            },
        ))?;
        let tool = self.tools_by_name.get(&call.name).cloned();
        let outcome = match tool {
            Some(tool) => tokio::select! {
                () = Self::wait_for_cancellation(cancellation) => {
                    cancellation_bridge.cancel();
                    Err(KernelError::Cancelled)
                }
                outcome = tool.call_with_cancellation_async(call.arguments.clone(), cancellation_bridge) => outcome,
            },
            None => Err(KernelError::UnknownTool(call.name.clone())),
        };
        if matches!(outcome, Err(KernelError::Cancelled)) {
            return Err(KernelError::Cancelled);
        }
        let web_fetch_url = (call.name == "web_fetch")
            .then(|| call.arguments.get("url").and_then(Value::as_str))
            .flatten();
        let result = match outcome {
            Ok(content) => ToolResult {
                tool_call_id: call.id,
                name: call.name,
                content,
                is_error: false,
            },
            Err(error) => ToolResult {
                tool_call_id: call.id,
                name: call.name,
                content: error.to_string(),
                is_error: true,
            },
        };
        let result = redacted_tool_result(result, web_fetch_url);
        Ok(result)
    }

    fn cancel(&self, state: &mut AgentState, sink: &dyn EventSink) -> Result<(), KernelError> {
        self.transition(state, AgentStatus::Cancelled, sink)?;
        Err(KernelError::Cancelled)
    }

    fn transition(
        &self,
        state: &mut AgentState,
        status: AgentStatus,
        sink: &dyn EventSink,
    ) -> Result<(), KernelError> {
        state.status = status;
        sink.emit(Event::now(
            state.session_id,
            EventKind::StateChanged {
                status,
                turn: state.turn,
            },
        ))
    }

    fn push_message(
        &self,
        state: &mut AgentState,
        message: Message,
        sink: &dyn EventSink,
    ) -> Result<(), KernelError> {
        state.transcript.push(message.clone());
        sink.emit(Event::now(
            state.session_id,
            EventKind::MessageAdded { message },
        ))
    }
}

/// An event sink that writes directly to an [`EventStore`].
pub struct PersistingEventSink {
    store: Arc<dyn EventStore>,
}

impl PersistingEventSink {
    /// Create a sink backed by an event store.
    #[must_use]
    pub fn new(store: Arc<dyn EventStore>) -> Self {
        Self { store }
    }
}

impl EventSink for PersistingEventSink {
    fn emit(&self, event: Event) -> Result<(), KernelError> {
        self.store.append(&event)
    }
}

/// A simple event sink for tests and embedding applications.
#[derive(Debug, Default)]
pub struct VecEventSink {
    /// Captured events in emission order.
    events: Mutex<Vec<Event>>,
}

impl VecEventSink {
    /// Return a point-in-time ordered snapshot of captured events.
    #[must_use]
    pub fn events(&self) -> Vec<Event> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl EventSink for VecEventSink {
    fn emit(&self, event: Event) -> Result<(), KernelError> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event);
        Ok(())
    }
}

/// Return a non-secret diagnostic string for a store location.
#[must_use]
pub fn store_location(store_root: &Path) -> String {
    store_root.display().to_string()
}

#[cfg(test)]
mod tests {
    use std::{
        process::Command,
        sync::atomic::{AtomicBool, Ordering},
        thread,
        time::Duration,
    };

    use super::*;

    struct EchoProvider;

    #[async_trait::async_trait]
    impl ModelProvider for EchoProvider {
        async fn stream(
            &self,
            _request: ModelRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ModelEventStream, KernelError> {
            Ok(Box::pin(futures::stream::iter(vec![
                Ok(ModelEvent::RequestStarted {
                    provider: "test".into(),
                    model: "test".into(),
                    endpoint: "test://provider".into(),
                }),
                Ok(ModelEvent::TextDelta {
                    text: "done".into(),
                }),
                Ok(ModelEvent::Completed {
                    response: ModelResponse {
                        content: "done".into(),
                        tool_calls: Vec::new(),
                    },
                }),
            ])))
        }
    }

    struct BlockingProvider;

    #[async_trait::async_trait]
    impl ModelProvider for BlockingProvider {
        async fn stream(
            &self,
            _request: ModelRequest,
            cancellation: &dyn CancellationSignal,
        ) -> Result<ModelEventStream, KernelError> {
            while !cancellation.is_cancelled() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            Err(KernelError::Cancelled)
        }
    }

    struct SharedCancellation(Arc<AtomicBool>);

    impl CancellationSignal for SharedCancellation {
        fn is_cancelled(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }

        fn shared_flag(&self) -> Option<Arc<AtomicBool>> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn kernel_cancels_an_in_flight_completion() {
        let flag = Arc::new(AtomicBool::new(false));
        let cancellation = SharedCancellation(flag.clone());
        let trigger = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            flag.store(true, Ordering::Release);
        });
        let mut state = AgentState::new(Uuid::nil());
        state.transcript.push(Message::text(Role::User, "hello"));
        let events = Arc::new(VecEventSink::default());
        let started = std::time::Instant::now();

        let error = AgentLoop::new(
            Arc::new(BlockingProvider),
            std::iter::empty::<Arc<dyn Tool>>(),
        )
        .run_with_cancellation(&mut state, events.clone(), &cancellation)
        .unwrap_err();
        trigger.join().unwrap();

        assert!(matches!(error, KernelError::Cancelled));
        assert_eq!(state.status, AgentStatus::Cancelled);
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn completes_a_tool_free_turn() {
        let provider = EchoProvider;
        let mut state = AgentState::new(Uuid::nil());
        state.transcript.push(Message::text(Role::User, "hello"));
        let events = Arc::new(VecEventSink::default());

        AgentLoop::new(Arc::new(provider), std::iter::empty::<Arc<dyn Tool>>())
            .run(&mut state, events.clone())
            .unwrap();

        assert_eq!(state.status, AgentStatus::Complete);
        assert_eq!(state.transcript.last().unwrap().content, "done");
        assert!(events.events().iter().any(|event| matches!(
            event.kind,
            EventKind::StateChanged {
                status: AgentStatus::Complete,
                ..
            }
        )));
        assert!(events.events().iter().any(|event| matches!(
            &event.kind,
            EventKind::Model {
                event: ModelEvent::TextDelta { text }
            } if text == "done"
        )));
    }

    struct EmptyProvider;

    #[async_trait::async_trait]
    impl ModelProvider for EmptyProvider {
        async fn stream(
            &self,
            _request: ModelRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ModelEventStream, KernelError> {
            Ok(Box::pin(futures::stream::iter(vec![Ok(
                ModelEvent::Completed {
                    response: ModelResponse {
                        content: String::new(),
                        tool_calls: Vec::new(),
                    },
                },
            )])))
        }
    }

    #[test]
    fn empty_tool_free_response_is_not_reported_as_success() {
        let mut state = AgentState::new(Uuid::nil());
        state.transcript.push(Message::text(Role::User, "hello"));

        let error = AgentLoop::new(Arc::new(EmptyProvider), std::iter::empty::<Arc<dyn Tool>>())
            .run(&mut state, Arc::new(VecEventSink::default()))
            .unwrap_err();

        assert!(
            matches!(error, KernelError::Model(message) if message.contains("empty assistant"))
        );
        assert_eq!(state.status, AgentStatus::Failed);
    }

    #[test]
    fn model_deltas_use_batch_durability_while_terminal_events_sync_immediately() {
        let session_id = Uuid::new_v4();
        let delta = Event::now(
            session_id,
            EventKind::Model {
                event: ModelEvent::TextDelta {
                    text: "partial".into(),
                },
            },
        );
        let terminal = Event::now(
            session_id,
            EventKind::StateChanged {
                status: AgentStatus::Complete,
                turn: 1,
            },
        );

        assert_eq!(Durability::for_event(&delta), Durability::Batch);
        assert_eq!(Durability::for_event(&terminal), Durability::Immediate);
    }

    #[test]
    fn independent_processes_preserve_jsonl_append_boundaries() {
        const ROOT: &str = "FOCUS_KERNEL_EVENT_STORE_CHILD_ROOT";
        const SESSION: &str = "FOCUS_KERNEL_EVENT_STORE_CHILD_SESSION";

        if let (Ok(root), Ok(session)) = (std::env::var(ROOT), std::env::var(SESSION)) {
            let store = JsonlEventStore::new(root).unwrap();
            let session_id = Uuid::parse_str(&session).unwrap();
            for index in 0..32 {
                store
                    .append(&Event::now(
                        session_id,
                        EventKind::Runtime {
                            name: "child_append".into(),
                            data: serde_json::json!({ "index": index }),
                        },
                    ))
                    .unwrap();
            }
            return;
        }

        let root = std::env::temp_dir().join(format!("focus-kernel-events-{}", Uuid::new_v4()));
        let session_id = Uuid::new_v4();
        let executable = std::env::current_exe().unwrap();
        let mut children = Vec::new();
        for _ in 0..2 {
            children.push(
                Command::new(&executable)
                    .args([
                        "--exact",
                        "tests::independent_processes_preserve_jsonl_append_boundaries",
                    ])
                    .env(ROOT, &root)
                    .env(SESSION, session_id.to_string())
                    .spawn()
                    .unwrap(),
            );
        }
        for mut child in children {
            assert!(child.wait().unwrap().success());
        }

        let store = JsonlEventStore::new(&root).unwrap();
        assert_eq!(store.load(session_id).unwrap().len(), 64);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn synchronous_runs_reuse_the_calling_threads_tokio_runtime() {
        #[derive(Debug)]
        struct RuntimeIdentityProvider(Arc<Mutex<Vec<tokio::runtime::Id>>>);

        #[async_trait::async_trait]
        impl ModelProvider for RuntimeIdentityProvider {
            async fn stream(
                &self,
                _request: ModelRequest,
                _cancellation: &dyn CancellationSignal,
            ) -> Result<ModelEventStream, KernelError> {
                self.0
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(tokio::runtime::Handle::current().id());
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed {
                        response: ModelResponse {
                            content: "done".into(),
                            tool_calls: Vec::new(),
                        },
                    },
                )])))
            }
        }

        let identities = Arc::new(Mutex::new(Vec::new()));
        let loop_ = AgentLoop::new(
            Arc::new(RuntimeIdentityProvider(identities.clone())),
            std::iter::empty::<Arc<dyn Tool>>(),
        );
        for _ in 0..2 {
            let mut state = AgentState::new(Uuid::new_v4());
            state.transcript.push(Message::text(Role::User, "run"));
            loop_
                .run(&mut state, Arc::new(VecEventSink::default()))
                .unwrap();
        }

        let identities = identities
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(identities.len(), 2);
        assert_eq!(identities[0], identities[1]);
    }

    #[test]
    fn tool_definitions_are_indexed_once_at_loop_construction() {
        #[derive(Debug)]
        struct DefinitionCountingTool(Arc<std::sync::atomic::AtomicUsize>);

        impl Tool for DefinitionCountingTool {
            fn definition(&self) -> ToolDefinition {
                self.0.fetch_add(1, Ordering::SeqCst);
                ToolDefinition {
                    name: "counted".into(),
                    description: "count definitions".into(),
                    input_schema: serde_json::json!({"type":"object"}),
                }
            }

            fn call_with_cancellation_async<'a>(
                &'a self,
                _arguments: Value,
                _cancellation: &'a dyn CancellationSignal,
            ) -> futures::future::BoxFuture<'a, Result<String, KernelError>> {
                Box::pin(async { Ok("ok".into()) })
            }
        }

        struct OneToolCallProvider(std::sync::atomic::AtomicUsize);
        #[async_trait::async_trait]
        impl ModelProvider for OneToolCallProvider {
            async fn stream(
                &self,
                _request: ModelRequest,
                _cancellation: &dyn CancellationSignal,
            ) -> Result<ModelEventStream, KernelError> {
                let response = if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    ModelResponse {
                        content: String::new(),
                        tool_calls: vec![ToolCall {
                            id: "counted-call".into(),
                            name: "counted".into(),
                            arguments: serde_json::json!({}),
                        }],
                    }
                } else {
                    ModelResponse {
                        content: "done".into(),
                        tool_calls: Vec::new(),
                    }
                };
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed { response },
                )])))
            }
        }

        let definition_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let loop_ = AgentLoop::new(
            Arc::new(OneToolCallProvider(std::sync::atomic::AtomicUsize::new(0))),
            [Arc::new(DefinitionCountingTool(definition_calls.clone())) as Arc<dyn Tool>],
        );
        let mut state = AgentState::new(Uuid::new_v4());
        state.transcript.push(Message::text(Role::User, "run"));

        loop_
            .run(&mut state, Arc::new(VecEventSink::default()))
            .unwrap();

        assert_eq!(definition_calls.load(Ordering::SeqCst), 1);
    }

    struct FailedEventProvider;

    #[async_trait::async_trait]
    impl ModelProvider for FailedEventProvider {
        async fn stream(
            &self,
            _request: ModelRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ModelEventStream, KernelError> {
            Ok(Box::pin(futures::stream::iter(vec![
                Ok(ModelEvent::RequestStarted {
                    provider: "fixture".into(),
                    model: "missing".into(),
                    endpoint: "fixture://failed".into(),
                }),
                Ok(ModelEvent::Failed {
                    error: "configured model does not exist".into(),
                }),
            ])))
        }
    }

    #[test]
    fn failed_model_event_cannot_become_an_empty_completion() {
        let mut state = AgentState::new(Uuid::nil());
        state.transcript.push(Message::text(Role::User, "hello"));
        let events = Arc::new(VecEventSink::default());

        let error = AgentLoop::new(
            Arc::new(FailedEventProvider),
            std::iter::empty::<Arc<dyn Tool>>(),
        )
        .run(&mut state, events)
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("configured model does not exist")
        );
        assert_eq!(state.status, AgentStatus::Failed);
        assert_eq!(state.transcript.len(), 1);
    }

    struct CancelNow;

    impl CancellationSignal for CancelNow {
        fn is_cancelled(&self) -> bool {
            true
        }
    }

    #[test]
    fn cancellation_is_emitted_before_a_model_request() {
        let provider = EchoProvider;
        let mut state = AgentState::new(Uuid::nil());
        let events = Arc::new(VecEventSink::default());

        let error = AgentLoop::new(Arc::new(provider), std::iter::empty::<Arc<dyn Tool>>())
            .run_with_cancellation(&mut state, events.clone(), &CancelNow)
            .unwrap_err();

        assert!(matches!(error, KernelError::Cancelled));
        assert_eq!(state.status, AgentStatus::Cancelled);
    }

    struct ToolThenErrorProvider(std::sync::atomic::AtomicUsize);

    #[async_trait::async_trait]
    impl ModelProvider for ToolThenErrorProvider {
        async fn stream(
            &self,
            _request: ModelRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ModelEventStream, KernelError> {
            if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                let call = ToolCall {
                    id: "call-1".into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"value":"evidence"}),
                };
                Ok(Box::pin(futures::stream::iter(vec![
                    Ok(ModelEvent::RequestStarted {
                        provider: "test".into(),
                        model: "test".into(),
                        endpoint: "test://provider".into(),
                    }),
                    Ok(ModelEvent::ToolCallStarted {
                        id: call.id.clone(),
                        name: call.name.clone(),
                    }),
                    Ok(ModelEvent::ToolArgumentsDelta {
                        id: call.id.clone(),
                        json_fragment: r#"{"value":"evidence"}"#.into(),
                    }),
                    Ok(ModelEvent::ToolCallReady { call: call.clone() }),
                    Ok(ModelEvent::Completed {
                        response: ModelResponse {
                            content: String::new(),
                            tool_calls: vec![call],
                        },
                    }),
                ])))
            } else {
                Err(KernelError::Model("provider failed after tool".into()))
            }
        }
    }

    struct EchoTool;

    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "echo a value".into(),
                input_schema: serde_json::json!({"type":"object"}),
            }
        }

        fn call_with_cancellation_async<'a>(
            &'a self,
            arguments: Value,
            _cancellation: &'a dyn CancellationSignal,
        ) -> BoxFuture<'a, Result<String, KernelError>> {
            Box::pin(async move { Ok(arguments["value"].as_str().unwrap().to_owned()) })
        }
    }

    #[test]
    fn provider_error_preserves_completed_tool_evidence() {
        let provider = ToolThenErrorProvider(std::sync::atomic::AtomicUsize::new(0));
        let mut state = AgentState::new(Uuid::nil());
        state
            .transcript
            .push(Message::text(Role::User, "use the tool"));
        let events = Arc::new(VecEventSink::default());

        let error = AgentLoop::new(Arc::new(provider), [Arc::new(EchoTool) as Arc<dyn Tool>])
            .run(&mut state, events.clone())
            .unwrap_err();

        assert!(matches!(error, KernelError::Model(_)));
        assert_eq!(state.status, AgentStatus::Failed);
        assert!(state.transcript.iter().any(|message| {
            message.role == Role::Assistant
                && message.tool_calls.iter().any(|call| call.id == "call-1")
        }));
        assert!(state.transcript.iter().any(|message| {
            message.role == Role::Tool
                && message.tool_call_id.as_deref() == Some("call-1")
                && message.content == "evidence"
        }));
        assert!(events.events().iter().any(|event| matches!(
            &event.kind,
            EventKind::ToolResultReceived { result }
                if result.tool_call_id == "call-1" && result.content == "evidence"
        )));
    }

    struct WebFetchThenFinishProvider(std::sync::atomic::AtomicUsize);

    #[async_trait::async_trait]
    impl ModelProvider for WebFetchThenFinishProvider {
        async fn stream(
            &self,
            request: ModelRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ModelEventStream, KernelError> {
            if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                let call = ToolCall {
                    id: "fetch-1".into(),
                    name: "web_fetch".into(),
                    arguments: serde_json::json!({
                        "url": "https://alice:password@example.com/report?access_token=secret#fragment"
                    }),
                };
                Ok(Box::pin(futures::stream::iter(vec![
                    Ok(ModelEvent::ToolCallStarted {
                        id: call.id.clone(),
                        name: call.name.clone(),
                    }),
                    Ok(ModelEvent::ToolArgumentsDelta {
                        id: call.id.clone(),
                        json_fragment: r#"{"url":"https://example.com/?access_token=secret"}"#
                            .into(),
                    }),
                    Ok(ModelEvent::ToolCallReady { call: call.clone() }),
                    Ok(ModelEvent::Completed {
                        response: ModelResponse {
                            content: String::new(),
                            tool_calls: vec![call],
                        },
                    }),
                ])))
            } else {
                let persisted_url = request.messages.iter().find_map(|message| {
                    message
                        .tool_calls
                        .iter()
                        .find(|call| call.name == "web_fetch")
                        .and_then(|call| call.arguments.get("url"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                });
                assert_eq!(
                    persisted_url.as_deref(),
                    Some("https://<redacted>@example.com/report?<redacted>")
                );
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed {
                        response: ModelResponse {
                            content: "done".into(),
                            tool_calls: Vec::new(),
                        },
                    },
                )])))
            }
        }
    }

    struct WebFetchEchoTool;

    impl Tool for WebFetchEchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "web_fetch".into(),
                description: "fixture".into(),
                input_schema: serde_json::json!({"type":"object"}),
            }
        }

        fn call_with_cancellation_async<'a>(
            &'a self,
            arguments: Value,
            _cancellation: &'a dyn CancellationSignal,
        ) -> BoxFuture<'a, Result<String, KernelError>> {
            Box::pin(async move {
                Ok(format!(
                    "url: {}; echoed query: access_token=secret",
                    arguments["url"]
                ))
            })
        }
    }

    #[test]
    fn web_fetch_persistence_redacts_urls_without_changing_execution_arguments() {
        let events = Arc::new(VecEventSink::default());
        let mut state = AgentState::new(Uuid::nil());
        state.transcript.push(Message::text(Role::User, "fetch"));

        AgentLoop::new(
            Arc::new(WebFetchThenFinishProvider(
                std::sync::atomic::AtomicUsize::new(0),
            )),
            [Arc::new(WebFetchEchoTool) as Arc<dyn Tool>],
        )
        .run(&mut state, events.clone())
        .unwrap();

        let serialized = serde_json::to_string(&events.events()).unwrap();
        assert!(!serialized.contains("access_token=secret"));
        assert!(!serialized.contains("password"));
        assert!(serialized.contains("<redacted web_fetch query>"));
        assert!(serialized.contains("<redacted web_fetch arguments>"));
        assert!(
            state
                .transcript
                .iter()
                .any(|message| message.content.contains("?<redacted>"))
        );
    }

    struct ModelSecretsThenFinishProvider {
        requests: Arc<Mutex<Vec<ModelRequest>>>,
        turn: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ModelProvider for ModelSecretsThenFinishProvider {
        async fn stream(
            &self,
            request: ModelRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ModelEventStream, KernelError> {
            self.requests.lock().unwrap().push(request);
            if self.turn.fetch_add(1, Ordering::AcqRel) == 0 {
                let call = ToolCall {
                    id: "echo-1".into(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"value":"evidence"}),
                };
                Ok(Box::pin(futures::stream::iter(vec![
                    Ok(ModelEvent::TextDelta {
                        text: "Authorization: Bearer model-stream-secret".into(),
                    }),
                    Ok(ModelEvent::Completed {
                        response: ModelResponse {
                            content: "token=model-completion-secret".into(),
                            tool_calls: vec![call],
                        },
                    }),
                ])))
            } else {
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed {
                        response: ModelResponse {
                            content: "done".into(),
                            tool_calls: Vec::new(),
                        },
                    },
                )])))
            }
        }
    }

    #[test]
    fn model_text_secrets_are_redacted_before_events_transcript_and_replay() {
        let provider = Arc::new(ModelSecretsThenFinishProvider {
            requests: Arc::new(Mutex::new(Vec::new())),
            turn: std::sync::atomic::AtomicUsize::new(0),
        });
        let events = Arc::new(VecEventSink::default());
        let mut state = AgentState::new(Uuid::nil());
        state
            .transcript
            .push(Message::text(Role::User, "use the tool"));

        AgentLoop::new(provider.clone(), [Arc::new(EchoTool) as Arc<dyn Tool>])
            .run(&mut state, events.clone())
            .unwrap();

        let events = serde_json::to_string(&events.events()).unwrap();
        let transcript = serde_json::to_string(&state.transcript).unwrap();
        let requests = provider.requests.lock().unwrap();
        let replay = serde_json::to_string(&requests[1]).unwrap();

        for boundary in [&events, &transcript, &replay] {
            assert!(!boundary.contains("model-stream-secret"));
            assert!(!boundary.contains("model-completion-secret"));
        }
        assert!(events.contains("Authorization: <redacted>"));
        assert!(transcript.contains("token=<redacted>"));
    }

    #[test]
    fn generic_tool_persistence_redacts_credentials_in_arguments_deltas_and_results() {
        let call = ToolCall {
            id: "shell-1".into(),
            name: "shell".into(),
            arguments: serde_json::json!({
                "command": "curl -H 'Authorization: Bearer top-secret' https://example.com/?api_key=top-secret",
                "token": "top-secret"
            }),
        };
        let persisted_call = call.redacted_for_persistence();
        let result = redacted_tool_result(
            ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                content: "Authorization: Bearer top-secret\npassword=top-secret\n-----BEGIN PRIVATE KEY-----\ntop-secret\n-----END PRIVATE KEY-----".into(),
                is_error: false,
            },
            None,
        );
        let mut tool_names = HashMap::new();
        let delta = redact_model_event(
            ModelEvent::ToolArgumentsDelta {
                id: call.id.clone(),
                json_fragment: r#"{"token":"top-secret"}"#.into(),
            },
            &mut tool_names,
        );

        assert!(
            serde_json::to_string(&persisted_call)
                .unwrap()
                .contains("<redacted>")
        );
        assert!(
            !serde_json::to_string(&persisted_call)
                .unwrap()
                .contains("top-secret")
        );
        assert!(!result.content.contains("top-secret"));
        assert!(
            !serde_json::to_string(&delta)
                .unwrap()
                .contains("top-secret")
        );
    }

    #[test]
    fn tool_lifecycle_events_measure_real_execution_time() {
        struct SlowEcho;

        impl Tool for SlowEcho {
            fn definition(&self) -> ToolDefinition {
                EchoTool.definition()
            }

            fn call_with_cancellation_async<'a>(
                &'a self,
                arguments: Value,
                _cancellation: &'a dyn CancellationSignal,
            ) -> BoxFuture<'a, Result<String, KernelError>> {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    Ok(arguments["value"].as_str().unwrap().to_owned())
                })
            }
        }

        let provider = ToolThenErrorProvider(std::sync::atomic::AtomicUsize::new(0));
        let mut state = AgentState::new(Uuid::new_v4());
        state
            .transcript
            .push(Message::text(Role::User, "measure the tool"));
        let events = Arc::new(VecEventSink::default());

        let _ = AgentLoop::new(Arc::new(provider), [Arc::new(SlowEcho) as Arc<dyn Tool>])
            .run(&mut state, events.clone());
        let events = events.events();
        let requested = events
            .iter()
            .find(|event| matches!(&event.kind, EventKind::ToolCallRequested { call } if call.id == "call-1"))
            .unwrap();
        let completed = events
            .iter()
            .find(|event| matches!(&event.kind, EventKind::ToolResultReceived { result } if result.tool_call_id == "call-1"))
            .unwrap();

        assert!(
            completed
                .timestamp_ms
                .saturating_sub(requested.timestamp_ms)
                >= 25
        );
    }

    #[test]
    fn bounded_tool_scheduler_caps_parallel_work_and_keeps_model_order() {
        struct BatchProvider(std::sync::atomic::AtomicUsize);

        #[async_trait::async_trait]
        impl ModelProvider for BatchProvider {
            async fn stream(
                &self,
                _request: ModelRequest,
                _cancellation: &dyn CancellationSignal,
            ) -> Result<ModelEventStream, KernelError> {
                let response = if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                    ModelResponse {
                        content: String::new(),
                        tool_calls: (0..6)
                            .map(|index| ToolCall {
                                id: format!("call-{index}"),
                                name: "parallel".into(),
                                arguments: serde_json::json!({}),
                            })
                            .collect(),
                    }
                } else {
                    ModelResponse {
                        content: "done".into(),
                        tool_calls: Vec::new(),
                    }
                };
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed { response },
                )])))
            }
        }

        struct TrackingTool {
            active: Arc<std::sync::atomic::AtomicUsize>,
            peak: Arc<std::sync::atomic::AtomicUsize>,
        }

        impl Tool for TrackingTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "parallel".into(),
                    description: "parallel test tool".into(),
                    input_schema: serde_json::json!({"type":"object"}),
                }
            }

            fn call_with_cancellation_async<'a>(
                &'a self,
                _arguments: Value,
                _cancellation: &'a dyn CancellationSignal,
            ) -> BoxFuture<'a, Result<String, KernelError>> {
                let active = self.active.clone();
                let peak = self.peak.clone();
                Box::pin(async move {
                    let current = active.fetch_add(1, Ordering::AcqRel) + 1;
                    peak.fetch_max(current, Ordering::AcqRel);
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    active.fetch_sub(1, Ordering::AcqRel);
                    Ok("complete".into())
                })
            }
        }

        let active = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut state = AgentState::new(Uuid::new_v4());
        state
            .transcript
            .push(Message::text(Role::User, "run batch"));
        AgentLoop::new(
            Arc::new(BatchProvider(std::sync::atomic::AtomicUsize::new(0))),
            [Arc::new(TrackingTool {
                active: active.clone(),
                peak: peak.clone(),
            }) as Arc<dyn Tool>],
        )
        .with_config(LoopConfig {
            max_turns: 3,
            max_parallel_tool_calls: 2,
        })
        .run(&mut state, Arc::new(VecEventSink::default()))
        .unwrap();

        assert!(peak.load(Ordering::Acquire) <= 2);
        let result_ids = state
            .transcript
            .iter()
            .filter(|message| message.role == Role::Tool)
            .map(|message| message.tool_call_id.as_deref().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            result_ids,
            vec!["call-0", "call-1", "call-2", "call-3", "call-4", "call-5"]
        );
    }

    #[test]
    fn exclusive_tool_is_a_submission_barrier() {
        struct BatchProvider(std::sync::atomic::AtomicUsize);

        #[async_trait::async_trait]
        impl ModelProvider for BatchProvider {
            async fn stream(
                &self,
                _request: ModelRequest,
                _cancellation: &dyn CancellationSignal,
            ) -> Result<ModelEventStream, KernelError> {
                let response = if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                    ModelResponse {
                        content: String::new(),
                        tool_calls: [
                            ("first", ToolExecutionClass::Parallel),
                            ("exclusive", ToolExecutionClass::Exclusive),
                            ("after", ToolExecutionClass::Parallel),
                        ]
                        .into_iter()
                        .map(|(name, _)| ToolCall {
                            id: format!("{name}-call"),
                            name: name.into(),
                            arguments: serde_json::json!({}),
                        })
                        .collect(),
                    }
                } else {
                    ModelResponse {
                        content: "done".into(),
                        tool_calls: Vec::new(),
                    }
                };
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed { response },
                )])))
            }
        }

        struct TraceTool {
            name: &'static str,
            class: ToolExecutionClass,
            trace: Arc<Mutex<Vec<String>>>,
        }

        impl Tool for TraceTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: self.name.into(),
                    description: self.name.into(),
                    input_schema: serde_json::json!({"type":"object"}),
                }
            }

            fn execution_class(&self) -> ToolExecutionClass {
                self.class
            }

            fn call_with_cancellation_async<'a>(
                &'a self,
                _arguments: Value,
                _cancellation: &'a dyn CancellationSignal,
            ) -> BoxFuture<'a, Result<String, KernelError>> {
                let name = self.name;
                let trace = self.trace.clone();
                Box::pin(async move {
                    trace.lock().unwrap().push(format!("{name}:start"));
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    trace.lock().unwrap().push(format!("{name}:end"));
                    Ok(name.into())
                })
            }
        }

        let trace = Arc::new(Mutex::new(Vec::new()));
        let mut state = AgentState::new(Uuid::new_v4());
        state
            .transcript
            .push(Message::text(Role::User, "run barrier"));
        AgentLoop::new(
            Arc::new(BatchProvider(std::sync::atomic::AtomicUsize::new(0))),
            [
                Arc::new(TraceTool {
                    name: "first",
                    class: ToolExecutionClass::Parallel,
                    trace: trace.clone(),
                }) as Arc<dyn Tool>,
                Arc::new(TraceTool {
                    name: "exclusive",
                    class: ToolExecutionClass::Exclusive,
                    trace: trace.clone(),
                }) as Arc<dyn Tool>,
                Arc::new(TraceTool {
                    name: "after",
                    class: ToolExecutionClass::Parallel,
                    trace: trace.clone(),
                }) as Arc<dyn Tool>,
            ],
        )
        .with_config(LoopConfig {
            max_turns: 3,
            max_parallel_tool_calls: 3,
        })
        .run(&mut state, Arc::new(VecEventSink::default()))
        .unwrap();

        let trace = trace.lock().unwrap().clone();
        let first_end = trace.iter().position(|entry| entry == "first:end").unwrap();
        let exclusive_start = trace
            .iter()
            .position(|entry| entry == "exclusive:start")
            .unwrap();
        let exclusive_end = trace
            .iter()
            .position(|entry| entry == "exclusive:end")
            .unwrap();
        let after_start = trace
            .iter()
            .position(|entry| entry == "after:start")
            .unwrap();
        assert!(first_end < exclusive_start, "{trace:?}");
        assert!(exclusive_end < after_start, "{trace:?}");
    }

    #[test]
    fn parallel_tools_commit_result_events_in_model_order() {
        struct BatchProvider(std::sync::atomic::AtomicUsize);

        #[async_trait::async_trait]
        impl ModelProvider for BatchProvider {
            async fn stream(
                &self,
                _request: ModelRequest,
                _cancellation: &dyn CancellationSignal,
            ) -> Result<ModelEventStream, KernelError> {
                let response = if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                    ModelResponse {
                        content: String::new(),
                        tool_calls: vec![
                            ToolCall {
                                id: "slow-call".into(),
                                name: "slow".into(),
                                arguments: serde_json::json!({}),
                            },
                            ToolCall {
                                id: "fast-call".into(),
                                name: "fast".into(),
                                arguments: serde_json::json!({}),
                            },
                        ],
                    }
                } else {
                    ModelResponse {
                        content: "done".into(),
                        tool_calls: Vec::new(),
                    }
                };
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed { response },
                )])))
            }
        }

        struct TimedTool {
            name: &'static str,
            delay: Duration,
        }

        impl Tool for TimedTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: self.name.into(),
                    description: self.name.into(),
                    input_schema: serde_json::json!({"type":"object"}),
                }
            }

            fn call_with_cancellation_async<'a>(
                &'a self,
                _arguments: Value,
                _cancellation: &'a dyn CancellationSignal,
            ) -> BoxFuture<'a, Result<String, KernelError>> {
                Box::pin(async move {
                    tokio::time::sleep(self.delay).await;
                    Ok(self.name.into())
                })
            }
        }

        let mut state = AgentState::new(Uuid::new_v4());
        state
            .transcript
            .push(Message::text(Role::User, "commit ordered results"));
        let events = Arc::new(VecEventSink::default());
        AgentLoop::new(
            Arc::new(BatchProvider(std::sync::atomic::AtomicUsize::new(0))),
            [
                Arc::new(TimedTool {
                    name: "slow",
                    delay: Duration::from_millis(40),
                }) as Arc<dyn Tool>,
                Arc::new(TimedTool {
                    name: "fast",
                    delay: Duration::from_millis(10),
                }) as Arc<dyn Tool>,
            ],
        )
        .run(&mut state, events.clone())
        .unwrap();

        let result_ids = events
            .events()
            .into_iter()
            .filter_map(|event| match event.kind {
                EventKind::ToolResultReceived { result } => Some(result.tool_call_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(result_ids, vec!["slow-call", "fast-call"]);
    }

    #[test]
    fn cancellation_drains_the_bounded_pool_without_starting_queued_calls() {
        struct BatchProvider(std::sync::atomic::AtomicUsize);

        #[async_trait::async_trait]
        impl ModelProvider for BatchProvider {
            async fn stream(
                &self,
                _request: ModelRequest,
                _cancellation: &dyn CancellationSignal,
            ) -> Result<ModelEventStream, KernelError> {
                let response = if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                    ModelResponse {
                        content: String::new(),
                        tool_calls: (0..4)
                            .map(|index| ToolCall {
                                id: format!("call-{index}"),
                                name: "wait".into(),
                                arguments: serde_json::json!({}),
                            })
                            .collect(),
                    }
                } else {
                    ModelResponse {
                        content: "done".into(),
                        tool_calls: Vec::new(),
                    }
                };
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed { response },
                )])))
            }
        }

        struct WaitingTool {
            starts: Arc<std::sync::atomic::AtomicUsize>,
            second_started: std::sync::mpsc::Sender<()>,
        }

        impl Tool for WaitingTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "wait".into(),
                    description: "cancellation fixture".into(),
                    input_schema: serde_json::json!({"type":"object"}),
                }
            }

            fn call_with_cancellation_async<'a>(
                &'a self,
                _arguments: Value,
                _cancellation: &'a dyn CancellationSignal,
            ) -> BoxFuture<'a, Result<String, KernelError>> {
                let starts = self.starts.clone();
                let second_started = self.second_started.clone();
                Box::pin(async move {
                    if starts.fetch_add(1, Ordering::AcqRel) + 1 == 2 {
                        let _ = second_started.send(());
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    Ok("late".into())
                })
            }
        }

        struct MutableCancellation(Arc<AtomicBool>);

        impl CancellationSignal for MutableCancellation {
            fn is_cancelled(&self) -> bool {
                self.0.load(Ordering::Acquire)
            }
        }

        let starts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cancellation_flag = Arc::new(AtomicBool::new(false));
        let cancellation = MutableCancellation(cancellation_flag.clone());
        let (second_started, started) = std::sync::mpsc::channel();
        let trigger = thread::spawn(move || {
            started
                .recv_timeout(Duration::from_secs(1))
                .expect("bounded pool must start two calls");
            cancellation_flag.store(true, Ordering::Release);
        });
        let mut state = AgentState::new(Uuid::new_v4());
        state
            .transcript
            .push(Message::text(Role::User, "cancel batch"));
        let error = AgentLoop::new(
            Arc::new(BatchProvider(std::sync::atomic::AtomicUsize::new(0))),
            [Arc::new(WaitingTool {
                starts: starts.clone(),
                second_started,
            }) as Arc<dyn Tool>],
        )
        .with_config(LoopConfig {
            max_turns: 3,
            max_parallel_tool_calls: 2,
        })
        .run_with_cancellation(&mut state, Arc::new(VecEventSink::default()), &cancellation)
        .unwrap_err();

        trigger.join().unwrap();
        assert!(matches!(error, KernelError::Cancelled));
        assert_eq!(starts.load(Ordering::Acquire), 2);
        assert_eq!(state.status, AgentStatus::Cancelled);
    }

    #[test]
    fn parallel_tools_overlap_but_results_preserve_call_order() {
        struct TwoToolProvider(std::sync::atomic::AtomicUsize);

        #[async_trait::async_trait]
        impl ModelProvider for TwoToolProvider {
            async fn stream(
                &self,
                _request: ModelRequest,
                _cancellation: &dyn CancellationSignal,
            ) -> Result<ModelEventStream, KernelError> {
                if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                    let calls = vec![
                        ToolCall {
                            id: "slow-call".into(),
                            name: "slow".into(),
                            arguments: serde_json::json!({}),
                        },
                        ToolCall {
                            id: "fast-call".into(),
                            name: "fast".into(),
                            arguments: serde_json::json!({}),
                        },
                    ];
                    Ok(Box::pin(futures::stream::iter(vec![Ok(
                        ModelEvent::Completed {
                            response: ModelResponse {
                                content: String::new(),
                                tool_calls: calls,
                            },
                        },
                    )])))
                } else {
                    Ok(Box::pin(futures::stream::iter(vec![Ok(
                        ModelEvent::Completed {
                            response: ModelResponse {
                                content: "done".into(),
                                tool_calls: Vec::new(),
                            },
                        },
                    )])))
                }
            }
        }

        struct TimedTool {
            name: &'static str,
            delay: Duration,
        }

        impl Tool for TimedTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: self.name.into(),
                    description: self.name.into(),
                    input_schema: serde_json::json!({"type":"object"}),
                }
            }

            fn call_with_cancellation_async<'a>(
                &'a self,
                _arguments: Value,
                _cancellation: &'a dyn CancellationSignal,
            ) -> BoxFuture<'a, Result<String, KernelError>> {
                Box::pin(async move {
                    tokio::time::sleep(self.delay).await;
                    Ok(self.name.into())
                })
            }
        }

        let mut state = AgentState::new(Uuid::new_v4());
        state.transcript.push(Message::text(Role::User, "run both"));
        let events = Arc::new(VecEventSink::default());
        let started = std::time::Instant::now();
        AgentLoop::new(
            Arc::new(TwoToolProvider(std::sync::atomic::AtomicUsize::new(0))),
            [
                Arc::new(TimedTool {
                    name: "slow",
                    delay: Duration::from_millis(60),
                }) as Arc<dyn Tool>,
                Arc::new(TimedTool {
                    name: "fast",
                    delay: Duration::from_millis(20),
                }) as Arc<dyn Tool>,
            ],
        )
        .run(&mut state, events)
        .unwrap();

        assert!(started.elapsed() < Duration::from_millis(95));
        let results = state
            .transcript
            .iter()
            .filter(|message| message.role == Role::Tool)
            .map(|message| message.tool_call_id.as_deref().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results, vec!["slow-call", "fast-call"]);
    }

    #[test]
    fn max_turns_converges_to_one_failed_terminal() {
        struct EndlessToolProvider;

        #[async_trait::async_trait]
        impl ModelProvider for EndlessToolProvider {
            async fn stream(
                &self,
                _request: ModelRequest,
                _cancellation: &dyn CancellationSignal,
            ) -> Result<ModelEventStream, KernelError> {
                let call = ToolCall {
                    id: Uuid::new_v4().to_string(),
                    name: "echo".into(),
                    arguments: serde_json::json!({"value":"again"}),
                };
                Ok(Box::pin(futures::stream::iter(vec![Ok(
                    ModelEvent::Completed {
                        response: ModelResponse {
                            content: String::new(),
                            tool_calls: vec![call],
                        },
                    },
                )])))
            }
        }

        let mut state = AgentState::new(Uuid::new_v4());
        state.transcript.push(Message::text(Role::User, "loop"));
        let events = Arc::new(VecEventSink::default());
        let error = AgentLoop::new(
            Arc::new(EndlessToolProvider),
            [Arc::new(EchoTool) as Arc<dyn Tool>],
        )
        .with_config(LoopConfig {
            max_turns: 2,
            ..LoopConfig::default()
        })
        .run(&mut state, events.clone())
        .unwrap_err();

        assert!(matches!(error, KernelError::TurnLimit(2)));
        assert_eq!(state.status, AgentStatus::Failed);
        assert_eq!(
            events
                .events()
                .iter()
                .filter(|event| matches!(
                    event.kind,
                    EventKind::StateChanged {
                        status: AgentStatus::Failed,
                        ..
                    }
                ))
                .count(),
            1
        );
    }

    #[test]
    fn failed_tool_results_keep_error_state_across_serialization() {
        let original = Message::tool_error("call-1", "shell", "command failed");

        let encoded = serde_json::to_vec(&original).unwrap();
        let restored: Message = serde_json::from_slice(&encoded).unwrap();

        assert!(restored.is_error);
        assert_eq!(restored.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(restored.tool_name.as_deref(), Some("shell"));
    }

    #[test]
    fn model_tool_calls_preserve_source_order() {
        let response = ModelResponse {
            content: "Inspecting source".into(),
            tool_calls: vec![
                ToolCall {
                    id: "call-1".into(),
                    name: "read_file".into(),
                    arguments: serde_json::json!({"path":"src/lib.rs"}),
                },
                ToolCall {
                    id: "call-2".into(),
                    name: "search".into(),
                    arguments: serde_json::json!({"query":"AgentLoop"}),
                },
            ],
        };

        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].id, "call-1");
        assert_eq!(response.tool_calls[0].arguments["path"], "src/lib.rs");
        assert_eq!(response.tool_calls[1].name, "search");
        assert_eq!(response.tool_calls[1].arguments["query"], "AgentLoop");
    }

    #[test]
    fn model_events_are_stable_replay_facts() {
        let event = ModelEvent::ToolCallReady {
            call: ToolCall {
                id: "call-42".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path":"src/lib.rs"}),
            },
        };

        assert_eq!(
            serde_json::to_value(event).unwrap(),
            serde_json::json!({
                "type":"tool_call_ready",
                "call":{
                    "id":"call-42",
                    "name":"read_file",
                    "arguments":{"path":"src/lib.rs"}
                }
            })
        );
    }

    #[test]
    fn kernel_event_persists_model_stream_facts() {
        let event = Event::now(
            Uuid::nil(),
            EventKind::Model {
                event: ModelEvent::TextDelta {
                    text: "visible before completion".into(),
                },
            },
        );

        assert_eq!(
            serde_json::to_value(event.kind).unwrap(),
            serde_json::json!({
                "type":"model",
                "event":{"type":"text_delta","text":"visible before completion"}
            })
        );
    }

    #[test]
    fn provider_stream_exposes_visible_deltas_before_completion() {
        use futures::TryStreamExt;

        let events = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                EchoProvider
                    .stream(
                        ModelRequest {
                            messages: vec![Message::text(Role::User, "hello")],
                            tools: Vec::new(),
                        },
                        &NoCancellation,
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
            });

        assert!(matches!(
            events.as_slice(),
            [
                ModelEvent::RequestStarted { .. },
                ModelEvent::TextDelta { text },
                ModelEvent::Completed { .. },
            ] if text == "done"
        ));
    }
}
