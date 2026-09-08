//! Runtime tool registry and canonical built-in coding tools.

use std::{collections::BTreeMap, sync::Arc};

use focus_kernel::{CancellationSignal, KernelError, Tool, ToolDefinition, ToolExecutionClass};
use futures::future::BoxFuture;
use reqwest::{Client, Url, redirect::Policy as RedirectPolicy};
use serde_json::{Value, json};

use crate::{
    RuntimeError,
    cancellation::wait_for_cancellation,
    network::NetworkConfig,
    policy::{PolicyEngine, ToolOperation},
    sandbox::{CommandRequest, CommandTermination, ResourceLimits, WorkspaceSandbox},
};

/// A Runtime tool handler after unified policy authorization.
pub trait ToolHandler: Send + Sync {
    /// Execute on the current Runtime executor while observing cancellation.
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>>;
}

/// Runtime-owned metadata and handler before policy is bound for a run.
#[derive(Clone)]
pub struct RuntimeToolSpec {
    definition: ToolDefinition,
    operation: ToolOperation,
    rationale: String,
    execution_class: ToolExecutionClass,
    handler: Arc<dyn ToolHandler>,
}

impl RuntimeToolSpec {
    /// Create one canonical Runtime tool specification.
    #[must_use]
    pub fn new(
        definition: ToolDefinition,
        operation: ToolOperation,
        rationale: impl Into<String>,
        handler: Arc<dyn ToolHandler>,
    ) -> Self {
        Self {
            definition,
            operation,
            rationale: rationale.into(),
            execution_class: ToolExecutionClass::Parallel,
            handler,
        }
    }

    /// Mark whether this tool may overlap other model-selected calls.
    #[must_use]
    pub fn with_execution_class(mut self, execution_class: ToolExecutionClass) -> Self {
        self.execution_class = execution_class;
        self
    }

    /// Return the provider-neutral definition advertised to models.
    #[must_use]
    pub fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
}

/// The single registry for built-in, MCP, and Runtime extension tools.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    specs: BTreeMap<String, RuntimeToolSpec>,
}

impl ToolRegistry {
    /// Register a unique canonical name.
    pub fn register(&mut self, spec: RuntimeToolSpec) -> Result<(), RuntimeError> {
        let name = spec.definition.name.clone();
        if self.specs.contains_key(&name) {
            return Err(RuntimeError::ToolInput(format!(
                "duplicate Runtime tool: {name}"
            )));
        }
        self.specs.insert(name, spec);
        Ok(())
    }

    /// Bind every registered tool to the policy engine for one run.
    #[must_use]
    pub fn bind(&self, policy: Arc<PolicyEngine>) -> Vec<Arc<dyn Tool>> {
        self.specs
            .values()
            .cloned()
            .map(|spec| {
                Arc::new(
                    RuntimeTool::new(
                        spec.definition,
                        spec.operation,
                        spec.rationale,
                        policy.clone(),
                        spec.handler,
                    )
                    .with_execution_class(spec.execution_class),
                ) as Arc<dyn Tool>
            })
            .collect()
    }

    /// Return the registered definitions in deterministic name order.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.specs
            .values()
            .map(|spec| spec.definition.clone())
            .collect()
    }
}

/// One Runtime-owned tool implementation decorated by the sole policy engine.
pub struct RuntimeTool {
    definition: ToolDefinition,
    operation: ToolOperation,
    rationale: String,
    execution_class: ToolExecutionClass,
    policy: Arc<PolicyEngine>,
    handler: Arc<dyn ToolHandler>,
}

impl RuntimeTool {
    /// Construct a tool that always routes through `policy` before its handler.
    #[must_use]
    pub fn new(
        definition: ToolDefinition,
        operation: ToolOperation,
        rationale: impl Into<String>,
        policy: Arc<PolicyEngine>,
        handler: Arc<dyn ToolHandler>,
    ) -> Self {
        Self {
            definition,
            operation,
            rationale: rationale.into(),
            execution_class: ToolExecutionClass::Parallel,
            policy,
            handler,
        }
    }

    fn with_execution_class(mut self, execution_class: ToolExecutionClass) -> Self {
        self.execution_class = execution_class;
        self
    }
}

impl Tool for RuntimeTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn execution_class(&self) -> ToolExecutionClass {
        self.execution_class
    }

    fn call_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, KernelError>> {
        Box::pin(async move {
            self.policy
                .authorize_with_cancellation(
                    crate::policy::ApprovalRequest {
                        tool: self.definition.name.clone(),
                        operation: self.operation,
                        arguments: redact_arguments(arguments.clone()),
                        rationale: self.rationale.clone(),
                    },
                    cancellation,
                )
                .map_err(runtime_as_kernel)?;
            self.handler
                .execute_with_cancellation_async(arguments, cancellation)
                .await
                .map_err(runtime_as_kernel)
        })
    }
}

/// Build the canonical registry containing the standard coding tools.
#[must_use]
pub fn builtin_tool_registry(sandbox: WorkspaceSandbox, network: NetworkConfig) -> ToolRegistry {
    let sandbox = Arc::new(sandbox);
    let mut specs = vec![
        RuntimeToolSpec::new(
            definition(
                "read_file",
                "Read a UTF-8 text file relative to the project root.",
                json!({"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}),
            ),
            ToolOperation::Read,
            "Read project source or configuration.",
            Arc::new(ReadFile {
                sandbox: sandbox.clone(),
            }),
        ),
        RuntimeToolSpec::new(
            definition(
                "search",
                "Search UTF-8 project files for a literal string. Skips .git, target, node_modules and symlinks.",
                json!({"type":"object","required":["query"],"properties":{"query":{"type":"string","minLength":1},"max_results":{"type":"integer","minimum":1,"maximum":500}}}),
            ),
            ToolOperation::Read,
            "Explore project source and configuration.",
            Arc::new(Search {
                sandbox: sandbox.clone(),
            }),
        ),
        RuntimeToolSpec::new(
            definition(
                "write_file",
                "Atomically create or replace a text file relative to the project root.",
                json!({"type":"object","required":["path","content"],"properties":{"path":{"type":"string"},"content":{"type":"string"}}}),
            ),
            ToolOperation::Write,
            "Apply a planned source-code or documentation change.",
            Arc::new(WriteFile {
                sandbox: sandbox.clone(),
            }),
        )
        .with_execution_class(ToolExecutionClass::Exclusive),
        RuntimeToolSpec::new(
            definition(
                "mkdir",
                "Create nested directories relative to the project root without traversing symlinks.",
                json!({"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}),
            ),
            ToolOperation::Write,
            "Create project directories for a planned change.",
            Arc::new(MakeDirectory {
                sandbox: sandbox.clone(),
            }),
        )
        .with_execution_class(ToolExecutionClass::Exclusive),
        RuntimeToolSpec::new(
            definition(
                "shell",
                shell_tool_description(),
                json!({"type":"object","required":["command"],"properties":{"command":{"type":"string"},"purpose":{"type":"string","enum":["explore","implement","verify","review"]},"timeout_ms":{"type":"integer","minimum":1,"maximum":3600000},"max_output_bytes":{"type":"integer","minimum":0,"maximum":8388608},"resources":{"type":"object","properties":{"memory_bytes":{"type":"integer","minimum":1},"cpus":{"type":"number","exclusiveMinimum":0},"pids":{"type":"integer","minimum":1}}}}}),
            ),
            ToolOperation::Execute,
            "Build, test, inspect, or format the current project.",
            Arc::new(Shell { sandbox }),
        )
        .with_execution_class(ToolExecutionClass::Exclusive),
    ];
    if network.enabled {
        specs.push(web_fetch_tool_spec(network));
    }
    let mut registry = ToolRegistry::default();
    for spec in specs {
        registry
            .register(spec)
            .expect("built-in tool names are unique");
    }
    registry
}

fn web_fetch_tool_spec(network: NetworkConfig) -> RuntimeToolSpec {
    RuntimeToolSpec::new(
        definition(
            "web_fetch",
            "Fetch an HTTP or HTTPS URL through the configured network policy. Follows bounded redirects, returns bounded UTF-8-lossy response text, and never sends credentials.",
            json!({"type":"object","required":["url"],"properties":{"url":{"type":"string","format":"uri"},"max_bytes":{"type":"integer","minimum":1,"maximum":8388608}}}),
        ),
        ToolOperation::Network,
        "Fetch an external web resource through the configured network policy.",
        Arc::new(WebFetch { network }),
    )
}

/// Build the workflow-only checkpoint capability for an engineering turn.
#[must_use]
pub(crate) fn workflow_checkpoint_tool_spec() -> RuntimeToolSpec {
    RuntimeToolSpec::new(
        definition(
            "workflow_checkpoint",
            "Record an explicit plan, no-change decision, or review summary for the executable workflow gate.",
            json!({"type":"object","required":["kind","summary"],"properties":{"kind":{"type":"string","enum":["plan","no_change","review"]},"summary":{"type":"string","minLength":1}}}),
        ),
        ToolOperation::Read,
        "Record non-mutating workflow evidence.",
        Arc::new(WorkflowCheckpoint),
    )
    .with_execution_class(ToolExecutionClass::Exclusive)
}

struct ReadFile {
    sandbox: Arc<WorkspaceSandbox>,
}

impl ToolHandler for ReadFile {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        let sandbox = self.sandbox.clone();
        Box::pin(async move {
            let path = required_string(&arguments, "path")?.to_owned();
            run_blocking_with_cancellation("read_file", cancellation, move |cancellation| {
                if cancellation.is_cancelled() {
                    return Err(RuntimeError::Cancelled);
                }
                sandbox.read_to_string(path)
            })
            .await
        })
    }
}

struct Search {
    sandbox: Arc<WorkspaceSandbox>,
}

impl ToolHandler for Search {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        let sandbox = self.sandbox.clone();
        Box::pin(async move {
            let query = required_string(&arguments, "query")?.to_owned();
            let max_results = arguments
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(100)
                .clamp(1, 500) as usize;
            let output =
                run_blocking_with_cancellation("search", cancellation, move |cancellation| {
                    sandbox.search_with_cancellation_report(&query, max_results, &cancellation)
                })
                .await?;
            Ok(if output.matches.is_empty() {
                "No matches.".into()
            } else {
                let mut rendered = output.matches.join("\n");
                if output.truncated {
                    rendered.push_str(&format!(
                        "\n[search truncated after {} files and {} bytes]",
                        output.files_scanned, output.bytes_scanned
                    ));
                }
                rendered
            })
        })
    }
}

struct WriteFile {
    sandbox: Arc<WorkspaceSandbox>,
}

impl ToolHandler for WriteFile {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        let sandbox = self.sandbox.clone();
        Box::pin(async move {
            let path = required_string(&arguments, "path")?.to_owned();
            let content = required_string(&arguments, "content")?.to_owned();
            run_blocking_with_cancellation("write_file", cancellation, move |cancellation| {
                if cancellation.is_cancelled() {
                    return Err(RuntimeError::Cancelled);
                }
                sandbox.write_string(path, &content)
            })
            .await?;
            Ok("File written atomically.".into())
        })
    }
}

struct Shell {
    sandbox: Arc<WorkspaceSandbox>,
}

struct WebFetch {
    network: NetworkConfig,
}

struct MakeDirectory {
    sandbox: Arc<WorkspaceSandbox>,
}

impl ToolHandler for MakeDirectory {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        let sandbox = self.sandbox.clone();
        Box::pin(async move {
            let path = required_string(&arguments, "path")?.to_owned();
            run_blocking_with_cancellation("mkdir", cancellation, move |cancellation| {
                if cancellation.is_cancelled() {
                    return Err(RuntimeError::Cancelled);
                }
                sandbox.create_dir_all(path)
            })
            .await?;
            Ok("Directory tree created.".into())
        })
    }
}

struct WorkflowCheckpoint;

impl ToolHandler for WorkflowCheckpoint {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        _cancellation: &dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        Box::pin(async move {
            let kind = required_string(&arguments, "kind")?;
            if !matches!(kind, "plan" | "no_change" | "review") {
                return Err(RuntimeError::ToolInput(
                    "`kind` must be plan, no_change, or review".into(),
                ));
            }
            let summary = required_string(&arguments, "summary")?;
            Ok(format!("Workflow checkpoint `{kind}` recorded: {summary}"))
        })
    }
}

impl ToolHandler for Shell {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        let sandbox = self.sandbox.clone();
        Box::pin(async move {
            run_blocking_with_cancellation("shell", cancellation, move |cancellation| {
                execute_shell(&sandbox, arguments, &cancellation)
            })
            .await
        })
    }
}

async fn run_blocking_with_cancellation<T>(
    operation: &'static str,
    cancellation: &dyn CancellationSignal,
    work: impl FnOnce(focus_kernel::CancellationBridge) -> Result<T, RuntimeError> + Send + 'static,
) -> Result<T, RuntimeError>
where
    T: Send + 'static,
{
    if cancellation.is_cancelled() {
        return Err(RuntimeError::Cancelled);
    }
    let bridge = focus_kernel::CancellationBridge::from_signal(cancellation);
    let worker_bridge = bridge.clone();
    let mut worker = tokio::task::spawn_blocking(move || work(worker_bridge));
    tokio::select! {
        () = wait_for_cancellation(cancellation) => {
            bridge.cancel();
            match worker.await {
                Ok(_) => Err(RuntimeError::Cancelled),
                Err(error) => Err(RuntimeError::ToolInput(format!("{operation} worker failed while cancelling: {error}"))),
            }
        }
        result = &mut worker => result
            .map_err(|error| RuntimeError::ToolInput(format!("{operation} worker failed: {error}")))?,
    }
}

impl ToolHandler for WebFetch {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        let network = self.network.clone();
        Box::pin(async move { execute_web_fetch(network, arguments, cancellation).await })
    }
}

async fn execute_web_fetch(
    network: NetworkConfig,
    arguments: Value,
    cancellation: &dyn CancellationSignal,
) -> Result<String, RuntimeError> {
    if cancellation.is_cancelled() {
        return Err(RuntimeError::Cancelled);
    }
    let mut url = Url::parse(required_string(&arguments, "url")?)
        .map_err(|error| RuntimeError::ToolInput(format!("`url` is invalid: {error}")))?;
    let max_bytes = arguments
        .get("max_bytes")
        .and_then(Value::as_u64)
        .unwrap_or(network.max_response_bytes as u64);
    if max_bytes == 0 || max_bytes > network.max_response_bytes as u64 {
        return Err(RuntimeError::ToolInput(format!(
            "`max_bytes` must be between 1 and {}",
            network.max_response_bytes
        )));
    }
    let max_bytes = max_bytes as usize;
    for redirect in 0..=network.max_redirects {
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let addresses = network.authorize_url(&url, cancellation).await?;
        let host = url
            .host_str()
            .ok_or_else(|| RuntimeError::Network("URL must include a host".into()))?;
        let client = Client::builder()
            .timeout(network.request_timeout)
            .no_proxy()
            .redirect(RedirectPolicy::none())
            .resolve_to_addrs(host, &addresses)
            .build()
            .map_err(|error| RuntimeError::Network(format!("HTTP client build failed: {error}")))?;
        let request = client.get(url.clone()).header(
            "accept",
            "text/plain, text/html, application/json, */*;q=0.1",
        );
        let response = tokio::select! {
            () = wait_for_cancellation(cancellation) => return Err(RuntimeError::Cancelled),
            response = request.send() => response.map_err(|error| RuntimeError::Network(format!("HTTP request failed: {error}")))?,
        };
        if response.status().is_redirection() {
            if redirect == network.max_redirects {
                return Err(RuntimeError::Network(format!(
                    "redirect limit of {} exceeded",
                    network.max_redirects
                )));
            }
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| {
                    RuntimeError::Network(
                        "redirect response did not include a valid Location header".into(),
                    )
                })?;
            url = url.join(location).map_err(|error| {
                RuntimeError::Network(format!("redirect Location was invalid: {error}"))
            })?;
            continue;
        }
        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("<missing>")
            .to_owned();
        let body = read_bounded_response(response, cancellation, max_bytes).await?;
        let mut output = format!(
            "url: {}\nstatus: {}\ncontent_type: {}\ntruncated: {}\nbody:\n{}",
            url,
            status.as_u16(),
            content_type,
            body.truncated,
            String::from_utf8_lossy(&body.bytes)
        );
        if !status.is_success() {
            output = format!("HTTP request completed with non-success status.\n{output}");
        }
        return Ok(output);
    }
    unreachable!("bounded redirect loop always returns")
}

struct ResponseBytes {
    bytes: Vec<u8>,
    truncated: bool,
}

async fn read_bounded_response(
    response: reqwest::Response,
    cancellation: &dyn CancellationSignal,
    max_bytes: usize,
) -> Result<ResponseBytes, RuntimeError> {
    use futures::StreamExt;

    let mut bytes = Vec::with_capacity(max_bytes.min(64 * 1024));
    let mut body = response.bytes_stream();
    loop {
        let next = tokio::select! {
            () = wait_for_cancellation(cancellation) => return Err(RuntimeError::Cancelled),
            chunk = body.next() => chunk,
        };
        let Some(chunk) = next else {
            return Ok(ResponseBytes {
                bytes,
                truncated: false,
            });
        };
        let chunk = chunk.map_err(|error| {
            RuntimeError::Network(format!("HTTP response read failed: {error}"))
        })?;
        let remaining = max_bytes.saturating_sub(bytes.len());
        if chunk.len() > remaining {
            bytes.extend_from_slice(&chunk[..remaining]);
            return Ok(ResponseBytes {
                bytes,
                truncated: true,
            });
        }
        bytes.extend_from_slice(&chunk);
    }
}

fn execute_shell(
    sandbox: &WorkspaceSandbox,
    arguments: Value,
    cancellation: &dyn CancellationSignal,
) -> Result<String, RuntimeError> {
    let mut request = CommandRequest::new(required_string(&arguments, "command")?);
    if let Some(value) = arguments.get("timeout_ms") {
        let timeout_ms = value
            .as_u64()
            .ok_or_else(|| RuntimeError::ToolInput("`timeout_ms` must be an integer".into()))?;
        if !(1..=3_600_000).contains(&timeout_ms) {
            return Err(RuntimeError::ToolInput(
                "`timeout_ms` must be between 1 and 3600000".into(),
            ));
        }
        request.timeout = std::time::Duration::from_millis(timeout_ms);
    }
    if let Some(value) = arguments.get("max_output_bytes") {
        let max_output_bytes = value.as_u64().ok_or_else(|| {
            RuntimeError::ToolInput("`max_output_bytes` must be an integer".into())
        })?;
        if max_output_bytes > 8 * 1024 * 1024 {
            return Err(RuntimeError::ToolInput(
                "`max_output_bytes` must be at most 8388608".into(),
            ));
        }
        request.max_output_bytes = max_output_bytes as usize;
    }
    if let Some(resources) = arguments.get("resources") {
        let resources = resources
            .as_object()
            .ok_or_else(|| RuntimeError::ToolInput("`resources` must be an object".into()))?;
        request.resources = ResourceLimits {
            memory_bytes: optional_u64(resources, "memory_bytes")?,
            cpus: optional_f64(resources, "cpus")?,
            pids: optional_u64(resources, "pids")?
                .map(u32::try_from)
                .transpose()
                .map_err(|_| RuntimeError::ToolInput("`pids` must fit in u32".into()))?,
        };
    }
    let output = sandbox.run_command(request, cancellation)?;
    if output.termination == CommandTermination::Cancelled {
        return Err(RuntimeError::Cancelled);
    }
    Ok(format!(
        "exit_code: {}\ntermination: {:?}\nstdout:\n{}\nstderr:\n{}",
        output
            .exit_code
            .map_or_else(|| "signal".into(), |code| code.to_string()),
        output.termination,
        output.stdout,
        output.stderr
    ))
}

fn optional_u64(
    values: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, RuntimeError> {
    values
        .get(field)
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| RuntimeError::ToolInput(format!("`{field}` must be an integer")))
        })
        .transpose()
}

fn optional_f64(
    values: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<f64>, RuntimeError> {
    values
        .get(field)
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| RuntimeError::ToolInput(format!("`{field}` must be a number")))
        })
        .transpose()
}

fn definition(name: &str, description: &str, input_schema: Value) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        input_schema,
    }
}

fn required_string<'a>(arguments: &'a Value, field: &str) -> Result<&'a str, RuntimeError> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| RuntimeError::ToolInput(format!("`{field}` must be a non-empty string")))
}

#[cfg(windows)]
fn shell_tool_description() -> &'static str {
    "Run a command in the project root and return bounded stdout and stderr. Native Windows execution uses cmd.exe /D /S /C: use cmd syntax such as `dir`, `type FILE`, or `cargo metadata`; invoke PowerShell explicitly when its syntax is required."
}

#[cfg(not(windows))]
fn shell_tool_description() -> &'static str {
    "Run a shell command in the project root and return bounded stdout and stderr. Native execution uses sh -lc."
}

fn runtime_as_kernel(error: RuntimeError) -> KernelError {
    match error {
        RuntimeError::Cancelled => KernelError::Cancelled,
        error => KernelError::Tool {
            name: "runtime".into(),
            message: error.to_string(),
        },
    }
}

fn redact_arguments(mut arguments: Value) -> Value {
    if let Some(Value::String(url)) = arguments.get_mut("url") {
        *url = redact_url_argument(url);
    }
    if let Some(object) = arguments.as_object_mut() {
        for (key, value) in object {
            if key.to_ascii_lowercase().contains("token")
                || key.to_ascii_lowercase().contains("secret")
                || key.to_ascii_lowercase().contains("password")
            {
                *value = Value::String("[redacted]".into());
            }
        }
    }
    arguments
}

fn redact_url_argument(url: &str) -> String {
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

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::atomic::{AtomicBool, Ordering},
        sync::mpsc,
        thread,
        time::Duration,
    };

    use super::*;
    use crate::policy::{FixedApproval, Policy};
    use focus_kernel::NoCancellation;
    use uuid::Uuid;

    struct Echo;

    impl ToolHandler for Echo {
        fn execute_with_cancellation_async<'a>(
            &'a self,
            arguments: Value,
            _cancellation: &'a dyn CancellationSignal,
        ) -> BoxFuture<'a, Result<String, RuntimeError>> {
            Box::pin(async move { Ok(arguments.to_string()) })
        }
    }

    struct AsyncEcho;

    impl ToolHandler for AsyncEcho {
        fn execute_with_cancellation_async<'a>(
            &'a self,
            arguments: Value,
            _cancellation: &'a dyn CancellationSignal,
        ) -> BoxFuture<'a, Result<String, RuntimeError>> {
            Box::pin(async move { Ok(format!("async:{arguments}")) })
        }
    }

    #[test]
    fn builtin_tools_declare_static_scheduler_classes() {
        let directory =
            std::env::temp_dir().join(format!("runtime-tool-classes-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let policy = Arc::new(PolicyEngine::new(
            Policy::default(),
            Arc::new(FixedApproval(true)),
        ));
        let tools = builtin_tool_registry(
            WorkspaceSandbox::new(&directory).unwrap(),
            NetworkConfig::default(),
        )
        .bind(policy);

        let class_for = |name: &str| {
            tools
                .iter()
                .find(|tool| tool.definition().name == name)
                .expect("canonical built-in must exist")
                .execution_class()
        };
        assert_eq!(
            class_for("read_file"),
            focus_kernel::ToolExecutionClass::Parallel
        );
        assert_eq!(
            class_for("search"),
            focus_kernel::ToolExecutionClass::Parallel
        );
        let search_schema = tools
            .iter()
            .find(|tool| tool.definition().name == "search")
            .expect("canonical search tool must exist")
            .definition()
            .input_schema
            .clone();
        assert_eq!(search_schema["properties"]["query"]["minLength"], 1);
        for name in ["write_file", "mkdir", "shell"] {
            assert_eq!(
                class_for(name),
                focus_kernel::ToolExecutionClass::Exclusive,
                "{name} must serialize filesystem or process side effects"
            );
        }
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn runtime_tool_awaits_an_async_handler_override() {
        let policy = Arc::new(PolicyEngine::new(
            Policy::default(),
            Arc::new(FixedApproval(true)),
        ));
        let tool = RuntimeTool::new(
            definition("async", "async", json!({"type":"object"})),
            ToolOperation::Read,
            "test async tool path",
            policy,
            Arc::new(AsyncEcho),
        );

        let output = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(tool.call_with_cancellation_async(json!({"value":7}), &NoCancellation))
            .unwrap();

        assert_eq!(output, "async:{\"value\":7}");
    }

    #[test]
    fn registry_rejects_duplicate_canonical_names() {
        let mut registry = ToolRegistry::default();
        let spec = RuntimeToolSpec::new(
            definition("extension", "extension", json!({"type":"object"})),
            ToolOperation::Other,
            "exercise extension policy",
            Arc::new(Echo),
        );

        registry.register(spec.clone()).unwrap();
        let error = registry.register(spec).unwrap_err();

        assert!(error.to_string().contains("duplicate Runtime tool"));
    }

    #[test]
    fn shell_rejects_an_out_of_range_timeout_instead_of_clamping_it() {
        let directory =
            std::env::temp_dir().join(format!("runtime-shell-limits-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();
        let command = if cfg!(windows) { "exit 0" } else { "true" };

        let error = execute_shell(
            &sandbox,
            json!({"command": command, "timeout_ms": 3_600_001}),
            &NoCancellation,
        )
        .unwrap_err();

        assert!(error.to_string().contains("timeout_ms"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[cfg(windows)]
    #[test]
    fn shell_tool_describes_the_native_cmd_interpreter() {
        let directory = std::env::temp_dir().join(format!(
            "runtime-shell-description-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let registry = builtin_tool_registry(
            WorkspaceSandbox::new(&directory).unwrap(),
            NetworkConfig::default(),
        );
        let shell = registry
            .definitions()
            .into_iter()
            .find(|definition| definition.name == "shell")
            .expect("shell must be a canonical built-in");

        assert!(shell.description.contains("cmd.exe"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn disabled_network_does_not_advertise_web_fetch() {
        let directory = std::env::temp_dir().join(format!(
            "runtime-web-fetch-disabled-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let registry = builtin_tool_registry(
            WorkspaceSandbox::new(&directory).unwrap(),
            NetworkConfig::default(),
        );

        assert!(
            !registry
                .definitions()
                .iter()
                .any(|definition| definition.name == "web_fetch")
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn web_fetch_requires_the_existing_network_policy_approval() {
        let directory =
            std::env::temp_dir().join(format!("runtime-web-fetch-policy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();
        let mut network = NetworkConfig::enabled();
        network
            .insert_domain_rule("example.com", crate::network::DomainAccess::Allow)
            .unwrap();
        let policy = Arc::new(PolicyEngine::new(
            Policy {
                network: crate::policy::PolicyDecision::RequireApproval,
                ..Policy::default()
            },
            Arc::new(FixedApproval(false)),
        ));
        let tools = builtin_tool_registry(sandbox, network).bind(policy);
        let fetch = tools
            .iter()
            .find(|tool| tool.definition().name == "web_fetch")
            .expect("enabled network must register web_fetch");

        let error = fetch
            .call_with_cancellation_async(json!({"url":"https://example.com/"}), &NoCancellation)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("approval"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn web_fetch_reads_a_bounded_local_fixture_and_reports_truncation() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (seen, received) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let count = stream.read(&mut request).unwrap();
            seen.send(String::from_utf8_lossy(&request[..count]).into_owned())
                .unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 10\r\nConnection: close\r\n\r\n0123456789",
                )
                .unwrap();
        });
        let mut network = NetworkConfig::enabled();
        network.allow_local = true;
        network.allow_port(address.port());
        let output = execute_web_fetch(
            network,
            json!({"url":format!("http://{address}/fixture"),"max_bytes":4}),
            &focus_kernel::NoCancellation,
        )
        .await
        .unwrap();

        assert!(
            received
                .recv()
                .unwrap()
                .starts_with("GET /fixture HTTP/1.1")
        );
        assert!(output.contains("status: 200"));
        assert!(output.contains("truncated: true"));
        assert!(output.ends_with("body:\n0123"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn web_fetch_revalidates_every_redirect_destination() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: http://example.com/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });
        let mut network = NetworkConfig::enabled();
        network.allow_local = true;
        network.allow_port(address.port());
        network
            .insert_domain_rule("127.0.0.1", crate::network::DomainAccess::Allow)
            .unwrap();
        let error = execute_web_fetch(
            network,
            json!({"url":format!("http://{address}/redirect")}),
            &focus_kernel::NoCancellation,
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("not permitted"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn web_fetch_cancels_an_in_flight_http_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (started, received) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            started.send(()).unwrap();
            thread::sleep(Duration::from_millis(200));
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
        });
        let directory =
            std::env::temp_dir().join(format!("runtime-web-fetch-cancel-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut network = NetworkConfig::enabled();
        network.allow_local = true;
        network.allow_port(address.port());
        let policy = Arc::new(PolicyEngine::new(
            Policy {
                network: crate::policy::PolicyDecision::Allow,
                ..Policy::default()
            },
            Arc::new(FixedApproval(true)),
        ));
        let tools =
            builtin_tool_registry(WorkspaceSandbox::new(&directory).unwrap(), network).bind(policy);
        let fetch = tools
            .iter()
            .find(|tool| tool.definition().name == "web_fetch")
            .unwrap();
        let cancellation = crate::subagent::CancellationToken::default();
        let trigger = cancellation.clone();
        let canceller = thread::spawn(move || {
            received
                .recv_timeout(Duration::from_secs(1))
                .expect("fixture must receive a request before cancellation");
            trigger.cancel();
        });

        let error = fetch
            .call_with_cancellation_async(
                json!({"url":format!("http://{address}/slow")}),
                &cancellation,
            )
            .await
            .unwrap_err();

        assert!(matches!(error, KernelError::Cancelled));
        canceller.join().unwrap();
        server.join().unwrap();
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn web_fetch_observes_a_non_shared_cancellation_signal() {
        struct MutableCancellation(Arc<AtomicBool>);

        impl CancellationSignal for MutableCancellation {
            fn is_cancelled(&self) -> bool {
                self.0.load(Ordering::Acquire)
            }
        }

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (started, received) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            started.send(()).unwrap();
            thread::sleep(Duration::from_millis(200));
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
        });
        let flag = Arc::new(AtomicBool::new(false));
        let cancellation = MutableCancellation(flag.clone());
        let trigger = thread::spawn(move || {
            received
                .recv_timeout(Duration::from_secs(1))
                .expect("fixture must receive a request before cancellation");
            flag.store(true, Ordering::Release);
        });
        let mut network = NetworkConfig::enabled();
        network.allow_local = true;
        network.allow_port(address.port());

        let error = execute_web_fetch(
            network,
            json!({"url":format!("http://{address}/slow")}),
            &cancellation,
        )
        .await
        .unwrap_err();

        assert!(matches!(error, RuntimeError::Cancelled));
        trigger.join().unwrap();
        server.join().unwrap();
    }

    #[test]
    fn runtime_cancellation_maps_to_kernel_cancellation() {
        assert!(matches!(
            runtime_as_kernel(RuntimeError::Cancelled),
            KernelError::Cancelled
        ));
    }

    #[test]
    fn approval_arguments_redact_web_fetch_credentials_and_query() {
        let arguments = redact_arguments(json!({
            "url": "https://alice:password@example.com/report?access_token=secret#fragment"
        }));

        assert_eq!(
            arguments["url"],
            "https://<redacted>@example.com/report?<redacted>"
        );
    }

    #[test]
    fn non_cancellation_runtime_errors_remain_kernel_tool_errors() {
        assert!(matches!(
            runtime_as_kernel(RuntimeError::ToolInput("bad input".into())),
            KernelError::Tool { .. }
        ));
    }

    #[tokio::test]
    async fn extension_tools_use_the_same_policy_binding_as_builtins() {
        let mut registry = ToolRegistry::default();
        registry
            .register(RuntimeToolSpec::new(
                definition("extension", "extension", json!({"type":"object"})),
                ToolOperation::Other,
                "exercise extension policy",
                Arc::new(Echo),
            ))
            .unwrap();
        let policy = Arc::new(PolicyEngine::new(
            Policy::default(),
            Arc::new(FixedApproval(false)),
        ));

        let tool = registry.bind(policy).into_iter().next().unwrap();
        let error = tool
            .call_with_cancellation_async(json!({"value":1}), &NoCancellation)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("approval"));
    }

    #[tokio::test]
    async fn builtin_mkdir_tool_creates_safe_nested_directories() {
        let directory =
            std::env::temp_dir().join(format!("runtime-mkdir-tool-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();
        let policy = Arc::new(PolicyEngine::new(
            Policy::default(),
            Arc::new(FixedApproval(true)),
        ));
        let tools = builtin_tool_registry(sandbox, NetworkConfig::default()).bind(policy);
        let mkdir = tools
            .iter()
            .find(|tool| tool.definition().name == "mkdir")
            .expect("mkdir must be a canonical built-in");

        mkdir
            .call_with_cancellation_async(json!({"path":"src/generated"}), &NoCancellation)
            .await
            .unwrap();

        assert!(directory.join("src/generated").is_dir());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn shell_observes_a_non_shared_cancellation_signal() {
        struct MutableCancellation(Arc<AtomicBool>);

        impl CancellationSignal for MutableCancellation {
            fn is_cancelled(&self) -> bool {
                self.0.load(Ordering::Acquire)
            }
        }

        let directory = std::env::temp_dir().join(format!("focus-shell-cancel-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();
        let registry = builtin_tool_registry(sandbox, NetworkConfig::default());
        let policy = Arc::new(PolicyEngine::new(
            Policy::default(),
            Arc::new(FixedApproval(true)),
        ));
        let shell = registry
            .bind(policy)
            .into_iter()
            .find(|tool| tool.definition().name == "shell")
            .unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let cancellation = MutableCancellation(flag.clone());
        let trigger = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            flag.store(true, Ordering::Release);
        });
        let command = if cfg!(windows) {
            "ping 127.0.0.1 -n 6 > NUL"
        } else {
            "sleep 5"
        };

        let error = shell
            .call_with_cancellation_async(json!({"command":command}), &cancellation)
            .await
            .unwrap_err();

        assert!(matches!(error, KernelError::Cancelled));
        trigger.join().unwrap();
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn write_tool_is_blocked_when_approval_is_denied() {
        let directory =
            std::env::temp_dir().join(format!("runtime-tools-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let sandbox = WorkspaceSandbox::new(&directory).unwrap();
        let policy = Arc::new(PolicyEngine::new(
            Policy::default(),
            Arc::new(FixedApproval(false)),
        ));
        let tools = builtin_tool_registry(sandbox, NetworkConfig::default()).bind(policy);
        let write = tools
            .iter()
            .find(|tool| tool.definition().name == "write_file")
            .unwrap();

        let error = write
            .call_with_cancellation_async(
                json!({"path":"blocked.txt","content":"x"}),
                &NoCancellation,
            )
            .await
            .unwrap_err();

        assert!(error.to_string().contains("approval"));
        let _ = std::fs::remove_dir_all(directory);
    }
}
