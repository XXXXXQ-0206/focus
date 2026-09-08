//! Deterministic event-derived ablation benchmark for the Focus runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use focus_kernel::{
    AgentLoop, AgentState, AgentStatus, CancellationSignal, Event, EventKind, EventStore,
    JsonlEventStore, KernelError, Message, ModelEvent, ModelEventStream, ModelProvider,
    ModelRequest, ModelResponse, NoCancellation, PersistingEventSink, Role, Tool, ToolDefinition,
};
use futures::{StreamExt, future::BoxFuture};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::policy::ToolOperation;
use crate::subagent::{CancellationToken, SubagentLimits};
use crate::tools::{RuntimeToolSpec, ToolHandler};
use crate::{FocusRuntime, RunOptions, RuntimeConfig, RuntimeError, approve_all};

/// One execution configuration in the ablation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkMode {
    /// Direct use of the Pi-backed kernel loop without Focus Runtime services.
    PiKernel,
    /// Focus's minimal core capability set.
    FocusCore,
    /// Focus core plus the explicit engineering workflow gate.
    FocusWorkflow,
    /// Focus core plus bounded model-visible delegation.
    FocusDelegation,
}

impl BenchmarkMode {
    /// Stable display name used in receipts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PiKernel => "pi-kernel",
            Self::FocusCore => "focus-core",
            Self::FocusWorkflow => "focus-workflow",
            Self::FocusDelegation => "focus-delegation",
        }
    }
}

/// Runtime dependency responsible for the longest observed blocking interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriticalPathKind {
    /// Provider request until its first visible text delta.
    Provider,
    /// Model-ready tool call waiting for handler execution to start.
    ToolQueue,
    /// Tool handler execution.
    ToolExecution,
    /// Child task waiting to start.
    SubagentQueue,
    /// Child task active execution.
    Subagent,
}

/// Metrics derived from canonical events plus explicit harness setup timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Delay from the first event to the first visible text delta.
    pub first_text_delta_ms: Option<u128>,
    /// First provider request to its first visible text delta.
    pub provider_first_delta_ms: Option<u128>,
    /// Sum of observed provider request-to-first-delta intervals.
    pub provider_wait_ms: Option<u128>,
    /// Duration between the first and last event.
    pub total_runtime_ms: Option<u128>,
    /// Active agent-run wall time measured at the harness boundary.
    pub active_runtime_us: Option<u128>,
    /// End-to-end harness duration including Runtime/session initialization.
    pub end_to_end_runtime_ms: Option<u128>,
    /// Runtime open and immutable setup duration in microseconds.
    pub runtime_initialization_us: Option<u128>,
    /// Fresh session metadata initialization duration in microseconds.
    pub session_initialization_us: Option<u128>,
    /// Sum of tool request-to-result waits.
    pub tool_wait_ms: Option<u128>,
    /// Sum of tool-ready to tool-start queue intervals.
    pub tool_queue_ms: Option<u128>,
    /// Sum of tool-start to tool-result execution intervals.
    pub tool_execution_ms: Option<u128>,
    /// Maximum number of active child tasks.
    pub subagent_peak: usize,
    /// Child active milliseconds divided by peak capacity and runtime.
    pub subagent_utilization_pct: Option<f64>,
    /// Child active milliseconds divided by peak capacity inside the child batch window.
    pub subagent_batch_utilization_pct: Option<f64>,
    /// Sum of child queued-to-start intervals.
    pub subagent_queue_ms: Option<u128>,
    /// Sum of child start-to-terminal active intervals.
    pub subagent_active_ms: Option<u128>,
    /// Longest individual provider, tool, or child blocking interval.
    pub critical_path_ms: Option<u128>,
    /// Dependency responsible for `critical_path_ms`.
    pub critical_path_kind: Option<CriticalPathKind>,
    /// Parent-session deterministic input usage units reported by the fixture.
    pub parent_input_units: u64,
    /// Parent-session deterministic output usage units reported by the fixture.
    pub parent_output_units: u64,
    /// Input usage units reported by isolated child sessions.
    pub child_input_units: u64,
    /// Output usage units reported by isolated child sessions.
    pub child_output_units: u64,
    /// Parent and child input usage units combined.
    pub total_input_units: u64,
    /// Parent and child output usage units combined.
    pub total_output_units: u64,
    /// Whether the run reached a complete terminal state with intact events.
    pub success: bool,
    /// Last observed lifecycle state.
    pub terminal: String,
    /// Whether event ordering and tool/child terminal pairing are valid.
    pub event_integrity: bool,
    /// Number of events in the analyzed stream.
    pub event_count: usize,
}

/// One mode/run result in a benchmark receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    /// Matrix mode.
    pub mode: BenchmarkMode,
    /// Repetition index starting at one.
    pub iteration: usize,
    /// Session identity used by the run.
    pub session_id: Uuid,
    /// Derived metrics, even when the run failed.
    pub metrics: BenchmarkMetrics,
    /// Redacted terminal error, when present.
    pub error: Option<String>,
}

/// Complete deterministic benchmark output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReceipt {
    /// Receipt schema version.
    pub schema_version: u32,
    /// Fixture identifier.
    pub fixture: String,
    /// Number of repetitions per mode.
    pub runs: usize,
    /// Individual samples in deterministic execution order.
    pub records: Vec<BenchmarkRecord>,
    /// Per-mode aggregates derived from all samples.
    pub summaries: Vec<BenchmarkSummary>,
    /// Deterministic provider-failure convergence probes.
    pub failure_probes: Vec<FailureProbe>,
    /// Deterministic in-flight cancellation convergence probes.
    pub cancellation_probes: Vec<CancellationProbe>,
}

/// Evidence that a provider failure remains a failed terminal run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureProbe {
    /// Matrix mode under test.
    pub mode: BenchmarkMode,
    /// Probe session identity.
    pub session_id: Uuid,
    /// Metrics derived without overriding the observed terminal state.
    pub metrics: BenchmarkMetrics,
    /// Actual run error.
    pub error: Option<String>,
    /// Whether the provider's normalized `Failed` event was persisted.
    pub failed_event_observed: bool,
    /// Whether all expected failure convergence facts were observed.
    pub passed: bool,
}

/// Evidence that in-flight cancellation converges to one cancelled terminal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancellationProbe {
    /// Matrix mode under test.
    pub mode: BenchmarkMode,
    /// Probe session identity.
    pub session_id: Uuid,
    /// Metrics derived without overriding the observed terminal state.
    pub metrics: BenchmarkMetrics,
    /// Actual cancellation error.
    pub error: Option<String>,
    /// Whether the provider's normalized `Cancelled` event was persisted.
    pub cancelled_event_observed: bool,
    /// Whether all expected cancellation convergence facts were observed.
    pub passed: bool,
}

/// Median and success aggregates for one matrix mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Matrix mode.
    pub mode: BenchmarkMode,
    /// Percentage of successful samples.
    pub success_rate_pct: f64,
    /// Percentage of samples with complete event integrity.
    pub event_integrity_rate_pct: f64,
    /// Median first visible text delay.
    pub median_first_text_delta_ms: Option<f64>,
    /// Median provider request-to-first-delta delay.
    pub median_provider_first_delta_ms: Option<f64>,
    /// Median summed provider wait.
    pub median_provider_wait_ms: Option<f64>,
    /// Median total event duration.
    pub median_total_runtime_ms: Option<f64>,
    /// Median active agent-run wall time at microsecond resolution.
    pub median_active_runtime_us: Option<f64>,
    /// P90 event-span duration.
    pub p90_total_runtime_ms: Option<f64>,
    /// P90 active agent-run wall time at microsecond resolution.
    pub p90_active_runtime_us: Option<f64>,
    /// Median end-to-end duration including setup.
    pub median_end_to_end_runtime_ms: Option<f64>,
    /// Median Runtime open and immutable setup duration.
    pub median_runtime_initialization_us: Option<f64>,
    /// Median fresh session metadata initialization duration.
    pub median_session_initialization_us: Option<f64>,
    /// Median summed tool wait.
    pub median_tool_wait_ms: Option<f64>,
    /// Median summed tool queue delay.
    pub median_tool_queue_ms: Option<f64>,
    /// Median summed tool execution time.
    pub median_tool_execution_ms: Option<f64>,
    /// Median parent-session deterministic input usage units.
    pub median_parent_input_units: f64,
    /// Median parent-session deterministic output usage units.
    pub median_parent_output_units: f64,
    /// Median child-session deterministic input usage units.
    pub median_child_input_units: f64,
    /// Median child-session deterministic output usage units.
    pub median_child_output_units: f64,
    /// Median combined parent and child input usage units.
    pub median_total_input_units: f64,
    /// Median combined parent and child output usage units.
    pub median_total_output_units: f64,
    /// Median peak active subagents.
    pub median_subagent_peak: f64,
    /// Median utilization among samples that launched subagents.
    pub median_subagent_utilization_pct: Option<f64>,
    /// Median utilization inside the first-start to last-terminal child batch window.
    pub median_subagent_batch_utilization_pct: Option<f64>,
    /// Median summed child queue delay.
    pub median_subagent_queue_ms: Option<f64>,
    /// Median summed child active time.
    pub median_subagent_active_ms: Option<f64>,
    /// Median longest dependency interval.
    pub median_critical_path_ms: Option<f64>,
}

/// Analyze a canonical event stream without consulting wall-clock telemetry.
pub fn analyze_events(
    events: &[Event],
    _mode: BenchmarkMode,
) -> Result<BenchmarkMetrics, RuntimeError> {
    if events.is_empty() {
        return Err(RuntimeError::Session("benchmark produced no events".into()));
    }
    let first = events.first().map_or(0, |event| event.timestamp_ms);
    let last = events.last().map_or(first, |event| event.timestamp_ms);
    let mut first_text_delta_ms = None;
    let mut provider_request_started = None;
    let mut provider_first_delta_ms = None;
    let mut provider_wait_ms = 0_u128;
    let mut provider_intervals = 0_usize;
    let mut tool_queued = BTreeMap::<String, u128>::new();
    let mut tool_requests = BTreeMap::<String, u128>::new();
    let mut tool_queue_ms = 0_u128;
    let mut tool_wait_ms = 0_u128;
    let mut integrity = true;
    let mut parent_input_units = 0_u64;
    let mut parent_output_units = 0_u64;
    let mut terminal = "unknown".to_owned();
    let mut seen_ids = BTreeSet::new();
    let mut queued_children = BTreeMap::<String, u128>::new();
    let mut active_children = BTreeMap::<String, u128>::new();
    let mut child_queue_total = 0_u128;
    let mut child_active_total = 0_u128;
    let mut subagent_peak = 0_usize;
    let mut first_child_start = None::<u128>;
    let mut last_child_terminal = None::<u128>;
    let mut previous_timestamp = first;
    let mut terminal_count = 0_usize;
    let mut terminal_seen = false;
    let mut critical_path = None::<(u128, CriticalPathKind)>;

    let update_critical =
        |duration: u128,
         kind: CriticalPathKind,
         critical: &mut Option<(u128, CriticalPathKind)>| {
            if critical.is_none_or(|(current, _)| duration > current) {
                *critical = Some((duration, kind));
            }
        };

    for event in events {
        if terminal_seen {
            integrity = false;
        }
        if event.session_id != events[0].session_id
            || event.timestamp_ms < previous_timestamp
            || !seen_ids.insert(event.id)
        {
            integrity = false;
        }
        previous_timestamp = event.timestamp_ms;
        match &event.kind {
            EventKind::Model {
                event: ModelEvent::RequestStarted { .. },
            } => {
                provider_request_started = Some(event.timestamp_ms);
            }
            EventKind::Model {
                event: ModelEvent::TextDelta { .. },
            } => {
                if first_text_delta_ms.is_none() {
                    first_text_delta_ms = event.timestamp_ms.checked_sub(first);
                }
                if let Some(start) = provider_request_started.take() {
                    if let Some(duration) = event.timestamp_ms.checked_sub(start) {
                        provider_first_delta_ms.get_or_insert(duration);
                        provider_wait_ms = provider_wait_ms.saturating_add(duration);
                        provider_intervals += 1;
                        update_critical(duration, CriticalPathKind::Provider, &mut critical_path);
                    } else {
                        integrity = false;
                    }
                }
            }
            EventKind::Model {
                event:
                    ModelEvent::Usage {
                        input_tokens: input,
                        output_tokens: output,
                    },
            } => {
                parent_input_units = parent_input_units.saturating_add(*input);
                parent_output_units = parent_output_units.saturating_add(*output);
            }
            EventKind::Model {
                event: ModelEvent::ToolCallReady { call },
            } => {
                if tool_queued
                    .insert(call.id.clone(), event.timestamp_ms)
                    .is_some()
                {
                    integrity = false;
                }
            }
            EventKind::ToolCallRequested { call } => {
                match tool_queued.remove(&call.id) {
                    Some(queued) if event.timestamp_ms >= queued => {
                        let duration = event.timestamp_ms - queued;
                        tool_queue_ms = tool_queue_ms.saturating_add(duration);
                        update_critical(duration, CriticalPathKind::ToolQueue, &mut critical_path);
                    }
                    _ => integrity = false,
                }
                if tool_requests
                    .insert(call.id.clone(), event.timestamp_ms)
                    .is_some()
                {
                    integrity = false;
                }
            }
            EventKind::ToolResultReceived { result } => {
                match tool_requests.remove(&result.tool_call_id) {
                    Some(start) if event.timestamp_ms >= start => {
                        let duration = event.timestamp_ms - start;
                        tool_wait_ms = tool_wait_ms.saturating_add(duration);
                        update_critical(
                            duration,
                            CriticalPathKind::ToolExecution,
                            &mut critical_path,
                        );
                    }
                    _ => integrity = false,
                }
            }
            EventKind::Runtime { name, data } if name == "subagent_queued" => {
                if let Some(id) = data.get("task_id").and_then(Value::as_str) {
                    if queued_children
                        .insert(id.to_owned(), event.timestamp_ms)
                        .is_some()
                    {
                        integrity = false;
                    }
                } else {
                    integrity = false;
                }
            }
            EventKind::Runtime { name, data } if name == "subagent_started" => {
                if let Some(id) = data.get("task_id").and_then(Value::as_str) {
                    match queued_children.remove(id) {
                        Some(queued) if event.timestamp_ms >= queued => {
                            let duration = event.timestamp_ms - queued;
                            child_queue_total = child_queue_total.saturating_add(duration);
                            update_critical(
                                duration,
                                CriticalPathKind::SubagentQueue,
                                &mut critical_path,
                            );
                        }
                        _ => integrity = false,
                    }
                    if active_children
                        .insert(id.to_owned(), event.timestamp_ms)
                        .is_some()
                    {
                        integrity = false;
                    }
                    first_child_start.get_or_insert(event.timestamp_ms);
                    subagent_peak = subagent_peak.max(active_children.len());
                } else {
                    integrity = false;
                }
            }
            EventKind::Runtime { name, data }
                if matches!(
                    name.as_str(),
                    "subagent_completed" | "subagent_failed" | "subagent_cancelled"
                ) =>
            {
                if let Some(id) = data.get("task_id").and_then(Value::as_str) {
                    match active_children.remove(id) {
                        Some(start) if event.timestamp_ms >= start => {
                            let duration = event.timestamp_ms - start;
                            child_active_total = child_active_total.saturating_add(duration);
                            last_child_terminal =
                                Some(last_child_terminal.map_or(event.timestamp_ms, |last| {
                                    last.max(event.timestamp_ms)
                                }));
                            update_critical(
                                duration,
                                CriticalPathKind::Subagent,
                                &mut critical_path,
                            );
                        }
                        _ => integrity = false,
                    }
                } else {
                    integrity = false;
                }
            }
            EventKind::StateChanged { status, .. } => {
                terminal = match status {
                    AgentStatus::Complete => "complete",
                    AgentStatus::Failed => "failed",
                    AgentStatus::Cancelled => "cancelled",
                    AgentStatus::AwaitingModel => "awaiting_model",
                    AgentStatus::ExecutingTools => "executing_tools",
                    AgentStatus::Idle => "idle",
                }
                .into();
                if matches!(
                    status,
                    AgentStatus::Complete | AgentStatus::Failed | AgentStatus::Cancelled
                ) {
                    terminal_count += 1;
                    terminal_seen = true;
                }
            }
            _ => {}
        }
    }
    if !tool_queued.is_empty()
        || !tool_requests.is_empty()
        || !queued_children.is_empty()
        || !active_children.is_empty()
        || terminal_count != 1
    {
        integrity = false;
    }
    let total_runtime_ms = last.checked_sub(first);
    let subagent_utilization_pct = (subagent_peak > 0 && last > first).then(|| {
        ((child_active_total as f64) / (subagent_peak as f64 * (last - first) as f64)
            * 100.0
            * 10_000.0)
            .round()
            / 10_000.0
    });
    let subagent_batch_utilization_pct = first_child_start
        .zip(last_child_terminal)
        .filter(|(start, terminal)| terminal > start)
        .map(|(start, terminal)| {
            ((child_active_total as f64) / (subagent_peak as f64 * (terminal - start) as f64)
                * 100.0
                * 10_000.0)
                .round()
                / 10_000.0
        });
    Ok(BenchmarkMetrics {
        first_text_delta_ms,
        provider_first_delta_ms,
        provider_wait_ms: (provider_intervals > 0).then_some(provider_wait_ms),
        total_runtime_ms,
        active_runtime_us: None,
        end_to_end_runtime_ms: None,
        runtime_initialization_us: None,
        session_initialization_us: None,
        tool_wait_ms: Some(tool_wait_ms),
        tool_queue_ms: Some(tool_queue_ms),
        tool_execution_ms: Some(tool_wait_ms),
        subagent_peak,
        subagent_utilization_pct,
        subagent_batch_utilization_pct,
        subagent_queue_ms: (subagent_peak > 0).then_some(child_queue_total),
        subagent_active_ms: (subagent_peak > 0).then_some(child_active_total),
        critical_path_ms: critical_path.map(|(duration, _)| duration),
        critical_path_kind: critical_path.map(|(_, kind)| kind),
        parent_input_units,
        parent_output_units,
        child_input_units: 0,
        child_output_units: 0,
        total_input_units: parent_input_units,
        total_output_units: parent_output_units,
        success: terminal == "complete" && integrity,
        terminal,
        event_integrity: integrity,
        event_count: events.len(),
    })
}

#[derive(Debug)]
struct FixtureProvider;

#[async_trait::async_trait]
impl ModelProvider for FixtureProvider {
    async fn stream(
        &self,
        request: ModelRequest,
        _cancellation: &dyn CancellationSignal,
    ) -> Result<ModelEventStream, KernelError> {
        wait_fixture(Duration::from_millis(2)).await;
        let child = request
            .messages
            .iter()
            .any(|message| message.content.contains("Act as the `"));
        if child {
            wait_fixture(Duration::from_millis(30)).await;
        }
        let names = request
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<BTreeSet<_>>();
        let has_result = |name: &str| {
            request.messages.iter().any(|message| {
                message.role == Role::Tool && message.tool_name.as_deref() == Some(name)
            })
        };
        let call = if child {
            None
        } else if names.contains("delegate") && !has_result("delegate") {
            Some((
                "delegate",
                json!({"mode":"parallel","tasks":[{"role":"reviewer","objective":"inspect fixture"},{"role":"verifier","objective":"verify fixture"}]}),
            ))
        } else if names.contains("workflow_checkpoint") && !has_result("search") {
            Some(("search", json!({"query":"fixture"})))
        } else if names.contains("workflow_checkpoint") && !has_result("workflow_checkpoint") {
            Some((
                "workflow_checkpoint",
                json!({"kind":"plan","summary":"fixture plan"}),
            ))
        } else if names.contains("workflow_checkpoint")
            && !request
                .messages
                .iter()
                .any(|message| message.role == Role::Tool && message.content.contains("no_change"))
        {
            Some((
                "workflow_checkpoint",
                json!({"kind":"no_change","summary":"fixture requires no mutation"}),
            ))
        } else if names.contains("workflow_checkpoint") && !has_result("shell") {
            Some(("shell", json!({"command":"exit 0","purpose":"verify"})))
        } else if names.contains("workflow_checkpoint")
            && !request
                .messages
                .iter()
                .any(|message| message.role == Role::Tool && message.content.contains("review"))
        {
            Some((
                "workflow_checkpoint",
                json!({"kind":"review","summary":"fixture reviewed"}),
            ))
        } else if names.contains("fixture_probe") && !has_result("fixture_probe") {
            Some(("fixture_probe", json!({})))
        } else {
            None
        };
        let mut events = vec![
            Ok(ModelEvent::RequestStarted {
                provider: "fixture".into(),
                model: "deterministic".into(),
                endpoint: "fixture://benchmark".into(),
            }),
            Ok(ModelEvent::Usage {
                input_tokens: request.messages.len() as u64,
                output_tokens: 4,
            }),
        ];
        if let Some((name, arguments)) = call {
            let progress = "fixture progress";
            let call = focus_kernel::ToolCall {
                id: format!("fixture-{}", Uuid::new_v4()),
                name: name.into(),
                arguments,
            };
            events.extend([
                Ok(ModelEvent::TextDelta {
                    text: "fixture ".into(),
                }),
                Ok(ModelEvent::TextDelta {
                    text: "progress".into(),
                }),
                Ok(ModelEvent::ToolCallStarted {
                    id: call.id.clone(),
                    name: call.name.clone(),
                }),
                Ok(ModelEvent::ToolArgumentsDelta {
                    id: call.id.clone(),
                    json_fragment: serde_json::to_string(&call.arguments).unwrap(),
                }),
                Ok(ModelEvent::ToolCallReady { call: call.clone() }),
                Ok(ModelEvent::Completed {
                    response: ModelResponse {
                        content: progress.into(),
                        tool_calls: vec![call],
                    },
                }),
            ]);
        } else {
            let content = if child {
                "child fixture complete"
            } else {
                "fixture complete"
            };
            events.extend([
                Ok(ModelEvent::TextDelta {
                    text: content[..content.len() / 2].into(),
                }),
                Ok(ModelEvent::TextDelta {
                    text: content[content.len() / 2..].into(),
                }),
                Ok(ModelEvent::Completed {
                    response: ModelResponse {
                        content: content.into(),
                        tool_calls: Vec::new(),
                    },
                }),
            ]);
        }
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

#[derive(Debug)]
struct FailingFixtureProvider;

#[async_trait::async_trait]
impl ModelProvider for FailingFixtureProvider {
    async fn stream(
        &self,
        _request: ModelRequest,
        _cancellation: &dyn CancellationSignal,
    ) -> Result<ModelEventStream, KernelError> {
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ModelEvent::RequestStarted {
                provider: "fixture".into(),
                model: "missing".into(),
                endpoint: "fixture://benchmark-failure".into(),
            }),
            Ok(ModelEvent::Failed {
                error: "deterministic provider failure".into(),
            }),
        ])))
    }
}

#[derive(Debug)]
struct CancellingFixtureProvider {
    started: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl ModelProvider for CancellingFixtureProvider {
    async fn stream(
        &self,
        _request: ModelRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ModelEventStream, KernelError> {
        self.started.store(true, Ordering::Release);
        let cancellation = cancellation.shared_flag().unwrap_or_default();
        let start = futures::stream::once(async {
            Ok(ModelEvent::RequestStarted {
                provider: "fixture".into(),
                model: "cancelled".into(),
                endpoint: "fixture://benchmark-cancellation".into(),
            })
        });
        let wait = futures::stream::unfold((), move |_| {
            let cancellation = cancellation.clone();
            async move {
                while !cancellation.load(std::sync::atomic::Ordering::Acquire) {
                    tokio::task::yield_now().await;
                }
                Some((Ok(ModelEvent::Cancelled), ()))
            }
        });
        Ok(Box::pin(start.chain(wait)))
    }
}

#[derive(Debug)]
struct FixtureSearchTool;

async fn execute_fixture_probe() -> String {
    wait_fixture(Duration::from_millis(8)).await;
    "fixture probe result".into()
}

async fn wait_fixture(duration: Duration) {
    let deadline = Instant::now() + duration;
    while Instant::now() < deadline {
        tokio::task::yield_now().await;
    }
}

impl Tool for FixtureSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fixture_probe".into(),
            description: "deterministic fixture probe".into(),
            input_schema: json!({"type":"object"}),
        }
    }
    fn call_with_cancellation_async<'a>(
        &'a self,
        _arguments: Value,
        _cancellation: &dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, KernelError>> {
        Box::pin(async { Ok(execute_fixture_probe().await) })
    }
}

#[derive(Debug)]
struct FixtureProbeHandler;

impl ToolHandler for FixtureProbeHandler {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        _arguments: Value,
        _cancellation: &dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        Box::pin(async { Ok(execute_fixture_probe().await) })
    }
}

fn fixture_probe_spec() -> RuntimeToolSpec {
    RuntimeToolSpec::new(
        ToolDefinition {
            name: "fixture_probe".into(),
            description: "deterministic fixture probe".into(),
            input_schema: json!({"type":"object"}),
        },
        ToolOperation::Other,
        "run deterministic ablation fixture",
        Arc::new(FixtureProbeHandler),
    )
}

/// Run the four deterministic matrix modes for `runs` repetitions.
pub fn run_deterministic(runs: usize) -> Result<BenchmarkReceipt, RuntimeError> {
    if runs == 0 {
        return Err(RuntimeError::ToolInput(
            "benchmark runs must be positive".into(),
        ));
    }
    let modes = [
        BenchmarkMode::PiKernel,
        BenchmarkMode::FocusCore,
        BenchmarkMode::FocusWorkflow,
        BenchmarkMode::FocusDelegation,
    ];
    let mut records = Vec::with_capacity(runs * modes.len());
    for iteration in 1..=runs {
        for mode in modes {
            let root_id = Uuid::new_v4();
            let observation = run_one(mode, root_id);
            let session_id = observation.session_id;
            let events = observation.events;
            let error = observation.error;
            let metrics = if events.is_empty() {
                BenchmarkMetrics {
                    first_text_delta_ms: None,
                    provider_first_delta_ms: None,
                    provider_wait_ms: None,
                    total_runtime_ms: None,
                    active_runtime_us: observation.active_runtime_us,
                    end_to_end_runtime_ms: Some(observation.end_to_end_runtime_ms),
                    runtime_initialization_us: observation.runtime_initialization_us,
                    session_initialization_us: observation.session_initialization_us,
                    tool_wait_ms: None,
                    tool_queue_ms: None,
                    tool_execution_ms: None,
                    subagent_peak: 0,
                    subagent_utilization_pct: None,
                    subagent_batch_utilization_pct: None,
                    subagent_queue_ms: None,
                    subagent_active_ms: None,
                    critical_path_ms: None,
                    critical_path_kind: None,
                    parent_input_units: 0,
                    parent_output_units: 0,
                    child_input_units: observation.child_input_units,
                    child_output_units: observation.child_output_units,
                    total_input_units: observation.child_input_units,
                    total_output_units: observation.child_output_units,
                    success: false,
                    terminal: "failed".into(),
                    event_integrity: false,
                    event_count: 0,
                }
            } else {
                analyze_events(&events, mode)?
            };
            let mut metrics = metrics;
            metrics.active_runtime_us = observation.active_runtime_us;
            metrics.end_to_end_runtime_ms = Some(
                observation
                    .end_to_end_runtime_ms
                    .max(metrics.total_runtime_ms.unwrap_or_default()),
            );
            metrics.runtime_initialization_us = observation.runtime_initialization_us;
            metrics.session_initialization_us = observation.session_initialization_us;
            metrics.child_input_units = observation.child_input_units;
            metrics.child_output_units = observation.child_output_units;
            metrics.total_input_units = metrics
                .parent_input_units
                .saturating_add(metrics.child_input_units);
            metrics.total_output_units = metrics
                .parent_output_units
                .saturating_add(metrics.child_output_units);
            metrics.event_integrity &= observation.live_matches_replay;
            metrics.success &= metrics.event_integrity && error.is_none();
            records.push(BenchmarkRecord {
                mode,
                iteration,
                session_id,
                metrics,
                error,
            });
        }
    }
    let summaries = modes
        .into_iter()
        .map(|mode| summarize(mode, &records))
        .collect();
    let failure_probes = [BenchmarkMode::PiKernel, BenchmarkMode::FocusCore]
        .into_iter()
        .map(run_failure_probe)
        .collect::<Result<Vec<_>, _>>()?;
    let cancellation_probes = [BenchmarkMode::PiKernel, BenchmarkMode::FocusCore]
        .into_iter()
        .map(run_cancellation_probe)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BenchmarkReceipt {
        schema_version: 3,
        fixture: "focus-ablation-v3".into(),
        runs,
        records,
        summaries,
        failure_probes,
        cancellation_probes,
    })
}

fn run_failure_probe(mode: BenchmarkMode) -> Result<FailureProbe, RuntimeError> {
    let root_id = Uuid::new_v4();
    let (session_id, events, live_matches_replay, error) = if mode == BenchmarkMode::PiKernel {
        let root = std::env::temp_dir().join(format!("focus-ablation-pi-failure-{root_id}"));
        let store = Arc::new(JsonlEventStore::new(root.join("events"))?);
        let sink = Arc::new(PersistingEventSink::new(store.clone()));
        let mut state = AgentState::new(root_id);
        state
            .transcript
            .push(Message::text(Role::User, "fail deterministically"));
        let error = AgentLoop::new(
            Arc::new(FailingFixtureProvider),
            std::iter::empty::<Arc<dyn Tool>>(),
        )
        .run_with_cancellation(&mut state, sink.clone(), &NoCancellation)
        .err()
        .map(|error| error.to_string());
        let events = store.load(root_id).map_err(RuntimeError::Kernel)?;
        let _ = std::fs::remove_dir_all(root);
        (root_id, events, true, error)
    } else {
        let root = std::env::temp_dir().join(format!("focus-ablation-failure-{root_id}"));
        std::fs::create_dir_all(&root).map_err(|error| RuntimeError::Session(error.to_string()))?;
        let mut config = RuntimeConfig::for_workspace(&root);
        config.data_root = root.join("state");
        let runtime = FocusRuntime::open(config)?;
        let live = runtime.subscribe();
        let session = runtime.create_session("benchmark failure")?;
        let error = runtime
            .run_in_session_with_options(
                session.id,
                "fail deterministically",
                Arc::new(FailingFixtureProvider),
                RunOptions::core(approve_all()),
            )
            .err()
            .map(|error| error.to_string());
        let events = runtime.replay(session.id)?;
        let live = live
            .try_iter()
            .filter(|event| event.session_id == session.id)
            .collect::<Vec<_>>();
        let live_matches_replay = live
            .iter()
            .map(|event| event.id)
            .eq(events.iter().map(|event| event.id));
        let _ = std::fs::remove_dir_all(root);
        (session.id, events, live_matches_replay, error)
    };
    let failed_event_observed = events.iter().any(|event| {
        matches!(
            event.kind,
            EventKind::Model {
                event: ModelEvent::Failed { .. }
            }
        )
    });
    let mut metrics = analyze_events(&events, mode)?;
    metrics.event_integrity &= live_matches_replay;
    metrics.success &= metrics.event_integrity && error.is_none();
    let passed = error.is_some()
        && metrics.terminal == "failed"
        && !metrics.success
        && metrics.event_integrity
        && failed_event_observed;
    Ok(FailureProbe {
        mode,
        session_id,
        metrics,
        error,
        failed_event_observed,
        passed,
    })
}

fn run_cancellation_probe(mode: BenchmarkMode) -> Result<CancellationProbe, RuntimeError> {
    let root_id = Uuid::new_v4();
    let cancellation = CancellationToken::default();
    let provider_started = Arc::new(AtomicBool::new(false));
    let trigger = {
        let cancellation = cancellation.clone();
        let provider_started = provider_started.clone();
        std::thread::spawn(move || {
            while !provider_started.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            cancellation.cancel();
        })
    };
    let (session_id, events, live_matches_replay, error) = if mode == BenchmarkMode::PiKernel {
        let root = std::env::temp_dir().join(format!("focus-ablation-pi-cancel-{root_id}"));
        let store = Arc::new(JsonlEventStore::new(root.join("events"))?);
        let sink = Arc::new(PersistingEventSink::new(store.clone()));
        let mut state = AgentState::new(root_id);
        state
            .transcript
            .push(Message::text(Role::User, "cancel deterministically"));
        let error = AgentLoop::new(
            Arc::new(CancellingFixtureProvider {
                started: provider_started.clone(),
            }),
            std::iter::empty::<Arc<dyn Tool>>(),
        )
        .run_with_cancellation(&mut state, sink, &cancellation)
        .err()
        .map(|error| error.to_string());
        let events = store.load(root_id).map_err(RuntimeError::Kernel)?;
        let _ = std::fs::remove_dir_all(root);
        (root_id, events, true, error)
    } else {
        let root = std::env::temp_dir().join(format!("focus-ablation-cancel-{root_id}"));
        std::fs::create_dir_all(&root).map_err(|error| RuntimeError::Session(error.to_string()))?;
        let mut config = RuntimeConfig::for_workspace(&root);
        config.data_root = root.join("state");
        let runtime = FocusRuntime::open(config)?;
        let live = runtime.subscribe();
        let session = runtime.create_session("benchmark cancellation")?;
        let mut options = RunOptions::core(approve_all());
        options.cancellation = Arc::new(cancellation.clone());
        let error = runtime
            .run_in_session_with_options(
                session.id,
                "cancel deterministically",
                Arc::new(CancellingFixtureProvider {
                    started: provider_started,
                }),
                options,
            )
            .err()
            .map(|error| error.to_string());
        let events = runtime.replay(session.id)?;
        let live = live
            .try_iter()
            .filter(|event| event.session_id == session.id)
            .collect::<Vec<_>>();
        let live_matches_replay = live_semantically_matches_replay(&live, &events);
        let _ = std::fs::remove_dir_all(root);
        (session.id, events, live_matches_replay, error)
    };
    trigger
        .join()
        .map_err(|_| RuntimeError::Session("cancellation trigger panicked".into()))?;
    let cancelled_event_observed = events.iter().any(|event| {
        matches!(
            event.kind,
            EventKind::Model {
                event: ModelEvent::Cancelled
            }
        )
    });
    let mut metrics = analyze_events(&events, mode)?;
    metrics.event_integrity &= live_matches_replay;
    metrics.success &= metrics.event_integrity && error.is_none();
    let passed = error.is_some()
        && metrics.terminal == "cancelled"
        && !metrics.success
        && metrics.event_integrity
        && cancelled_event_observed;
    Ok(CancellationProbe {
        mode,
        session_id,
        metrics,
        error,
        cancelled_event_observed,
        passed,
    })
}

fn summarize(mode: BenchmarkMode, records: &[BenchmarkRecord]) -> BenchmarkSummary {
    let samples = records
        .iter()
        .filter(|record| record.mode == mode)
        .collect::<Vec<_>>();
    let sample_count = samples.len() as f64;
    let percentage = |count: usize| {
        if sample_count == 0.0 {
            0.0
        } else {
            count as f64 / sample_count * 100.0
        }
    };
    BenchmarkSummary {
        mode,
        success_rate_pct: percentage(
            samples
                .iter()
                .filter(|record| record.metrics.success)
                .count(),
        ),
        event_integrity_rate_pct: percentage(
            samples
                .iter()
                .filter(|record| record.metrics.event_integrity)
                .count(),
        ),
        median_first_text_delta_ms: optional_median(&samples, |metrics| {
            metrics.first_text_delta_ms.map(|value| value as f64)
        }),
        median_provider_first_delta_ms: optional_median(&samples, |metrics| {
            metrics.provider_first_delta_ms.map(|value| value as f64)
        }),
        median_provider_wait_ms: optional_median(&samples, |metrics| {
            metrics.provider_wait_ms.map(|value| value as f64)
        }),
        median_total_runtime_ms: optional_median(&samples, |metrics| {
            metrics.total_runtime_ms.map(|value| value as f64)
        }),
        median_active_runtime_us: optional_median(&samples, |metrics| {
            metrics.active_runtime_us.map(|value| value as f64)
        }),
        p90_total_runtime_ms: optional_percentile(&samples, |metrics| {
            metrics.total_runtime_ms.map(|value| value as f64)
        }),
        p90_active_runtime_us: optional_percentile(&samples, |metrics| {
            metrics.active_runtime_us.map(|value| value as f64)
        }),
        median_end_to_end_runtime_ms: optional_median(&samples, |metrics| {
            metrics.end_to_end_runtime_ms.map(|value| value as f64)
        }),
        median_runtime_initialization_us: optional_median(&samples, |metrics| {
            metrics.runtime_initialization_us.map(|value| value as f64)
        }),
        median_session_initialization_us: optional_median(&samples, |metrics| {
            metrics.session_initialization_us.map(|value| value as f64)
        }),
        median_tool_wait_ms: optional_median(&samples, |metrics| {
            metrics.tool_wait_ms.map(|value| value as f64)
        }),
        median_tool_queue_ms: optional_median(&samples, |metrics| {
            metrics.tool_queue_ms.map(|value| value as f64)
        }),
        median_tool_execution_ms: optional_median(&samples, |metrics| {
            metrics.tool_execution_ms.map(|value| value as f64)
        }),
        median_parent_input_units: required_median(&samples, |metrics| {
            metrics.parent_input_units as f64
        }),
        median_parent_output_units: required_median(&samples, |metrics| {
            metrics.parent_output_units as f64
        }),
        median_child_input_units: required_median(&samples, |metrics| {
            metrics.child_input_units as f64
        }),
        median_child_output_units: required_median(&samples, |metrics| {
            metrics.child_output_units as f64
        }),
        median_total_input_units: required_median(&samples, |metrics| {
            metrics.total_input_units as f64
        }),
        median_total_output_units: required_median(&samples, |metrics| {
            metrics.total_output_units as f64
        }),
        median_subagent_peak: required_median(&samples, |metrics| metrics.subagent_peak as f64),
        median_subagent_utilization_pct: optional_median(&samples, |metrics| {
            metrics.subagent_utilization_pct
        }),
        median_subagent_batch_utilization_pct: optional_median(&samples, |metrics| {
            metrics.subagent_batch_utilization_pct
        }),
        median_subagent_queue_ms: optional_median(&samples, |metrics| {
            metrics.subagent_queue_ms.map(|value| value as f64)
        }),
        median_subagent_active_ms: optional_median(&samples, |metrics| {
            metrics.subagent_active_ms.map(|value| value as f64)
        }),
        median_critical_path_ms: optional_median(&samples, |metrics| {
            metrics.critical_path_ms.map(|value| value as f64)
        }),
    }
}

fn optional_median(
    records: &[&BenchmarkRecord],
    metric: impl Fn(&BenchmarkMetrics) -> Option<f64>,
) -> Option<f64> {
    median(
        records
            .iter()
            .filter_map(|record| metric(&record.metrics))
            .collect(),
    )
}

fn required_median(records: &[&BenchmarkRecord], metric: impl Fn(&BenchmarkMetrics) -> f64) -> f64 {
    optional_median(records, |metrics| Some(metric(metrics))).unwrap_or_default()
}

fn optional_percentile(
    records: &[&BenchmarkRecord],
    metric: impl Fn(&BenchmarkMetrics) -> Option<f64>,
) -> Option<f64> {
    percentile(
        records
            .iter()
            .filter_map(|record| metric(&record.metrics))
            .collect(),
        0.90,
    )
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[midpoint - 1] + values[midpoint]) / 2.0)
    } else {
        Some(values[midpoint])
    }
}

fn percentile(mut values: Vec<f64>, quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * quantile).ceil() as usize).saturating_sub(1);
    values.get(index.min(values.len() - 1)).copied()
}

struct RunObservation {
    session_id: Uuid,
    events: Vec<Event>,
    live_matches_replay: bool,
    error: Option<String>,
    end_to_end_runtime_ms: u128,
    active_runtime_us: Option<u128>,
    runtime_initialization_us: Option<u128>,
    session_initialization_us: Option<u128>,
    child_input_units: u64,
    child_output_units: u64,
}

fn canonical_events(events: &[Event]) -> Vec<Event> {
    let mut canonical: Vec<Event> = Vec::with_capacity(events.len());
    for event in events {
        let merged =
            canonical
                .last_mut()
                .is_some_and(|previous| match (&mut previous.kind, &event.kind) {
                    (
                        EventKind::Model {
                            event: ModelEvent::TextDelta { text: existing },
                        },
                        EventKind::Model {
                            event: ModelEvent::TextDelta { text: incoming },
                        },
                    ) => {
                        existing.push_str(incoming);
                        true
                    }
                    (
                        EventKind::Model {
                            event:
                                ModelEvent::ToolArgumentsDelta {
                                    id: existing_id,
                                    json_fragment: existing,
                                },
                        },
                        EventKind::Model {
                            event:
                                ModelEvent::ToolArgumentsDelta {
                                    id: incoming_id,
                                    json_fragment: incoming,
                                },
                        },
                    ) if existing_id == incoming_id => {
                        existing.push_str(incoming);
                        true
                    }
                    _ => false,
                });
        if !merged {
            canonical.push(event.clone());
        }
    }
    canonical
}

fn live_semantically_matches_replay(live: &[Event], replay: &[Event]) -> bool {
    let live = canonical_events(live);
    let replay = canonical_events(replay);
    live.len() == replay.len()
        && live.iter().zip(&replay).all(|(live, replay)| {
            live.id == replay.id
                && live.session_id == replay.session_id
                && serde_json::to_value(&live.kind).ok() == serde_json::to_value(&replay.kind).ok()
        })
}

fn child_usage_units(runtime: &FocusRuntime, root_id: Uuid) -> Result<(u64, u64), RuntimeError> {
    let sessions = runtime.list_sessions()?;
    let by_id = sessions
        .iter()
        .map(|session| (session.id, session))
        .collect::<BTreeMap<_, _>>();
    let is_descendant = |session_id: Uuid| {
        let mut current = by_id.get(&session_id).and_then(|session| session.parent_id);
        while let Some(parent_id) = current {
            if parent_id == root_id {
                return true;
            }
            current = by_id.get(&parent_id).and_then(|session| session.parent_id);
        }
        false
    };
    let mut input = 0_u64;
    let mut output = 0_u64;
    for session in sessions.iter().filter(|session| is_descendant(session.id)) {
        for event in runtime
            .replay(session.id)?
            .into_iter()
            .filter(|event| event.session_id == session.id)
        {
            if let EventKind::Model {
                event:
                    ModelEvent::Usage {
                        input_tokens,
                        output_tokens,
                    },
            } = event.kind
            {
                input = input.saturating_add(input_tokens);
                output = output.saturating_add(output_tokens);
            }
        }
    }
    Ok((input, output))
}

fn run_one(mode: BenchmarkMode, session_id: Uuid) -> RunObservation {
    let started = Instant::now();
    if mode == BenchmarkMode::PiKernel {
        let sink = Arc::new(focus_kernel::VecEventSink::default());
        let mut state = AgentState::new(session_id);
        state
            .transcript
            .push(Message::text(Role::User, "benchmark fixture task"));
        let active_started = Instant::now();
        let error = AgentLoop::new(
            Arc::new(FixtureProvider),
            [Arc::new(FixtureSearchTool) as Arc<dyn Tool>],
        )
        .run_with_cancellation(&mut state, sink.clone(), &NoCancellation)
        .err()
        .map(|error| error.to_string());
        let active_runtime_us = active_started.elapsed().as_micros();
        return RunObservation {
            session_id,
            events: sink.events(),
            live_matches_replay: true,
            error,
            end_to_end_runtime_ms: started.elapsed().as_millis(),
            active_runtime_us: Some(active_runtime_us),
            runtime_initialization_us: None,
            session_initialization_us: None,
            child_input_units: 0,
            child_output_units: 0,
        };
    }
    let root = std::env::temp_dir().join(format!("focus-ablation-{session_id}"));
    if let Err(error) = std::fs::create_dir_all(&root) {
        return RunObservation {
            session_id,
            events: Vec::new(),
            live_matches_replay: false,
            error: Some(error.to_string()),
            end_to_end_runtime_ms: started.elapsed().as_millis(),
            active_runtime_us: None,
            runtime_initialization_us: None,
            session_initialization_us: None,
            child_input_units: 0,
            child_output_units: 0,
        };
    }
    let mut config = RuntimeConfig::for_workspace(&root);
    config.data_root = root.join("state");
    let runtime_started = Instant::now();
    let mut runtime = match FocusRuntime::open(config) {
        Ok(runtime) => runtime,
        Err(error) => {
            return RunObservation {
                session_id,
                events: Vec::new(),
                live_matches_replay: false,
                error: Some(error.to_string()),
                end_to_end_runtime_ms: started.elapsed().as_millis(),
                active_runtime_us: None,
                runtime_initialization_us: None,
                session_initialization_us: None,
                child_input_units: 0,
                child_output_units: 0,
            };
        }
    };
    let runtime_initialization_us = runtime_started.elapsed().as_micros();
    if let Err(error) = runtime.register_tool(fixture_probe_spec()) {
        return RunObservation {
            session_id,
            events: Vec::new(),
            live_matches_replay: false,
            error: Some(error.to_string()),
            end_to_end_runtime_ms: started.elapsed().as_millis(),
            active_runtime_us: None,
            runtime_initialization_us: Some(runtime_initialization_us),
            session_initialization_us: None,
            child_input_units: 0,
            child_output_units: 0,
        };
    }
    let live = runtime.subscribe();
    let options = match mode {
        BenchmarkMode::FocusCore => RunOptions::core(approve_all()),
        BenchmarkMode::FocusWorkflow => RunOptions::engineering(approve_all()),
        BenchmarkMode::FocusDelegation => {
            RunOptions::core(approve_all()).with_delegation(SubagentLimits {
                max_tasks: 4,
                max_parallel: 2,
                max_depth: 2,
            })
        }
        BenchmarkMode::PiKernel => unreachable!(),
    };
    let session_started = Instant::now();
    let session = match runtime
        .sessions
        .create_with_phase("benchmark", crate::session::SessionPhase::Running)
    {
        Ok(session) => session,
        Err(error) => {
            return RunObservation {
                session_id,
                events: Vec::new(),
                live_matches_replay: false,
                error: Some(error.to_string()),
                end_to_end_runtime_ms: started.elapsed().as_millis(),
                active_runtime_us: None,
                runtime_initialization_us: Some(runtime_initialization_us),
                session_initialization_us: None,
                child_input_units: 0,
                child_output_units: 0,
            };
        }
    };
    let session_initialization_us = session_started.elapsed().as_micros();
    let active_started = Instant::now();
    let error = FocusRuntime::block_on(runtime.run_prepared_session_with_options_async(
        session.id,
        "benchmark fixture task",
        Arc::new(FixtureProvider),
        options,
    ))
    .err()
    .map(|error| error.to_string());
    let active_runtime_us = active_started.elapsed().as_micros();
    let events = match runtime.replay(session.id) {
        Ok(events) => events,
        Err(replay_error) => {
            let combined = error.map_or_else(
                || replay_error.to_string(),
                |error| format!("{error}; replay: {replay_error}"),
            );
            return RunObservation {
                session_id: session.id,
                events: Vec::new(),
                live_matches_replay: false,
                error: Some(combined),
                end_to_end_runtime_ms: started.elapsed().as_millis(),
                active_runtime_us: Some(active_runtime_us),
                runtime_initialization_us: Some(runtime_initialization_us),
                session_initialization_us: Some(session_initialization_us),
                child_input_units: 0,
                child_output_units: 0,
            };
        }
    };
    let live = live
        .try_iter()
        .filter(|event| event.session_id == session.id)
        .collect::<Vec<_>>();
    let live_matches_replay = live_semantically_matches_replay(&live, &events);
    let (child_input_units, child_output_units) =
        child_usage_units(&runtime, session.id).unwrap_or_default();
    let end_to_end_runtime_ms = started.elapsed().as_millis();
    let _ = std::fs::remove_dir_all(root);
    RunObservation {
        session_id: session.id,
        events,
        live_matches_replay,
        error,
        end_to_end_runtime_ms,
        active_runtime_us: Some(active_runtime_us),
        runtime_initialization_us: Some(runtime_initialization_us),
        session_initialization_us: Some(session_initialization_us),
        child_input_units,
        child_output_units,
    }
}

#[cfg(test)]
mod tests {
    use focus_kernel::{AgentStatus, Event, EventKind, ModelEvent, ToolCall, ToolResult};
    use serde_json::json;
    use uuid::Uuid;

    use super::{BenchmarkMode, analyze_events};

    #[test]
    fn derives_latency_usage_tool_wait_and_integrity_from_events() {
        let session = Uuid::new_v4();
        let call = ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            arguments: json!({"query":"x"}),
        };
        let events = vec![
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 100,
                kind: EventKind::StateChanged {
                    status: AgentStatus::AwaitingModel,
                    turn: 0,
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 105,
                kind: EventKind::Model {
                    event: ModelEvent::TextDelta {
                        text: "first".into(),
                    },
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 110,
                kind: EventKind::Model {
                    event: ModelEvent::Usage {
                        input_tokens: 7,
                        output_tokens: 3,
                    },
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 114,
                kind: EventKind::Model {
                    event: ModelEvent::ToolCallReady { call: call.clone() },
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 115,
                kind: EventKind::ToolCallRequested { call: call.clone() },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 125,
                kind: EventKind::ToolResultReceived {
                    result: ToolResult {
                        tool_call_id: call.id.clone(),
                        name: call.name.clone(),
                        content: "ok".into(),
                        is_error: false,
                    },
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 135,
                kind: EventKind::Runtime {
                    name: "subagent_queued".into(),
                    data: json!({"task_id":"task-1"}),
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 140,
                kind: EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({"task_id":"task-1"}),
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 150,
                kind: EventKind::Runtime {
                    name: "subagent_completed".into(),
                    data: json!({"task_id":"task-1"}),
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 160,
                kind: EventKind::StateChanged {
                    status: AgentStatus::Complete,
                    turn: 1,
                },
            },
        ];
        let metrics = analyze_events(&events, BenchmarkMode::FocusCore).unwrap();
        assert_eq!(metrics.first_text_delta_ms, Some(5));
        assert_eq!(metrics.total_runtime_ms, Some(60));
        assert_eq!(metrics.tool_wait_ms, Some(10));
        assert_eq!(metrics.parent_input_units, 7);
        assert_eq!(metrics.parent_output_units, 3);
        assert_eq!(metrics.total_input_units, 7);
        assert_eq!(metrics.total_output_units, 3);
        assert_eq!(metrics.subagent_peak, 1);
        assert_eq!(metrics.subagent_utilization_pct, Some(16.6667));
        assert_eq!(metrics.subagent_batch_utilization_pct, Some(100.0));
        assert!(metrics.success);
        assert_eq!(metrics.terminal, "complete");
        assert!(metrics.event_integrity);
    }

    #[test]
    fn marks_unpaired_tools_and_failed_terminal_as_incomplete() {
        let session = Uuid::new_v4();
        let events = vec![
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 10,
                kind: EventKind::ToolCallRequested {
                    call: ToolCall {
                        id: "orphan".into(),
                        name: "search".into(),
                        arguments: json!({}),
                    },
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 20,
                kind: EventKind::StateChanged {
                    status: AgentStatus::Failed,
                    turn: 1,
                },
            },
        ];
        let metrics = analyze_events(&events, BenchmarkMode::PiKernel).unwrap();
        assert!(!metrics.success);
        assert_eq!(metrics.terminal, "failed");
        assert!(!metrics.event_integrity);
    }

    #[test]
    fn rejects_repeated_terminal_and_lifecycle_events_without_task_ids() {
        let session = Uuid::new_v4();
        let events = vec![
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 10,
                kind: EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({}),
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 20,
                kind: EventKind::StateChanged {
                    status: AgentStatus::Complete,
                    turn: 1,
                },
            },
            Event {
                id: Uuid::new_v4(),
                session_id: session,
                timestamp_ms: 21,
                kind: EventKind::StateChanged {
                    status: AgentStatus::Complete,
                    turn: 1,
                },
            },
        ];
        let metrics = analyze_events(&events, BenchmarkMode::FocusDelegation).unwrap();
        assert!(!metrics.event_integrity);
        assert!(!metrics.success);
    }

    #[test]
    fn derives_provider_tool_child_and_critical_path_intervals() {
        let session = Uuid::new_v4();
        let call = ToolCall {
            id: "call-observe".into(),
            name: "fixture_probe".into(),
            arguments: json!({}),
        };
        let event = |timestamp_ms, kind| Event {
            id: Uuid::new_v4(),
            session_id: session,
            timestamp_ms,
            kind,
        };
        let events = vec![
            event(
                100,
                EventKind::Model {
                    event: ModelEvent::RequestStarted {
                        provider: "fixture".into(),
                        model: "deterministic".into(),
                        endpoint: "fixture://observability".into(),
                    },
                },
            ),
            event(
                112,
                EventKind::Model {
                    event: ModelEvent::TextDelta {
                        text: "visible".into(),
                    },
                },
            ),
            event(
                120,
                EventKind::Model {
                    event: ModelEvent::ToolCallReady { call: call.clone() },
                },
            ),
            event(125, EventKind::ToolCallRequested { call: call.clone() }),
            event(
                145,
                EventKind::ToolResultReceived {
                    result: ToolResult {
                        tool_call_id: call.id,
                        name: call.name,
                        content: "ok".into(),
                        is_error: false,
                    },
                },
            ),
            event(
                150,
                EventKind::Runtime {
                    name: "subagent_queued".into(),
                    data: json!({"task_id":"child-1"}),
                },
            ),
            event(
                158,
                EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({"task_id":"child-1"}),
                },
            ),
            event(
                198,
                EventKind::Runtime {
                    name: "subagent_completed".into(),
                    data: json!({"task_id":"child-1"}),
                },
            ),
            event(
                200,
                EventKind::StateChanged {
                    status: AgentStatus::Complete,
                    turn: 1,
                },
            ),
        ];

        let metrics = analyze_events(&events, BenchmarkMode::FocusDelegation).unwrap();
        assert_eq!(metrics.provider_first_delta_ms, Some(12));
        assert_eq!(metrics.provider_wait_ms, Some(12));
        assert_eq!(metrics.tool_queue_ms, Some(5));
        assert_eq!(metrics.tool_execution_ms, Some(20));
        assert_eq!(metrics.subagent_queue_ms, Some(8));
        assert_eq!(metrics.subagent_active_ms, Some(40));
        assert_eq!(metrics.subagent_batch_utilization_pct, Some(100.0));
        assert_eq!(metrics.critical_path_ms, Some(40));
        assert_eq!(
            metrics.critical_path_kind,
            Some(super::CriticalPathKind::Subagent)
        );
        assert!(metrics.event_integrity);
    }

    #[test]
    fn derives_child_batch_window_utilization_from_overlapping_children() {
        let session = Uuid::new_v4();
        let event = |timestamp_ms, name: &str, task_id: &str| Event {
            id: Uuid::new_v4(),
            session_id: session,
            timestamp_ms,
            kind: EventKind::Runtime {
                name: name.into(),
                data: json!({"task_id":task_id}),
            },
        };
        let mut events = vec![
            event(90, "subagent_queued", "a"),
            event(91, "subagent_queued", "b"),
            event(100, "subagent_started", "a"),
            event(110, "subagent_started", "b"),
            event(150, "subagent_completed", "a"),
            event(160, "subagent_completed", "b"),
        ];
        events.push(Event {
            id: Uuid::new_v4(),
            session_id: session,
            timestamp_ms: 170,
            kind: EventKind::StateChanged {
                status: AgentStatus::Complete,
                turn: 1,
            },
        });

        let metrics = analyze_events(&events, BenchmarkMode::FocusDelegation).unwrap();

        assert_eq!(metrics.subagent_active_ms, Some(100));
        assert_eq!(metrics.subagent_batch_utilization_pct, Some(83.3333));
    }

    #[test]
    fn canonical_comparison_accepts_coalesced_model_deltas() {
        let session = Uuid::new_v4();
        let first = Event {
            id: Uuid::new_v4(),
            session_id: session,
            timestamp_ms: 10,
            kind: EventKind::Model {
                event: ModelEvent::TextDelta { text: "hel".into() },
            },
        };
        let second = Event {
            id: Uuid::new_v4(),
            session_id: session,
            timestamp_ms: 11,
            kind: EventKind::Model {
                event: ModelEvent::TextDelta { text: "lo".into() },
            },
        };
        let mut persisted = first.clone();
        persisted.kind = EventKind::Model {
            event: ModelEvent::TextDelta {
                text: "hello".into(),
            },
        };
        assert!(super::live_semantically_matches_replay(
            &[first, second],
            &[persisted]
        ));
    }

    #[test]
    fn canonical_comparison_accepts_live_and_replay_delta_boundaries() {
        let session = Uuid::new_v4();
        let first = Event {
            id: Uuid::new_v4(),
            session_id: session,
            timestamp_ms: 10,
            kind: EventKind::Model {
                event: ModelEvent::TextDelta { text: "hel".into() },
            },
        };
        let second = Event {
            id: Uuid::new_v4(),
            session_id: session,
            timestamp_ms: 11,
            kind: EventKind::Model {
                event: ModelEvent::TextDelta { text: "lo".into() },
            },
        };

        assert!(super::live_semantically_matches_replay(
            &[first.clone(), second.clone()],
            &[first, second]
        ));
    }

    #[test]
    fn deterministic_matrix_reaches_all_four_modes() {
        let receipt = super::run_deterministic(1).expect("fixture matrix should complete");
        assert_eq!(receipt.records.len(), 4);
        assert_eq!(receipt.summaries.len(), 4);
        assert_eq!(receipt.failure_probes.len(), 2);
        assert_eq!(receipt.cancellation_probes.len(), 2);
        for record in &receipt.records {
            assert!(
                record.metrics.success,
                "{} failed: error={:?}; metrics={:?}",
                record.mode.as_str(),
                record.error,
                record.metrics
            );
            assert!(
                record.metrics.event_integrity,
                "{} emitted incomplete events",
                record.mode.as_str()
            );
            assert!(
                record.metrics.end_to_end_runtime_ms >= record.metrics.total_runtime_ms,
                "{} end-to-end duration must cover its event span",
                record.mode.as_str()
            );
            if record.mode != BenchmarkMode::PiKernel {
                assert!(record.metrics.runtime_initialization_us.is_some());
                assert!(record.metrics.session_initialization_us.is_some());
            }
        }
        let delegated = receipt
            .records
            .iter()
            .find(|record| record.mode == BenchmarkMode::FocusDelegation)
            .unwrap();
        assert_eq!(delegated.metrics.subagent_peak, 2);
        assert!(delegated.metrics.child_input_units > 0);
        assert!(delegated.metrics.child_output_units > 0);
        assert_eq!(
            delegated.metrics.total_input_units,
            delegated.metrics.parent_input_units + delegated.metrics.child_input_units
        );
        let delegated_summary = receipt
            .summaries
            .iter()
            .find(|summary| summary.mode == BenchmarkMode::FocusDelegation)
            .unwrap();
        assert_eq!(delegated_summary.success_rate_pct, 100.0);
        assert_eq!(delegated_summary.median_subagent_peak, 2.0);
        for probe in &receipt.failure_probes {
            assert!(probe.passed);
            assert!(probe.failed_event_observed);
            assert!(!probe.metrics.success);
            assert_eq!(probe.metrics.terminal, "failed");
            assert!(probe.metrics.event_integrity);
            assert!(
                probe
                    .error
                    .as_deref()
                    .is_some_and(|error| !error.is_empty())
            );
        }
        for probe in &receipt.cancellation_probes {
            assert!(probe.passed);
            assert!(probe.cancelled_event_observed);
            assert!(!probe.metrics.success);
            assert_eq!(probe.metrics.terminal, "cancelled");
            assert!(probe.metrics.event_integrity);
            assert!(
                probe
                    .error
                    .as_deref()
                    .is_some_and(|error| !error.is_empty())
            );
        }
    }
}
