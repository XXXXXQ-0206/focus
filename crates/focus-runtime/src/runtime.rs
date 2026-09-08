use std::{
    future::Future,
    sync::{Arc, Mutex},
};

use crate::{
    config::{RunOptions, RunResult, RuntimeConfig},
    context::{self, ContextBuilder, ContextItem, ContextKind},
    error::RuntimeError,
    events, goal, host, mcp, memory,
    policy::{ApprovalHandler, PolicyEngine},
    sandbox::WorkspaceSandbox,
    session::{self, SessionManager, SessionMetadata, SessionPhase},
    subagent, tools, workflow,
};
use focus_kernel::{
    AgentLoop, AgentState, AgentStatus, CancellationSignal, Event, EventKind, EventSink,
    EventStore, JsonlEventStore, KernelError, Message, ModelProvider, PersistingEventSink, Role,
    ToolCall, ToolResult,
};
use serde_json::json;
use uuid::Uuid;

struct RunExecution<'a> {
    approval: Arc<dyn ApprovalHandler>,
    cancellation: &'a dyn CancellationSignal,
    coding_workflow: Option<workflow::CodingWorkflow>,
    delegation: Option<subagent::SubagentLimits>,
    subagent_depth: usize,
}

fn terminal_state(result: &Result<String, RuntimeError>) -> (AgentStatus, SessionPhase) {
    if result.is_ok() {
        (AgentStatus::Complete, SessionPhase::Complete)
    } else if matches!(result, Err(RuntimeError::Cancelled)) {
        (AgentStatus::Cancelled, SessionPhase::Cancelled)
    } else {
        (AgentStatus::Failed, SessionPhase::Failed)
    }
}

fn failed_session_phase(error: &RuntimeError) -> SessionPhase {
    if matches!(error, RuntimeError::Cancelled) {
        SessionPhase::Cancelled
    } else {
        SessionPhase::Failed
    }
}

/// One canonical Runtime API shared by CLI, TUI, subagent, and IDE hosts.
#[derive(Clone)]
pub struct FocusRuntime {
    pub(crate) config: RuntimeConfig,
    pub(crate) sandbox: WorkspaceSandbox,
    pub(crate) events: Arc<events::EventHub>,
    pub(crate) sessions: SessionManager,
    pub(crate) memory: memory::MemoryStore,
    pub(crate) goals: goal::GoalStore,
    pub(crate) tools: tools::ToolRegistry,
    pub(crate) mcp_clients: Vec<Arc<mcp::McpClient>>,
}

struct TranscriptCapturingProvider {
    inner: Arc<dyn ModelProvider>,
    latest_messages: Arc<Mutex<Vec<Message>>>,
}

impl TranscriptCapturingProvider {
    fn new(inner: Arc<dyn ModelProvider>, latest_messages: Arc<Mutex<Vec<Message>>>) -> Self {
        Self {
            inner,
            latest_messages,
        }
    }

    fn capture(&self, request: &focus_kernel::ModelRequest) {
        *self
            .latest_messages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = request.messages.clone();
    }
}

#[async_trait::async_trait]
impl ModelProvider for TranscriptCapturingProvider {
    async fn stream(
        &self,
        request: focus_kernel::ModelRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<focus_kernel::ModelEventStream, KernelError> {
        self.capture(&request);
        self.inner.stream(request, cancellation).await
    }
}

impl std::fmt::Debug for FocusRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FocusRuntime")
            .field("workspace_root", &self.config.workspace_root)
            .field("data_root", &self.config.data_root)
            .finish_non_exhaustive()
    }
}

impl FocusRuntime {
    /// Open the Runtime and initialize its sole state layout.
    pub fn open(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        config.validate()?;
        let sandbox = WorkspaceSandbox::new(&config.workspace_root)?
            .with_executor(config.sandbox_backend.executor());
        std::fs::create_dir_all(&config.data_root)
            .map_err(|error| RuntimeError::Session(error.to_string()))?;
        let event_store = Arc::new(JsonlEventStore::new(config.data_root.join("events"))?);
        let events = Arc::new(events::EventHub::new(event_store));
        let sessions = SessionManager::new(config.data_root.join("sessions"), events.clone())?;
        let memory = memory::MemoryStore::new(config.data_root.join("memory"))?;
        let goals = goal::GoalStore::new(config.data_root.join("goals"))?;
        let mut tools = tools::builtin_tool_registry(sandbox.clone(), config.network.clone());
        let mut mcp_clients = Vec::with_capacity(config.mcp_servers.len());
        if config.enable_mcp {
            for server in &config.mcp_servers {
                mcp_clients.push(mcp::register_mcp_server(&mut tools, server)?);
            }
        }
        Ok(Self {
            config,
            sandbox,
            events,
            sessions,
            memory,
            goals,
            tools,
            mcp_clients,
        })
    }

    /// Return immutable Runtime configuration for interface diagnostics.
    #[must_use]
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Register one extension tool before starting runs on this Runtime.
    pub fn register_tool(&mut self, spec: tools::RuntimeToolSpec) -> Result<(), RuntimeError> {
        self.tools.register(spec)
    }

    /// Create a session before a host begins planning or collecting context.
    pub fn create_session(
        &self,
        title: impl Into<String>,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.sessions.create(title)
    }

    /// Create a session with an identifier allocated by an external host.
    pub fn create_session_with_id(
        &self,
        id: Uuid,
        title: impl Into<String>,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.sessions.create_with_id(id, title)
    }

    /// Fork an existing session by ancestry rather than copying event logs.
    pub fn fork_session(
        &self,
        parent: Uuid,
        title: impl Into<String>,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.sessions.fork(parent, title)
    }

    /// Create an ancestry-linked child with no parent transcript inheritance.
    pub fn fork_subagent_session(
        &self,
        parent: Uuid,
        title: impl Into<String>,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.sessions
            .fork_with_inheritance(parent, title, session::TranscriptInheritance::None)
    }

    pub(crate) fn fork_running_subagent_session(
        &self,
        parent: Uuid,
        title: impl Into<String>,
    ) -> Result<SessionMetadata, RuntimeError> {
        self.sessions.fork_with_inheritance_and_phase(
            parent,
            title,
            session::TranscriptInheritance::None,
            SessionPhase::Running,
        )
    }

    /// List sessions for any interface using the Runtime state directory.
    pub fn list_sessions(&self) -> Result<Vec<SessionMetadata>, RuntimeError> {
        self.sessions.list()
    }

    /// Load one session metadata record for a CLI or GUI detail view.
    pub fn session(&self, id: Uuid) -> Result<SessionMetadata, RuntimeError> {
        self.sessions.load(id)
    }

    /// Reconstruct a session's state including parent transcript inheritance.
    pub fn resume(&self, id: Uuid) -> Result<AgentState, RuntimeError> {
        self.sessions.resume(id)
    }

    /// Return the ordered inherited and local session events.
    pub fn replay(&self, id: Uuid) -> Result<Vec<Event>, RuntimeError> {
        self.sessions.replay(id)
    }

    /// Project canonical replay events for a resumable external host connection.
    pub fn host_replay(
        &self,
        id: Uuid,
        after_event_id: Option<Uuid>,
    ) -> Result<host::HostReplayV1, RuntimeError> {
        let events = self.replay(id)?;
        Ok(host::project_replay(id, &events, after_event_id))
    }

    /// Tail only newly persisted local events after an external-host replay baseline.
    pub fn host_event_tail(&self, id: Uuid) -> Result<host::JsonlEventTail, RuntimeError> {
        host::JsonlEventTail::open(
            self.config
                .data_root
                .join("events")
                .join(format!("{id}.jsonl")),
        )
        .map_err(RuntimeError::Kernel)
    }

    /// Subscribe to future events after they have been persisted successfully.
    #[must_use]
    pub fn subscribe(&self) -> std::sync::mpsc::Receiver<Event> {
        self.events.subscribe()
    }

    /// Retain a concise session or project fact for future bounded context.
    pub fn remember(
        &self,
        scope: memory::MemoryScope,
        session_id: Option<Uuid>,
        content: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<memory::MemoryEntry, RuntimeError> {
        self.memory.remember(scope, session_id, content, source)
    }

    /// List one explicit memory scope for a CLI or GUI control surface.
    pub fn list_memory(
        &self,
        scope: memory::MemoryScope,
        session_id: Option<Uuid>,
    ) -> Result<Vec<memory::MemoryEntry>, RuntimeError> {
        self.memory.list(scope, session_id)
    }

    /// Create durable goal metadata without creating a second execution model.
    pub fn create_goal(
        &self,
        title: impl Into<String>,
        objective: impl Into<String>,
    ) -> Result<goal::Goal, RuntimeError> {
        self.goals.create(title, objective)
    }

    /// List the Runtime-owned goal index.
    pub fn list_goals(&self) -> Result<Vec<goal::Goal>, RuntimeError> {
        self.goals.list()
    }

    /// Load one goal metadata record.
    pub fn goal(&self, id: Uuid) -> Result<goal::Goal, RuntimeError> {
        self.goals.load(id)
    }

    /// Mark a session as the active session for an operator-owned goal.
    pub fn attach_goal_session(
        &self,
        goal_id: Uuid,
        session_id: Uuid,
    ) -> Result<goal::Goal, RuntimeError> {
        self.sessions.load(session_id)?;
        self.goals.attach_session(goal_id, session_id)
    }

    /// Apply an operator-selected lifecycle phase to a goal.
    pub fn set_goal_phase(
        &self,
        id: Uuid,
        phase: goal::GoalPhase,
    ) -> Result<goal::Goal, RuntimeError> {
        self.goals.set_phase(id, phase)
    }

    pub(crate) fn block_on<T>(
        operation: impl Future<Output = Result<T, RuntimeError>>,
    ) -> Result<T, RuntimeError> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(RuntimeError::Subagent(
                "synchronous FocusRuntime entry points cannot run inside Tokio; use the matching async method".into(),
            ));
        }
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| RuntimeError::Subagent(error.to_string()))?
            .block_on(operation)
    }

    /// Run a full coding-agent turn in a fresh session.
    pub fn run(
        &self,
        title: impl Into<String>,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Result<RunResult, RuntimeError> {
        Self::block_on(self.run_async(title, task, provider, approval))
    }

    /// Run a full coding-agent turn on the caller's Tokio runtime.
    pub async fn run_async(
        &self,
        title: impl Into<String>,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Result<RunResult, RuntimeError> {
        self.run_with_options_async(title, task, provider, RunOptions::new(approval))
            .await
    }

    /// Run a fresh session with explicit shared Runtime behavior.
    pub fn run_with_options(
        &self,
        title: impl Into<String>,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        options: RunOptions,
    ) -> Result<RunResult, RuntimeError> {
        Self::block_on(self.run_with_options_async(title, task, provider, options))
    }

    /// Run a fresh session with explicit shared Runtime behavior on Tokio.
    pub async fn run_with_options_async(
        &self,
        title: impl Into<String>,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        options: RunOptions,
    ) -> Result<RunResult, RuntimeError> {
        options.validate()?;
        let session = self
            .sessions
            .create_with_phase(title, SessionPhase::Running)?;
        self.run_prepared_session_with_options_async(session.id, task, provider, options)
            .await
    }

    pub(crate) async fn run_prepared_session_with_options_async(
        &self,
        session_id: Uuid,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        options: RunOptions,
    ) -> Result<RunResult, RuntimeError> {
        options.validate()?;
        let result = self
            .run_loaded_session_async(
                AgentState::new(session_id),
                Vec::new(),
                task.into(),
                provider,
                RunExecution {
                    approval: options.approval,
                    cancellation: options.cancellation.as_ref(),
                    coding_workflow: options.workflow,
                    delegation: options.delegation,
                    subagent_depth: options.subagent_depth,
                },
            )
            .await;
        if let Err(error) = &result {
            self.sessions
                .set_phase(session_id, failed_session_phase(error))?;
        }
        result
    }

    /// Continue an existing session. New context is built incrementally for this request.
    pub fn run_in_session(
        &self,
        session_id: Uuid,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Result<RunResult, RuntimeError> {
        Self::block_on(self.run_in_session_async(session_id, task, provider, approval))
    }

    /// Continue an existing session on the caller's Tokio runtime.
    pub async fn run_in_session_async(
        &self,
        session_id: Uuid,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Result<RunResult, RuntimeError> {
        self.run_in_session_with_options_async(
            session_id,
            task,
            provider,
            RunOptions::new(approval),
        )
        .await
    }

    /// Continue a session with explicit shared Runtime behavior.
    pub fn run_in_session_with_options(
        &self,
        session_id: Uuid,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        options: RunOptions,
    ) -> Result<RunResult, RuntimeError> {
        Self::block_on(self.run_in_session_with_options_async(session_id, task, provider, options))
    }

    /// Continue an existing session with explicit behavior on the caller's Tokio runtime.
    pub async fn run_in_session_with_options_async(
        &self,
        session_id: Uuid,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        options: RunOptions,
    ) -> Result<RunResult, RuntimeError> {
        options.validate()?;
        let RunOptions {
            approval,
            cancellation,
            workflow,
            delegation,
            subagent_depth,
        } = options;
        self.run_in_session_configured_async(
            session_id,
            task,
            provider,
            RunExecution {
                approval,
                cancellation: cancellation.as_ref(),
                coding_workflow: workflow,
                delegation,
                subagent_depth,
            },
        )
        .await
    }

    /// Continue a session while observing Runtime-owned cooperative cancellation.
    pub fn run_in_session_cancellable(
        &self,
        session_id: Uuid,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        approval: Arc<dyn ApprovalHandler>,
        cancellation: &dyn CancellationSignal,
    ) -> Result<RunResult, RuntimeError> {
        Self::block_on(self.run_in_session_configured_async(
            session_id,
            task,
            provider,
            RunExecution {
                approval,
                cancellation,
                coding_workflow: None,
                delegation: None,
                subagent_depth: 0,
            },
        ))
    }

    async fn run_in_session_configured_async(
        &self,
        session_id: Uuid,
        task: impl Into<String>,
        provider: Arc<dyn ModelProvider>,
        execution: RunExecution<'_>,
    ) -> Result<RunResult, RuntimeError> {
        let _lease = self.sessions.acquire_lease(session_id)?;
        self.sessions.set_phase(session_id, SessionPhase::Running)?;
        let result = self
            .run_active_session_async(session_id, task.into(), provider, execution)
            .await;
        if let Err(error) = &result {
            self.sessions
                .set_phase(session_id, failed_session_phase(error))?;
        }
        result
    }

    async fn run_active_session_async(
        &self,
        session_id: Uuid,
        task: String,
        provider: Arc<dyn ModelProvider>,
        execution: RunExecution<'_>,
    ) -> Result<RunResult, RuntimeError> {
        let RunExecution {
            approval,
            cancellation,
            coding_workflow,
            delegation,
            subagent_depth,
        } = execution;
        let persisted_events = self.events.load(session_id)?;
        let state = self.sessions.resume(session_id)?;
        self.run_loaded_session_async(
            state,
            persisted_events,
            task,
            provider,
            RunExecution {
                approval,
                cancellation,
                coding_workflow,
                delegation,
                subagent_depth,
            },
        )
        .await
    }

    async fn run_loaded_session_async(
        &self,
        mut state: AgentState,
        persisted_events: Vec<Event>,
        task: String,
        provider: Arc<dyn ModelProvider>,
        execution: RunExecution<'_>,
    ) -> Result<RunResult, RuntimeError> {
        let session_id = state.session_id;
        let RunExecution {
            approval,
            cancellation,
            coding_workflow,
            delegation,
            subagent_depth,
        } = execution;
        let recovered_workflow = coding_workflow
            .as_ref()
            .and_then(|_| workflow::WorkflowState::recover_latest(&persisted_events));
        let mut workflow_state = recovered_workflow.clone().unwrap_or_default();
        if coding_workflow.is_some() {
            if recovered_workflow.is_some() {
                let recovered_messages =
                    workflow::persisted_workflow_messages(&persisted_events, workflow_state.run_id);
                self.reconcile_workflow_evidence(
                    session_id,
                    &mut workflow_state,
                    &recovered_messages,
                )?;
                self.emit_runtime(
                    session_id,
                    "workflow_recovered",
                    json!({"run_id": workflow_state.run_id, "stage": workflow_state.current}),
                )?;
            } else {
                self.emit_runtime(
                    session_id,
                    "workflow_started",
                    serde_json::to_value(&workflow_state)
                        .map_err(|error| RuntimeError::Workflow(error.to_string()))?,
                )?;
                self.emit_runtime(
                    session_id,
                    "workflow_stage_entered",
                    json!({"run_id": workflow_state.run_id, "stage": workflow_state.current}),
                )?;
            }
        }
        state.transcript = context::compact_transcript(
            &state.transcript,
            self.config.context_budget.saturating_mul(2) / 5,
        );
        let window = self.build_context(session_id, &task)?;
        let sink = Arc::new(PersistingEventSink::new(self.events.clone()));
        if state.transcript.is_empty() {
            let instruction = coding_workflow.as_ref().map_or_else(
                || workflow::CodingWorkflow::core_instruction().into(),
                workflow::CodingWorkflow::instruction,
            );
            self.append_message(
                &mut state,
                Message::text(Role::System, instruction),
                sink.as_ref(),
            )?;
        }
        self.append_message(
            &mut state,
            Message::text(Role::System, window.render()),
            sink.as_ref(),
        )?;
        self.append_message(&mut state, Message::text(Role::User, task), sink.as_ref())?;
        let workflow_transcript_start = state.transcript.len();

        let latest_provider_messages = Arc::new(Mutex::new(state.transcript.clone()));
        let run_provider: Arc<dyn ModelProvider> = Arc::new(TranscriptCapturingProvider::new(
            provider.clone(),
            latest_provider_messages.clone(),
        ));
        let mut tool_registry = self.tools.clone();
        if coding_workflow.is_some() {
            tool_registry.register(tools::workflow_checkpoint_tool_spec())?;
        }
        if let Some(limits) = delegation
            && subagent_depth < limits.max_depth
        {
            tool_registry.register(subagent::delegate_tool_spec(
                self.clone(),
                session_id,
                provider.clone(),
                approval.clone(),
                latest_provider_messages,
                limits,
                subagent_depth,
            ))?;
        }
        let engine = Arc::new(PolicyEngine::new(self.config.policy, approval));
        let tools = tool_registry.bind(engine);
        loop {
            let agent_loop = AgentLoop::new(run_provider.clone(), tools.clone());
            let outcome = if coding_workflow.is_some() {
                agent_loop
                    .run_with_deferred_completion_async(&mut state, sink.clone(), cancellation)
                    .await
            } else {
                agent_loop
                    .run_with_cancellation_async(&mut state, sink.clone(), cancellation)
                    .await
            };
            if coding_workflow.is_some() {
                self.reconcile_workflow_evidence(
                    session_id,
                    &mut workflow_state,
                    state
                        .transcript
                        .get(workflow_transcript_start..)
                        .unwrap_or_default(),
                )?;
            }
            if let Err(error) = outcome {
                return Err(error.into());
            }

            let missing = coding_workflow.as_ref().map_or_else(Vec::new, |workflow| {
                workflow_state.missing_requirements(workflow.require_verification)
            });
            if missing.is_empty() {
                if coding_workflow.is_some() {
                    self.emit_runtime(
                        session_id,
                        "workflow_gate_passed",
                        json!({"run_id": workflow_state.run_id}),
                    )?;
                    self.emit_runtime(
                        session_id,
                        "workflow_completed",
                        json!({"run_id": workflow_state.run_id}),
                    )?;
                    agent_loop.complete_run(&mut state, sink.as_ref())?;
                }
                self.sessions
                    .set_phase(session_id, SessionPhase::Complete)?;
                return Ok(RunResult {
                    session_id,
                    status: state.status,
                    final_response: state
                        .transcript
                        .iter()
                        .rev()
                        .find(|message| message.role == Role::Assistant)
                        .map_or_else(String::new, |message| message.content.clone()),
                    estimated_context_tokens: window.estimated_context_tokens,
                    omitted_context_items: window.omitted.len(),
                });
            }

            workflow_state.gate_failures = workflow_state.gate_failures.saturating_add(1);
            self.emit_runtime(
                session_id,
                "workflow_gate_failed",
                json!({
                    "run_id": workflow_state.run_id,
                    "attempt": workflow_state.gate_failures,
                    "missing": missing,
                }),
            )?;
            if workflow_state.gate_failures > self.config.max_workflow_gate_retries {
                return Err(RuntimeError::Workflow(format!(
                    "completion gate still missing {} after {} retries",
                    missing.join(", "),
                    self.config.max_workflow_gate_retries
                )));
            }
            self.append_message(
                &mut state,
                Message::text(
                    Role::System,
                    format!(
                        "Workflow completion gate blocked this response. Missing evidence: {}. The previous response was not accepted; do not return an empty assistant message. Continue the same task and satisfy each missing item with the exact canonical evidence: explore = successful read_file/search or shell with purpose=explore; plan = workflow_checkpoint(kind=plan, summary=...); implement_or_no_change = a successful mutation tool or workflow_checkpoint(kind=no_change, summary=...); verify = a successful shell call with purpose=verify; review = workflow_checkpoint(kind=review, summary=...). Delegated child activity is not parent evidence until you inspect or summarize it. After the missing evidence is recorded, return a concise non-empty final answer.",
                        missing.join(", ")
                    ),
                ),
                sink.as_ref(),
            )?;
        }
    }

    /// Execute a built-in tool through the same policy and sandbox used by agent calls.
    pub fn invoke_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Result<String, RuntimeError> {
        Self::block_on(self.invoke_tool_async(name, arguments, approval))
    }

    /// Execute a built-in tool on the caller's Tokio runtime.
    pub async fn invoke_tool_async(
        &self,
        name: &str,
        arguments: serde_json::Value,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Result<String, RuntimeError> {
        let session = self
            .sessions
            .create_with_phase(format!("tool: {name}"), SessionPhase::Running)?;
        self.invoke_tool_in_session_async(session.id, name, arguments, approval)
            .await
    }

    /// Execute one host-initiated tool in an existing event-backed session.
    pub async fn invoke_tool_in_session_async(
        &self,
        session_id: Uuid,
        name: &str,
        arguments: serde_json::Value,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Result<String, RuntimeError> {
        let _lease = self.sessions.acquire_lease(session_id)?;
        self.sessions.set_phase(session_id, SessionPhase::Running)?;
        let result = self
            .invoke_tool_in_active_session_async(session_id, name, arguments, approval)
            .await;
        let (status, phase) = terminal_state(&result);
        self.events
            .append(&Event::now(
                session_id,
                EventKind::StateChanged { status, turn: 0 },
            ))
            .map_err(RuntimeError::Kernel)?;
        self.sessions.set_phase(session_id, phase)?;
        result
    }

    async fn invoke_tool_in_active_session_async(
        &self,
        session_id: Uuid,
        name: &str,
        arguments: serde_json::Value,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Result<String, RuntimeError> {
        let engine = Arc::new(PolicyEngine::new(self.config.policy, approval));
        let tools = self.tools.bind(engine);
        let tool = tools
            .iter()
            .find(|tool| tool.definition().name == name)
            .ok_or_else(|| RuntimeError::ToolInput(format!("unknown Runtime tool: {name}")))?;
        let call = ToolCall {
            id: format!("host-{}", Uuid::new_v4()),
            name: name.into(),
            arguments,
        };
        self.events
            .append(&Event::now(
                session_id,
                EventKind::ToolCallRequested {
                    call: call.redacted_for_persistence(),
                },
            ))
            .map_err(RuntimeError::Kernel)?;
        match tool
            .call_with_cancellation_async(call.arguments.clone(), &focus_kernel::NoCancellation)
            .await
        {
            Ok(content) => {
                self.events
                    .append(&Event::now(
                        session_id,
                        EventKind::ToolResultReceived {
                            result: ToolResult {
                                tool_call_id: call.id,
                                name: call.name,
                                content: content.clone(),
                                is_error: false,
                            },
                        },
                    ))
                    .map_err(RuntimeError::Kernel)?;
                Ok(content)
            }
            Err(error) => {
                let error = RuntimeError::from(error);
                self.events
                    .append(&Event::now(
                        session_id,
                        EventKind::ToolResultReceived {
                            result: ToolResult {
                                tool_call_id: call.id,
                                name: call.name,
                                content: error.to_string(),
                                is_error: true,
                            },
                        },
                    ))
                    .map_err(RuntimeError::Kernel)?;
                Err(error)
            }
        }
    }

    pub(crate) fn build_context(
        &self,
        session_id: Uuid,
        task: &str,
    ) -> Result<context::ContextWindow, RuntimeError> {
        let mut builder = self.base_context();
        builder.upsert(ContextItem {
            id: format!("task:{session_id}"),
            kind: ContextKind::Task,
            priority: 950,
            source: "current request".into(),
            content: task.into(),
        });
        for item in self.memory.context_items(session_id)? {
            builder.upsert(item);
        }
        Ok(builder.build())
    }

    fn base_context(&self) -> ContextBuilder {
        let mut builder = ContextBuilder::new(self.config.context_budget);
        builder.upsert(ContextItem {
            id: "runtime-principles".into(),
            kind: ContextKind::System,
            priority: 1_000,
            source: "runtime".into(),
            content: "Small Kernel, Codex-inspired Workflow, Unified Runtime, One Implementation, Everything Composable.".into(),
        });
        builder.upsert(ContextItem {
            id: "workspace".into(),
            kind: ContextKind::Project,
            priority: 700,
            source: "workspace".into(),
            content: format!("Project root: {}", self.sandbox.root().display()),
        });
        builder
    }

    pub(crate) fn build_subagent_context_from_messages(
        &self,
        parent_session_id: Uuid,
        parent_messages: &[Message],
    ) -> Result<String, RuntimeError> {
        let mut builder = self.base_context();
        let transcript = context::compact_transcript(
            parent_messages,
            self.config.context_budget.saturating_mul(2) / 5,
        );
        if !transcript.is_empty() {
            builder.upsert(ContextItem {
                id: format!("parent-transcript:{parent_session_id}"),
                kind: ContextKind::Summary,
                priority: 850,
                source: "bounded parent transcript".into(),
                content: serde_json::to_string(&transcript)
                    .map_err(|error| RuntimeError::Subagent(error.to_string()))?,
            });
        }
        for item in self.memory.context_items(parent_session_id)? {
            builder.upsert(item);
        }
        Ok(builder.build().render())
    }

    fn reconcile_workflow_evidence(
        &self,
        session_id: Uuid,
        workflow_state: &mut workflow::WorkflowState,
        messages: &[Message],
    ) -> Result<(), RuntimeError> {
        for evidence in workflow::transcript_evidence(messages) {
            let previous_stage = workflow_state.current;
            if workflow_state.record(evidence.clone()) {
                self.emit_runtime(
                    session_id,
                    "workflow_evidence_recorded",
                    json!({"run_id": workflow_state.run_id, "evidence": evidence}),
                )?;
                if workflow_state.current != previous_stage {
                    self.emit_runtime(
                        session_id,
                        "workflow_stage_entered",
                        json!({"run_id": workflow_state.run_id, "stage": workflow_state.current}),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn append_message(
        &self,
        state: &mut AgentState,
        message: Message,
        sink: &PersistingEventSink,
    ) -> Result<(), RuntimeError> {
        state.transcript.push(message.clone());
        sink.emit(Event::now(
            state.session_id,
            EventKind::MessageAdded { message },
        ))?;
        Ok(())
    }

    pub(crate) fn emit_runtime(
        &self,
        session_id: Uuid,
        name: &str,
        data: serde_json::Value,
    ) -> Result<(), RuntimeError> {
        self.events
            .append(&Event::now(
                session_id,
                EventKind::Runtime {
                    name: name.into(),
                    data,
                },
            ))
            .map_err(RuntimeError::Kernel)
    }
}

#[cfg(test)]
mod async_tests {
    use std::sync::Arc;

    use crate::{
        config::RuntimeConfig,
        diagnostics::{StaticProvider, deny_approvals},
    };

    use super::*;

    #[test]
    fn failed_session_phase_distinguishes_cancellation_from_failure() {
        assert_eq!(
            super::failed_session_phase(&RuntimeError::Cancelled),
            SessionPhase::Cancelled,
        );
        assert_eq!(
            super::failed_session_phase(&RuntimeError::Session("fixture".into())),
            SessionPhase::Failed,
        );
    }

    #[tokio::test]
    async fn async_run_persists_the_same_terminal_session_state() {
        let directory =
            std::env::temp_dir().join(format!("focus-runtime-async-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&directory)).unwrap();

        let result = runtime
            .run_with_options_async(
                "async turn",
                "answer directly",
                Arc::new(StaticProvider::new("done")),
                crate::config::RunOptions::core(deny_approvals()),
            )
            .await
            .unwrap();

        assert_eq!(result.final_response, "done");
        assert!(matches!(
            runtime
                .replay(result.session_id)
                .unwrap()
                .last()
                .map(|event| &event.kind),
            Some(EventKind::StateChanged {
                status: focus_kernel::AgentStatus::Complete,
                turn: 1,
            })
        ));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn synchronous_tool_host_wrapper_rejects_nested_tokio() {
        let directory =
            std::env::temp_dir().join(format!("focus-runtime-tool-host-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&directory)).unwrap();

        let error = runtime
            .invoke_tool(
                "read_file",
                json!({"path":"missing.txt"}),
                crate::approve_all(),
            )
            .unwrap_err();

        assert!(error.to_string().contains("cannot run inside Tokio"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn host_tool_invocation_persists_a_canonical_session_lifecycle() {
        let directory =
            std::env::temp_dir().join(format!("focus-runtime-host-tool-events-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("note.txt"), "recorded").unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&directory)).unwrap();

        let output = runtime
            .invoke_tool_async(
                "read_file",
                json!({"path":"note.txt"}),
                crate::approve_all(),
            )
            .await
            .unwrap();
        assert_eq!(output, "recorded");
        let session = runtime
            .list_sessions()
            .unwrap()
            .into_iter()
            .find(|session| session.title == "tool: read_file")
            .unwrap();
        assert_eq!(session.phase, SessionPhase::Complete);
        let events = runtime.replay(session.id).unwrap();
        assert!(matches!(
            events.first().map(|event| &event.kind),
            Some(EventKind::ToolCallRequested { call }) if call.name == "read_file"
        ));
        assert!(matches!(
            events.get(1).map(|event| &event.kind),
            Some(EventKind::ToolResultReceived { result }) if result.content == "recorded"
        ));
        assert!(matches!(
            events.last().map(|event| &event.kind),
            Some(EventKind::StateChanged {
                status: focus_kernel::AgentStatus::Complete,
                ..
            })
        ));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn host_tool_invocation_marks_a_pre_execution_failure_terminal() {
        let directory = std::env::temp_dir().join(format!(
            "focus-runtime-host-tool-preflight-failure-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&directory)).unwrap();
        let session = runtime.sessions.create("host failure").unwrap();

        let error = runtime
            .invoke_tool_in_session_async(
                session.id,
                "missing_tool",
                json!({}),
                crate::approve_all(),
            )
            .await
            .unwrap_err();

        assert!(matches!(error, RuntimeError::ToolInput(_)));
        assert_eq!(
            runtime.sessions.load(session.id).unwrap().phase,
            SessionPhase::Failed
        );
        assert!(matches!(
            runtime
                .replay(session.id)
                .unwrap()
                .last()
                .map(|event| &event.kind),
            Some(EventKind::StateChanged {
                status: focus_kernel::AgentStatus::Failed,
                ..
            })
        ));
        let _ = std::fs::remove_dir_all(directory);
    }
}
