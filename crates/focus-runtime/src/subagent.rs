//! First-class, cancellable, parallel subagent orchestration primitives.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use focus_kernel::{CancellationSignal, ModelProvider};
use futures::{FutureExt, StreamExt, future::BoxFuture, stream};

use crate::{
    FocusRuntime, RunOptions, RuntimeError,
    policy::{ApprovalHandler, ToolOperation},
    tools::{RuntimeToolSpec, ToolHandler},
};

/// Compact context inherited by a child rather than copying its full parent transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentContext {
    /// Parent session identity for tracing and memory lookup.
    pub session_id: Uuid,
    /// The bounded rendered parent context.
    pub inherited_context: String,
    /// Project-root-relative paths the child should prioritize.
    pub focus_paths: Vec<String>,
}

/// A requested specialized unit of work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentTask {
    /// Stable id used to aggregate results in request order.
    pub id: Uuid,
    /// Specialist role, such as `explorer`, `implementer`, `reviewer`, or `verifier`.
    pub role: String,
    /// Explicit bounded assignment.
    pub objective: String,
    /// Minimal inherited context.
    pub context: SubagentContext,
}

/// Final child outcome fed back into parent context or workflow gates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentResult {
    /// ID from [`SubagentTask`].
    pub task_id: Uuid,
    /// Isolated child session containing the canonical event trail.
    pub child_session_id: Uuid,
    /// Specialist role that produced the result.
    pub role: String,
    /// Final child lifecycle outcome.
    pub status: SubagentStatus,
    /// Concise evidence-bearing result.
    pub summary: String,
}

/// Stable child lifecycle state used in aggregation and Runtime events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    /// Child returned a normal final response.
    Completed,
    /// Child run returned an error.
    Failed,
    /// Parent cancellation stopped the child.
    Cancelled,
}

/// Bounds applied to model-requested delegation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentLimits {
    /// Maximum tasks accepted in one delegate call.
    pub max_tasks: usize,
    /// Maximum concurrently active child workers.
    pub max_parallel: usize,
    /// Maximum nested delegate depth.
    pub max_depth: usize,
}

impl Default for SubagentLimits {
    fn default() -> Self {
        Self {
            max_tasks: 8,
            max_parallel: 4,
            max_depth: 2,
        }
    }
}

impl SubagentLimits {
    /// Validate delegation bounds before a run exposes the delegate tool.
    pub fn validate(self) -> Result<(), RuntimeError> {
        if !(1..=32).contains(&self.max_tasks) {
            return Err(RuntimeError::Configuration(
                "subagent max_tasks must be between 1 and 32".into(),
            ));
        }
        if !(1..=16).contains(&self.max_parallel) {
            return Err(RuntimeError::Configuration(
                "subagent max_parallel must be between 1 and 16".into(),
            ));
        }
        if self.max_parallel > self.max_tasks {
            return Err(RuntimeError::Configuration(
                "subagent max_parallel must not exceed max_tasks".into(),
            ));
        }
        if self.max_depth > 4 {
            return Err(RuntimeError::Configuration(
                "subagent max_depth must be at most 4".into(),
            ));
        }
        Ok(())
    }
}

/// Cooperative cancellation token shared by all child workers in a batch.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Request cancellation. Runners should observe this before irreversible work.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Return whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Wrap an existing Runtime cancellation flag.
    #[must_use]
    pub fn from_shared(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }
}

impl CancellationSignal for CancellationToken {
    fn is_cancelled(&self) -> bool {
        self.is_cancelled()
    }

    fn shared_flag(&self) -> Option<Arc<AtomicBool>> {
        Some(self.0.clone())
    }
}

/// Runs one child task. Implementations can host an agent loop or a specialized deterministic worker.
pub trait SubagentRunner: Send + Sync + 'static {
    /// Execute the task on the active Runtime executor.
    fn run_async<'a>(
        &'a self,
        task: SubagentTask,
        cancellation: CancellationToken,
    ) -> BoxFuture<'a, Result<SubagentResult, RuntimeError>>;
}

/// Canonical Runtime orchestration for parallel child work.
#[derive(Clone)]
pub struct SubagentManager {
    runner: Arc<dyn SubagentRunner>,
}

impl std::fmt::Debug for SubagentManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubagentManager")
            .finish_non_exhaustive()
    }
}

impl SubagentManager {
    /// Create a manager around an application-owned child runner.
    #[must_use]
    pub fn new(runner: Arc<dyn SubagentRunner>) -> Self {
        Self { runner }
    }

    /// Execute bounded child tasks on the caller's Tokio runtime and retain submission order.
    pub async fn run_parallel_limited_async(
        &self,
        tasks: Vec<SubagentTask>,
        cancellation: CancellationToken,
        max_parallel: usize,
    ) -> Vec<Result<SubagentResult, RuntimeError>> {
        if tasks.is_empty() {
            return Vec::new();
        }
        let task_count = tasks.len();
        let runner = self.runner.clone();
        let mut executions = stream::iter(tasks.into_iter().enumerate())
            .map(move |(index, task)| {
                let runner = runner.clone();
                let cancellation = cancellation.clone();
                async move {
                    let result = std::panic::AssertUnwindSafe(async move {
                        if cancellation.is_cancelled() {
                            Err(RuntimeError::Cancelled)
                        } else {
                            runner.run_async(task, cancellation).await
                        }
                    })
                    .catch_unwind()
                    .await
                    .unwrap_or_else(|_| {
                        Err(RuntimeError::Subagent("subagent runner panicked".into()))
                    });
                    (index, result)
                }
            })
            .buffer_unordered(max_parallel.max(1));

        let mut ordered = (0..task_count).map(|_| None).collect::<Vec<_>>();
        while let Some((index, result)) = executions.next().await {
            ordered[index] = Some(result);
        }
        ordered
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| {
                    Err(RuntimeError::Subagent(
                        "subagent worker result was not returned".into(),
                    ))
                })
            })
            .collect()
    }
}

/// A child runner that creates a forked Runtime session for every specialized task.
#[derive(Clone)]
pub struct RuntimeSubagentRunner {
    runtime: Arc<FocusRuntime>,
    provider: Arc<dyn ModelProvider>,
    approval: Arc<dyn ApprovalHandler>,
    limits: SubagentLimits,
    depth: usize,
}

impl std::fmt::Debug for RuntimeSubagentRunner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeSubagentRunner")
            .finish_non_exhaustive()
    }
}

impl RuntimeSubagentRunner {
    /// Connect child lifecycle execution to the shared Runtime and model provider.
    #[must_use]
    pub fn new(
        runtime: Arc<FocusRuntime>,
        provider: Arc<dyn ModelProvider>,
        approval: Arc<dyn ApprovalHandler>,
        limits: SubagentLimits,
        depth: usize,
    ) -> Self {
        Self {
            runtime,
            provider,
            approval,
            limits,
            depth,
        }
    }
}

impl SubagentRunner for RuntimeSubagentRunner {
    fn run_async<'a>(
        &'a self,
        task: SubagentTask,
        cancellation: CancellationToken,
    ) -> BoxFuture<'a, Result<SubagentResult, RuntimeError>> {
        Box::pin(async move {
            if cancellation.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            let session = self.runtime.fork_running_subagent_session(
                task.context.session_id,
                format!("{}: {}", task.role, task.objective),
            )?;
            self.runtime.emit_runtime(
                task.context.session_id,
                "subagent_started",
                json!({
                    "task_id": task.id,
                    "child_session_id": session.id,
                    "role": task.role,
                }),
            )?;
            let prompt = format!(
                "Act as the `{}` specialist. Objective: {}\n\nInherited context:\n{}\n\nFocus paths: {}\n\nReturn concise findings with concrete evidence for the parent agent.",
                task.role,
                task.objective,
                task.context.inherited_context,
                task.context.focus_paths.join(", "),
            );
            let result = std::panic::AssertUnwindSafe(
                self.runtime.run_prepared_session_with_options_async(
                    session.id,
                    prompt,
                    self.provider.clone(),
                    RunOptions {
                        approval: self.approval.clone(),
                        cancellation: Arc::new(cancellation.clone()),
                        workflow: None,
                        delegation: Some(self.limits),
                        subagent_depth: self.depth.saturating_add(1),
                    },
                ),
            )
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err(RuntimeError::Subagent("subagent runtime panicked".into())));
            let (event_name, child_result) = match result {
                Ok(result) => (
                    "subagent_completed",
                    SubagentResult {
                        task_id: task.id,
                        child_session_id: session.id,
                        role: task.role,
                        status: SubagentStatus::Completed,
                        summary: result.final_response,
                    },
                ),
                Err(error) => {
                    let cancelled = cancellation.is_cancelled()
                        || matches!(
                            error,
                            RuntimeError::Cancelled
                                | RuntimeError::Kernel(focus_kernel::KernelError::Cancelled)
                        );
                    (
                        if cancelled {
                            "subagent_cancelled"
                        } else {
                            "subagent_failed"
                        },
                        SubagentResult {
                            task_id: task.id,
                            child_session_id: session.id,
                            role: task.role,
                            status: if cancelled {
                                SubagentStatus::Cancelled
                            } else {
                                SubagentStatus::Failed
                            },
                            summary: error.to_string(),
                        },
                    )
                }
            };
            self.runtime.emit_runtime(
                task.context.session_id,
                event_name,
                serde_json::to_value(&child_result)
                    .map_err(|error| RuntimeError::Subagent(error.to_string()))?,
            )?;
            Ok(child_result)
        })
    }
}

/// Build the canonical model-visible delegation tool for one parent run.
pub fn delegate_tool_spec(
    runtime: FocusRuntime,
    parent_session_id: Uuid,
    provider: Arc<dyn ModelProvider>,
    approval: Arc<dyn ApprovalHandler>,
    parent_messages: Arc<Mutex<Vec<focus_kernel::Message>>>,
    limits: SubagentLimits,
    depth: usize,
) -> RuntimeToolSpec {
    RuntimeToolSpec::new(
        focus_kernel::ToolDefinition {
            name: "delegate".into(),
            description: "Run bounded specialist tasks in isolated child sessions and return ordered evidence to the parent.".into(),
            input_schema: json!({
                "type":"object",
                "required":["tasks"],
                "properties":{
                    "mode":{"type":"string","enum":["parallel","serial"]},
                    "tasks":{"type":"array","minItems":1,"items":{"type":"object","required":["role","objective"],"properties":{
                        "role":{"type":"string","minLength":1},
                        "objective":{"type":"string","minLength":1},
                        "focus_paths":{"type":"array","items":{"type":"string"}}
                    }}}
                }
            }),
        },
        ToolOperation::Other,
        "Spawn bounded specialist agents using the shared Runtime.",
        Arc::new(DelegateToolHandler {
            runtime: Arc::new(runtime),
            parent_session_id,
            provider,
            approval,
            parent_messages,
            limits,
            depth,
        }),
    )
    .with_execution_class(focus_kernel::ToolExecutionClass::Exclusive)
}

struct DelegateToolHandler {
    runtime: Arc<FocusRuntime>,
    parent_session_id: Uuid,
    provider: Arc<dyn ModelProvider>,
    approval: Arc<dyn ApprovalHandler>,
    parent_messages: Arc<Mutex<Vec<focus_kernel::Message>>>,
    limits: SubagentLimits,
    depth: usize,
}

#[derive(Debug, Deserialize)]
struct DelegateRequest {
    #[serde(default = "default_delegate_mode")]
    mode: String,
    tasks: Vec<DelegateTaskInput>,
}

#[derive(Debug, Deserialize)]
struct DelegateTaskInput {
    role: String,
    objective: String,
    #[serde(default)]
    focus_paths: Vec<String>,
}

fn default_delegate_mode() -> String {
    "parallel".into()
}

impl ToolHandler for DelegateToolHandler {
    fn execute_with_cancellation_async<'a>(
        &'a self,
        arguments: Value,
        cancellation: &'a dyn CancellationSignal,
    ) -> BoxFuture<'a, Result<String, RuntimeError>> {
        Box::pin(async move {
            if self.depth >= self.limits.max_depth {
                return Err(RuntimeError::Subagent(format!(
                    "delegate depth {} reached the configured limit {}",
                    self.depth, self.limits.max_depth
                )));
            }
            let request: DelegateRequest = serde_json::from_value(arguments).map_err(|error| {
                RuntimeError::ToolInput(format!("invalid delegate input: {error}"))
            })?;
            if request.tasks.is_empty() || request.tasks.len() > self.limits.max_tasks {
                return Err(RuntimeError::Subagent(format!(
                    "delegate task count must be between 1 and {}",
                    self.limits.max_tasks
                )));
            }
            if !matches!(request.mode.as_str(), "parallel" | "serial") {
                return Err(RuntimeError::ToolInput(
                    "delegate mode must be parallel or serial".into(),
                ));
            }
            let parent_messages = self
                .parent_messages
                .lock()
                .map_err(|_| RuntimeError::Subagent("parent context lock was poisoned".into()))?
                .clone();
            let inherited_context = self
                .runtime
                .build_subagent_context_from_messages(self.parent_session_id, &parent_messages)?;
            let mut tasks = Vec::with_capacity(request.tasks.len());
            for input in request.tasks {
                if input.role.trim().is_empty() || input.objective.trim().is_empty() {
                    return Err(RuntimeError::ToolInput(
                        "delegate role and objective must not be empty".into(),
                    ));
                }
                let task = SubagentTask {
                    id: Uuid::new_v4(),
                    role: input.role,
                    objective: input.objective,
                    context: SubagentContext {
                        session_id: self.parent_session_id,
                        inherited_context: inherited_context.clone(),
                        focus_paths: input.focus_paths,
                    },
                };
                self.runtime.emit_runtime(
                    self.parent_session_id,
                    "subagent_queued",
                    json!({"task_id": task.id, "role": task.role}),
                )?;
                tasks.push(task);
            }
            let cancellation = CancellationToken::from_shared(
                focus_kernel::CancellationBridge::from_signal(cancellation)
                    .shared_flag()
                    .expect("cancellation bridge always exposes a shared flag"),
            );
            if cancellation.is_cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            let runner = Arc::new(RuntimeSubagentRunner::new(
                self.runtime.clone(),
                self.provider.clone(),
                self.approval.clone(),
                self.limits,
                self.depth,
            ));
            let manager = SubagentManager::new(runner.clone());
            let task_metadata = tasks
                .iter()
                .map(|task| (task.id, task.role.clone()))
                .collect::<Vec<_>>();
            let results = if request.mode == "serial" {
                let mut results = Vec::with_capacity(tasks.len());
                for task in tasks {
                    results.push(runner.run_async(task, cancellation.clone()).await);
                }
                results
            } else {
                manager
                    .run_parallel_limited_async(
                        tasks,
                        cancellation.clone(),
                        self.limits.max_parallel,
                    )
                    .await
            };
            let results = results
                .into_iter()
                .enumerate()
                .map(|(index, result)| match result {
                    Ok(result) => serde_json::to_value(result)
                        .map_err(|error| RuntimeError::Subagent(error.to_string())),
                    Err(error) => {
                        let cancelled = matches!(
                            error,
                            RuntimeError::Cancelled
                                | RuntimeError::Kernel(focus_kernel::KernelError::Cancelled)
                        );
                        let status = if cancelled {
                            SubagentStatus::Cancelled
                        } else {
                            SubagentStatus::Failed
                        };
                        self.runtime.emit_runtime(
                            self.parent_session_id,
                            if cancelled {
                                "subagent_cancelled"
                            } else {
                                "subagent_failed"
                            },
                            json!({
                                "task_id": task_metadata[index].0,
                                "child_session_id": Value::Null,
                                "role": task_metadata[index].1,
                                "status": status,
                                "summary": error.to_string(),
                            }),
                        )?;
                        Ok(json!({
                            "task_id": task_metadata[index].0,
                            "child_session_id": Value::Null,
                            "role": task_metadata[index].1,
                            "status": status,
                            "summary": error.to_string(),
                        }))
                    }
                })
                .collect::<Result<Vec<_>, RuntimeError>>()?;
            let cancelled = cancellation.is_cancelled();
            self.runtime.emit_runtime(
                self.parent_session_id,
                "subagent_batch_completed",
                json!({"mode": request.mode, "cancelled": cancelled, "results": results}),
            )?;
            if cancelled {
                return Err(RuntimeError::Cancelled);
            }
            serde_json::to_string(&json!({"mode": request.mode, "results": results}))
                .map_err(|error| RuntimeError::Subagent(error.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    struct AsyncConcurrencyProbe {
        active: AtomicUsize,
        peak: AtomicUsize,
    }

    struct PanickingProbe;

    impl SubagentRunner for AsyncConcurrencyProbe {
        fn run_async<'a>(
            &'a self,
            task: SubagentTask,
            _cancellation: CancellationToken,
        ) -> futures::future::BoxFuture<'a, Result<SubagentResult, RuntimeError>> {
            Box::pin(async move {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(active, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(25)).await;
                self.active.fetch_sub(1, Ordering::SeqCst);
                Ok(SubagentResult {
                    task_id: task.id,
                    child_session_id: Uuid::new_v4(),
                    role: task.role,
                    status: SubagentStatus::Completed,
                    summary: task.objective,
                })
            })
        }
    }

    impl SubagentRunner for PanickingProbe {
        fn run_async<'a>(
            &'a self,
            _task: SubagentTask,
            _cancellation: CancellationToken,
        ) -> futures::future::BoxFuture<'a, Result<SubagentResult, RuntimeError>> {
            Box::pin(async move { panic!("subagent fixture panic") })
        }
    }

    #[tokio::test]
    async fn async_parallel_execution_bounds_active_tasks_and_preserves_order() {
        let runner = Arc::new(AsyncConcurrencyProbe {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let manager = SubagentManager::new(runner.clone());
        let tasks = (0..4)
            .map(|index| SubagentTask {
                id: Uuid::new_v4(),
                role: "worker".into(),
                objective: index.to_string(),
                context: SubagentContext {
                    session_id: Uuid::nil(),
                    inherited_context: String::new(),
                    focus_paths: Vec::new(),
                },
            })
            .collect::<Vec<_>>();

        let results = manager
            .run_parallel_limited_async(tasks, CancellationToken::default(), 2)
            .await;

        assert_eq!(runner.peak.load(Ordering::SeqCst), 2);
        assert_eq!(
            results
                .into_iter()
                .map(|result| result.unwrap().summary)
                .collect::<Vec<_>>(),
            ["0", "1", "2", "3"]
        );
    }

    #[tokio::test]
    async fn parallel_execution_converts_runner_panics_into_ordered_errors() {
        let manager = SubagentManager::new(Arc::new(PanickingProbe));
        let task = SubagentTask {
            id: Uuid::new_v4(),
            role: "worker".into(),
            objective: "panic".into(),
            context: SubagentContext {
                session_id: Uuid::nil(),
                inherited_context: String::new(),
                focus_paths: Vec::new(),
            },
        };

        let results = manager
            .run_parallel_limited_async(vec![task], CancellationToken::default(), 1)
            .await;

        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .as_ref()
                .is_err_and(|error| error.to_string().contains("subagent runner panicked"))
        );
    }

    #[tokio::test]
    async fn delegate_rejects_depth_and_batch_limits_before_spawning() {
        let directory = std::env::temp_dir().join(format!("delegate-limits-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut config = crate::RuntimeConfig::for_workspace(&directory);
        config.data_root = directory.join("state");
        let runtime = FocusRuntime::open(config).unwrap();
        let parent = runtime.create_session("parent").unwrap();
        let mut handler = DelegateToolHandler {
            runtime: Arc::new(runtime),
            parent_session_id: parent.id,
            provider: Arc::new(crate::StaticProvider::new("unused")),
            approval: crate::approve_all(),
            parent_messages: Arc::new(Mutex::new(Vec::new())),
            limits: SubagentLimits {
                max_tasks: 1,
                max_parallel: 1,
                max_depth: 1,
            },
            depth: 1,
        };

        let depth_error = handler
            .execute_with_cancellation_async(
                json!({"tasks":[{"role":"reviewer","objective":"one"}]}),
                &CancellationToken::default(),
            )
            .await
            .unwrap_err();
        handler.depth = 0;
        let batch_error = handler
            .execute_with_cancellation_async(
                json!({"tasks":[
                    {"role":"reviewer","objective":"one"},
                    {"role":"verifier","objective":"two"}
                ]}),
                &CancellationToken::default(),
            )
            .await
            .unwrap_err();

        assert!(depth_error.to_string().contains("configured limit 1"));
        assert!(batch_error.to_string().contains("between 1 and 1"));
        assert_eq!(handler.runtime.list_sessions().unwrap().len(), 1);
        drop(handler);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn inherited_parent_context_does_not_duplicate_the_child_objective() {
        let directory = std::env::temp_dir().join(format!("delegate-context-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut config = crate::RuntimeConfig::for_workspace(&directory);
        config.data_root = directory.join("state");
        let runtime = FocusRuntime::open(config).unwrap();
        let parent = runtime.create_session("parent").unwrap();
        let objective = "UNIQUE_CHILD_OBJECTIVE";

        let inherited = runtime
            .build_subagent_context_from_messages(
                parent.id,
                &[focus_kernel::Message::text(
                    focus_kernel::Role::User,
                    "shared parent evidence",
                )],
            )
            .unwrap();

        assert!(!inherited.contains(objective));
        assert!(inherited.contains("shared parent evidence"));
        let _ = std::fs::remove_dir_all(directory);
    }
}
