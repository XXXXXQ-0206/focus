//! Codex-inspired software-engineering workflow built solely on Runtime primitives.

use std::collections::HashMap;

use focus_kernel::{Event, EventKind, Message, Role};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Explicit stage surfaced to every Runtime interface through session events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStage {
    /// Inspect repository structure and task-relevant code before acting.
    Explore,
    /// State a concrete implementation plan and constraints.
    Plan,
    /// Apply focused source changes.
    Implement,
    /// Run task-relevant checks and diagnose failures.
    Verify,
    /// Review diffs for regressions, missing tests, and scope drift.
    Review,
    /// Produce a concise evidence-bearing handoff.
    Complete,
}

/// Evidence accepted by the executable workflow gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkflowEvidence {
    /// A canonical read/search tool inspected the repository.
    RepositoryInspected {
        /// Canonical tool which produced the evidence.
        tool: String,
    },
    /// The model explicitly recorded an implementation plan.
    PlanRecorded {
        /// Concise implementation plan.
        summary: String,
    },
    /// A canonical mutation tool changed the workspace.
    MutationApplied {
        /// Project-relative path when the tool supplies one.
        path: Option<String>,
    },
    /// The model explicitly established that no mutation is required.
    NoChangeRequired {
        /// Evidence-bearing reason no mutation is required.
        reason: String,
    },
    /// A real verification command completed with its observed status.
    VerificationRun {
        /// Exact command reported by the canonical shell tool.
        command: String,
        /// Whether the observed exit status was successful.
        success: bool,
    },
    /// The model explicitly recorded a post-implementation review.
    ReviewRecorded {
        /// Concise review outcome.
        summary: String,
    },
}

/// Persistable state for one software-engineering workflow run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowState {
    /// Stable identity across retries and interrupted recovery.
    pub run_id: Uuid,
    /// Current stage derived from accepted evidence.
    pub current: WorkflowStage,
    /// Deduplicated evidence retained for gates and replay.
    pub evidence: Vec<WorkflowEvidence>,
    /// Number of model completions rejected by the gate.
    pub gate_failures: usize,
}

fn matches_run_id(data: &Value, serialized_run_id: &str) -> bool {
    data.get("run_id").and_then(Value::as_str) == Some(serialized_run_id)
}

impl WorkflowState {
    /// Start a fresh workflow in the explore stage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            run_id: Uuid::new_v4(),
            current: WorkflowStage::Explore,
            evidence: Vec::new(),
            gate_failures: 0,
        }
    }

    /// Retain one unique evidence item and advance the derived stage.
    pub fn record(&mut self, evidence: WorkflowEvidence) -> bool {
        if self.evidence.contains(&evidence) {
            return false;
        }
        self.evidence.push(evidence);
        self.current = self.next_stage(true);
        true
    }

    /// Return stable gate requirement identifiers that are still missing.
    #[must_use]
    pub fn missing_requirements(&self, require_verification: bool) -> Vec<&'static str> {
        let (mut explored, mut planned, mut implemented, mut verified, mut reviewed) =
            (false, false, false, false, false);
        for item in &self.evidence {
            match item {
                WorkflowEvidence::RepositoryInspected { .. } => explored = true,
                WorkflowEvidence::PlanRecorded { .. } => planned = true,
                WorkflowEvidence::MutationApplied { .. }
                | WorkflowEvidence::NoChangeRequired { .. } => implemented = true,
                WorkflowEvidence::VerificationRun { success: true, .. } => verified = true,
                WorkflowEvidence::ReviewRecorded { .. } => reviewed = true,
                WorkflowEvidence::VerificationRun { success: false, .. } => {}
            }
        }
        let mut missing = Vec::new();
        if !explored {
            missing.push("explore");
        }
        if !planned {
            missing.push("plan");
        }
        if !implemented {
            missing.push("implement_or_no_change");
        }
        if require_verification && !verified {
            missing.push("verify");
        }
        if !reviewed {
            missing.push("review");
        }
        missing
    }

    /// Recover the most recent incomplete workflow from persisted Runtime events.
    #[must_use]
    pub fn recover_latest(events: &[Event]) -> Option<Self> {
        let mut active = None;
        for event in events {
            let EventKind::Runtime { name, data } = &event.kind else {
                continue;
            };
            match name.as_str() {
                "workflow_started" => {
                    active = serde_json::from_value::<Self>(data.clone())
                        .ok()
                        .map(|state| {
                            let run_id = state.run_id.to_string();
                            (state, run_id)
                        });
                }
                "workflow_evidence_recorded" => {
                    let Some((state, active_run_id)) = active.as_mut() else {
                        continue;
                    };
                    if matches_run_id(data, active_run_id)
                        && let Some(evidence) = data.get("evidence")
                        && let Ok(evidence) = serde_json::from_value(evidence.clone())
                    {
                        state.record(evidence);
                    }
                }
                "workflow_gate_failed" => {
                    if let Some((state, active_run_id)) = active.as_mut()
                        && matches_run_id(data, active_run_id)
                    {
                        state.gate_failures = state.gate_failures.saturating_add(1);
                    }
                }
                "workflow_completed"
                    if active
                        .as_ref()
                        .is_some_and(|(_, active_run_id)| matches_run_id(data, active_run_id)) =>
                {
                    active = None;
                }
                _ => {}
            }
        }
        active.map(|(state, _)| state)
    }

    fn next_stage(&self, require_verification: bool) -> WorkflowStage {
        match self
            .missing_requirements(require_verification)
            .first()
            .copied()
        {
            Some("explore") => WorkflowStage::Explore,
            Some("plan") => WorkflowStage::Plan,
            Some("implement_or_no_change") => WorkflowStage::Implement,
            Some("verify") => WorkflowStage::Verify,
            Some("review") => WorkflowStage::Review,
            _ => WorkflowStage::Complete,
        }
    }
}

impl Default for WorkflowState {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract evidence from canonical assistant tool calls and matching tool results.
#[must_use]
pub fn transcript_evidence(messages: &[Message]) -> Vec<WorkflowEvidence> {
    let results = messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .filter_map(|message| {
            message
                .tool_call_id
                .as_ref()
                .map(|id| (id.as_str(), message))
        })
        .collect::<HashMap<_, _>>();
    let mut evidence = Vec::new();
    for call in messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .flat_map(|message| &message.tool_calls)
    {
        let Some(result) = results.get(call.id.as_str()) else {
            continue;
        };
        match call.name.as_str() {
            "read_file" | "search" | "delegate" => {
                if !result.is_error {
                    evidence.push(WorkflowEvidence::RepositoryInspected {
                        tool: call.name.clone(),
                    });
                }
            }
            "write_file" | "mkdir" if !result.is_error => {
                evidence.push(WorkflowEvidence::MutationApplied {
                    path: call
                        .arguments
                        .get("path")
                        .and_then(|value| value.as_str())
                        .map(str::to_owned),
                });
            }
            "shell" => match call
                .arguments
                .get("purpose")
                .and_then(|value| value.as_str())
            {
                Some("verify") => evidence.push(WorkflowEvidence::VerificationRun {
                    command: call
                        .arguments
                        .get("command")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .into(),
                    success: !result.is_error && result.content.starts_with("exit_code: 0\n"),
                }),
                Some("explore") | Some("review") if result.content.starts_with("exit_code: ") => {
                    evidence.push(WorkflowEvidence::RepositoryInspected {
                        tool: "shell".into(),
                    });
                }
                Some(_) if result.content.starts_with("exit_code: ") => {
                    evidence.push(WorkflowEvidence::RepositoryInspected {
                        tool: "shell".into(),
                    });
                }
                _ => {}
            },
            "workflow_checkpoint" => {
                let summary = call
                    .arguments
                    .get("summary")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_owned();
                if !result.is_error {
                    match call.arguments.get("kind").and_then(|value| value.as_str()) {
                        Some("plan") => evidence.push(WorkflowEvidence::PlanRecorded { summary }),
                        Some("no_change") => {
                            evidence.push(WorkflowEvidence::NoChangeRequired { reason: summary });
                        }
                        Some("review") => {
                            evidence.push(WorkflowEvidence::ReviewRecorded { summary });
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    evidence
}

/// Reconstruct the transcript belonging to one persisted workflow run.
///
/// Runtime events remain the canonical source so recovery can reconcile tool
/// evidence that was persisted immediately before a model, process, or host
/// interruption.
#[must_use]
pub fn persisted_workflow_messages(events: &[Event], run_id: Uuid) -> Vec<Message> {
    let run_id = run_id.to_string();
    let mut collecting = false;
    let mut messages = Vec::new();
    for event in events {
        match &event.kind {
            EventKind::Runtime { name, data } if name == "workflow_started" => {
                collecting = matches_run_id(data, &run_id);
                if collecting {
                    messages.clear();
                }
            }
            EventKind::MessageAdded { message } if collecting => messages.push(message.clone()),
            EventKind::Runtime { name, data }
                if name == "workflow_completed" && matches_run_id(data, &run_id) =>
            {
                collecting = false;
            }
            _ => {}
        }
    }
    messages
}

/// The stable workflow contract injected into every coding turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodingWorkflow {
    /// Ordered stages used for an engineering task.
    pub stages: Vec<WorkflowStage>,
    /// Whether verification is a required completion gate.
    pub require_verification: bool,
}

impl Default for CodingWorkflow {
    fn default() -> Self {
        Self {
            stages: vec![
                WorkflowStage::Explore,
                WorkflowStage::Plan,
                WorkflowStage::Implement,
                WorkflowStage::Verify,
                WorkflowStage::Review,
                WorkflowStage::Complete,
            ],
            require_verification: true,
        }
    }
}

impl CodingWorkflow {
    /// Render the direct-response instruction used by the minimal core turn.
    #[must_use]
    pub const fn core_instruction() -> &'static str {
        "You are a coding assistant. Use registered tools only when they materially help, report only facts you observed, and do not claim checks or file changes that did not occur."
    }

    /// Render the system instruction used by CLI, TUI and future IDE hosts.
    #[must_use]
    pub fn instruction(&self) -> String {
        let verification = if self.require_verification {
            "Run the narrowest relevant checks after edits and report their actual result."
        } else {
            "Report which verification remains outstanding."
        };
        format!(
            "You are a code agent. Work through this lifecycle: explore the repository before edits; plan focused changes; implement with the registered tools; verify and review the resulting diff; report facts, changed files, and remaining uncertainty. Evidence must come from canonical tools, not prose: use `read_file` or `search` to explore; call `workflow_checkpoint` with `plan`, then `no_change` when no mutation is required (or use a canonical mutation tool); run `shell` with `purpose` set to `verify`; and call `workflow_checkpoint` with `review` before your final response. Do not claim checks that were not run. {verification}"
        )
    }
}

#[cfg(test)]
mod tests {
    use focus_kernel::{Event, EventKind, Message, ToolCall};
    use serde_json::json;
    use uuid::Uuid;

    use super::{WorkflowEvidence, WorkflowState, transcript_evidence};

    #[test]
    fn completion_gate_requires_real_engineering_evidence() {
        let mut state = WorkflowState::new();
        assert_eq!(
            state.missing_requirements(true),
            vec![
                "explore",
                "plan",
                "implement_or_no_change",
                "verify",
                "review"
            ]
        );

        state.record(WorkflowEvidence::RepositoryInspected {
            tool: "search".into(),
        });
        state.record(WorkflowEvidence::PlanRecorded {
            summary: "focused plan".into(),
        });
        state.record(WorkflowEvidence::NoChangeRequired {
            reason: "already correct".into(),
        });
        state.record(WorkflowEvidence::VerificationRun {
            command: "cargo test".into(),
            success: true,
        });
        state.record(WorkflowEvidence::ReviewRecorded {
            summary: "reviewed".into(),
        });

        assert!(state.missing_requirements(true).is_empty());
    }

    #[test]
    fn interrupted_workflow_recovers_from_runtime_events() {
        let session_id = Uuid::new_v4();
        let state = WorkflowState::new();
        let evidence = WorkflowEvidence::PlanRecorded {
            summary: "resume this".into(),
        };
        let events = vec![
            Event::now(
                session_id,
                EventKind::Runtime {
                    name: "workflow_started".into(),
                    data: serde_json::to_value(&state).unwrap(),
                },
            ),
            Event::now(
                session_id,
                EventKind::Runtime {
                    name: "workflow_evidence_recorded".into(),
                    data: json!({"run_id": state.run_id, "evidence": evidence}),
                },
            ),
        ];

        let recovered = WorkflowState::recover_latest(&events).unwrap();

        assert_eq!(recovered.run_id, state.run_id);
        assert!(recovered.evidence.contains(&evidence));
    }

    #[test]
    fn workflow_run_id_matcher_accepts_only_the_expected_serialized_id() {
        let run_id = Uuid::new_v4();
        let serialized = run_id.to_string();

        assert!(super::matches_run_id(
            &json!({"run_id": serialized}),
            &serialized
        ));
        assert!(!super::matches_run_id(
            &json!({"run_id": Uuid::new_v4()}),
            &serialized
        ));
        assert!(!super::matches_run_id(
            &json!({"run_id": "invalid"}),
            &serialized
        ));
        assert!(!super::matches_run_id(&json!({}), &serialized));
    }

    #[test]
    fn transcript_evidence_uses_the_latest_matching_tool_result() {
        let call = ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            arguments: json!({"query": "Focus"}),
        };
        let messages = vec![
            Message::assistant("", vec![call]),
            Message::tool_error("call-1", "search", "temporary failure"),
            Message::tool_result("call-1", "search", "matched"),
        ];

        assert_eq!(
            super::transcript_evidence(&messages),
            vec![WorkflowEvidence::RepositoryInspected {
                tool: "search".into(),
            }]
        );
    }

    #[test]
    fn persisted_messages_keep_only_the_requested_workflow_run() {
        let session_id = Uuid::new_v4();
        let requested = WorkflowState::new();
        let other = WorkflowState::new();
        let events = vec![
            Event::now(
                session_id,
                EventKind::Runtime {
                    name: "workflow_started".into(),
                    data: json!({"run_id": requested.run_id}),
                },
            ),
            Event::now(
                session_id,
                EventKind::MessageAdded {
                    message: Message::text(focus_kernel::Role::User, "requested"),
                },
            ),
            Event::now(
                session_id,
                EventKind::Runtime {
                    name: "workflow_started".into(),
                    data: json!({"run_id": other.run_id}),
                },
            ),
            Event::now(
                session_id,
                EventKind::MessageAdded {
                    message: Message::text(focus_kernel::Role::User, "other"),
                },
            ),
        ];

        assert_eq!(
            super::persisted_workflow_messages(&events, requested.run_id)
                .into_iter()
                .map(|message| message.content)
                .collect::<Vec<_>>(),
            vec!["requested"]
        );
    }

    #[test]
    fn instruction_names_the_canonical_workflow_tools() {
        let instruction = super::CodingWorkflow::default().instruction();

        for tool in ["read_file", "search", "workflow_checkpoint", "shell"] {
            assert!(instruction.contains(tool), "instruction omitted {tool}");
        }
    }

    #[test]
    fn successful_exploratory_shell_contributes_repository_evidence() {
        let call = ToolCall {
            id: "explore-shell".into(),
            name: "shell".into(),
            arguments: json!({"command":"dir","purpose":"explore"}),
        };
        let messages = vec![
            Message::assistant(String::new(), vec![call]),
            Message::tool_result("explore-shell", "shell", "exit_code: 0\noutput"),
        ];

        assert_eq!(
            transcript_evidence(&messages),
            vec![WorkflowEvidence::RepositoryInspected {
                tool: "shell".into()
            }]
        );
    }

    #[test]
    fn exploratory_shell_with_partial_output_keeps_repository_evidence() {
        let call = ToolCall {
            id: "explore-shell-partial".into(),
            name: "shell".into(),
            arguments: json!({"command":"dir && findstr missing","purpose":"explore"}),
        };
        let result = Message::tool_error(
            "explore-shell-partial",
            "shell",
            "exit_code: 1\nstdout:\nCargo.toml\nstderr:\nFINDSTR: missing",
        );
        let messages = vec![Message::assistant(String::new(), vec![call]), result];

        assert_eq!(
            transcript_evidence(&messages),
            vec![WorkflowEvidence::RepositoryInspected {
                tool: "shell".into()
            }]
        );
    }

    #[test]
    fn successful_delegate_contributes_parent_exploration_evidence() {
        let messages = vec![
            Message::assistant(
                String::new(),
                vec![ToolCall {
                    id: "delegate".into(),
                    name: "delegate".into(),
                    arguments: json!({"tasks":[{"role":"reviewer","objective":"inspect"}]}),
                }],
            ),
            Message::tool_result("delegate", "delegate", "[{\"status\":\"completed\"}]"),
        ];

        assert_eq!(
            transcript_evidence(&messages),
            vec![WorkflowEvidence::RepositoryInspected {
                tool: "delegate".into()
            }]
        );
    }
}
