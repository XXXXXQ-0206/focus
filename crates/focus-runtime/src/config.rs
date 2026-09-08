use std::path::PathBuf;
use std::sync::Arc;

use crate::{
    RuntimeError, mcp,
    network::NetworkConfig,
    policy::{ApprovalHandler, Policy},
    sandbox, subagent, workflow,
};
use focus_kernel::{AgentStatus, CancellationSignal, NoCancellation};
use uuid::Uuid;

const MAX_CONTEXT_BUDGET: usize = 1_000_000;

/// All immutable Runtime configuration lives here, not in individual interfaces.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Project root subject to the Runtime workspace boundary.
    pub workspace_root: PathBuf,
    /// Persistent state root. Default callers should use `.focus-harness` below workspace root.
    pub data_root: PathBuf,
    /// Bounded context budget used for all provider calls.
    pub context_budget: usize,
    /// Policy applied to every built-in and extension tool invocation.
    pub policy: Policy,
    /// Bounded outbound HTTP configuration for the first-party web tool.
    pub network: NetworkConfig,
    /// Command isolation backend shared by the canonical shell tool.
    pub sandbox_backend: sandbox::SandboxBackend,
    /// Persistent stdio MCP server definitions available for explicit enablement.
    pub mcp_servers: Vec<mcp::McpServerConfig>,
    /// Whether this Runtime may start configured MCP server processes.
    pub enable_mcp: bool,
    /// Maximum times a tool-free model response may fail the workflow gate.
    pub max_workflow_gate_retries: usize,
}

impl RuntimeConfig {
    /// Create a configuration with the standard local state directory.
    #[must_use]
    pub fn for_workspace(workspace_root: impl Into<PathBuf>) -> Self {
        let workspace_root = workspace_root.into();
        Self {
            data_root: workspace_root.join(".focus-harness"),
            workspace_root,
            context_budget: 24_000,
            policy: Policy::default(),
            network: NetworkConfig::default(),
            sandbox_backend: sandbox::SandboxBackend::default(),
            mcp_servers: Vec::new(),
            enable_mcp: false,
            max_workflow_gate_retries: 2,
        }
    }

    /// Validate configuration invariants before any state or child process is created.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if !(1..=MAX_CONTEXT_BUDGET).contains(&self.context_budget) {
            return Err(RuntimeError::Configuration(format!(
                "context budget must be between 1 and {MAX_CONTEXT_BUDGET}"
            )));
        }
        self.network.validate()?;
        for server in &self.mcp_servers {
            server.validate()?;
        }
        Ok(())
    }
}

/// Result of a completed agent run, independent of CLI/TUI rendering.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Session receiving the event sequence.
    pub session_id: Uuid,
    /// Last completed lifecycle status.
    pub status: AgentStatus,
    /// Final assistant text when present.
    pub final_response: String,
    /// Local character-based context-size estimate provided for this run.
    pub estimated_context_tokens: usize,
    /// Items excluded from the model context by the configured budget.
    pub omitted_context_items: usize,
}

/// Per-run behavior shared by CLI, TUI, subagents, tests, and future IDE hosts.
#[derive(Clone)]
pub struct RunOptions {
    /// Approval path used by the canonical policy engine.
    pub approval: Arc<dyn ApprovalHandler>,
    /// Shared cancellation source observed by Pi and Runtime operations.
    pub cancellation: Arc<dyn CancellationSignal>,
    /// Explicit executable workflow contract.
    pub workflow: Option<workflow::CodingWorkflow>,
    /// Explicit model-requested delegation bounds shared by descendants.
    pub delegation: Option<subagent::SubagentLimits>,
    /// Current delegation depth, incremented only for isolated children.
    pub subagent_depth: usize,
}

impl RunOptions {
    /// Create the minimal core behavior used by default Runtime entry points.
    #[must_use]
    pub fn new(approval: Arc<dyn ApprovalHandler>) -> Self {
        Self::core(approval)
    }

    /// Create a core turn without workflow gates or model-visible delegation.
    #[must_use]
    pub fn core(approval: Arc<dyn ApprovalHandler>) -> Self {
        Self {
            approval,
            cancellation: Arc::new(NoCancellation),
            workflow: None,
            delegation: None,
            subagent_depth: 0,
        }
    }

    /// Create a core turn with the executable engineering workflow enabled.
    #[must_use]
    pub fn engineering(approval: Arc<dyn ApprovalHandler>) -> Self {
        let mut options = Self::core(approval);
        options.workflow = Some(workflow::CodingWorkflow::default());
        options
    }

    /// Expose bounded model-requested delegation for this turn.
    #[must_use]
    pub fn with_delegation(mut self, limits: subagent::SubagentLimits) -> Self {
        self.delegation = Some(limits);
        self
    }

    /// Validate per-run options before a session or child worker is created.
    pub fn validate(&self) -> Result<(), RuntimeError> {
        if let Some(limits) = self.delegation {
            limits.validate()?;
            if self.subagent_depth > limits.max_depth {
                return Err(RuntimeError::Configuration(format!(
                    "subagent depth {} exceeds configured maximum {}",
                    self.subagent_depth, limits.max_depth
                )));
            }
        } else if self.subagent_depth != 0 {
            return Err(RuntimeError::Configuration(
                "subagent depth requires delegation limits".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{RunOptions, RuntimeConfig};
    use crate::{FocusRuntime, approve_all, subagent::SubagentLimits};
    use std::path::PathBuf;
    use uuid::Uuid;

    #[test]
    fn workspace_configuration_uses_the_focus_state_directory() {
        let config = RuntimeConfig::for_workspace("workspace");
        assert_eq!(config.data_root, PathBuf::from("workspace/.focus-harness"));
    }

    #[test]
    fn opening_runtime_rejects_a_zero_context_budget() {
        let workspace = std::env::temp_dir().join(format!("focus-config-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&workspace).unwrap();
        for context_budget in [0, 1_000_001] {
            let mut config = RuntimeConfig::for_workspace(&workspace);
            config.context_budget = context_budget;

            let error = FocusRuntime::open(config).unwrap_err();

            assert!(error.to_string().contains("context budget"));
        }
        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn opening_runtime_rejects_invalid_mcp_configuration_before_it_is_enabled() {
        let workspace = std::env::temp_dir().join(format!("focus-config-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&workspace).unwrap();
        let mut config = RuntimeConfig::for_workspace(&workspace);
        config
            .mcp_servers
            .push(crate::mcp::McpServerConfig::new("   ", "fixture"));

        let error = FocusRuntime::open(config).unwrap_err();

        assert!(error.to_string().contains("MCP server name"));
        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn capabilities_are_explicit_on_run_options() {
        let core = RunOptions::core(approve_all());
        let engineering = RunOptions::engineering(approve_all());
        let limits = SubagentLimits::default();
        let delegated = RunOptions::core(approve_all()).with_delegation(limits);

        assert!(core.workflow.is_none());
        assert!(core.delegation.is_none());
        assert!(engineering.workflow.is_some());
        assert!(engineering.delegation.is_none());
        assert_eq!(delegated.delegation, Some(limits));
    }

    #[test]
    fn run_options_reject_invalid_subagent_limits() {
        let options = RunOptions::core(approve_all()).with_delegation(SubagentLimits {
            max_tasks: 0,
            max_parallel: 1,
            max_depth: 1,
        });

        let error = options.validate().unwrap_err();

        assert!(error.to_string().contains("subagent max_tasks"));
    }
}
