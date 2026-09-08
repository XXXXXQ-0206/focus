use crate::policy::ToolOperation;
use focus_kernel::KernelError;
use thiserror::Error;

/// Runtime errors deliberately separate policy, storage, and model/kernel failures.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Runtime configuration violated a construction-time invariant.
    #[error("configuration error: {0}")]
    Configuration(String),
    /// A kernel primitive returned an error.
    #[error(transparent)]
    Kernel(KernelError),
    /// The workspace boundary rejected an operation.
    #[error("sandbox rejected operation: {0}")]
    Sandbox(String),
    /// A memory record could not be persisted or decoded.
    #[error("memory error: {0}")]
    Memory(String),
    /// A session metadata or replay operation failed.
    #[error("session error: {0}")]
    Session(String),
    /// A policy permanently denied a capability.
    #[error("policy denied {operation:?} for tool `{tool}`")]
    PolicyDenied {
        /// Tool name.
        tool: String,
        /// Requested operation.
        operation: ToolOperation,
    },
    /// A host approval handler declined a capability.
    #[error("approval was not granted for {operation:?} in tool `{tool}`")]
    ApprovalRequired {
        /// Tool name.
        tool: String,
        /// Requested operation.
        operation: ToolOperation,
    },
    /// A tool argument did not satisfy its contract.
    #[error("invalid tool input: {0}")]
    ToolInput(String),
    /// The configured network boundary rejected a request or transport error.
    #[error("network error: {0}")]
    Network(String),
    /// MCP tool catalog registration failed.
    #[error("MCP error: {0}")]
    Mcp(String),
    /// A child agent failed.
    #[error("subagent error: {0}")]
    Subagent(String),
    /// The executable coding workflow rejected completion or recovery.
    #[error("workflow error: {0}")]
    Workflow(String),
    /// A caller cancelled the task.
    #[error("task was cancelled")]
    Cancelled,
    /// Shared UI/approval interaction failed.
    #[error("interaction error: {0}")]
    Interaction(String),
}

impl From<KernelError> for RuntimeError {
    fn from(error: KernelError) -> Self {
        match error {
            KernelError::Cancelled => Self::Cancelled,
            error => Self::Kernel(error),
        }
    }
}

#[cfg(test)]
mod runtime_error_tests {
    use super::*;

    #[test]
    fn kernel_cancellation_maps_to_runtime_cancellation() {
        assert!(matches!(
            RuntimeError::from(KernelError::Cancelled),
            RuntimeError::Cancelled
        ));
    }
}
