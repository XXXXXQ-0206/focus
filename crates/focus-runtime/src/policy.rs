//! Single policy decision point for all Runtime tool executions.

use std::{
    collections::BTreeSet,
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

use focus_kernel::{CancellationSignal, NoCancellation};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::RuntimeError;

/// Capability requested by a tool invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOperation {
    /// Read a file below the workspace root.
    Read,
    /// Create or modify a workspace file.
    Write,
    /// Start a process in the workspace.
    Execute,
    /// Make a network request.
    Network,
    /// An extension-specific action.
    Other,
}

/// The outcome before applying an interactive approval handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyDecision {
    /// Run immediately.
    Allow,
    /// Require a host-provided approval decision.
    RequireApproval,
    /// Do not run.
    Deny,
}

/// Human-readable approval payload for CLI, TUI and future IDE adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Tool requesting the capability.
    pub tool: String,
    /// Requested operation.
    pub operation: ToolOperation,
    /// Sanitized tool arguments.
    pub arguments: Value,
    /// Reason supplied by the tool implementation.
    pub rationale: String,
}

/// Host decision for one approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    /// Approve only the current invocation.
    ApproveOnce,
    /// Approve matching tool/operation requests for the current policy engine.
    ApproveSession,
    /// Deny the invocation.
    Deny,
}

/// Interface-specific approval mechanism.
pub trait ApprovalHandler: Send + Sync {
    /// Decide whether one request may run.
    fn approve(&self, request: &ApprovalRequest) -> bool;
    /// Decide with cancellation support for interactive brokers.
    fn request(
        &self,
        request: &ApprovalRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ApprovalDecision, RuntimeError> {
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        Ok(if self.approve(request) {
            ApprovalDecision::ApproveOnce
        } else {
            ApprovalDecision::Deny
        })
    }
}

/// A non-interactive handler for automated environments.
#[derive(Debug, Clone, Copy)]
pub struct FixedApproval(pub bool);

impl ApprovalHandler for FixedApproval {
    fn approve(&self, _request: &ApprovalRequest) -> bool {
        self.0
    }
}

/// Default coding-agent policy. Reads are automatic; writes and commands are explicit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Policy {
    /// Default decision for workspace reads.
    pub read: PolicyDecision,
    /// Default decision for edits.
    pub write: PolicyDecision,
    /// Default decision for command execution.
    pub execute: PolicyDecision,
    /// Default decision for networking.
    pub network: PolicyDecision,
    /// Default decision for extension actions.
    pub other: PolicyDecision,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            read: PolicyDecision::Allow,
            write: PolicyDecision::RequireApproval,
            execute: PolicyDecision::RequireApproval,
            network: PolicyDecision::Deny,
            other: PolicyDecision::RequireApproval,
        }
    }
}

impl Policy {
    /// Resolve the configured decision for an operation.
    #[must_use]
    pub fn decide(&self, operation: ToolOperation) -> PolicyDecision {
        match operation {
            ToolOperation::Read => self.read,
            ToolOperation::Write => self.write,
            ToolOperation::Execute => self.execute,
            ToolOperation::Network => self.network,
            ToolOperation::Other => self.other,
        }
    }
}

/// Shared, immutable policy engine used by every tool path.
#[derive(Clone)]
pub struct PolicyEngine {
    policy: Policy,
    approval: Arc<dyn ApprovalHandler>,
    session_approvals: Arc<(Mutex<SessionApprovals>, Condvar)>,
}

#[derive(Default)]
struct SessionApprovals {
    approved: BTreeSet<(String, ToolOperation)>,
    in_flight: BTreeSet<(String, ToolOperation)>,
}

impl std::fmt::Debug for PolicyEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PolicyEngine")
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl PolicyEngine {
    /// Create the sole Runtime policy gate.
    #[must_use]
    pub fn new(policy: Policy, approval: Arc<dyn ApprovalHandler>) -> Self {
        Self {
            policy,
            approval,
            session_approvals: Arc::new((Mutex::new(SessionApprovals::default()), Condvar::new())),
        }
    }

    /// Authorize a requested capability, delegating approvals to the host.
    pub fn authorize(&self, request: ApprovalRequest) -> Result<(), RuntimeError> {
        self.authorize_with_cancellation(request, &NoCancellation)
    }

    /// Authorize while allowing interactive hosts to unblock on cancellation.
    pub fn authorize_with_cancellation(
        &self,
        request: ApprovalRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<(), RuntimeError> {
        match self.policy.decide(request.operation) {
            PolicyDecision::Allow => Ok(()),
            PolicyDecision::RequireApproval => {
                let key = (request.tool.clone(), request.operation);
                let (cache, changed) = &*self.session_approvals;
                loop {
                    if cancellation.is_cancelled() {
                        return Err(RuntimeError::Cancelled);
                    }
                    let mut state = cache.lock().map_err(|_| {
                        RuntimeError::Interaction("approval cache was poisoned".into())
                    })?;
                    if state.approved.contains(&key) {
                        return Ok(());
                    }
                    if !state.in_flight.contains(&key) {
                        state.in_flight.insert(key.clone());
                        break;
                    }
                    let _ = changed
                        .wait_timeout(state, Duration::from_millis(25))
                        .map_err(|_| {
                            RuntimeError::Interaction("approval cache was poisoned".into())
                        })?;
                }

                let decision = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.approval.request(&request, cancellation)
                }))
                .unwrap_or_else(|_| {
                    Err(RuntimeError::Interaction(
                        "approval handler panicked".into(),
                    ))
                });
                let mut state = cache
                    .lock()
                    .map_err(|_| RuntimeError::Interaction("approval cache was poisoned".into()))?;
                state.in_flight.remove(&key);
                if matches!(decision, Ok(ApprovalDecision::ApproveSession)) {
                    state.approved.insert(key);
                }
                changed.notify_all();
                drop(state);

                match decision? {
                    ApprovalDecision::ApproveOnce => Ok(()),
                    ApprovalDecision::ApproveSession => Ok(()),
                    ApprovalDecision::Deny => Err(RuntimeError::ApprovalRequired {
                        tool: request.tool,
                        operation: request.operation,
                    }),
                }
            }
            PolicyDecision::Deny => Err(RuntimeError::PolicyDenied {
                tool: request.tool,
                operation: request.operation,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    use std::{thread, time::Duration};

    use serde_json::json;

    use super::*;

    struct SessionApproval {
        requests: AtomicUsize,
    }

    impl ApprovalHandler for SessionApproval {
        fn approve(&self, _request: &ApprovalRequest) -> bool {
            false
        }

        fn request(
            &self,
            _request: &ApprovalRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ApprovalDecision, RuntimeError> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            Ok(ApprovalDecision::ApproveSession)
        }
    }

    #[test]
    fn cloned_policy_engines_share_session_approval_cache() {
        let approval = Arc::new(SessionApproval {
            requests: AtomicUsize::new(0),
        });
        let engine = PolicyEngine::new(Policy::default(), approval.clone());
        let cloned = engine.clone();

        engine
            .authorize(ApprovalRequest {
                tool: "shell".into(),
                operation: ToolOperation::Execute,
                arguments: json!({"command":"cargo test"}),
                rationale: "verify".into(),
            })
            .unwrap();
        cloned
            .authorize(ApprovalRequest {
                tool: "shell".into(),
                operation: ToolOperation::Execute,
                arguments: json!({"command":"cargo clippy"}),
                rationale: "lint".into(),
            })
            .unwrap();

        assert_eq!(approval.requests.load(Ordering::SeqCst), 1);
    }

    struct SlowSessionApproval {
        requests: AtomicUsize,
    }

    impl ApprovalHandler for SlowSessionApproval {
        fn approve(&self, _request: &ApprovalRequest) -> bool {
            false
        }

        fn request(
            &self,
            _request: &ApprovalRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ApprovalDecision, RuntimeError> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(100));
            Ok(ApprovalDecision::ApproveSession)
        }
    }

    #[test]
    fn concurrent_matching_requests_share_one_session_approval_prompt() {
        let approval = Arc::new(SlowSessionApproval {
            requests: AtomicUsize::new(0),
        });
        let engine = PolicyEngine::new(Policy::default(), approval.clone());
        let barrier = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for command in ["cargo test", "cargo clippy"] {
            let engine = engine.clone();
            let barrier = barrier.clone();
            workers.push(thread::spawn(move || {
                barrier.wait();
                engine.authorize(ApprovalRequest {
                    tool: "shell".into(),
                    operation: ToolOperation::Execute,
                    arguments: json!({"command":command}),
                    rationale: "verify".into(),
                })
            }));
        }
        barrier.wait();

        for worker in workers {
            worker.join().unwrap().unwrap();
        }

        assert_eq!(approval.requests.load(Ordering::SeqCst), 1);
    }

    struct SlowNonSessionApproval {
        requests: AtomicUsize,
        decision: ApprovalDecision,
    }

    impl ApprovalHandler for SlowNonSessionApproval {
        fn approve(&self, _request: &ApprovalRequest) -> bool {
            false
        }

        fn request(
            &self,
            _request: &ApprovalRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ApprovalDecision, RuntimeError> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(50));
            Ok(self.decision)
        }
    }

    #[test]
    fn non_session_decisions_wake_waiters_without_caching() {
        for decision in [ApprovalDecision::ApproveOnce, ApprovalDecision::Deny] {
            let approval = Arc::new(SlowNonSessionApproval {
                requests: AtomicUsize::new(0),
                decision,
            });
            let engine = PolicyEngine::new(Policy::default(), approval.clone());
            let barrier = Arc::new(Barrier::new(3));
            let mut workers = Vec::new();
            for command in ["cargo test", "cargo clippy"] {
                let engine = engine.clone();
                let barrier = barrier.clone();
                workers.push(thread::spawn(move || {
                    barrier.wait();
                    engine.authorize(ApprovalRequest {
                        tool: "shell".into(),
                        operation: ToolOperation::Execute,
                        arguments: json!({"command":command}),
                        rationale: "verify".into(),
                    })
                }));
            }
            barrier.wait();

            for worker in workers {
                let result = worker.join().unwrap();
                assert_eq!(result.is_ok(), decision == ApprovalDecision::ApproveOnce);
            }
            assert_eq!(approval.requests.load(Ordering::SeqCst), 2);
        }
    }

    struct PanicOnceApproval {
        requests: AtomicUsize,
    }

    impl ApprovalHandler for PanicOnceApproval {
        fn approve(&self, _request: &ApprovalRequest) -> bool {
            false
        }

        fn request(
            &self,
            _request: &ApprovalRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ApprovalDecision, RuntimeError> {
            if self.requests.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("approval handler panic payload must not escape");
            }
            Ok(ApprovalDecision::ApproveSession)
        }
    }

    #[test]
    fn panicking_approval_handler_clears_singleflight_state() {
        let approval = Arc::new(PanicOnceApproval {
            requests: AtomicUsize::new(0),
        });
        let engine = PolicyEngine::new(Policy::default(), approval.clone());
        let request = || ApprovalRequest {
            tool: "shell".into(),
            operation: ToolOperation::Execute,
            arguments: json!({"command":"cargo test"}),
            rationale: "verify".into(),
        };

        let first = engine.authorize(request()).unwrap_err();
        assert!(matches!(first, RuntimeError::Interaction(_)));

        let (result_tx, result_rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            result_tx.send(engine.authorize(request())).unwrap();
        });
        result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("panic cleanup must release the singleflight key")
            .unwrap();
        assert_eq!(approval.requests.load(Ordering::SeqCst), 2);
    }
}
