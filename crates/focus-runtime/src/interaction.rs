//! Shared approval broker for CLI, TUI, subagents, and future IDE adapters.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};

use focus_kernel::{CancellationSignal, NoCancellation};
use uuid::Uuid;

use crate::{
    RuntimeError,
    policy::{ApprovalDecision, ApprovalHandler, ApprovalRequest},
};

const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Default)]
struct ApprovalQueue {
    pending: BTreeMap<Uuid, mpsc::Sender<ApprovalDecision>>,
    inbox_closed: bool,
}

type PendingApprovals = Arc<Mutex<ApprovalQueue>>;

/// Identified approval request consumed by exactly one interface inbox.
#[derive(Debug, Clone)]
pub struct ApprovalEnvelope {
    /// Stable identifier used to resolve this request once.
    pub id: Uuid,
    /// Sanitized approval payload produced by the Runtime policy layer.
    pub request: ApprovalRequest,
}

/// Runtime-side approval handler shared by worker threads and child agents.
#[derive(Clone)]
pub struct ApprovalBroker {
    requests: mpsc::Sender<ApprovalEnvelope>,
    pending: PendingApprovals,
}

impl std::fmt::Debug for ApprovalBroker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ApprovalBroker")
            .finish_non_exhaustive()
    }
}

impl ApprovalBroker {
    /// Create a broker and its single serialized host inbox.
    #[must_use]
    pub fn new() -> (Self, ApprovalInbox) {
        let (request_tx, request_rx) = mpsc::channel();
        let pending = Arc::new(Mutex::new(ApprovalQueue::default()));
        (
            Self {
                requests: request_tx,
                pending: pending.clone(),
            },
            ApprovalInbox {
                requests: request_rx,
                pending,
            },
        )
    }

    fn request_decision(
        &self,
        request: &ApprovalRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ApprovalDecision, RuntimeError> {
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }

        let id = Uuid::new_v4();
        let (response_tx, response_rx) = mpsc::channel();
        {
            let mut queue = self
                .pending
                .lock()
                .map_err(|_| RuntimeError::Interaction("approval queue was poisoned".into()))?;
            if queue.inbox_closed {
                return Err(RuntimeError::Interaction(
                    "approval inbox is no longer available".into(),
                ));
            }
            queue.pending.insert(id, response_tx);
        }
        if self
            .requests
            .send(ApprovalEnvelope {
                id,
                request: request.clone(),
            })
            .is_err()
        {
            self.remove_pending(id)?;
            return Err(RuntimeError::Interaction(
                "approval inbox is no longer available".into(),
            ));
        }

        loop {
            if cancellation.is_cancelled() {
                self.remove_pending(id)?;
                return Err(RuntimeError::Cancelled);
            }
            match response_rx.recv_timeout(CANCELLATION_POLL_INTERVAL) {
                Ok(decision) => return Ok(decision),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.remove_pending(id)?;
                    return Err(RuntimeError::Interaction(
                        "approval response channel was closed".into(),
                    ));
                }
            }
        }
    }

    fn remove_pending(&self, id: Uuid) -> Result<(), RuntimeError> {
        self.pending
            .lock()
            .map_err(|_| RuntimeError::Interaction("approval queue was poisoned".into()))?
            .pending
            .remove(&id);
        Ok(())
    }
}

impl ApprovalHandler for ApprovalBroker {
    fn approve(&self, request: &ApprovalRequest) -> bool {
        matches!(
            self.request_decision(request, &NoCancellation),
            Ok(ApprovalDecision::ApproveOnce | ApprovalDecision::ApproveSession)
        )
    }

    fn request(
        &self,
        request: &ApprovalRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ApprovalDecision, RuntimeError> {
        self.request_decision(request, cancellation)
    }
}

/// Host-side serialized approval stream used by CLI, TUI, and IDE adapters.
pub struct ApprovalInbox {
    requests: mpsc::Receiver<ApprovalEnvelope>,
    pending: PendingApprovals,
}

impl std::fmt::Debug for ApprovalInbox {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ApprovalInbox")
            .finish_non_exhaustive()
    }
}

impl ApprovalInbox {
    /// Wait for the next queued request, returning `None` on timeout or shutdown.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<ApprovalEnvelope>, RuntimeError> {
        match self.requests.recv_timeout(timeout) {
            Ok(request) => Ok(Some(request)),
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => Ok(None),
        }
    }

    /// Resolve one live request. Each identifier is accepted at most once.
    pub fn resolve(&self, id: Uuid, decision: ApprovalDecision) -> Result<(), RuntimeError> {
        let response = self
            .pending
            .lock()
            .map_err(|_| RuntimeError::Interaction("approval queue was poisoned".into()))?
            .pending
            .remove(&id)
            .ok_or_else(|| {
                RuntimeError::Interaction(format!("approval request `{id}` is stale"))
            })?;
        response.send(decision).map_err(|_| {
            RuntimeError::Interaction(format!("approval request `{id}` was cancelled"))
        })
    }
}

impl Drop for ApprovalInbox {
    fn drop(&mut self) {
        if let Ok(mut queue) = self.pending.lock() {
            queue.inbox_closed = true;
            queue.pending.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use focus_kernel::NoCancellation;
    use serde_json::json;

    use crate::{
        policy::{ApprovalDecision, ApprovalHandler, ApprovalRequest, ToolOperation},
        subagent::CancellationToken,
    };

    use super::ApprovalBroker;

    #[test]
    fn broker_round_trips_identified_approval_requests() {
        let (broker, inbox) = ApprovalBroker::new();
        let worker = thread::spawn(move || {
            broker.request(
                &ApprovalRequest {
                    tool: "shell".into(),
                    operation: ToolOperation::Execute,
                    arguments: json!({"command":"cargo test"}),
                    rationale: "verify".into(),
                },
                &NoCancellation,
            )
        });

        let pending = inbox.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
        assert_eq!(pending.request.tool, "shell");
        inbox
            .resolve(pending.id, ApprovalDecision::ApproveOnce)
            .unwrap();

        assert_eq!(
            worker.join().unwrap().unwrap(),
            ApprovalDecision::ApproveOnce
        );
    }

    #[test]
    fn cancellation_releases_a_blocked_approval_worker() {
        let (broker, inbox) = ApprovalBroker::new();
        let cancellation = CancellationToken::default();
        let worker_cancellation = cancellation.clone();
        let worker = thread::spawn(move || {
            broker.request(
                &ApprovalRequest {
                    tool: "write_file".into(),
                    operation: ToolOperation::Write,
                    arguments: json!({"path":"x"}),
                    rationale: "edit".into(),
                },
                &worker_cancellation,
            )
        });
        let pending = inbox.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();

        cancellation.cancel();

        assert!(worker.join().unwrap().is_err());
        assert!(
            inbox
                .resolve(pending.id, ApprovalDecision::ApproveOnce)
                .is_err()
        );
    }

    #[test]
    fn resolved_approval_ids_become_stale() {
        let (broker, inbox) = ApprovalBroker::new();
        let worker = thread::spawn(move || {
            broker.request(
                &ApprovalRequest {
                    tool: "shell".into(),
                    operation: ToolOperation::Execute,
                    arguments: json!({"command":"cargo check"}),
                    rationale: "check".into(),
                },
                &NoCancellation,
            )
        });

        let pending = inbox.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
        inbox
            .resolve(pending.id, ApprovalDecision::ApproveOnce)
            .unwrap();
        assert_eq!(
            worker.join().unwrap().unwrap(),
            ApprovalDecision::ApproveOnce
        );

        assert!(inbox.resolve(pending.id, ApprovalDecision::Deny).is_err());
        assert!(
            inbox
                .resolve(uuid::Uuid::new_v4(), ApprovalDecision::ApproveOnce)
                .is_err()
        );
    }

    #[test]
    fn dropping_inbox_releases_waiters_and_clears_pending_requests() {
        let (broker, inbox) = ApprovalBroker::new();
        let worker_broker = broker.clone();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let worker = thread::spawn(move || {
            let result = worker_broker.request(
                &ApprovalRequest {
                    tool: "shell".into(),
                    operation: ToolOperation::Execute,
                    arguments: json!({"command":"cargo test"}),
                    rationale: "verify".into(),
                },
                &NoCancellation,
            );
            result_tx.send(result).unwrap();
        });

        inbox.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
        drop(inbox);

        let error = result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("dropping the inbox must release the waiter")
            .unwrap_err();
        assert!(matches!(error, crate::RuntimeError::Interaction(_)));
        assert!(broker.pending.lock().unwrap().pending.is_empty());
        worker.join().unwrap();
    }
}
