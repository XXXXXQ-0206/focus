//! Focus Runtime: the single coding-agent implementation above the small Pi kernel.
//!
//! Interfaces (CLI, TUI, subagents, and future IDE adapters) use this facade;
//! lifecycle and policy behavior live in the modules below so there is one
//! canonical implementation.

pub mod benchmark;
mod cancellation;
pub mod context;
pub mod events;
pub mod goal;
pub mod host;
pub mod interaction;
pub mod mcp;
pub mod memory;
pub mod network;
pub mod policy;
pub mod provider;
pub mod sandbox;
pub mod session;
pub mod subagent;
pub mod tools;
pub mod update;
pub mod workflow;

mod config;
mod diagnostics;
mod error;
mod runtime;

pub use config::{RunOptions, RunResult, RuntimeConfig};
pub use diagnostics::{StaticProvider, approve_all, deny_approvals, doctor, workspace_root};
pub use error::RuntimeError;
pub use runtime::FocusRuntime;

#[cfg(test)]
include!("tests.rs");

#[cfg(test)]
mod cancellation_tests {
    use std::{
        sync::{Arc, atomic::AtomicBool},
        time::Duration,
    };

    use crate::subagent::CancellationToken;

    #[tokio::test]
    async fn shared_wait_for_cancellation_returns_after_the_signal_is_set() {
        let cancellation = CancellationToken::from_shared(Arc::new(AtomicBool::new(false)));
        let trigger = cancellation.clone();
        let waiter = tokio::spawn(async move {
            crate::cancellation::wait_for_cancellation(&cancellation).await;
        });

        tokio::time::sleep(Duration::from_millis(1)).await;
        trigger.cancel();
        waiter.await.unwrap();
    }
}

#[cfg(test)]
mod host_protocol_tests {
    use std::sync::Arc;

    use focus_kernel::{Event, EventKind};
    use uuid::Uuid;

    #[test]
    fn host_protocol_resumes_by_event_id_and_resets_an_unknown_cursor() {
        let session_id = Uuid::new_v4();
        let first = Event::now(
            session_id,
            EventKind::Runtime {
                name: "first".into(),
                data: serde_json::json!({}),
            },
        );
        let second = Event::now(
            session_id,
            EventKind::Runtime {
                name: "second".into(),
                data: serde_json::json!({}),
            },
        );
        let request: crate::host::HostRequestV1 = serde_json::from_value(serde_json::json!({
            "protocol": "focus-host-v1",
            "operation": "resume",
            "session_id": session_id,
            "after_event_id": first.id,
        }))
        .unwrap();
        request.validate().unwrap();

        let resumed = crate::host::project_replay(
            session_id,
            &[first.clone(), second.clone()],
            request.after_event_id(),
        );
        assert!(!resumed.reset);
        assert_eq!(resumed.events.len(), 1);
        assert_eq!(resumed.events[0].cursor, second.id);
        assert_eq!(resumed.events[0].event.id, second.id);

        let reset = crate::host::project_replay(session_id, &[first, second], Some(Uuid::new_v4()));
        assert!(reset.reset);
        assert_eq!(reset.events.len(), 2);
    }

    #[test]
    fn host_projection_deduplicates_a_live_event_already_seen_in_replay() {
        let session_id = Uuid::new_v4();
        let replayed = Event::now(
            session_id,
            EventKind::Runtime {
                name: "replayed".into(),
                data: serde_json::json!({}),
            },
        );
        let live = Event::now(
            session_id,
            EventKind::Runtime {
                name: "live".into(),
                data: serde_json::json!({}),
            },
        );
        let projection =
            crate::host::project_replay(session_id, std::slice::from_ref(&replayed), None);
        let mut deduplicator = projection.deduplicator();

        assert!(deduplicator.accept(replayed).is_none());
        let frame = deduplicator.accept(live.clone()).unwrap();
        assert_eq!(frame.session_id, session_id);
        assert_eq!(frame.cursor, live.id);
        assert_eq!(frame.event.id, live.id);
    }

    #[test]
    fn runtime_host_replay_projects_the_canonical_session_stream() {
        let directory = std::env::temp_dir().join(format!("focus-host-replay-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let runtime =
            crate::FocusRuntime::open(crate::RuntimeConfig::for_workspace(&directory)).unwrap();
        let result = runtime
            .run(
                "host replay",
                "return a fixture response",
                Arc::new(crate::StaticProvider::new("done")),
                crate::deny_approvals(),
            )
            .unwrap();

        let projection = runtime.host_replay(result.session_id, None).unwrap();
        let replay = runtime.replay(result.session_id).unwrap();
        assert_eq!(
            projection
                .events
                .iter()
                .map(|frame| frame.event.id)
                .collect::<Vec<_>>(),
            replay.iter().map(|event| event.id).collect::<Vec<_>>()
        );
        assert!(
            projection
                .events
                .iter()
                .all(|frame| frame.session_id == result.session_id)
        );
        let _ = std::fs::remove_dir_all(directory);
    }
}
