//! Versioned, resumable event projections for external Focus hosts.

use std::{
    collections::HashSet,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};

use focus_kernel::{Event, KernelError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Stable identifier for the first Focus host protocol version.
pub const HOST_PROTOCOL_V1: &str = "focus-host-v1";

/// One JSONL request received from an external host.
#[derive(Debug, Clone, Deserialize)]
pub struct HostRequestV1 {
    /// Protocol identifier selected by the client.
    pub protocol: String,
    /// Keep the attachment open across terminal turns in a long-lived chat session.
    #[serde(default)]
    pub follow: bool,
    /// Requested attachment behavior.
    #[serde(flatten)]
    pub operation: HostOperationV1,
}

/// The two read-only session attachment operations supported by v1.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum HostOperationV1 {
    /// Replay a full baseline before subscribing to future events.
    Attach {
        /// Focus session whose event projection is requested.
        session_id: Uuid,
    },
    /// Replay only events after a durable event cursor when it remains available.
    Resume {
        /// Focus session whose event projection is requested.
        session_id: Uuid,
        /// Last event the client durably incorporated.
        #[serde(default)]
        after_event_id: Option<Uuid>,
    },
}

impl HostRequestV1 {
    /// Reject requests that target an unsupported wire protocol.
    pub fn validate(&self) -> Result<(), HostProtocolError> {
        if self.protocol == HOST_PROTOCOL_V1 {
            Ok(())
        } else {
            Err(HostProtocolError::UnsupportedProtocol(
                self.protocol.clone(),
            ))
        }
    }

    /// Return the session selected by this request.
    #[must_use]
    pub fn session_id(&self) -> Uuid {
        match self.operation {
            HostOperationV1::Attach { session_id } | HostOperationV1::Resume { session_id, .. } => {
                session_id
            }
        }
    }

    /// Return the resume cursor, if this is a resume operation.
    #[must_use]
    pub fn after_event_id(&self) -> Option<Uuid> {
        match self.operation {
            HostOperationV1::Attach { .. } => None,
            HostOperationV1::Resume { after_event_id, .. } => after_event_id,
        }
    }

    /// Whether this peer needs one persistent attachment across multiple chat turns.
    #[must_use]
    pub fn follow(&self) -> bool {
        self.follow
    }
}

/// Request validation failure for the host transport boundary.
#[derive(Debug, Error)]
pub enum HostProtocolError {
    /// The peer selected a protocol this Focus binary does not implement.
    #[error("unsupported host protocol `{0}`; expected `{HOST_PROTOCOL_V1}`")]
    UnsupportedProtocol(String),
}

/// One event projected to a selected session by the host protocol.
#[derive(Debug, Clone, Serialize)]
pub struct HostEventEnvelopeV1 {
    /// Protocol identifier for this line.
    pub protocol: &'static str,
    /// Discriminant for JSONL consumers.
    #[serde(rename = "type")]
    pub frame_type: &'static str,
    /// The selected session, including when a replay contains inherited events.
    pub session_id: Uuid,
    /// Stable replay cursor, equal to the source event ID.
    pub cursor: Uuid,
    /// Canonical persisted event; this is never a host-created transcript event.
    pub event: Event,
}

impl HostEventEnvelopeV1 {
    fn from_event(session_id: Uuid, event: Event) -> Self {
        Self {
            protocol: HOST_PROTOCOL_V1,
            frame_type: "event",
            session_id,
            cursor: event.id,
            event,
        }
    }
}

/// A reset notice sent before a full baseline when the requested cursor is absent.
#[derive(Debug, Clone, Serialize)]
pub struct HostResetEnvelopeV1 {
    /// Protocol identifier for this line.
    pub protocol: &'static str,
    /// Discriminant for JSONL consumers.
    #[serde(rename = "type")]
    pub frame_type: &'static str,
    /// The selected session requiring a replay baseline.
    pub session_id: Uuid,
    /// Reason the host started a full baseline.
    pub reason: &'static str,
}

/// A structured protocol failure written to stdout instead of terminal prose.
#[derive(Debug, Clone, Serialize)]
pub struct HostErrorEnvelopeV1 {
    /// Protocol identifier for this line.
    pub protocol: &'static str,
    /// Discriminant for JSONL consumers.
    #[serde(rename = "type")]
    pub frame_type: &'static str,
    /// Non-secret protocol or replay error.
    pub message: String,
}

/// Any JSONL frame produced by a v1 host connection.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum HostFrameV1 {
    /// Canonical persisted event projection.
    Event(HostEventEnvelopeV1),
    /// Full baseline is required before resuming.
    Reset(HostResetEnvelopeV1),
    /// A request or replay error.
    Error(HostErrorEnvelopeV1),
}

impl HostFrameV1 {
    /// Create an explicit reset notification.
    #[must_use]
    pub fn reset(session_id: Uuid) -> Self {
        Self::Reset(HostResetEnvelopeV1 {
            protocol: HOST_PROTOCOL_V1,
            frame_type: "reset",
            session_id,
            reason: "cursor_not_found",
        })
    }

    /// Create a structured non-secret host error.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(HostErrorEnvelopeV1 {
            protocol: HOST_PROTOCOL_V1,
            frame_type: "error",
            message: message.into(),
        })
    }
}

/// Replay projection plus the deduplicator needed to join a live subscription.
#[derive(Debug, Clone)]
pub struct HostReplayV1 {
    session_id: Uuid,
    /// Whether the requested cursor was absent and clients must rebuild from this baseline.
    pub reset: bool,
    /// Ordered canonical event frames for this attachment operation.
    pub events: Vec<HostEventEnvelopeV1>,
}

impl HostReplayV1 {
    /// Seed a live-event deduplicator with every event emitted by this projection.
    #[must_use]
    pub fn deduplicator(&self) -> HostEventDeduplicator {
        HostEventDeduplicator {
            session_id: self.session_id,
            seen: self.events.iter().map(|event| event.cursor).collect(),
        }
    }
}

/// Ephemeral per-connection live/replay deduplication state.
#[derive(Debug, Clone)]
pub struct HostEventDeduplicator {
    session_id: Uuid,
    seen: HashSet<Uuid>,
}

impl HostEventDeduplicator {
    /// Create a deduplicator for an empty attachment baseline.
    #[must_use]
    pub fn empty(session_id: Uuid) -> Self {
        Self {
            session_id,
            seen: HashSet::new(),
        }
    }

    /// Return a frame only when this event has not been emitted before.
    pub fn accept(&mut self, event: Event) -> Option<HostEventEnvelopeV1> {
        if !self.seen.insert(event.id) {
            return None;
        }
        Some(HostEventEnvelopeV1::from_event(self.session_id, event))
    }
}

/// Project ordered persisted events after an optional durable event cursor.
#[must_use]
pub fn project_replay(
    session_id: Uuid,
    events: &[Event],
    after_event_id: Option<Uuid>,
) -> HostReplayV1 {
    let (start, reset) = match after_event_id {
        None => (0, false),
        Some(cursor) => match events.iter().position(|event| event.id == cursor) {
            Some(index) => (index.saturating_add(1), false),
            None => (0, true),
        },
    };
    HostReplayV1 {
        session_id,
        reset,
        events: events[start..]
            .iter()
            .cloned()
            .map(|event| HostEventEnvelopeV1::from_event(session_id, event))
            .collect(),
    }
}

/// Incremental reader for the append-only JSONL event file owned by one session.
///
/// This is deliberately ephemeral: reconnects still derive their baseline from
/// the canonical event store and cursor protocol. The reader avoids rescanning
/// the full history while an external host observes another Focus process.
#[derive(Debug)]
pub struct JsonlEventTail {
    path: PathBuf,
    offset: u64,
    incomplete_line: Vec<u8>,
}

impl JsonlEventTail {
    /// Start tailing after the current durable end of a session event file.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, KernelError> {
        let path = path.into();
        let offset = match std::fs::metadata(&path) {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => return Err(KernelError::Store(error.to_string())),
        };
        Ok(Self {
            path,
            offset,
            incomplete_line: Vec::new(),
        })
    }

    /// Read only complete events appended since the prior call.
    pub fn read_new(&mut self) -> Result<Vec<Event>, KernelError> {
        let mut file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(KernelError::Store(error.to_string())),
        };
        let length = file
            .metadata()
            .map_err(|error| KernelError::Store(error.to_string()))?
            .len();
        if length < self.offset {
            self.offset = 0;
            self.incomplete_line.clear();
        }
        file.seek(SeekFrom::Start(self.offset))
            .map_err(|error| KernelError::Store(error.to_string()))?;
        let mut appended = Vec::new();
        file.read_to_end(&mut appended)
            .map_err(|error| KernelError::Store(error.to_string()))?;
        self.offset = self.offset.saturating_add(appended.len() as u64);
        self.incomplete_line.extend_from_slice(&appended);

        let Some(last_newline) = self.incomplete_line.iter().rposition(|byte| *byte == b'\n')
        else {
            return Ok(Vec::new());
        };
        let complete = self
            .incomplete_line
            .drain(..=last_newline)
            .collect::<Vec<_>>();
        complete
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
            .map(|line| {
                serde_json::from_slice(line).map_err(|error| KernelError::Store(error.to_string()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Write};

    use focus_kernel::EventKind;

    use super::*;

    #[test]
    fn jsonl_tail_reads_only_complete_new_events() {
        let directory = std::env::temp_dir().join(format!("focus-host-tail-{}", Uuid::new_v4()));
        let path = directory.join("events.jsonl");
        std::fs::create_dir_all(&directory).unwrap();
        let mut tail = JsonlEventTail::open(&path).unwrap();
        let session_id = Uuid::new_v4();
        let event = Event::now(
            session_id,
            EventKind::Runtime {
                name: "tail_fixture".into(),
                data: serde_json::json!({}),
            },
        );
        let encoded = serde_json::to_vec(&event).unwrap();
        let split = encoded.len() / 2;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(&encoded[..split]).unwrap();
        file.flush().unwrap();

        assert!(tail.read_new().unwrap().is_empty());

        file.write_all(&encoded[split..]).unwrap();
        file.write_all(b"\n").unwrap();
        file.flush().unwrap();
        let received = tail.read_new().unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].id, event.id);
        assert!(tail.read_new().unwrap().is_empty());

        drop(file);
        let _ = std::fs::remove_dir_all(directory);
    }
}
