//! Persist-first Runtime event fan-out shared by CLI, TUI, subagents, and IDE hosts.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use focus_kernel::{Event, EventKind, EventStore, KernelError, ModelEvent};
use uuid::Uuid;

const DEFAULT_SUBSCRIBER_CAPACITY: usize = 256;
const MODEL_DELTA_FLUSH_AFTER: Duration = Duration::from_millis(50);
const MODEL_DELTA_MAX_BYTES: usize = 4 * 1024;

#[derive(Debug, Default)]
struct EventHubCounters {
    overflow_disconnects: AtomicUsize,
    disconnected_subscribers: AtomicUsize,
    flush_worker_starts: AtomicUsize,
}

#[derive(Debug, Default)]
struct EventHubHealth {
    persistence_error: Mutex<Option<String>>,
    persistence_failures: AtomicUsize,
    last_successful_flush_ms: Mutex<Option<u128>>,
}

#[derive(Debug)]
struct PendingModelDelta {
    event: Event,
    started_at: Instant,
    bytes: usize,
}

/// Snapshot of non-blocking event fan-out health.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EventHubStats {
    /// Subscribers disconnected because their bounded queue was full.
    pub overflow_disconnects: usize,
    /// Subscribers removed after their receiver was dropped.
    pub disconnected_subscribers: usize,
    /// Dedicated model-delta flush workers created for this hub.
    pub flush_worker_starts: usize,
    /// Number of persistence failures observed by this hub.
    pub persistence_failures: usize,
    /// Number of model-delta batches waiting to be committed.
    pub pending_model_deltas: usize,
    /// Unix epoch milliseconds for the latest successful event commit.
    pub last_successful_flush_ms: Option<u128>,
    /// First persistence error observed by this hub, if any.
    pub persistence_error: Option<String>,
}

/// Canonical event store decorator which broadcasts only committed events.
pub struct EventHub {
    store: Arc<dyn EventStore>,
    pending_model_deltas: Arc<Mutex<HashMap<Uuid, PendingModelDelta>>>,
    scheduled_model_delta_flushes: Arc<Mutex<HashSet<Uuid>>>,
    persistence: Arc<Mutex<()>>,
    subscribers: Arc<Mutex<Vec<mpsc::SyncSender<Event>>>>,
    counters: Arc<EventHubCounters>,
    health: Arc<EventHubHealth>,
    flush_tx: Option<mpsc::Sender<Uuid>>,
}

impl std::fmt::Debug for EventHub {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("EventHub").finish_non_exhaustive()
    }
}

impl EventHub {
    /// Wrap the canonical persistent store.
    #[must_use]
    pub fn new(store: Arc<dyn EventStore>) -> Self {
        let pending_model_deltas = Arc::new(Mutex::new(HashMap::new()));
        let scheduled_model_delta_flushes = Arc::new(Mutex::new(HashSet::new()));
        let persistence = Arc::new(Mutex::new(()));
        let subscribers = Arc::new(Mutex::new(Vec::new()));
        let counters = Arc::new(EventHubCounters::default());
        let health = Arc::new(EventHubHealth::default());
        let (flush_tx, flush_rx) = mpsc::channel();
        let worker = Self {
            store: store.clone(),
            pending_model_deltas: pending_model_deltas.clone(),
            scheduled_model_delta_flushes: scheduled_model_delta_flushes.clone(),
            persistence: persistence.clone(),
            subscribers: subscribers.clone(),
            counters: counters.clone(),
            health: health.clone(),
            flush_tx: None,
        };
        counters.flush_worker_starts.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(move || worker.run_flush_worker(flush_rx));
        Self {
            store,
            pending_model_deltas,
            scheduled_model_delta_flushes,
            persistence,
            subscribers,
            counters,
            health,
            flush_tx: Some(flush_tx),
        }
    }

    /// Subscribe to future committed events.
    #[must_use]
    pub fn subscribe(&self) -> mpsc::Receiver<Event> {
        self.subscribe_with_capacity(DEFAULT_SUBSCRIBER_CAPACITY)
    }

    /// Subscribe with an explicit bounded queue capacity.
    #[must_use]
    pub fn subscribe_with_capacity(&self, capacity: usize) -> mpsc::Receiver<Event> {
        let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
        if let Ok(mut subscribers) = self.subscribers.lock() {
            subscribers.push(sender);
        }
        receiver
    }

    /// Return fan-out health counters without touching persisted event state.
    #[must_use]
    pub fn stats(&self) -> EventHubStats {
        let (last_successful_flush_ms, persistence_error) = self
            .health
            .persistence_error
            .lock()
            .ok()
            .map_or((None, None), |error| {
                let last_successful_flush_ms = self
                    .health
                    .last_successful_flush_ms
                    .lock()
                    .ok()
                    .and_then(|value| *value);
                (last_successful_flush_ms, error.clone())
            });
        EventHubStats {
            overflow_disconnects: self.counters.overflow_disconnects.load(Ordering::Relaxed),
            disconnected_subscribers: self
                .counters
                .disconnected_subscribers
                .load(Ordering::Relaxed),
            flush_worker_starts: self.counters.flush_worker_starts.load(Ordering::Relaxed),
            persistence_failures: self.health.persistence_failures.load(Ordering::Relaxed),
            pending_model_deltas: self
                .pending_model_deltas
                .lock()
                .map_or(0, |pending| pending.len()),
            last_successful_flush_ms,
            persistence_error,
        }
    }

    /// Return the number of currently connected subscribers.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.subscribers
            .lock()
            .map_or(0, |subscribers| subscribers.len())
    }

    fn queue_model_delta(&self, event: Event) -> Result<bool, KernelError> {
        let delta_bytes = model_delta_bytes(&event).expect("only model deltas are queued");
        let mut pending = self
            .pending_model_deltas
            .lock()
            .map_err(|_| KernelError::Store("pending model delta lock was poisoned".into()))?;

        if let Some(current) = pending.get_mut(&event.session_id) {
            let is_due = current.started_at.elapsed() >= MODEL_DELTA_FLUSH_AFTER;
            let would_exceed_limit =
                current.bytes.saturating_add(delta_bytes) > MODEL_DELTA_MAX_BYTES;
            if !is_due && !would_exceed_limit && merge_model_delta(&mut current.event, &event) {
                current.bytes += delta_bytes;
                return Ok(false);
            }
            drop(pending);
            self.flush_pending(event.session_id)?;
            return self.queue_model_delta(event);
        }

        if delta_bytes >= MODEL_DELTA_MAX_BYTES {
            drop(pending);
            self.commit_or_record(&event)?;
            return Ok(false);
        } else {
            pending.insert(
                event.session_id,
                PendingModelDelta {
                    event,
                    started_at: Instant::now(),
                    bytes: delta_bytes,
                },
            );
        }
        Ok(true)
    }

    fn pending_event(&self, session_id: Uuid) -> Result<Option<Event>, KernelError> {
        self.pending_model_deltas
            .lock()
            .map_err(|_| KernelError::Store("pending model delta lock was poisoned".into()))
            .map(|pending| {
                pending
                    .get(&session_id)
                    .map(|pending| pending.event.clone())
            })
    }

    fn remove_pending_if_matches(
        &self,
        session_id: Uuid,
        event_id: Uuid,
    ) -> Result<(), KernelError> {
        let mut pending = self
            .pending_model_deltas
            .lock()
            .map_err(|_| KernelError::Store("pending model delta lock was poisoned".into()))?;
        if pending
            .get(&session_id)
            .is_some_and(|pending| pending.event.id == event_id)
        {
            pending.remove(&session_id);
        }
        Ok(())
    }

    fn pending_session_ids(&self) -> Result<Vec<Uuid>, KernelError> {
        self.pending_model_deltas
            .lock()
            .map_err(|_| KernelError::Store("pending model delta lock was poisoned".into()))
            .map(|pending| pending.keys().copied().collect())
    }

    fn flush_pending(&self, session_id: Uuid) -> Result<(), KernelError> {
        let pending = match self.pending_event(session_id) {
            Ok(pending) => pending,
            Err(error) => {
                self.record_persistence_failure(&error);
                return Err(error);
            }
        };
        let Some(event) = pending else {
            return Ok(());
        };
        self.commit_or_record(&event)?;
        if let Err(error) = self.remove_pending_if_matches(session_id, event.id) {
            self.record_persistence_failure(&error);
            return Err(error);
        }
        Ok(())
    }

    fn ensure_healthy(&self) -> Result<(), KernelError> {
        let error = self
            .health
            .persistence_error
            .lock()
            .map_err(|_| KernelError::Store("event hub health lock was poisoned".into()))?;
        match error.as_deref() {
            Some(error) => Err(KernelError::Store(format!(
                "event persistence previously failed: {error}"
            ))),
            None => Ok(()),
        }
    }

    fn record_persistence_failure(&self, error: &KernelError) {
        self.health
            .persistence_failures
            .fetch_add(1, Ordering::Relaxed);
        if let Ok(mut recorded) = self.health.persistence_error.lock()
            && recorded.is_none()
        {
            *recorded = Some(error.to_string());
        }
    }

    fn record_successful_flush(&self) {
        if let Ok(mut timestamp) = self.health.last_successful_flush_ms.lock() {
            *timestamp = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis(),
            );
        }
    }

    fn commit_or_record(&self, event: &Event) -> Result<(), KernelError> {
        match self.commit_locked(event) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.record_persistence_failure(&error);
                Err(error)
            }
        }
    }

    fn schedule_model_delta_flush(&self, session_id: Uuid) {
        let Ok(mut scheduled) = self.scheduled_model_delta_flushes.lock() else {
            self.record_persistence_failure(&KernelError::Store(
                "model delta schedule lock was poisoned".into(),
            ));
            return;
        };
        if scheduled.insert(session_id)
            && let Some(flush_tx) = &self.flush_tx
            && flush_tx.send(session_id).is_err()
        {
            self.record_persistence_failure(&KernelError::Store(
                "model delta flush worker stopped".into(),
            ));
        }
    }

    fn run_flush_worker(self, receiver: mpsc::Receiver<Uuid>) {
        let mut deadlines: HashMap<Uuid, Instant> = HashMap::new();
        loop {
            let now = Instant::now();
            let next = deadlines
                .values()
                .copied()
                .min()
                .map(|deadline| deadline.saturating_duration_since(now));
            let received = match next {
                Some(wait) => receiver.recv_timeout(wait),
                None => match receiver.recv() {
                    Ok(session_id) => Ok(session_id),
                    Err(_) => break,
                },
            };
            match received {
                Ok(session_id) => {
                    deadlines
                        .entry(session_id)
                        .or_insert_with(|| Instant::now() + MODEL_DELTA_FLUSH_AFTER);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let now = Instant::now();
                    let due = deadlines
                        .iter()
                        .filter_map(|(session_id, deadline)| {
                            (*deadline <= now).then_some(*session_id)
                        })
                        .collect::<Vec<_>>();
                    for session_id in due {
                        deadlines.remove(&session_id);
                        self.flush_scheduled_session(session_id);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        match self.persistence.lock() {
            Ok(_persistence) => match self.pending_session_ids() {
                Ok(session_ids) => {
                    for session_id in session_ids {
                        if self.flush_pending(session_id).is_err() {
                            break;
                        }
                    }
                }
                Err(error) => self.record_persistence_failure(&error),
            },
            Err(_) => self.record_persistence_failure(&KernelError::Store(
                "event persistence lock was poisoned".into(),
            )),
        }
    }

    fn flush_scheduled_session(&self, session_id: Uuid) {
        if let Ok(mut scheduled) = self.scheduled_model_delta_flushes.lock() {
            scheduled.remove(&session_id);
        }
        match self.persistence.lock() {
            Ok(_persistence) => {
                let _ = self.flush_pending(session_id);
            }
            Err(_) => self.record_persistence_failure(&KernelError::Store(
                "event persistence lock was poisoned".into(),
            )),
        }
    }

    fn commit_locked(&self, event: &Event) -> Result<(), KernelError> {
        self.store.append(event)?;
        self.record_successful_flush();
        self.broadcast(event)
    }

    fn broadcast(&self, event: &Event) -> Result<(), KernelError> {
        let mut subscribers = self
            .subscribers
            .lock()
            .map_err(|_| KernelError::Store("event subscriber lock was poisoned".into()))?;
        subscribers.retain(|subscriber| match subscriber.try_send(event.clone()) {
            Ok(()) => true,
            Err(mpsc::TrySendError::Full(_)) => {
                self.counters
                    .overflow_disconnects
                    .fetch_add(1, Ordering::Relaxed);
                false
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.counters
                    .disconnected_subscribers
                    .fetch_add(1, Ordering::Relaxed);
                false
            }
        });
        Ok(())
    }
}

impl EventStore for EventHub {
    fn append(&self, event: &Event) -> Result<(), KernelError> {
        self.ensure_healthy()?;
        if model_delta_bytes(event).is_some() {
            let schedule_flush = {
                let _persistence = self.persistence.lock().map_err(|_| {
                    KernelError::Store("event persistence lock was poisoned".into())
                })?;
                self.ensure_healthy()?;
                self.queue_model_delta(event.clone())?
            };
            if schedule_flush {
                self.schedule_model_delta_flush(event.session_id);
            }
            self.ensure_healthy()?;
            return Ok(());
        }

        let _persistence = self
            .persistence
            .lock()
            .map_err(|_| KernelError::Store("event persistence lock was poisoned".into()))?;
        self.ensure_healthy()?;
        self.flush_pending(event.session_id)?;
        self.commit_or_record(event)
    }

    fn load(&self, session_id: Uuid) -> Result<Vec<Event>, KernelError> {
        self.ensure_healthy()?;
        let _persistence = self
            .persistence
            .lock()
            .map_err(|_| KernelError::Store("event persistence lock was poisoned".into()))?;
        self.ensure_healthy()?;
        self.flush_pending(session_id)?;
        self.ensure_healthy()?;
        self.store.load(session_id)
    }
}

fn model_delta_bytes(event: &Event) -> Option<usize> {
    match &event.kind {
        EventKind::Model {
            event: ModelEvent::TextDelta { text },
        } => Some(text.len()),
        EventKind::Model {
            event: ModelEvent::ToolArgumentsDelta { json_fragment, .. },
        } => Some(json_fragment.len()),
        _ => None,
    }
}

fn merge_model_delta(existing: &mut Event, incoming: &Event) -> bool {
    match (&mut existing.kind, &incoming.kind) {
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
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use focus_kernel::{EventKind, JsonlEventStore, ModelEvent, ModelResponse};

    use super::*;

    #[derive(Debug, Default)]
    struct RecordingStore {
        events: Mutex<Vec<Event>>,
    }

    impl EventStore for RecordingStore {
        fn append(&self, event: &Event) -> Result<(), KernelError> {
            self.events
                .lock()
                .map_err(|_| KernelError::Store("recording store lock poisoned".into()))?
                .push(event.clone());
            Ok(())
        }

        fn load(&self, session_id: Uuid) -> Result<Vec<Event>, KernelError> {
            Ok(self
                .events
                .lock()
                .map_err(|_| KernelError::Store("recording store lock poisoned".into()))?
                .iter()
                .filter(|event| event.session_id == session_id)
                .cloned()
                .collect())
        }
    }

    #[derive(Debug)]
    struct FailOnceStore {
        failures_remaining: AtomicUsize,
        append_attempts: AtomicUsize,
        events: Mutex<Vec<Event>>,
    }

    impl EventStore for FailOnceStore {
        fn append(&self, event: &Event) -> Result<(), KernelError> {
            self.append_attempts.fetch_add(1, Ordering::Relaxed);
            if self
                .failures_remaining
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                    if remaining > 0 {
                        Some(remaining - 1)
                    } else {
                        None
                    }
                })
                .is_ok()
            {
                return Err(KernelError::Store("injected persistence failure".into()));
            }
            self.events
                .lock()
                .map_err(|_| KernelError::Store("failing store lock poisoned".into()))?
                .push(event.clone());
            Ok(())
        }

        fn load(&self, session_id: Uuid) -> Result<Vec<Event>, KernelError> {
            Ok(self
                .events
                .lock()
                .map_err(|_| KernelError::Store("failing store lock poisoned".into()))?
                .iter()
                .filter(|event| event.session_id == session_id)
                .cloned()
                .collect())
        }
    }

    #[test]
    fn concurrent_commits_broadcast_in_persisted_order() {
        let store = Arc::new(RecordingStore::default());
        let hub = Arc::new(EventHub::new(store.clone()));
        let receiver = hub.subscribe();
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

        let subscribers = hub.subscribers.lock().unwrap();
        let first_thread = {
            let hub = hub.clone();
            let first = first.clone();
            std::thread::spawn(move || hub.append(&first).unwrap())
        };
        let deadline = std::time::Instant::now() + Duration::from_millis(250);
        while store.events.lock().unwrap().is_empty() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        let second_thread = {
            let hub = hub.clone();
            let second = second.clone();
            std::thread::spawn(move || hub.append(&second).unwrap())
        };
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(
            store.events.lock().unwrap().len(),
            1,
            "a later append must not persist before the earlier commit is broadcast"
        );
        drop(subscribers);
        first_thread.join().unwrap();
        second_thread.join().unwrap();

        let live = [
            receiver.recv_timeout(Duration::from_millis(50)).unwrap(),
            receiver.recv_timeout(Duration::from_millis(50)).unwrap(),
        ];
        let persisted = store.load(session_id).unwrap();
        assert_eq!(
            live.iter().map(|event| event.id).collect::<Vec<_>>(),
            persisted.iter().map(|event| event.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn slow_subscriber_is_bounded_and_disconnected_on_overflow() {
        let directory = std::env::temp_dir().join(format!("event-hub-{}", Uuid::new_v4()));
        let store = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let hub = EventHub::new(store);
        let receiver = hub.subscribe_with_capacity(1);
        let first = Event::now(
            Uuid::new_v4(),
            EventKind::Runtime {
                name: "first".into(),
                data: serde_json::json!({}),
            },
        );
        let second = Event::now(
            first.session_id,
            EventKind::Runtime {
                name: "second".into(),
                data: serde_json::json!({}),
            },
        );

        hub.append(&first).unwrap();
        hub.append(&second).unwrap();

        assert_eq!(
            receiver.recv_timeout(Duration::from_millis(50)).unwrap().id,
            first.id
        );
        assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
        assert_eq!(hub.stats().overflow_disconnects, 1);
        assert_eq!(hub.subscriber_count(), 0);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn model_deltas_have_one_committed_live_and_replay_boundary() {
        let directory = std::env::temp_dir().join(format!("event-hub-model-{}", Uuid::new_v4()));
        let store = Arc::new(JsonlEventStore::new(directory.join("events")).unwrap());
        let hub = EventHub::new(store.clone());
        let receiver = hub.subscribe();
        let session_id = Uuid::new_v4();
        let input = vec![
            Event::now(
                session_id,
                EventKind::Model {
                    event: ModelEvent::TextDelta { text: "hel".into() },
                },
            ),
            Event::now(
                session_id,
                EventKind::Model {
                    event: ModelEvent::TextDelta { text: "lo ".into() },
                },
            ),
            Event::now(
                session_id,
                EventKind::Model {
                    event: ModelEvent::TextDelta {
                        text: "world".into(),
                    },
                },
            ),
            Event::now(
                session_id,
                EventKind::Model {
                    event: ModelEvent::ToolArgumentsDelta {
                        id: "call-1".into(),
                        json_fragment: "{\"query\":".into(),
                    },
                },
            ),
            Event::now(
                session_id,
                EventKind::Model {
                    event: ModelEvent::ToolArgumentsDelta {
                        id: "call-1".into(),
                        json_fragment: "\"focus\"}".into(),
                    },
                },
            ),
        ];

        for event in &input {
            hub.append(event).unwrap();
        }
        let terminal = Event::now(
            session_id,
            EventKind::Model {
                event: ModelEvent::Completed {
                    response: ModelResponse {
                        content: "hello world".into(),
                        tool_calls: Vec::new(),
                    },
                },
            },
        );
        hub.append(&terminal).unwrap();

        let persisted = store.load(session_id).unwrap();
        let live = (0..persisted.len())
            .map(|_| receiver.recv_timeout(Duration::from_millis(50)).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            live.iter().map(|event| event.id).collect::<Vec<_>>(),
            persisted.iter().map(|event| event.id).collect::<Vec<_>>()
        );
        let text = persisted
            .iter()
            .filter_map(|event| match &event.kind {
                EventKind::Model {
                    event: ModelEvent::TextDelta { text },
                } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        let arguments = persisted
            .iter()
            .filter_map(|event| match &event.kind {
                EventKind::Model {
                    event: ModelEvent::ToolArgumentsDelta { json_fragment, .. },
                } => Some(json_fragment.as_str()),
                _ => None,
            })
            .collect::<String>();

        assert_eq!(text, "hello world");
        assert_eq!(arguments, "{\"query\":\"focus\"}");
        assert!(
            persisted.len() < input.len() + 1,
            "durable JSONL must coalesce model fragments"
        );
        assert!(matches!(
            persisted.last().map(|event| &event.kind),
            Some(EventKind::Model {
                event: ModelEvent::Completed { .. }
            })
        ));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn alternating_model_deltas_schedule_one_flush_worker_per_session() {
        let hub = EventHub::new(Arc::new(RecordingStore::default()));
        let session_id = Uuid::new_v4();

        for index in 0..10_000 {
            let event = if index % 2 == 0 {
                Event::now(
                    session_id,
                    EventKind::Model {
                        event: ModelEvent::TextDelta { text: "x".into() },
                    },
                )
            } else {
                Event::now(
                    session_id,
                    EventKind::Model {
                        event: ModelEvent::ToolArgumentsDelta {
                            id: "call-1".into(),
                            json_fragment: "x".into(),
                        },
                    },
                )
            };
            hub.append(&event).unwrap();
        }

        assert_eq!(hub.stats().flush_worker_starts, 1);
    }

    #[test]
    fn model_delta_is_durably_flushed_after_the_coalescing_deadline() {
        let directory = std::env::temp_dir().join(format!("event-hub-deadline-{}", Uuid::new_v4()));
        let event_directory = directory.join("events");
        let store = Arc::new(JsonlEventStore::new(&event_directory).unwrap());
        let hub = EventHub::new(store);
        let session_id = Uuid::new_v4();
        hub.append(&Event::now(
            session_id,
            EventKind::Model {
                event: ModelEvent::TextDelta {
                    text: "durable delta".into(),
                },
            },
        ))
        .unwrap();

        let deadline = std::time::Instant::now() + Duration::from_millis(250);
        let event_path = event_directory.join(format!("{session_id}.jsonl"));
        let persisted = loop {
            if let Ok(contents) = std::fs::read_to_string(&event_path)
                && contents.contains("durable delta")
            {
                break true;
            }
            if std::time::Instant::now() >= deadline {
                break false;
            }
            std::thread::sleep(Duration::from_millis(5));
        };

        assert!(
            persisted,
            "pending model delta was not flushed by its deadline"
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn background_persistence_failure_blocks_future_append_and_load() {
        let store = Arc::new(FailOnceStore {
            failures_remaining: AtomicUsize::new(1),
            append_attempts: AtomicUsize::new(0),
            events: Mutex::new(Vec::new()),
        });
        let hub = EventHub::new(store.clone());
        let session_id = Uuid::new_v4();
        hub.append(&Event::now(
            session_id,
            EventKind::Model {
                event: ModelEvent::TextDelta {
                    text: "pending".into(),
                },
            },
        ))
        .unwrap();

        let deadline = Instant::now() + Duration::from_millis(250);
        while store.append_attempts.load(Ordering::Acquire) == 0 {
            assert!(
                Instant::now() < deadline,
                "flush worker did not attempt persistence"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        let terminal = Event::now(
            session_id,
            EventKind::Runtime {
                name: "terminal".into(),
                data: serde_json::json!({}),
            },
        );
        assert!(hub.append(&terminal).is_err());
        assert!(hub.load(session_id).is_err());
        assert_eq!(
            hub.pending_model_deltas.lock().unwrap().len(),
            1,
            "failed background persistence must retain the pending model delta"
        );
    }
}
