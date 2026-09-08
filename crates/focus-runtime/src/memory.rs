//! Durable project and session memory with atomic JSON persistence.

use std::{
    io,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    RuntimeError,
    context::{ContextItem, ContextKind},
    session::now_ms,
};

/// Lifetime of an explicitly stored memory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    /// Survives across sessions in the current repository.
    Project,
    /// Visible only to the owning session and its descendants.
    Session,
}

/// A concise, attributable retained fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique entry identifier.
    pub id: Uuid,
    /// Retention scope.
    pub scope: MemoryScope,
    /// Content surfaced to future context assembly.
    pub content: String,
    /// Why the content is trusted or useful.
    pub source: String,
    /// Unix epoch milliseconds.
    pub created_at_ms: u128,
}

/// Canonical memory repository for one Runtime workspace.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    root: PathBuf,
}

/// Exclusive short-lived guard for one JSON record across Runtime processes.
#[derive(Debug)]
pub(crate) struct RecordLock {
    _file: std::fs::File,
}

/// Acquire a per-record lock before a read-modify-write transaction.
pub(crate) fn acquire_record_lock(path: &Path) -> Result<RecordLock, RuntimeError> {
    const WAIT_TIMEOUT: Duration = Duration::from_secs(5);
    let parent = path
        .parent()
        .ok_or_else(|| RuntimeError::Memory("storage path has no parent".into()))?;
    let name = path
        .file_name()
        .ok_or_else(|| RuntimeError::Memory("storage path has no file name".into()))?
        .to_string_lossy();
    let lock_path = parent.join(format!(".{name}.lock"));
    acquire_lock_file(&lock_path, WAIT_TIMEOUT).map_err(io_error)
}

/// Acquire a handle-bound advisory lock at an explicit path.
///
/// The lock file is retained after release; ownership ends when the file handle is dropped, so a
/// crashed process releases its lock without any PID probing or path deletion.
pub(crate) fn acquire_lock_file(path: &Path, timeout: Duration) -> io::Result<RecordLock> {
    const POLL: Duration = Duration::from_millis(10);

    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "lock path has no parent"))?;
    std::fs::create_dir_all(parent)?;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let started = Instant::now();
    loop {
        if file.try_lock_exclusive()? {
            return Ok(RecordLock { _file: file });
        }
        if started.elapsed() >= timeout {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("durable record lock is active: {}", path.display()),
            ));
        }
        thread::sleep(POLL);
    }
}

impl MemoryStore {
    /// Open or create memory storage below the Runtime data directory.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, RuntimeError> {
        let root = root.into();
        std::fs::create_dir_all(root.join("sessions")).map_err(io_error)?;
        Ok(Self { root })
    }

    /// Store one fact. Large raw logs belong in event persistence, not memory.
    pub fn remember(
        &self,
        scope: MemoryScope,
        session_id: Option<Uuid>,
        content: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<MemoryEntry, RuntimeError> {
        let session_id = match scope {
            MemoryScope::Project => None,
            MemoryScope::Session => Some(session_id.ok_or_else(|| {
                RuntimeError::Memory("session memory requires a session id".into())
            })?),
        };
        let path = self.path(scope, session_id)?;
        let _lock = acquire_record_lock(&path)?;
        let mut entries = self.load_scope(scope, session_id)?;
        let entry = MemoryEntry {
            id: Uuid::new_v4(),
            scope,
            content: content.into(),
            source: source.into(),
            created_at_ms: now_ms(),
        };
        entries.push(entry.clone());
        self.write_scope(scope, session_id, &entries)?;
        Ok(entry)
    }

    /// Load project memory plus the requested session's memory.
    pub fn relevant(&self, session_id: Uuid) -> Result<Vec<MemoryEntry>, RuntimeError> {
        let mut entries = self.load_scope(MemoryScope::Project, None)?;
        entries.extend(self.load_scope(MemoryScope::Session, Some(session_id))?);
        Ok(entries)
    }

    /// List entries from one explicit scope without widening session visibility.
    pub fn list(
        &self,
        scope: MemoryScope,
        session_id: Option<Uuid>,
    ) -> Result<Vec<MemoryEntry>, RuntimeError> {
        self.load_scope(scope, session_id)
    }

    /// Transform retained facts into low-priority context items.
    pub fn context_items(&self, session_id: Uuid) -> Result<Vec<ContextItem>, RuntimeError> {
        let mut items = Vec::new();
        items.extend(
            self.load_scope(MemoryScope::Project, None)?
                .into_iter()
                .map(memory_entry_to_context_item),
        );
        items.extend(
            self.load_scope(MemoryScope::Session, Some(session_id))?
                .into_iter()
                .map(memory_entry_to_context_item),
        );
        Ok(items)
    }

    fn load_scope(
        &self,
        scope: MemoryScope,
        session_id: Option<Uuid>,
    ) -> Result<Vec<MemoryEntry>, RuntimeError> {
        let path = self.path(scope, session_id)?;
        if !path.exists() {
            return Ok(Vec::new());
        }
        let contents = std::fs::read_to_string(path).map_err(io_error)?;
        serde_json::from_str(&contents).map_err(|error| RuntimeError::Memory(error.to_string()))
    }

    fn write_scope(
        &self,
        scope: MemoryScope,
        session_id: Option<Uuid>,
        entries: &[MemoryEntry],
    ) -> Result<(), RuntimeError> {
        let path = self.path(scope, session_id)?;
        atomic_json_write(&path, entries)
    }

    fn path(&self, scope: MemoryScope, session_id: Option<Uuid>) -> Result<PathBuf, RuntimeError> {
        match scope {
            MemoryScope::Project => Ok(self.root.join("project.json")),
            MemoryScope::Session => session_id
                .map(|id| self.root.join("sessions").join(format!("{id}.json")))
                .ok_or_else(|| RuntimeError::Memory("session memory requires a session id".into())),
        }
    }
}

fn memory_entry_to_context_item(entry: MemoryEntry) -> ContextItem {
    ContextItem {
        id: format!("memory:{}", entry.id),
        kind: ContextKind::Memory,
        priority: if entry.scope == MemoryScope::Project {
            500
        } else {
            600
        },
        source: entry.source,
        content: entry.content,
    }
}

pub(crate) fn atomic_json_write(
    path: &Path,
    value: &(impl Serialize + ?Sized),
) -> Result<(), RuntimeError> {
    let encoded = serde_json::to_vec_pretty(value)
        .map_err(|error| RuntimeError::Memory(error.to_string()))?;
    let parent = path
        .parent()
        .ok_or_else(|| RuntimeError::Memory("storage path has no parent".into()))?;
    std::fs::create_dir_all(parent).map_err(io_error)?;
    let temporary = parent.join(format!(".{}.tmp", Uuid::new_v4()));
    let mut file = std::fs::File::create(&temporary).map_err(io_error)?;
    std::io::Write::write_all(&mut file, &encoded).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    std::fs::rename(temporary, path).map_err(io_error)
}

fn io_error(error: std::io::Error) -> RuntimeError {
    RuntimeError::Memory(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_keeps_project_and_session_memory_separate() {
        let root = std::env::temp_dir().join(format!("focus-memory-list-{}", Uuid::new_v4()));
        let store = MemoryStore::new(&root).unwrap();
        let session_id = Uuid::new_v4();
        store
            .remember(MemoryScope::Project, None, "project fact", "test")
            .unwrap();
        store
            .remember(
                MemoryScope::Session,
                Some(session_id),
                "session fact",
                "test",
            )
            .unwrap();

        let project = store.list(MemoryScope::Project, None).unwrap();
        let session = store.list(MemoryScope::Session, Some(session_id)).unwrap();

        assert_eq!(project.len(), 1);
        assert_eq!(project[0].content, "project fact");
        assert_eq!(session.len(), 1);
        assert_eq!(session[0].content, "session fact");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn atomic_json_write_replaces_an_existing_record() {
        let root = std::env::temp_dir().join(format!("memory-replace-{}", Uuid::new_v4()));
        let path = root.join("record.json");

        atomic_json_write(&path, &serde_json::json!({"value": "first"})).unwrap();
        atomic_json_write(&path, &serde_json::json!({"value": "second"})).unwrap();

        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["value"], "second");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn concurrent_process_like_writes_preserve_both_memory_entries() {
        let root = std::env::temp_dir().join(format!("focus-memory-lock-{}", Uuid::new_v4()));
        let first = MemoryStore::new(&root).unwrap();
        let second = MemoryStore::new(&root).unwrap();
        let left = std::thread::spawn(move || {
            first
                .remember(MemoryScope::Project, None, "left", "test")
                .unwrap();
        });
        let right = std::thread::spawn(move || {
            second
                .remember(MemoryScope::Project, None, "right", "test")
                .unwrap();
        });
        left.join().unwrap();
        right.join().unwrap();
        let entries = MemoryStore::new(&root)
            .unwrap()
            .list(MemoryScope::Project, None)
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().any(|entry| entry.content == "left"));
        assert!(entries.iter().any(|entry| entry.content == "right"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn remembers_when_a_stale_lock_path_remains() {
        let root = std::env::temp_dir().join(format!("focus-memory-stale-lock-{}", Uuid::new_v4()));
        let store = MemoryStore::new(&root).unwrap();
        let lock = root.join(".project.json.lock");
        std::fs::write(&lock, "stale path contents").unwrap();

        store
            .remember(MemoryScope::Project, None, "recovered", "test")
            .unwrap();

        assert_eq!(store.list(MemoryScope::Project, None).unwrap().len(), 1);
        assert!(lock.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn lock_file_survives_handle_release_and_is_reacquired_by_a_new_handle() {
        let root =
            std::env::temp_dir().join(format!("focus-memory-handle-lock-{}", Uuid::new_v4()));
        let path = root.join("record.lock");
        let first = acquire_lock_file(&path, Duration::ZERO).unwrap();

        let active = acquire_lock_file(&path, Duration::ZERO).unwrap_err();
        assert_eq!(active.kind(), io::ErrorKind::AlreadyExists);

        drop(first);
        assert!(
            path.exists(),
            "the lock path must not encode lock ownership"
        );

        let second = acquire_lock_file(&path, Duration::ZERO).unwrap();
        drop(second);
        let _ = std::fs::remove_dir_all(root);
    }
}
