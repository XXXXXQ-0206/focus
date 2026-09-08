//! Thin durable goal metadata linked to Runtime sessions.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    RuntimeError,
    memory::{acquire_record_lock, atomic_json_write},
    session::now_ms,
};

/// Explicit lifecycle selected by the operator rather than inferred from a turn result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalPhase {
    /// The goal can receive new work in its active session.
    Active,
    /// The operator marked the objective complete.
    Complete,
    /// The operator stopped work before completion.
    Cancelled,
}

/// Metadata binding an objective to the sessions that pursue it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Goal {
    /// Stable goal identifier.
    pub id: Uuid,
    /// Short operator-facing name.
    pub title: String,
    /// Durable statement of the intended outcome.
    pub objective: String,
    /// Operator-owned lifecycle state.
    pub phase: GoalPhase,
    /// First session attached to this objective.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_session_id: Option<Uuid>,
    /// Session currently receiving work for this objective.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_session_id: Option<Uuid>,
    /// Unix epoch milliseconds.
    pub created_at_ms: u128,
    /// Unix epoch milliseconds.
    pub updated_at_ms: u128,
}

/// Runtime-owned persistence for goal metadata only.
#[derive(Debug, Clone)]
pub struct GoalStore {
    root: PathBuf,
}

impl GoalStore {
    /// Open or create the Runtime goal metadata directory.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, RuntimeError> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(io_error)?;
        Ok(Self { root })
    }

    /// Create an unbound active goal.
    pub fn create(
        &self,
        title: impl Into<String>,
        objective: impl Into<String>,
    ) -> Result<Goal, RuntimeError> {
        let now = now_ms();
        let goal = Goal {
            id: Uuid::new_v4(),
            title: title.into(),
            objective: objective.into(),
            phase: GoalPhase::Active,
            root_session_id: None,
            active_session_id: None,
            created_at_ms: now,
            updated_at_ms: now,
        };
        self.save(&goal)?;
        Ok(goal)
    }

    /// Load one goal by identity.
    pub fn load(&self, id: Uuid) -> Result<Goal, RuntimeError> {
        let text = std::fs::read_to_string(self.path(id)).map_err(io_error)?;
        serde_json::from_str(&text).map_err(|error| RuntimeError::Session(error.to_string()))
    }

    /// List goals in most-recently-updated order.
    pub fn list(&self) -> Result<Vec<Goal>, RuntimeError> {
        let mut goals = std::fs::read_dir(&self.root)
            .map_err(io_error)?
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "json")
            })
            .map(|entry| {
                let text = std::fs::read_to_string(entry.path()).map_err(io_error)?;
                serde_json::from_str::<Goal>(&text)
                    .map_err(|error| RuntimeError::Session(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        goals.sort_by_key(|goal| std::cmp::Reverse(goal.updated_at_ms));
        Ok(goals)
    }

    /// Attach an existing session without duplicating its events or transcript.
    pub fn attach_session(&self, id: Uuid, session_id: Uuid) -> Result<Goal, RuntimeError> {
        self.update(id, |goal| {
            if goal.phase != GoalPhase::Active {
                return Err(RuntimeError::Session(format!(
                    "goal {id} is not active and cannot receive a session"
                )));
            }
            if goal.root_session_id.is_none() {
                goal.root_session_id = Some(session_id);
            }
            goal.active_session_id = Some(session_id);
            Ok(())
        })
    }

    /// Apply an operator-selected terminal goal phase.
    pub fn set_phase(&self, id: Uuid, phase: GoalPhase) -> Result<Goal, RuntimeError> {
        self.update(id, |goal| {
            goal.phase = phase;
            Ok(())
        })
    }

    fn update(
        &self,
        id: Uuid,
        update: impl FnOnce(&mut Goal) -> Result<(), RuntimeError>,
    ) -> Result<Goal, RuntimeError> {
        let _lock = acquire_record_lock(&self.path(id))
            .map_err(|error| RuntimeError::Session(error.to_string()))?;
        let mut goal = self.load(id)?;
        update(&mut goal)?;
        goal.updated_at_ms = now_ms();
        self.save(&goal)?;
        Ok(goal)
    }

    fn save(&self, goal: &Goal) -> Result<(), RuntimeError> {
        atomic_json_write(&self.path(goal.id), goal)
            .map_err(|error| RuntimeError::Session(error.to_string()))
    }

    fn path(&self, id: Uuid) -> PathBuf {
        self.root.join(format!("{id}.json"))
    }
}

fn io_error(error: std::io::Error) -> RuntimeError {
    RuntimeError::Session(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_tracks_session_without_owning_execution_events() {
        let root = std::env::temp_dir().join(format!("focus-goal-store-{}", Uuid::new_v4()));
        let store = GoalStore::new(&root).unwrap();
        let created = store.create("Release", "publish verified build").unwrap();
        let session_id = Uuid::new_v4();

        let attached = store.attach_session(created.id, session_id).unwrap();

        assert_eq!(attached.root_session_id, Some(session_id));
        assert_eq!(attached.active_session_id, Some(session_id));
        assert_eq!(store.list().unwrap(), vec![attached]);
        let _ = std::fs::remove_dir_all(root);
    }
}
