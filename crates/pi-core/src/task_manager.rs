use crate::agent_types::{Subtask, TaskStatus};
use chrono::Utc;
use uuid::Uuid;

/// Manages the hierarchy of subtasks.
#[derive(Debug, Clone, Default)]
pub struct TaskManager {
    subtasks: Vec<Subtask>,
    active_subtask: Option<Uuid>,
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            subtasks: Vec::new(),
            active_subtask: None,
        }
    }

    /// Add a new subtask to the tree.
    pub fn add_subtask(
        &mut self,
        title: String,
        description: Option<String>,
        parent_id: Option<Uuid>,
    ) -> Uuid {
        let id = Uuid::new_v4();
        let subtask = Subtask {
            id,
            parent_id,
            title,
            description,
            status: TaskStatus::Pending,
            result: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        self.subtasks.push(subtask);
        id
    }

    /// Update the status of a subtask.
    pub fn update_status(&mut self, id: Uuid, status: TaskStatus) -> bool {
        if let Some(task) = self.subtasks.iter_mut().find(|t| t.id == id) {
            task.status = status;
            task.updated_at = Utc::now();

            // If completed or failed, and it was active, clear active
            if matches!(task.status, TaskStatus::Completed | TaskStatus::Failed)
                && self.active_subtask == Some(id)
            {
                self.active_subtask = None;
            }

            // If set to running, make it active
            if matches!(task.status, TaskStatus::Running) {
                self.active_subtask = Some(id);
            }

            true
        } else {
            false
        }
    }

    /// Get the current tree snapshot.
    pub fn get_tree(&self) -> Vec<Subtask> {
        self.subtasks.clone()
    }

    pub fn get_active_subtask(&self) -> Option<Uuid> {
        self.active_subtask
    }
}
