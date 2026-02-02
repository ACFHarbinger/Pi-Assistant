//! Transaction management for undoable agent actions.

use anyhow::Result;
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::fs;
use tracing::{info, warn};

/// Trait for an action that can be rolled back.
#[async_trait]
pub trait UndoableAction: Send + Sync {
    /// Human-readable description of the action.
    fn description(&self) -> &str;

    /// Roll back the action.
    async fn rollback(&self) -> Result<()>;
}

/// Action representing a file change (write or patch).
pub struct FileChange {
    pub path: PathBuf,
    pub original_content: Option<String>,
    pub is_new: bool,
}

#[async_trait]
impl UndoableAction for FileChange {
    fn description(&self) -> &str {
        if self.is_new {
            "Create file"
        } else {
            "Modify file"
        }
    }

    async fn rollback(&self) -> Result<()> {
        if self.is_new {
            info!(path = %self.path.display(), "Rolling back file creation (deleting)");
            if self.path.exists() {
                fs::remove_file(&self.path).await?;
            }
        } else if let Some(content) = &self.original_content {
            info!(path = %self.path.display(), "Rolling back file modification (restoring content)");
            fs::write(&self.path, content).await?;
        }
        Ok(())
    }
}

/// Manages a stack of undoable actions (transactions).
pub struct TransactionManager {
    transactions: Vec<Vec<Box<dyn UndoableAction>>>,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            transactions: Vec::new(),
        }
    }

    /// Start a new transaction (usually per planning iteration).
    pub fn start_transaction(&mut self) {
        self.transactions.push(Vec::new());
    }

    /// Push an action to the current transaction.
    pub fn push_action(&mut self, action: Box<dyn UndoableAction>) {
        if let Some(current) = self.transactions.last_mut() {
            current.push(action);
        } else {
            warn!("No active transaction to push action to");
        }
    }

    /// Roll back the most recent transaction.
    pub async fn rollback_last(&mut self) -> Result<bool> {
        if let Some(actions) = self.transactions.pop() {
            info!(count = actions.len(), "Rolling back last transaction");
            // Roll back in reverse order
            for action in actions.into_iter().rev() {
                action.rollback().await?;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Roll back all transactions and clear the stack.
    pub async fn rollback_all(&mut self) -> Result<()> {
        info!("Rolling back all transactions");
        while self.rollback_last().await? {}
        Ok(())
    }

    /// Commit the current stack (clear it without rolling back).
    pub fn commit_all(&mut self) {
        self.transactions.clear();
    }

    /// Get the count of transactions.
    pub fn len(&self) -> usize {
        self.transactions.len()
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}
