//! SQLite storage for sessions, messages, and tool logs.

use crate::tools::{ToolCall, ToolResult};
use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::sync::Mutex;
use tracing::{debug, info};
use uuid::Uuid;

/// Memory manager for persistent storage.
pub struct MemoryManager {
    conn: Mutex<Connection>,
}

impl MemoryManager {
    /// Create a new memory manager with SQLite database.
    pub fn new(db_path: Option<PathBuf>) -> Result<Self> {
        let path = db_path.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".pi-assistant")
                .join("memory.db")
        });

        // Create parent directory
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        info!(path = %path.display(), "Opening SQLite database");
        let conn = Connection::open(&path)?;
        let manager = Self {
            conn: Mutex::new(conn),
        };
        manager.init_schema()?;
        Ok(manager)
    }

    /// Create an in-memory database (for testing).
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let manager = Self {
            conn: Mutex::new(conn),
        };
        manager.init_schema()?;
        Ok(manager)
    }

    /// Initialize database schema.
    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute_batch(
            r#"
            -- Sessions
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                title TEXT
            );

            -- Messages
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- Tasks
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- Tool executions
            CREATE TABLE IF NOT EXISTS tool_executions (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                parameters TEXT NOT NULL,
                result_success INTEGER NOT NULL,
                result_output TEXT,
                result_error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );

            -- Permission cache
            CREATE TABLE IF NOT EXISTS permission_cache (
                pattern TEXT PRIMARY KEY,
                allowed INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
            CREATE INDEX IF NOT EXISTS idx_tool_executions_task ON tool_executions(task_id);
        "#,
        )?;

        info!("Database schema initialized");
        Ok(())
    }

    /// Create a new session.
    pub fn create_session(&self, title: Option<&str>) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO sessions (id, title) VALUES (?1, ?2)",
            params![id.to_string(), title],
        )?;

        debug!(session_id = %id, "Created session");
        Ok(id)
    }

    /// Store a message.
    pub async fn store_message(
        &self,
        session_id: &Uuid,
        role: &str,
        content: &str,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO messages (id, session_id, role, content) VALUES (?1, ?2, ?3, ?4)",
            params![id.to_string(), session_id.to_string(), role, content],
        )?;

        debug!(message_id = %id, session_id = %session_id, role = role, "Stored message");
        Ok(id)
    }

    /// Create a new task.
    pub fn create_task(&self, session_id: &Uuid, description: &str) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO tasks (id, session_id, description, status, started_at) VALUES (?1, ?2, ?3, 'running', datetime('now'))",
            params![id.to_string(), session_id.to_string(), description],
        )?;

        debug!(task_id = %id, "Created task");
        Ok(id)
    }

    /// Update task status.
    pub fn update_task_status(&self, task_id: &Uuid, status: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        let completed_at = if status == "completed" || status == "failed" {
            Some("datetime('now')".to_string())
        } else {
            None
        };

        if completed_at.is_some() {
            conn.execute(
                "UPDATE tasks SET status = ?1, completed_at = datetime('now') WHERE id = ?2",
                params![status, task_id.to_string()],
            )?;
        } else {
            conn.execute(
                "UPDATE tasks SET status = ?1 WHERE id = ?2",
                params![status, task_id.to_string()],
            )?;
        }

        Ok(())
    }

    /// Store a tool execution result.
    pub async fn store_tool_result(
        &self,
        task_id: &Uuid,
        call: &ToolCall,
        result: &ToolResult,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO tool_executions (id, task_id, tool_name, parameters, result_success, result_output, result_error) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                id.to_string(),
                task_id.to_string(),
                call.tool_name,
                serde_json::to_string(&call.parameters)?,
                result.success as i32,
                result.output,
                result.error,
            ],
        )?;

        debug!(execution_id = %id, tool = %call.tool_name, "Stored tool result");
        Ok(id)
    }

    /// Retrieve recent messages for a session.
    pub fn get_recent_messages(&self, session_id: &Uuid, limit: usize) -> Result<Vec<Message>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, role, content, created_at FROM messages WHERE session_id = ?1 ORDER BY created_at DESC LIMIT ?2"
        )?;

        let messages = stmt
            .query_map(params![session_id.to_string(), limit as i64], |row| {
                Ok(Message {
                    id: row.get(0)?,
                    role: row.get(1)?,
                    content: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(messages.into_iter().rev().collect())
    }

    /// Retrieve context for the agent planner.
    pub async fn retrieve_context(
        &self,
        _task_description: &str,
        session_id: &Uuid,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>> {
        let messages = self.get_recent_messages(session_id, limit)?;

        Ok(messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect())
    }

    /// Cache a permission decision.
    pub fn cache_permission(&self, pattern: &str, allowed: bool) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT OR REPLACE INTO permission_cache (pattern, allowed) VALUES (?1, ?2)",
            params![pattern, allowed as i32],
        )?;

        Ok(())
    }

    /// Check cached permission.
    pub fn get_cached_permission(&self, pattern: &str) -> Result<Option<bool>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT allowed FROM permission_cache WHERE pattern = ?1")?;

        let result = stmt.query_row(params![pattern], |row| Ok(row.get::<_, i32>(0)? != 0));

        match result {
            Ok(allowed) => Ok(Some(allowed)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Message from the database.
#[derive(Debug)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
}
