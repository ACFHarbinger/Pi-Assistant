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

            -- Subtasks
            CREATE TABLE IF NOT EXISTS subtasks (
                id TEXT PRIMARY KEY,
                root_task_id TEXT NOT NULL,
                parent_id TEXT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                result TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(root_task_id) REFERENCES tasks(id)
            );

            -- Permission cache
            CREATE TABLE IF NOT EXISTS permission_cache (
                pattern TEXT PRIMARY KEY,
                allowed INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- RAG Chunks
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT, -- JSON
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
            CREATE INDEX IF NOT EXISTS idx_tool_executions_task ON tool_executions(task_id);
            CREATE INDEX IF NOT EXISTS idx_subtasks_root ON subtasks(root_task_id);
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_session ON rag_chunks(session_id);

            -- Knowledge Graph: Entities
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                metadata TEXT, -- JSON
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(name, entity_type)
            );

            -- Knowledge Graph: Relations
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                from_entity_id TEXT NOT NULL,
                to_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT, -- JSON
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(from_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY(to_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                UNIQUE(from_entity_id, to_entity_id, relation_type)
            );

            -- Graph Indexes
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

            -- Episode Summaries
            CREATE TABLE IF NOT EXISTS episode_summaries (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_episode_summaries_task ON episode_summaries(task_id);
        "#,
        )?;

        // Migration: add duration_ms column to tool_executions if not present
        let has_duration: bool = conn
            .prepare("SELECT COUNT(*) FROM pragma_table_info('tool_executions') WHERE name='duration_ms'")?
            .query_row([], |row| row.get::<_, i64>(0))
            .unwrap_or(0)
            > 0;

        if !has_duration {
            conn.execute(
                "ALTER TABLE tool_executions ADD COLUMN duration_ms INTEGER DEFAULT NULL",
                [],
            )?;
            info!("Migration: added duration_ms column to tool_executions");
        }

        // Migration: add expires_at column to permission_cache if not present
        let has_expires: bool = conn
            .prepare("SELECT COUNT(*) FROM pragma_table_info('permission_cache') WHERE name='expires_at'")?
            .query_row([], |row| row.get::<_, i64>(0))
            .unwrap_or(0)
            > 0;

        if !has_expires {
            conn.execute(
                "ALTER TABLE permission_cache ADD COLUMN expires_at TEXT DEFAULT NULL",
                [],
            )?;
            info!("Migration: added expires_at column to permission_cache");
        }

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

    /// Store or update subtasks.
    pub async fn store_subtasks(
        &self,
        task_id: &Uuid,
        subtasks: &[pi_core::agent_types::Subtask],
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        for subtask in subtasks {
            conn.execute(
                "INSERT OR REPLACE INTO subtasks (id, root_task_id, parent_id, title, description, status, result, created_at, updated_at) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    subtask.id.to_string(),
                    task_id.to_string(),
                    subtask.parent_id.map(|id| id.to_string()),
                    subtask.title,
                    subtask.description,
                    format!("{:?}", subtask.status).to_lowercase(),
                    subtask.result,
                    subtask.created_at.to_rfc3339(),
                    subtask.updated_at.to_rfc3339(),
                ],
            )?;
        }

        Ok(())
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
        duration_ms: Option<u64>,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO tool_executions (id, task_id, tool_name, parameters, result_success, result_output, result_error, duration_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                id.to_string(),
                task_id.to_string(),
                call.tool_name,
                serde_json::to_string(&call.parameters)?,
                result.success as i32,
                result.output,
                result.error,
                duration_ms.map(|d| d as i64),
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
    /// This implementation includes token-aware pruning to fit within context windows.
    pub async fn retrieve_context(
        &self,
        _task_description: &str,
        session_id: &Uuid,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>> {
        // Fetch more than needed initially to allow for pruning
        let messages = self.get_recent_messages(session_id, limit * 2)?;

        const MAX_CONTEXT_CHARS: usize = 16000; // ~4000 tokens
        let mut total_chars = 0;
        let mut pruned_messages = Vec::new();

        // Iterate backwards from most recent
        for m in messages.iter().rev() {
            let msg_chars = m.content.len();
            if total_chars + msg_chars > MAX_CONTEXT_CHARS && !pruned_messages.is_empty() {
                debug!(
                    total_chars,
                    excess = msg_chars,
                    "Pruning context to fit window"
                );
                break;
            }
            total_chars += msg_chars;
            pruned_messages.push(serde_json::json!({
                "role": m.role,
                "content": m.content,
            }));
        }

        // Reverse back to chronological order
        pruned_messages.reverse();
        Ok(pruned_messages)
    }

    /// Retrieve the execution timeline for a task.
    pub fn get_execution_timeline(&self, task_id: &Uuid) -> Result<Vec<serde_json::Value>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, tool_name, parameters, result_success, result_output, result_error, duration_ms, created_at
             FROM tool_executions
             WHERE task_id = ?1
             ORDER BY created_at ASC",
        )?;

        let rows = stmt
            .query_map(params![task_id.to_string()], |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "tool_name": row.get::<_, String>(1)?,
                    "parameters": row.get::<_, String>(2)?,
                    "success": row.get::<_, i32>(3)? != 0,
                    "output": row.get::<_, Option<String>>(4)?,
                    "error": row.get::<_, Option<String>>(5)?,
                    "duration_ms": row.get::<_, Option<i64>>(6)?,
                    "created_at": row.get::<_, String>(7)?,
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// Cache a permission decision.
    pub fn cache_permission(
        &self,
        pattern: &str,
        allowed: bool,
        expires_at: Option<String>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT OR REPLACE INTO permission_cache (pattern, allowed, expires_at) VALUES (?1, ?2, ?3)",
            params![pattern, allowed as i32, expires_at],
        )?;

        Ok(())
    }

    /// Check cached permission.
    pub fn get_cached_permission(&self, pattern: &str) -> Result<Option<bool>> {
        let conn = self.conn.lock().unwrap();
        // Clean up expired entries first (lazy cleanup)
        conn.execute(
            "DELETE FROM permission_cache WHERE expires_at IS NOT NULL AND expires_at < datetime('now')",
            [],
        )?;

        let mut stmt = conn.prepare("SELECT allowed FROM permission_cache WHERE pattern = ?1")?;

        let result = stmt.query_row(params![pattern], |row| Ok(row.get::<_, i32>(0)? != 0));

        match result {
            Ok(allowed) => Ok(Some(allowed)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // --- Knowledge Graph Operations ---

    /// Upsert an entity. Returns the entity ID.
    pub fn upsert_entity(
        &self,
        name: &str,
        entity_type: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<Uuid> {
        let conn = self.conn.lock().unwrap();
        let id = Uuid::new_v4();

        // Check if exists
        let existing_id: Option<String> = conn
            .query_row(
                "SELECT id FROM entities WHERE name = ?1 AND entity_type = ?2",
                params![name, entity_type],
                |row| row.get(0),
            )
            .ok();

        if let Some(existing) = existing_id {
            let uuid = Uuid::parse_str(&existing)?;
            // Update metadata if provided
            if let Some(meta) = metadata {
                conn.execute(
                    "UPDATE entities SET metadata = ?1, updated_at = datetime('now') WHERE id = ?2",
                    params![meta.to_string(), existing],
                )?;
            }
            return Ok(uuid);
        }

        conn.execute(
            "INSERT INTO entities (id, name, entity_type, metadata) VALUES (?1, ?2, ?3, ?4)",
            params![
                id.to_string(),
                name,
                entity_type,
                metadata.map(|m| m.to_string())
            ],
        )?;

        Ok(id)
    }

    /// Add a relation between two entities.
    pub fn add_relation(
        &self,
        from_id: &Uuid,
        to_id: &Uuid,
        relation_type: &str,
        weight: f64,
    ) -> Result<Uuid> {
        let conn = self.conn.lock().unwrap();
        let id = Uuid::new_v4();

        conn.execute(
            "INSERT OR REPLACE INTO relations (id, from_entity_id, to_entity_id, relation_type, weight) 
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id.to_string(),
                from_id.to_string(),
                to_id.to_string(),
                relation_type,
                weight
            ],
        )?;

        Ok(id)
    }

    /// Find related entities for a given entity.
    pub fn find_related_entities(&self, entity_name: &str, limit: usize) -> Result<Vec<GraphNode>> {
        let conn = self.conn.lock().unwrap();

        // simple query: find outgoing relations from entities matching the name
        let mut stmt = conn.prepare(
            r#"
             SELECT e2.name, e2.entity_type, r.relation_type, r.weight
             FROM entities e1
             JOIN relations r ON e1.id = r.from_entity_id
             JOIN entities e2 ON r.to_entity_id = e2.id
             WHERE e1.name = ?1
             ORDER BY r.weight DESC
             LIMIT ?2
             "#,
        )?;

        let rows = stmt.query_map(params![entity_name, limit as i64], |row| {
            Ok(GraphNode {
                name: row.get(0)?,
                entity_type: row.get(1)?,
                relation: row.get(2)?,
                weight: row.get(3)?,
            })
        })?;

        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.into())
    }

    // --- RAG Operations ---

    /// Store a RAG chunk with its embedding.
    pub fn store_rag_chunk(
        &self,
        session_id: &Uuid,
        document_name: &str,
        content: &str,
        embedding: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let conn = self.conn.lock().unwrap();

        // Convert f32 array to little-endian bytes for the BLOB
        let mut blob = Vec::with_capacity(embedding.len() * 4);
        for &f in embedding {
            blob.extend_from_slice(&f.to_le_bytes());
        }

        let metadata_str = metadata.map(|m| m.to_string());

        conn.execute(
            "INSERT INTO rag_chunks (id, session_id, document_name, content, embedding, metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id.to_string(),
                session_id.to_string(),
                document_name,
                content,
                blob,
                metadata_str
            ],
        )?;

        Ok(id)
    }

    /// Retrieve all RAG chunks for a session to perform similarity search in Rust.
    pub fn get_rag_chunks(
        &self,
        session_id: &Uuid,
    ) -> Result<Vec<(String, String, Vec<f32>, Option<serde_json::Value>)>> {
        self.get_rag_chunks_multi(&[*session_id])
    }

    /// Retrieve all RAG chunks for multiple sessions to perform similarity search in Rust.
    pub fn get_rag_chunks_multi(
        &self,
        session_ids: &[Uuid],
    ) -> Result<Vec<(String, String, Vec<f32>, Option<serde_json::Value>)>> {
        if session_ids.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().unwrap();

        // Build placeholders like (?1, ?2, ...)
        let mut placeholders = String::from("(");
        for i in 1..=session_ids.len() {
            placeholders.push_str(&format!("?{}", i));
            if i < session_ids.len() {
                placeholders.push(',');
            }
        }
        placeholders.push(')');

        let query = format!(
            "SELECT content, document_name, embedding, metadata FROM rag_chunks WHERE session_id IN {}",
            placeholders
        );

        let mut stmt = conn.prepare(&query)?;

        let id_strs: Vec<String> = session_ids.iter().map(|id| id.to_string()).collect();
        let params_vec: Vec<&dyn rusqlite::ToSql> =
            id_strs.iter().map(|s| s as &dyn rusqlite::ToSql).collect();

        let rows = stmt.query_map(rusqlite::params_from_iter(params_vec), |row| {
            let content: String = row.get(0)?;
            let doc_name: String = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let metadata_str: Option<String> = row.get(3)?;

            // Convert little-endian bytes back to f32
            let mut floats = Vec::with_capacity(blob.len() / 4);
            for chunk in blob.chunks_exact(4) {
                let bytes: [u8; 4] = match chunk.try_into() {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                floats.push(f32::from_le_bytes(bytes));
            }

            let metadata = metadata_str.and_then(|s| serde_json::from_str(&s).ok());

            Ok((content, doc_name, floats, metadata))
        })?;

        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row?);
        }
        Ok(chunks)
    }

    /// Delete all RAG chunks for a specific document in a session.
    pub fn delete_document_chunks(&self, session_id: &Uuid, document_name: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM rag_chunks WHERE session_id = ?1 AND document_name = ?2",
            params![session_id.to_string(), document_name],
        )?;
        Ok(())
    }

    // =========================================================================
    // Episode Summaries
    // =========================================================================

    /// Store an episode summary for a completed task.
    pub fn store_episode_summary(
        &self,
        task_id: &Uuid,
        summary: &str,
        embedding: Option<&[f32]>,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let conn = self.conn.lock().unwrap();

        let blob: Option<Vec<u8>> = embedding.map(|emb| {
            let mut bytes = Vec::with_capacity(emb.len() * 4);
            for &f in emb {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            bytes
        });

        conn.execute(
            "INSERT OR REPLACE INTO episode_summaries (id, task_id, summary, embedding) VALUES (?1, ?2, ?3, ?4)",
            params![id.to_string(), task_id.to_string(), summary, blob],
        )?;

        debug!(summary_id = %id, task_id = %task_id, "Stored episode summary");
        Ok(id)
    }

    /// Retrieve the episode summary for a specific task.
    pub fn get_episode_summary(&self, task_id: &Uuid) -> Result<Option<EpisodeSummary>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT id, summary, created_at FROM episode_summaries WHERE task_id = ?1")?;

        let mut rows = stmt.query(params![task_id.to_string()])?;
        if let Some(row) = rows.next()? {
            Ok(Some(EpisodeSummary {
                id: row.get(0)?,
                task_id: task_id.to_string(),
                summary: row.get(1)?,
                created_at: row.get(2)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get recent episode summaries for a session (most recent first).
    pub fn get_recent_summaries(
        &self,
        session_id: &Uuid,
        limit: usize,
    ) -> Result<Vec<EpisodeSummary>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT es.id, es.task_id, es.summary, es.created_at 
             FROM episode_summaries es
             JOIN tasks t ON es.task_id = t.id
             WHERE t.session_id = ?1
             ORDER BY es.created_at DESC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![session_id.to_string(), limit as i64], |row| {
            Ok(EpisodeSummary {
                id: row.get(0)?,
                task_id: row.get(1)?,
                summary: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;

        let mut summaries = Vec::new();
        for row in rows {
            summaries.push(row?);
        }
        Ok(summaries)
    }
}

/// Episode summary from the database.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EpisodeSummary {
    pub id: String,
    pub task_id: String,
    pub summary: String,
    pub created_at: String,
}

/// A node in the knowledge graph response.
#[derive(Debug, serde::Serialize)]
pub struct GraphNode {
    pub name: String,
    pub entity_type: String,
    pub relation: String,
    pub weight: f64,
}

/// Message from the database.
#[derive(Debug)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph() {
        let memory = MemoryManager::in_memory().unwrap();

        let id1 = memory.upsert_entity("Rust", "Language", None).unwrap();
        let id2 = memory
            .upsert_entity("Performance", "Attribute", None)
            .unwrap();

        memory.add_relation(&id1, &id2, "has_feature", 0.9).unwrap();

        // Use a short delay or ensure ordering isn't an issue if we had multiple
        let related = memory.find_related_entities("Rust", 5).unwrap();
        assert_eq!(related[0].name, "Performance");
        assert_eq!(related[0].relation, "has_feature");
    }

    #[test]
    fn test_rag_storage() {
        let memory = MemoryManager::in_memory().unwrap();
        let session_id = memory.create_session(Some("Test")).unwrap();
        let embedding = vec![0.1, 0.2, 0.3];

        memory
            .store_rag_chunk(&session_id, "test.txt", "hello world", &embedding, None)
            .unwrap();

        let chunks = memory.get_rag_chunks(&session_id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, "hello world");
        assert_eq!(chunks[0].1, "test.txt");
        assert_eq!(chunks[0].2, embedding);
    }

    #[test]
    fn test_rag_storage_multi() {
        let memory = MemoryManager::in_memory().unwrap();
        let sid1 = memory.create_session(Some("S1")).unwrap();
        let sid2 = memory.create_session(Some("S2")).unwrap();

        let emb = vec![0.1, 0.2, 0.3];
        memory
            .store_rag_chunk(&sid1, "d1.txt", "content 1", &emb, None)
            .unwrap();
        memory
            .store_rag_chunk(&sid2, "d2.txt", "content 2", &emb, None)
            .unwrap();

        let chunks = memory.get_rag_chunks_multi(&[sid1, sid2]).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_episode_summary_storage() {
        let memory = MemoryManager::in_memory().unwrap();
        let session_id = memory.create_session(Some("Test")).unwrap();
        let task_id = memory.create_task(&session_id, "Test task").unwrap();

        let summary = "User asked to build a web page. Used 3 tools. Task completed successfully.";
        memory
            .store_episode_summary(&task_id, summary, None)
            .unwrap();

        let retrieved = memory.get_episode_summary(&task_id).unwrap();
        assert!(retrieved.is_some());
        let ep = retrieved.unwrap();
        assert_eq!(ep.summary, summary);
        assert_eq!(ep.task_id, task_id.to_string());

        // Test retrieval by session
        let recent = memory.get_recent_summaries(&session_id, 10).unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].summary, summary);
    }
}
