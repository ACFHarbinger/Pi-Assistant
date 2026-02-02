use crate::memory::EmbeddingGenerator;
use crate::tools::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// Tool for managing markdown-based persistent knowledge.
pub struct KnowledgeTool {
    embedding_generator: Arc<EmbeddingGenerator>,
    knowledge_dir: PathBuf,
}

impl KnowledgeTool {
    /// Create a new Knowledge tool.
    pub fn new(embedding_generator: Arc<EmbeddingGenerator>) -> Self {
        let knowledge_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".pi-assistant")
            .join("knowledge");

        // Ensure directory exists
        if !knowledge_dir.exists() {
            let _ = std::fs::create_dir_all(&knowledge_dir);
        }

        Self {
            embedding_generator,
            knowledge_dir,
        }
    }

    /// Sanitize topic name for filename use.
    fn sanitize_topic(&self, topic: &str) -> String {
        topic
            .to_lowercase()
            .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "-")
            .trim_matches('-')
            .to_string()
    }

    /// Get file path for a topic.
    fn get_path(&self, topic: &str) -> PathBuf {
        let filename = format!("{}.md", self.sanitize_topic(topic));
        self.knowledge_dir.join(filename)
    }
}

#[async_trait]
impl Tool for KnowledgeTool {
    fn name(&self) -> &str {
        "knowledge"
    }

    fn description(&self) -> &str {
        "Maintain a persistent, human-readable knowledge base in markdown format.
Actions:
- upsert(topic, content): Create or update a knowledge file.
- read(topic): Read a knowledge file.
- list(): List all available topics.
- index(): Trigger RAG indexing for all knowledge files."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["upsert", "read", "list", "index"],
                    "description": "Action to perform"
                },
                "topic": {
                    "type": "string",
                    "description": "The topic name (used as filename)"
                },
                "content": {
                    "type": "string",
                    "description": "The markdown content (for action='upsert')"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, params: serde_json::Value, context: ToolContext) -> Result<ToolResult> {
        let action = params.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let memory = context
            .memory
            .ok_or_else(|| anyhow::anyhow!("MemoryManager not available"))?;

        // Global knowledge uses a nil UUID
        let global_session_id = uuid::Uuid::nil();

        match action {
            "upsert" => {
                let topic = params
                    .get("topic")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing topic"))?;
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing content"))?;

                let path = self.get_path(topic);
                std::fs::write(&path, content)?;

                info!(?path, "Saved knowledge file");

                // Automatically index the updated file for RAG
                let chunks = content.split("\n\n").collect::<Vec<_>>(); // Simple chunking by paragraph

                // Clear old chunks for this document first if they exist
                let doc_name = path.file_name().and_then(|n| n.to_str()).unwrap_or(topic);
                memory.delete_document_chunks(&global_session_id, doc_name)?;

                for chunk in chunks {
                    if chunk.trim().is_empty() {
                        continue;
                    }
                    let embedding = self.embedding_generator.embed(chunk).await?;
                    memory.store_rag_chunk(
                        &global_session_id,
                        doc_name,
                        chunk,
                        &embedding,
                        None,
                    )?;
                }

                Ok(ToolResult::success(format!(
                    "Knowledge saved and indexed for topic: {}",
                    topic
                )))
            }
            "read" => {
                let topic = params
                    .get("topic")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing topic"))?;

                let path = self.get_path(topic);
                if !path.exists() {
                    return Ok(ToolResult::error(format!(
                        "Knowledge for '{}' not found.",
                        topic
                    )));
                }

                let content = std::fs::read_to_string(path)?;
                Ok(ToolResult::success(content))
            }
            "list" => {
                let entries = std::fs::read_dir(&self.knowledge_dir)?
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
                    .filter_map(|e| {
                        e.path()
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string())
                    })
                    .collect::<Vec<_>>();

                Ok(ToolResult::success(format!(
                    "Available topics: {}",
                    entries.join(", ")
                )))
            }
            "index" => {
                let mut total_chunks = 0;
                let mut total_files = 0;

                for entry in std::fs::read_dir(&self.knowledge_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("md") {
                        let content = std::fs::read_to_string(&path)?;
                        let doc_name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown");

                        // Clear old chunks
                        memory.delete_document_chunks(&global_session_id, doc_name)?;

                        let chunks = content.split("\n\n").collect::<Vec<_>>();
                        for chunk in chunks {
                            if chunk.trim().is_empty() {
                                continue;
                            }
                            let embedding = self.embedding_generator.embed(chunk).await?;
                            memory.store_rag_chunk(
                                &global_session_id,
                                doc_name,
                                chunk,
                                &embedding,
                                None,
                            )?;
                            total_chunks += 1;
                        }
                        total_files += 1;
                    }
                }

                Ok(ToolResult::success(format!(
                    "Indexed {} chunks from {} knowledge files.",
                    total_chunks, total_files
                )))
            }
            _ => Err(anyhow::anyhow!("Unknown action: {}", action)),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Low
    }
}
