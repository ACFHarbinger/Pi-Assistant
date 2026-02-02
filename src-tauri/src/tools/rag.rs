use crate::memory::{EmbeddingGenerator, VectorStore};
use crate::tools::{PermissionTier, Tool, ToolContext, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tracing::info;

/// Tool for Retrieval-Augmented Generation (RAG).
pub struct RagTool {
    embedding_generator: Arc<EmbeddingGenerator>,
    vector_store: VectorStore,
}

impl RagTool {
    /// Create a new RAG tool.
    pub fn new(embedding_generator: Arc<EmbeddingGenerator>) -> Self {
        Self {
            embedding_generator,
            vector_store: VectorStore::new(),
        }
    }

    /// Split text into chunks with overlap.
    fn chunk_text(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut i = 0;
        while i < words.len() {
            let end = (i + chunk_size).min(words.len());
            let chunk = words[i..end].join(" ");
            chunks.push(chunk);
            if end == words.len() {
                break;
            }
            i += chunk_size - overlap;
        }
        chunks
    }
}

#[async_trait]
impl Tool for RagTool {
    fn name(&self) -> &str {
        "rag"
    }

    fn description(&self) -> &str {
        "Retrieval-Augmented Generation: ingest documents and query them using vector search."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ingest", "query", "delete"],
                    "description": "Action to perform"
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file to ingest (required for action='ingest')"
                },
                "query": {
                    "type": "string",
                    "description": "Query string (required for action='query')"
                },
                "document_name": {
                    "type": "string",
                    "description": "Optional name for the document"
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results to return"
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
        let session_id = context.session_id;

        match action {
            "ingest" => {
                let path = params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing path for ingest"))?;
                let doc_name = params
                    .get("document_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| {
                        std::path::Path::new(path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(path)
                    });

                let content = std::fs::read_to_string(path)?;
                let chunks = self.chunk_text(&content, 200, 50);

                info!(doc = %doc_name, chunks = chunks.len(), "Ingesting document into RAG");

                for chunk in &chunks {
                    let embedding = self.embedding_generator.embed(chunk).await?;
                    memory.store_rag_chunk(&session_id, doc_name, chunk, &embedding, None)?;
                }

                Ok(ToolResult::success(format!(
                    "Successfully ingested {} in {} chunks",
                    doc_name,
                    chunks.len()
                )))
            }
            "query" => {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing query"))?;
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

                let query_embedding = self.embedding_generator.embed(query).await?;

                // Search both current session and global knowledge (nil UUID)
                let session_ids = [session_id, uuid::Uuid::nil()];
                let all_chunks = memory.get_rag_chunks_multi(&session_ids)?;

                let search_results =
                    self.vector_store
                        .search_linear(&query_embedding, all_chunks, limit);

                if search_results.is_empty() {
                    return Ok(ToolResult::success("No relevant information found."));
                }

                let mut output = String::from("Relevant excerpts found:\n\n");
                for res in search_results {
                    output.push_str(&format!(
                        "--- Source: {} (Score: {:.2}) ---\n",
                        res.document_name, res.score
                    ));
                    output.push_str(&res.content);
                    output.push('\n');
                    output.push('\n');
                }

                Ok(ToolResult::success(output))
            }
            "delete" => {
                let doc_name = params
                    .get("document_name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing document_name for delete"))?;

                memory.delete_document_chunks(&session_id, doc_name)?;
                Ok(ToolResult::success(format!(
                    "Deleted all chunks for document: {}",
                    doc_name
                )))
            }
            _ => Err(anyhow::anyhow!("Unknown action: {}", action)),
        }
    }

    fn permission_tier(&self) -> PermissionTier {
        PermissionTier::Low
    }
}
