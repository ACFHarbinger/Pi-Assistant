//! Embedding generation via Python sidecar.

use crate::ipc::SidecarHandle;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

/// Embedding generator using the Python sidecar.
pub struct EmbeddingGenerator {
    sidecar: Arc<Mutex<SidecarHandle>>,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator.
    pub fn new(sidecar: Arc<Mutex<SidecarHandle>>) -> Self {
        Self { sidecar }
    }

    /// Generate an embedding for the given text.
    /// Returns a 384-dimensional vector (all-MiniLM-L6-v2).
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        debug!(text_len = text.len(), "Generating embedding");

        let mut sidecar = self.sidecar.lock().await;
        let response = sidecar
            .request(
                "inference.embed",
                serde_json::json!({
                    "text": text,
                }),
            )
            .await?;

        let embedding: Vec<f32> = serde_json::from_value(
            response
                .get("embedding")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Missing embedding in response"))?,
        )?;

        debug!(dimensions = embedding.len(), "Embedding generated");
        Ok(embedding)
    }

    /// Generate embeddings for multiple texts in batch.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        debug!(count = texts.len(), "Generating batch embeddings");

        let mut sidecar = self.sidecar.lock().await;
        let response = sidecar
            .request(
                "inference.embed_batch",
                serde_json::json!({
                    "texts": texts,
                }),
            )
            .await?;

        let embeddings: Vec<Vec<f32>> = serde_json::from_value(
            response
                .get("embeddings")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Missing embeddings in response"))?,
        )?;

        Ok(embeddings)
    }
}
