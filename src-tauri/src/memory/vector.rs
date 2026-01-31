//! Vector search placeholder.
//!
//! Note: sqlite-vec requires special compilation or the extension file.
//! This is a stub that returns empty results when not available.

use anyhow::Result;
use tracing::{debug, info};

/// Vector store for similarity search (placeholder).
pub struct VectorStore {
    /// Flag indicating if sqlite-vec is available.
    enabled: bool,
}

impl VectorStore {
    /// Create a new vector store.
    pub fn new() -> Self {
        info!("VectorStore: sqlite-vec not bundled, vector search disabled");
        Self { enabled: false }
    }

    /// Check if vector search is available.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Store an embedding (no-op when disabled).
    pub fn store(&self, _content_type: &str, _content_id: &str, _embedding: &[f32]) -> Result<i64> {
        if !self.enabled {
            // Return a dummy ID when disabled
            return Ok(0);
        }
        Ok(0)
    }

    /// Search for similar embeddings (returns empty when disabled).
    pub fn search(&self, _query_embedding: &[f32], _limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Vector search disabled, returning empty results");
        Ok(Vec::new())
    }
}

/// Result of a vector similarity search.
#[derive(Debug)]
pub struct SearchResult {
    pub id: i64,
    pub content_type: String,
    pub content_id: String,
    pub distance: f32,
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}
