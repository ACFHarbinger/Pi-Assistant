/// Vector similarity search result.
#[derive(Debug, serde::Serialize)]
pub struct SearchResult {
    pub content: String,
    pub document_name: String,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

/// Calculate cosine similarity between two vectors.
pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }
    let dot_product: f32 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
    let mag1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let mag2: f32 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();
    if mag1 == 0.0 || mag2 == 0.0 {
        return 0.0;
    }
    dot_product / (mag1 * mag2)
}

/// Vector store for similarity search.
/// Currently uses a linear scan over SQLite-stored chunks.
pub struct VectorStore {
    enabled: bool,
}

impl VectorStore {
    /// Create a new vector store.
    pub fn new() -> Self {
        Self { enabled: true }
    }

    /// Check if vector search is available.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Perform a linear scan search over provided chunks.
    pub fn search_linear(
        &self,
        query_embedding: &[f32],
        chunks: Vec<(String, String, Vec<f32>, Option<serde_json::Value>)>,
        limit: usize,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = chunks
            .into_iter()
            .map(|(content, document_name, embedding, metadata)| {
                let score = cosine_similarity(query_embedding, &embedding);
                SearchResult {
                    content,
                    document_name,
                    score,
                    metadata,
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}
