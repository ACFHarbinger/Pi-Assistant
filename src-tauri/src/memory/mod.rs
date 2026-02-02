//! Memory subsystem for persistent storage.

pub mod embeddings;
pub mod sqlite;
pub mod summarization;
pub mod vector;

pub use embeddings::EmbeddingGenerator;
pub use sqlite::{EpisodeSummary, MemoryManager};
pub use summarization::{generate_task_summary, TaskOutcome};
pub use vector::VectorStore;
