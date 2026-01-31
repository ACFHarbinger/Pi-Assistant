//! Memory subsystem for persistent storage.

pub mod embeddings;
pub mod sqlite;
pub mod vector;

pub use embeddings::EmbeddingGenerator;
pub use sqlite::MemoryManager;
pub use vector::VectorStore;
