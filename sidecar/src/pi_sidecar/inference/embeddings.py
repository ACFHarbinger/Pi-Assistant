"""Embedding generation using sentence-transformers."""

from typing import List, Optional
import numpy as np


class EmbeddingEngine:
    """Generate text embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise RuntimeError("sentence-transformers not installed")

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.model is None:
            self._load_model()
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.model is None:
            self._load_model()
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self.model is None:
            return 384  # Default for MiniLM
        return self.model.get_sentence_embedding_dimension()


# Singleton instance
_embedding_engine: Optional[EmbeddingEngine] = None


def get_embedding_engine() -> EmbeddingEngine:
    """Get the singleton embedding engine."""
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine()
    return _embedding_engine
