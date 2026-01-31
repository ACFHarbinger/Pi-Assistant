"""Model registry for loading and managing models."""
from __future__ import annotations

import logging
from pathlib import Path

from ..configs.model import LoadedModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing loaded models."""

    def __init__(self, models_dir: Path | None = None):
        """
        Initialize the model registry.
        Args:
            models_dir: The directory to store models.
        """
        self.models_dir = models_dir or Path.home() / ".pi-assistant" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, LoadedModel] = {}

    def list_models(self) -> list[dict]:
        """
        List available models.
        Returns:
            A list of dictionaries containing the model information.
        """
        models = []

        # List models in models_dir
        for item in self.models_dir.iterdir():
            if item.is_dir():
                models.append({
                    "model_id": item.name,
                    "path": str(item),
                    "loaded": item.name in self._loaded,
                    "backend": "transformers",
                })
            elif item.suffix == ".gguf":
                models.append({
                    "model_id": item.name,
                    "path": str(item),
                    "loaded": item.name in self._loaded,
                    "backend": "llama.cpp",
                })

        # Add loaded models not in directory
        for model_id, model in self._loaded.items():
            if not any(m["model_id"] == model_id for m in models):
                models.append({
                    "model_id": model_id,
                    "path": None,
                    "loaded": True,
                    "backend": getattr(model, "backend", "unknown"),
                })

        return models

    async def load_model(self, model_id: str, backend: str | None = None) -> LoadedModel:
        """
        Load a model by ID.
        Args:
            model_id: The model ID to load.
            backend: Optional backend override ("transformers" or "llama.cpp")
        Returns:
            A LoadedModel object containing the model and tokenizer.
        """
        if model_id in self._loaded:
            logger.info("Model already loaded: %s", model_id)
            return self._loaded[model_id]

        logger.info("Loading model: %s (backend: %s)", model_id, backend or "auto")

        # Check path
        model_path = self.models_dir / model_id
        is_gguf = model_id.endswith(".gguf") or model_path.suffix == ".gguf"
        
        # Decide backend
        effective_backend = backend
        if not effective_backend:
            effective_backend = "llama.cpp" if is_gguf else "transformers"

        if effective_backend == "llama.cpp":
            from llama_cpp import Llama
            
            path = str(model_path) if model_path.exists() else model_id
            model = Llama(model_path=path, n_ctx=2048) # Default context
            
            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=None,
                backend="llama.cpp",
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Check if it's a local path or HuggingFace ID
            if model_path.exists():
                path = str(model_path)
            else:
                path = model_id  # Treat as HuggingFace model ID

            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path)

            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=tokenizer,
                backend="transformers",
            )
            
        self._loaded[model_id] = loaded
        return loaded

    def get_model(self, model_id: str) -> LoadedModel | None:
        """
        Get a loaded model, or None if not loaded.
        Args:
            model_id: The model ID to get.
        Returns:
            A LoadedModel object containing the model and tokenizer.
        """
        return self._loaded.get(model_id)

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model to free memory.
        Args:
            model_id: The model ID to unload.
        Returns:
            A boolean indicating whether the model was unloaded.
        """
        if model_id in self._loaded:
            del self._loaded[model_id]
            logger.info("Unloaded model: %s", model_id)
            return True
        return False
