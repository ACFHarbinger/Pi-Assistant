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

        # List HuggingFace models in models_dir
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                models.append({
                    "model_id": model_dir.name,
                    "path": str(model_dir),
                    "loaded": model_dir.name in self._loaded,
                })

        # Add loaded models not in directory
        for model_id, model in self._loaded.items():
            if not any(m["model_id"] == model_id for m in models):
                models.append({
                    "model_id": model_id,
                    "path": None,
                    "loaded": True,
                })

        return models

    async def load_model(self, model_id: str) -> LoadedModel:
        """
        Load a model by ID.
        Args:
            model_id: The model ID to load.
        Returns:
            A LoadedModel object containing the model and tokenizer.
        """
        if model_id in self._loaded:
            logger.info("Model already loaded: %s", model_id)
            return self._loaded[model_id]

        logger.info("Loading model: %s", model_id)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Check if it's a local path or HuggingFace ID
        model_path = self.models_dir / model_id
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
