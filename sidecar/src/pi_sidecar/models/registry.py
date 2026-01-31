"""Model registry for loading and managing models."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A loaded model with its tokenizer."""

    model_id: str
    model: Any
    tokenizer: Any
    metadata: dict = field(default_factory=dict)


class ModelRegistry:
    """Registry for managing loaded models."""

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or Path.home() / ".pi-assistant" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, LoadedModel] = {}

    def list_models(self) -> list[dict]:
        """List available models."""
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
        """Load a model by ID."""
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
        """Get a loaded model, or None if not loaded."""
        return self._loaded.get(model_id)

    def unload_model(self, model_id: str) -> bool:
        """Unload a model to free memory."""
        if model_id in self._loaded:
            del self._loaded[model_id]
            logger.info("Unloaded model: %s", model_id)
            return True
        return False
