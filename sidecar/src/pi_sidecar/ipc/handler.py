"""Request handler routing IPC methods to subsystems."""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Callable

from pi_sidecar.inference.engine import InferenceEngine
from pi_sidecar.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class RequestHandler:
    """Routes incoming IPC method calls to the correct subsystem."""

    def __init__(
        self,
        engine: InferenceEngine,
        registry: ModelRegistry,
    ):
        """
        Initialize the request handler.
        Args:
            engine: The inference engine to use for text completion.
            registry: The model registry to use for model management.
        """
        self.engine = engine
        self.registry = registry

        self._handlers: dict[str, Callable] = {
            "health.ping": self._health_ping,
            "lifecycle.shutdown": self._lifecycle_shutdown,
            "inference.complete": self._inference_complete,
            "inference.embed": self._inference_embed,
            "inference.plan": self._inference_plan,
            "model.list": self._model_list,
            "model.load": self._model_load,
        }

    async def dispatch(
        self,
        method: str,
        params: dict,
        progress_callback: Callable | None = None,
    ) -> Any:
        """
        Dispatch an IPC method call to the appropriate handler.
        Args:
            method: The method to dispatch.
            params: The parameters for the method.
            progress_callback: A callback to report progress.
        Returns:
            The result of the method call.
        """
        handler = self._handlers.get(method)
        if handler is None:
            raise ValueError(f"Unknown method: {method}")
        return await handler(params, progress_callback)

    # ── Built-in handlers ─────────────────────────────────────────

    async def _health_ping(self, params, _cb):
        """
        Handle health ping requests.
        Args:
            params: The parameters for the method.
            _cb: The progress callback.
        Returns:
            A dictionary containing the health status.
        """
        return {"status": "ok", "version": "0.1.0"}

    async def _lifecycle_shutdown(self, params, _cb):
        """
        Handle lifecycle shutdown requests.
        Args:
            params: The parameters for the method.
            _cb: The progress callback.
        Returns:
            A dictionary containing the shutdown status.
        """
        logger.info("Shutdown requested by Rust core")
        asyncio.get_event_loop().call_later(0.5, sys.exit, 0)
        return {"status": "shutting_down"}

    async def _inference_complete(self, params, _cb):
        return await self.engine.complete(
            prompt=params["prompt"],
            provider=params.get("provider", "local"),
            model_id=params.get("model_id"),
            max_tokens=params.get("max_tokens", 1024),
            temperature=params.get("temperature", 0.7),
        )

    async def _inference_embed(self, params, _cb):
        """
        Handle inference embed requests.
        Args:
            params: The parameters for the method.
            _cb: The progress callback.
        Returns:
            A dictionary containing the embeddings.
        """
        vector = await self.engine.embed(
            text=params["text"],
            model_id=params.get("model_id", "all-MiniLM-L6-v2"),
        )
        return {"embedding": vector}

    async def _inference_plan(self, params, _cb):
        """
        Handle inference plan requests.
        Args:
            params: The parameters for the method.
            _cb: The progress callback.
        Returns:
            A dictionary containing the plan.
        """
        return await self.engine.plan(
            task=params["task"],
            iteration=params["iteration"],
            context=params.get("context", []),
            provider=params.get("provider", "local"),
        )

    async def _model_list(self, params, _cb):
        """
        Handle model list requests.
        Args:
            params: The parameters for the method.
            _cb: The progress callback.
        Returns:
            A dictionary containing the list of models.
        """
        return {"models": self.registry.list_models()}

    async def _model_load(self, params, _cb):
        """
        Handle model load requests.
        Args:
            params: The parameters for the method.
            _cb: The progress callback.
        Returns:
            A dictionary containing the load status.
        """
        await self.registry.load_model(params["model_id"])
        return {"status": "loaded", "model_id": params["model_id"]}
