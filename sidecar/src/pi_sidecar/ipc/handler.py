"""Request handler routing IPC methods to subsystems."""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Callable

from pi_sidecar.inference.engine import InferenceEngine
from pi_sidecar.models.registry import ModelRegistry
from pi_sidecar.training import TrainingService

logger = logging.getLogger(__name__)


class RequestHandler:
    """Routes incoming IPC method calls to the correct subsystem."""

    def __init__(
        self,
        engine: InferenceEngine,
        registry: ModelRegistry,
        training_service: TrainingService | None = None,
    ):
        """
        Initialize the request handler.
        Args:
            engine: The inference engine to use for text completion.
            registry: The model registry to use for model management.
            training_service: Optional training service for training operations.
        """
        self.engine = engine
        self.registry = registry
        self.training = training_service or TrainingService()

        self._handlers: dict[str, Callable] = {
            "health.ping": self._health_ping,
            "lifecycle.shutdown": self._lifecycle_shutdown,
            "inference.complete": self._inference_complete,
            "inference.embed": self._inference_embed,
            "inference.plan": self._inference_plan,
            "model.list": self._model_list,
            "model.load": self._model_load,
            "inference.load_model": self._inference_load_model,
            "personality.get_hatching": self._personality_get_hatching,
            "personality.get_prompt": self._personality_get_prompt,
            "personality.get_name": self._personality_get_name,
            "personality.update_name": self._personality_update_name,
            "personality.hatch_chat": self._personality_hatch_chat,
            # Training handlers
            "training.start": self._training_start,
            "training.stop": self._training_stop,
            "training.status": self._training_status,
            "training.list": self._training_list,
            "voice.synthesize": self._voice_synthesize,
            "voice.transcribe": self._voice_transcribe,
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
    
    # ... (existing handlers) ...

    async def _inference_load_model(self, params, _cb):
        """
        Handle inference model load requests.
        """
        model_name = params.get("model_id") or params.get("path")
        if not model_name:
             raise ValueError("Missing model_id or path")
             
        # Use the multi-provider engine to handle loading
        return await self.engine.load_model(model_name)

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
            tools=params.get("tools", []),
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

    # ── Personality handlers ─────────────────────────────────────

    async def _personality_get_hatching(self, params, _cb):
        """
        Get the hatching (first-run) message from personality.
        Returns:
            A dictionary containing the hatching message.
        """
        from pi_sidecar.personality import get_personality
        personality = get_personality()
        return {"message": personality.hatching_message}

    async def _personality_get_prompt(self, params, _cb):
        """
        Get the personality system prompt.
        Returns:
            A dictionary containing the personality prompt.
        """
        from pi_sidecar.personality import get_personality
        personality = get_personality()
        return {"prompt": personality.system_prompt}

    async def _personality_get_name(self, params, _cb):
        """
        Get the agent's name from personality.
        """
        from pi_sidecar.personality import get_personality
        personality = get_personality()
        return {"name": personality.name}

    async def _personality_update_name(self, params, _cb):
        """
        Update the agent's name in soul.md.
        """
        name = params.get("name")
        if not name:
             raise ValueError("Missing name")
             
        from pi_sidecar.personality import get_personality
        personality = get_personality()
        success = personality.update_name(name)
        return {"success": success, "name": personality.name}
    async def _personality_hatch_chat(self, params, _cb):
        """
        Handle interactive hatching chat.
        """
        from pi_sidecar.personality import get_personality
        personality = get_personality()
        
        # Build context with hatching system prompt
        system_prompt = f"{personality.system_prompt}\n\n# Hatching Context\nYou are in the 'hatching' phase. Be extremely welcoming and discuss your identity with the user."
        
        history = params.get("history", [])
        
        # Prepare messages for inference
        prompt = f"{system_prompt}\n\n"
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"[{role}]: {content}\n"
        
        # Get completion
        result = await self.engine.complete(
            prompt=prompt,
            provider=params.get("provider", "local"),
            model_id=params.get("model_id"),
        )
        
        return {"text": result.get("text", "")}

    # ── Training handlers ─────────────────────────────────────────

    async def _training_start(self, params, _cb):
        """
        Start a new training run.
        Args:
            params: Training configuration dict.
        Returns:
            Dictionary with run_id.
        """
        run_id = await self.training.start(params)
        return {"run_id": run_id, "status": "started"}

    async def _training_stop(self, params, _cb):
        """
        Stop a running training job.
        Args:
            params: Dictionary with run_id.
        Returns:
            Dictionary with stop status.
        """
        run_id = params.get("run_id")
        if not run_id:
            raise ValueError("Missing run_id")
        stopped = await self.training.stop(run_id)
        return {"stopped": stopped, "run_id": run_id}

    async def _training_status(self, params, _cb):
        """
        Get status of a training run.
        Args:
            params: Dictionary with run_id.
        Returns:
            Dictionary with run status and metrics.
        """
        run_id = params.get("run_id")
        if not run_id:
            raise ValueError("Missing run_id")
        return await self.training.status(run_id)

    async def _training_list(self, params, _cb):
        """
        List all training runs.
        Returns:
            Dictionary with list of runs.
        """
        runs = await self.training.list_runs()
        return {"runs": runs}

    async def _voice_synthesize(self, params, _cb):
        """
        Handle TTS synthesis requests.
        """
        text = params.get("text")
        output_path = params.get("output_path")
        if not text or not output_path:
            raise ValueError("Missing text or output_path")
            
        from pi_sidecar.tts.elevenlabs import ElevenLabsTTS
        tts = ElevenLabsTTS(api_key=params.get("api_key"))
        success = await tts.synthesize(text, output_path)
        return {"success": success, "output_path": output_path}

    async def _voice_transcribe(self, params, _cb):
        """
        Handle STT transcription requests.
        """
        audio_path = params.get("audio_path")
        if not audio_path:
            raise ValueError("Missing audio_path")
            
        from pi_sidecar.stt.whisper import WhisperSTT
        stt = WhisperSTT(
            model_size=params.get("model_size", "base"),
            device=params.get("device", "cpu")
        )
        text = await stt.transcribe(audio_path)
        return {"text": text, "audio_path": audio_path}
