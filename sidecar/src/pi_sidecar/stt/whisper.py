import os
import logging
from typing import Optional
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class WhisperSTT:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the Whisper STT engine.
        Args:
            model_size: Model size (base, small, medium, large-v3)
            device: cpu or cuda
            compute_type: int8, float16, etc.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def _ensure_model(self):
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_size} ({self.device})")
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    async def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe an audio file.
        """
        try:
            self._ensure_model()
            segments, info = self._model.transcribe(audio_path, beam_size=5)
            
            text = ""
            for segment in segments:
                text += segment.text
            
            return text.strip()
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return None
