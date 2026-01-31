"""
Personality module for Pi-Assistant.

Loads soul.md and provides personality-aware system prompts.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default hatching message if soul.md is not found
DEFAULT_HATCHING_MESSAGE = """Hey! I'm Pi, your personal AI assistant. 🎉

I run right here on your device—your conversations stay private, and I can help with all sorts of tasks:

- **Code & Shell**: Write, debug, and run code  
- **Files & Browser**: Read, edit files, and browse the web
- **Memory**: Remember our conversations for context

I'll ask before doing anything that could affect your system. Ready to get started?"""

DEFAULT_PERSONALITY = """You are Pi, a friendly and helpful AI assistant.
Be warm, curious, and concise. Use natural language and match the user's energy."""


class Personality:
    """Manages agent personality from soul.md."""

    def __init__(self, workspace_path: str | None = None):
        """
        Initialize personality from soul.md.
        
        Args:
            workspace_path: Path to workspace containing soul.md.
                           Defaults to Pi-Assistant root.
        """
        self._workspace = workspace_path or self._find_workspace()
        self._soul_content: str | None = None
        self._load_soul()

    def _find_workspace(self) -> str:
        """Find the Pi-Assistant workspace root."""
        # Try common locations
        candidates = [
            Path.home() / "Repositories" / "Pi-Assistant",
            Path.home() / ".pi-assistant",
            Path(__file__).parents[4],  # Navigate up from this module
        ]
        for candidate in candidates:
            if (candidate / "soul.md").exists():
                return str(candidate)
        return str(Path.home())

    def _load_soul(self) -> None:
        """Load soul.md content."""
        soul_path = Path(self._workspace) / "soul.md"
        if soul_path.exists():
            try:
                self._soul_content = soul_path.read_text(encoding="utf-8")
                logger.info("Loaded personality from: %s", soul_path)
            except Exception as e:
                logger.warning("Failed to load soul.md: %s", e)
                self._soul_content = None
        else:
            logger.info("No soul.md found, using default personality")
            self._soul_content = None

    @property
    def system_prompt(self) -> str:
        """Get the personality-aware system prompt."""
        if self._soul_content:
            return f"# Personality Guide\n\n{self._soul_content}\n\n# Instructions\n\nFollow the personality guidelines above in all your responses."
        return DEFAULT_PERSONALITY

    @property
    def hatching_message(self) -> str:
        """Get the first-run hatching message."""
        if self._soul_content:
            # Extract hatching message from soul.md if present
            lines = self._soul_content.split("\n")
            in_hatching = False
            hatching_lines = []
            
            for line in lines:
                if "First Encounter" in line or "Hatching" in line:
                    in_hatching = True
                    continue
                if in_hatching:
                    if line.startswith("#"):
                        break
                    if line.startswith(">"):
                        hatching_lines.append(line[1:].strip())
                    elif hatching_lines and line.strip():
                        hatching_lines.append(line.strip())
            
            if hatching_lines:
                return "\n".join(hatching_lines)
        
        return DEFAULT_HATCHING_MESSAGE

    def reload(self) -> None:
        """Reload soul.md from disk."""
        self._load_soul()


# Singleton instance
_personality: Personality | None = None


def get_personality(workspace_path: str | None = None) -> Personality:
    """Get or create the personality singleton."""
    global _personality
    if _personality is None:
        _personality = Personality(workspace_path)
    return _personality
