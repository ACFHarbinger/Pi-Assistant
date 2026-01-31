"""
Personality module for Pi-Assistant.

Loads soul.md and provides personality-aware system prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from .constants import DEFAULT_HATCHING_MESSAGE, DEFAULT_PERSONALITY

logger = logging.getLogger(__name__)


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

    @property
    def name(self) -> str:
        """Get the agent's name from soul.md."""
        if self._soul_content:
            # Find "You are **[Name]**"
            import re

            match = re.search(r"You are \*\*([^\*]+)\*\*", self._soul_content)
            if match:
                return match.group(1).strip()
        return "Pi"

    def update_name(self, name: str) -> bool:
        """Update the agent's name in soul.md."""
        soul_path = Path(self._workspace) / "soul.md"
        if not soul_path.exists():
            return False

        content = soul_path.read_text(encoding="utf-8")
        import re

        # Replace the first occurrence of "You are **[Name]**"
        new_content = re.sub(r"You are \*\*([^\*]+)\*\*", f"You are **{name}**", content, count=1)

        try:
            soul_path.write_text(new_content, encoding="utf-8")
            self._soul_content = new_content
            return True
        except Exception as e:
            logger.error("Failed to update agent name in soul.md: %s", e)
            return False

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
