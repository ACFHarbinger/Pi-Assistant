"""Logic sidecar test configuration and fixtures."""

from __future__ import annotations

import pytest
from pathlib import Path


# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    """Create a temporary workspace directory for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


# ============================================================================
# Soul/Personality Fixtures
# ============================================================================


@pytest.fixture
def soul_file(workspace_dir: Path) -> Path:
    """Create a soul.md file in the workspace."""
    soul_path = workspace_dir / "soul.md"
    soul_content = "You are **TestAgent**\n\n# First Encounter\n> Welcome!"
    soul_path.write_text(soul_content, encoding="utf-8")
    return soul_path


@pytest.fixture
def mock_personality_manager(workspace_dir: Path, soul_file: Path, monkeypatch):
    """
    Mock the Personality manager to return a configured instance.
    This avoids needing to instantiate the full Personality logic in every test.
    """
    from pi_sidecar.personality import Personality

    # Create a real personality instance to be returned by the mock
    p = Personality(workspace_path=str(workspace_dir))

    # Mock the get_personality function or wherever the singleton is accessed
    # Assuming logic_main.py uses a global or singleton pattern, but based on
    # typical patterns we might need to mock where it's imported.
    # For now, we'll return the instance so tests can use it to patch
    return p
