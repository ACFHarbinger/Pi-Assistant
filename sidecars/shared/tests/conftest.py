"""Shared sidecar test configuration and fixtures."""

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

    """Return sample soul.md content."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    soul_path = fixtures_dir / "sample_soul.md"
    return soul_path.read_text(encoding="utf-8")


@pytest.fixture
def soul_file(workspace_dir: Path, sample_soul_content: str) -> Path:
    """Create a soul.md file in the workspace."""
    soul_path = workspace_dir / "soul.md"
    soul_path.write_text(sample_soul_content, encoding="utf-8")
    return soul_path


@pytest.fixture
def minimal_soul_file(workspace_dir: Path) -> Path:
    """Create a minimal soul.md with just a name."""
    soul_path = workspace_dir / "soul.md"
    soul_path.write_text("You are **MinimalAgent**", encoding="utf-8")
    return soul_path


# ============================================================================
# Personality Instance Fixtures
# ============================================================================


@pytest.fixture
def personality_with_soul(workspace_dir: Path, soul_file: Path):
    """Create a Personality instance with soul.md loaded."""
    from pi_sidecar.personality import Personality

    return Personality(workspace_path=str(workspace_dir))


@pytest.fixture
def personality_without_soul(workspace_dir: Path):
    """Create a Personality instance without soul.md (uses defaults)."""
    from pi_sidecar.personality import Personality

    return Personality(workspace_path=str(workspace_dir))


# ============================================================================
# IPC Transport Fixtures
# ============================================================================


@pytest.fixture
def transport():
    """Create an NdjsonTransport instance."""
    from pi_sidecar.ipc.ndjson_transport import NdjsonTransport

    return NdjsonTransport()


@pytest.fixture
def mock_stdout(monkeypatch):
    """Mock stdout for capturing transport output."""
    import io
    import sys
    from unittest.mock import MagicMock

    buffer = io.BytesIO()
    mock = MagicMock()
    mock.buffer = buffer
    mock.buffer.write = buffer.write
    mock.buffer.flush = lambda: None

    monkeypatch.setattr(sys, "stdout", mock)
    return buffer
