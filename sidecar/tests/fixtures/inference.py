import pytest
import pytest_asyncio
from unittest.mock import MagicMock
from pi_sidecar.inference.engine import InferenceEngine

@pytest.fixture
def mock_registry():
    """Fixture to provide a mocked model registry."""
    registry = MagicMock()
    # Mock return values if necessary
    return registry

@pytest.fixture
def engine(mock_registry):
    """Fixture to provide an InferenceEngine with mocked registry."""
    return InferenceEngine(registry=mock_registry)
