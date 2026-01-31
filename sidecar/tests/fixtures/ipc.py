import pytest
from pi_sidecar.ipc.ndjson_transport import NdjsonTransport

@pytest.fixture
def transport():
    """Fixture to provide an NdjsonTransport instance."""
    return NdjsonTransport()
