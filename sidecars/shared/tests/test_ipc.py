import pytest
import json
import io
import sys
from unittest.mock import MagicMock

from pi_sidecar.ipc.ndjson_transport import NdjsonTransport


@pytest.mark.asyncio
async def test_transport_send_response(monkeypatch):
    """Test sending a successful response."""
    transport = NdjsonTransport()

    buffer = io.BytesIO()
    mock_stdout = MagicMock()
    mock_stdout.buffer = buffer
    mock_stdout.buffer.write = buffer.write
    mock_stdout.buffer.flush = lambda: None

    monkeypatch.setattr(sys, "stdout", mock_stdout)

    await transport.send_response("req-123", result={"status": "ok"})

    output = buffer.getvalue().decode("utf-8")
    assert "req-123" in output
    assert "status" in output
    assert "ok" in output


@pytest.mark.asyncio
async def test_transport_send_error(monkeypatch):
    """Test sending an error response."""
    transport = NdjsonTransport()

    buffer = io.BytesIO()
    mock_stdout = MagicMock()
    mock_stdout.buffer = buffer
    mock_stdout.buffer.write = buffer.write
    mock_stdout.buffer.flush = lambda: None

    monkeypatch.setattr(sys, "stdout", mock_stdout)

    await transport.send_error("req-456", code="TestError", message="Something went wrong")

    output = buffer.getvalue().decode("utf-8")
    parsed = json.loads(output.strip())
    assert parsed["id"] == "req-456"
    assert parsed["error"]["code"] == "TestError"
    assert parsed["error"]["message"] == "Something went wrong"


@pytest.mark.asyncio
async def test_transport_send_progress(monkeypatch):
    """Test sending progress updates."""
    transport = NdjsonTransport()

    buffer = io.BytesIO()
    mock_stdout = MagicMock()
    mock_stdout.buffer = buffer
    mock_stdout.buffer.write = buffer.write
    mock_stdout.buffer.flush = lambda: None

    monkeypatch.setattr(sys, "stdout", mock_stdout)

    await transport.send_progress("req-789", {"percent": 50, "message": "Halfway"})

    output = buffer.getvalue().decode("utf-8")
    parsed = json.loads(output.strip())
    assert parsed["id"] == "req-789"
    assert parsed["progress"]["percent"] == 50
