import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from pi_sidecar.ipc.ndjson_transport import NdjsonTransport

@pytest.mark.asyncio
async def test_ndjson_read_requests(transport):
    """Verify that read_requests yields parsed JSON objects from a stream."""
    
    # Mock the reader
    mock_reader = AsyncMock()
    # Simulate lines: valid json, empty line, malformed json, valid json, EOF
    mock_reader.readline.side_effect = [
        b'{"id": "1", "method": "test"}\n',
        b'\n',
        b'INVALID_JSON\n',
        b'{"id": "2", "method": "test2"}\n',
        b''
    ]
    transport._reader = mock_reader

    # Collect requests
    requests = []
    async for req in transport.read_requests():
        requests.append(req)

    # We expect 2 valid requests. The empty line and invalid json should be skipped.
    assert len(requests) == 2
    assert requests[0]["id"] == "1"
    assert requests[0]["method"] == "test"
    assert requests[1]["id"] == "2"

@pytest.mark.asyncio
async def test_send_response_format(transport):
    """Verify send_response calls write with correct format."""
    transport._write = AsyncMock()

    await transport.send_response("req-123", result={"foo": "bar"})

    # Check that _write was called with expected dict
    transport._write.assert_called_once()
    call_arg = transport._write.call_args[0][0]
    assert call_arg == {"id": "req-123", "result": {"foo": "bar"}}

@pytest.mark.asyncio
async def test_send_error_format(transport):
    """Verify send_error calls write with correct format."""
    transport._write = AsyncMock()

    await transport.send_error("req-999", "ErrorCode", "Something went wrong")

    transport._write.assert_called_once()
    call_arg = transport._write.call_args[0][0]
    assert call_arg == {
        "id": "req-999", 
        "error": {"code": "ErrorCode", "message": "Something went wrong"}
    }
