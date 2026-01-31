"""
Pi-Assistant Python Sidecar — Entry Point.

Communicates with the Rust core via NDJSON over stdin/stdout.
stderr is used exclusively for logging (not protocol traffic).
"""
from __future__ import annotations

import asyncio
import logging
import sys

from pi_sidecar.ipc.ndjson_transport import NdjsonTransport
from pi_sidecar.ipc.handler import RequestHandler
from pi_sidecar.inference.engine import InferenceEngine
from pi_sidecar.models.registry import ModelRegistry

# All logging goes to stderr so stdout stays clean for NDJSON
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[sidecar] %(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Pi-Assistant sidecar starting (version 0.1.0)")

    registry = ModelRegistry()
    engine = InferenceEngine(registry)
    handler = RequestHandler(engine=engine, registry=registry)
    transport = NdjsonTransport()

    async for request in transport.read_requests():
        # Each request is handled concurrently
        asyncio.create_task(_handle_request(handler, transport, request))


async def _handle_request(
    handler: RequestHandler,
    transport: NdjsonTransport,
    request: dict,
):
    request_id = request.get("id", "unknown")
    method = request.get("method", "")
    params = request.get("params", {})

    logger.info("Handling request %s: %s", request_id, method)

    try:
        result = await handler.dispatch(
            method=method,
            params=params,
            progress_callback=lambda p: asyncio.ensure_future(
                transport.send_progress(request_id, p)
            ),
        )
        await transport.send_response(request_id, result=result)
    except Exception as e:
        logger.exception("Error handling %s: %s", method, e)
        await transport.send_error(
            request_id,
            code=type(e).__name__,
            message=str(e),
        )


if __name__ == "__main__":
    asyncio.run(main())
