import asyncio
import logging
import sys
from typing import Any, Callable

# Add shared and logic src to path
# (The Rust core will set PYTHONPATH properly)

from pi_sidecar.ipc.ndjson_transport import NdjsonTransport

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[logic-sidecar] %(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class LogicRequestHandler:
    def __init__(self):
        self._handlers = {
            "health.ping": self._health_ping,
            "lifecycle.shutdown": self._lifecycle_shutdown,
            "personality.get_hatching": self._personality_get_hatching,
            "personality.get_prompt": self._personality_get_prompt,
            "personality.get_name": self._personality_get_name,
            "personality.update_name": self._personality_update_name,
        }

    async def dispatch(
        self, method: str, params: dict, progress_callback: Callable | None = None
    ) -> Any:
        handler = self._handlers.get(method)
        if not handler:
            raise ValueError(f"Method {method} not supported by Logic sidecar")
        return await handler(params, progress_callback)

    async def _health_ping(self, p, _cb):
        return {"status": "ok", "sidecar": "logic"}

    async def _lifecycle_shutdown(self, p, _cb):
        asyncio.get_event_loop().call_later(0.5, sys.exit, 0)
        return {"status": "shutting_down"}

    async def _personality_get_hatching(self, p, _cb):
        from pi_sidecar.personality import get_personality

        return {"message": get_personality().hatching_message}

    async def _personality_get_prompt(self, p, _cb):
        from pi_sidecar.personality import get_personality

        return {"prompt": get_personality().system_prompt}

    async def _personality_get_name(self, p, _cb):
        from pi_sidecar.personality import get_personality

        return {"name": get_personality().name}

    async def _personality_update_name(self, p, _cb):
        from pi_sidecar.personality import get_personality

        success = get_personality().update_name(p["name"])
        return {"success": success, "name": get_personality().name}


async def main():
    logger.info("Logic sidecar starting")
    handler = LogicRequestHandler()
    transport = NdjsonTransport()
    async for request in transport.read_requests():
        asyncio.create_task(_handle_request(handler, transport, request))


async def _handle_request(handler, transport, request):
    req_id = request.get("id", "unknown")
    try:
        result = await handler.dispatch(
            request["method"],
            request.get("params", {}),
            lambda p: asyncio.ensure_future(transport.send_progress(req_id, p)),
        )
        await transport.send_response(req_id, result=result)
    except Exception as e:
        logger.exception("Error handling %s", request["method"])
        await transport.send_error(req_id, code=type(e).__name__, message=str(e))


if __name__ == "__main__":
    asyncio.run(main())
