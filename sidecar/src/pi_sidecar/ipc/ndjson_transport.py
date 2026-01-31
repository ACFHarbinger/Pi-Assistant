"""NDJSON transport over stdin/stdout for Rust <-> Python IPC."""
from __future__ import annotations

import asyncio
import json
import sys
from typing import AsyncIterator


class NdjsonTransport:
    """
    Reads JSON lines from stdin, writes JSON lines to stdout.
    Thread-safe for concurrent writes via asyncio Lock.
    """

    def __init__(self):
        self._write_lock = asyncio.Lock()
        self._reader: asyncio.StreamReader | None = None

    async def _ensure_reader(self) -> asyncio.StreamReader:
        if self._reader is None:
            loop = asyncio.get_event_loop()
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
            self._reader = reader
        return self._reader

    async def read_requests(self) -> AsyncIterator[dict]:
        """Async generator yielding parsed JSON request dicts from stdin."""
        reader = await self._ensure_reader()

        while True:
            line = await reader.readline()
            if not line:
                break  # stdin closed — parent process exited

            try:
                decoded = line.decode("utf-8").strip()
                if decoded:
                    yield json.loads(decoded)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Log to stderr, never to stdout
                print(f"[sidecar] Malformed input line: {e}", file=sys.stderr)
                continue

    async def send_response(self, request_id: str, result=None, error=None):
        msg: dict = {"id": request_id}
        if result is not None:
            msg["result"] = result
        if error is not None:
            msg["error"] = error
        await self._write(msg)

    async def send_error(self, request_id: str, code: str, message: str):
        await self.send_response(
            request_id,
            error={"code": code, "message": message},
        )

    async def send_progress(self, request_id: str, progress: dict):
        await self._write({"id": request_id, "progress": progress})

    async def _write(self, msg: dict):
        async with self._write_lock:
            line = json.dumps(msg, separators=(",", ":")) + "\n"
            sys.stdout.buffer.write(line.encode("utf-8"))
            sys.stdout.buffer.flush()
