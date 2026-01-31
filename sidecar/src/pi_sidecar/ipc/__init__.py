"""IPC module for Rust <-> Python communication via NDJSON."""

from .ndjson_transport import NdjsonTransport
from .handler import RequestHandler

__all__ = ["NdjsonTransport", "RequestHandler"]
