"""IPC module for Rust <-> Python communication via NDJSON."""

from .ndjson_transport import NdjsonTransport

__all__ = ["NdjsonTransport", "RequestHandler"]
