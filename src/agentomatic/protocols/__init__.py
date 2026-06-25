"""Protocol support (A2A, decorators)."""

from __future__ import annotations

from .decorators import (
    APIResponse,
    create_streaming_response,
    handle_api_errors,
    log_api_call,
)

__all__ = [
    "APIResponse",
    "create_streaming_response",
    "handle_api_errors",
    "log_api_call",
]
