"""API response envelope and error handling decorators."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from agentomatic.core.errors import client_safe_detail


class APIResponse(BaseModel):
    """Standard JSON response envelope."""

    success: bool = Field(True)
    data: Any = Field(default=None)
    message: str = Field("")
    error: str | None = Field(default=None)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def handle_api_errors(fn: Callable[..., Any]):
    """Decorator that catches unhandled exceptions and wraps them."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Unhandled error in {fn.__name__}: {exc}")
            raise HTTPException(500, detail=client_safe_detail(exc, context="Request failed"))

    return wrapper


def log_api_call(fn: Callable[..., Any]):
    """Decorator that logs function call timing."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"{fn.__name__} completed in {elapsed:.1f}ms")
            return result
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error(f"{fn.__name__} failed after {elapsed:.1f}ms")
            raise

    return wrapper


def create_streaming_response(
    generator: Any,
    agent_name: str = "",
    media_type: str = "text/event-stream",
    *,
    stream_id: str | None = None,
    owner: str = "",
) -> StreamingResponse:
    """Create an SSE streaming response.

    Args:
        generator: Async iterator yielding SSE-encoded strings.
        agent_name: Reported in the ``X-Agent`` header.
        media_type: Response media type.
        stream_id: Retain the stream's frames under this identity and number
            them, so a client that drops mid-response can collect what it
            missed. It **must** be a fresh
            :func:`~agentomatic.streaming.new_stream_id` — never a
            caller-supplied value such as a path parameter, which would let
            one caller name another's stream. The value is echoed back in
            ``X-Stream-Id``. Omit it to stream without retention, as before.
        owner: Ownership tag required to read these frames back. Retention is
            process-wide, so without a tag scoping the stream to its endpoint
            and principal, anything that can reach a replay route could read
            it by id alone.

    Returns:
        The streaming response.
    """
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Agent": agent_name,
    }
    body = generator
    if stream_id:
        from agentomatic.streaming import numbered_stream

        body = numbered_stream(generator, stream_id=stream_id, owner=owner)
        headers["X-Stream-Id"] = stream_id

    return StreamingResponse(body, media_type=media_type, headers=headers)
