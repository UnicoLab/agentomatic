"""API response envelope and error handling decorators."""
from __future__ import annotations

import functools
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Standard JSON response envelope."""
    success: bool = Field(True)
    data: Any = Field(None)
    message: str = Field("")
    error: str | None = Field(None)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def handle_api_errors(fn):
    """Decorator that catches unhandled exceptions and wraps them."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Unhandled error in {fn.__name__}: {exc}")
            raise HTTPException(500, detail=str(exc))
    return wrapper


def log_api_call(fn):
    """Decorator that logs function call timing."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
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
) -> StreamingResponse:
    """Create an SSE streaming response."""
    return StreamingResponse(
        generator,
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Agent": agent_name,
        },
    )
