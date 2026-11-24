"""Request logging middleware with X-Request-ID."""

from __future__ import annotations

import time
import uuid
from typing import Any

from loguru import logger
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import Message, Receive, Scope, Send

_SKIP_PATHS = {"/health", "/healthz", "/readiness", "/metrics", "/favicon.ico"}


class LoggingMiddleware:
    """Pure ASGI middleware logging every HTTP request with timing and request ID."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process ASGI request with logging, timing, and isolated LLM usage scope."""
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        if path in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        from agentomatic.logs.runtime import ensure_llm_usage, reset_llm_usage

        method = str(scope.get("method") or "GET").upper()
        headers = Headers(scope=scope)
        request_id = headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()

        logger.info(f"→ {method} {path} [{request_id}]")
        status_code = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 0) or 0)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                hdrs = MutableHeaders(scope=message)
                hdrs["X-Request-ID"] = request_id
                hdrs["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
            await send(message)

        _, usage_token = ensure_llm_usage(force_new=True)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_llm_usage(usage_token)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"← {method} {path} → {status_code or 500} ({elapsed_ms:.1f}ms) [{request_id}]"
            )
