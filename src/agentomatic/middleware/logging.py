"""Request logging middleware with X-Request-ID."""
from __future__ import annotations

import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_SKIP_PATHS = {"/health", "/healthz", "/readiness", "/metrics", "/favicon.ico"}


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with timing and request ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with logging and timing."""
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        t0 = time.perf_counter()

        logger.info(f"→ {request.method} {request.url.path} [{request_id}]")

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"← {request.method} {request.url.path} → {response.status_code} "
            f"({elapsed_ms:.1f}ms) [{request_id}]"
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response
