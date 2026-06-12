"""In-memory sliding-window rate limiter.

Enabled via ``FEATURES__ENABLE_RATE_LIMIT=true``.
Configured via ``RATE_LIMIT__REQUESTS`` and ``RATE_LIMIT__WINDOW_SECONDS``.
"""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_SKIP_PATHS = {"/health", "/healthz", "/readiness"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket style rate limiter per client IP.

    Args:
        app: ASGI application.
        max_requests: Maximum requests per window.
        window_seconds: Sliding window duration.
    """

    def __init__(
        self,
        app,
        *,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()

        # Purge expired entries
        self._hits[key] = [t for t in self._hits[key] if now - t < self._window]

        if len(self._hits[key]) >= self._max:
            retry_after = int(self._window - (now - self._hits[key][0]))
            return JSONResponse(
                {"detail": "Rate limit exceeded", "retry_after": max(retry_after, 1)},
                status_code=429,
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        self._hits[key].append(now)
        response = await call_next(request)
        remaining = self._max - len(self._hits[key])
        response.headers["X-RateLimit-Limit"] = str(self._max)
        response.headers["X-RateLimit-Remaining"] = str(max(remaining, 0))
        return response
