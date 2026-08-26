"""In-memory sliding-window rate limiter.

Enabled via ``FEATURES__ENABLE_RATE_LIMIT=true``.
Configured via ``RATE_LIMIT__REQUESTS`` and ``RATE_LIMIT__WINDOW_SECONDS``.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from agentomatic.middleware.pathutils import OPERATIONAL_PATHS

#: Probe and scrape endpoints are exempt — see ``OPERATIONAL_PATHS``.
_SKIP_PATHS = OPERATIONAL_PATHS


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket style rate limiter per client IP.

    Args:
        app: ASGI application.
        max_requests: Maximum requests per window.
        window_seconds: Sliding window duration.
    """

    def __init__(
        self,
        app: Any,
        *,
        max_requests: int = 100,
        window_seconds: int = 60,
        trust_proxy_headers: bool = False,
    ) -> None:
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        # X-Forwarded-For is client-controlled unless a trusted reverse proxy
        # sets/overwrites it — trusting it by default lets any caller rotate
        # the header per request and bypass the limiter entirely. Only honour
        # it when the deployer explicitly confirms a trusted proxy is in front.
        #
        # This flag governs *this* middleware only. Uvicorn's own
        # ``--proxy-headers`` (on by default) rewrites ``request.client`` from
        # X-Forwarded-For for peers listed in ``--forwarded-allow-ips``
        # (default ``127.0.0.1``), and that rewrite happens before any of this
        # runs — the original peer address is not recoverable. So a caller who
        # can reach the server *from an allowed peer address* can still steer
        # the key. Keep ``--forwarded-allow-ips`` limited to your real proxy.
        self._trust_proxy_headers = trust_proxy_headers

    def _client_key(self, request: Request) -> str:
        if self._trust_proxy_headers:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    @staticmethod
    def _retry_after_seconds(*, window: float, now: float, oldest_hit: float) -> int:
        """Return a safe integral delay until a sliding-window slot opens."""
        return max(math.ceil(window - (now - oldest_hit)), 1)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in _SKIP_PATHS:
            response: Response = await call_next(request)
            return response

        key = self._client_key(request)
        now = time.monotonic()

        # Purge expired entries
        self._hits[key] = [t for t in self._hits[key] if now - t < self._window]

        if len(self._hits[key]) >= self._max:
            # HTTP Retry-After is integral seconds.  Rounding down tells a
            # caller to retry before the oldest sliding-window hit expires;
            # round up so the advertised delay is always safe to honour.
            retry_after = self._retry_after_seconds(
                window=self._window, now=now, oldest_hit=self._hits[key][0]
            )
            return JSONResponse(
                {"detail": "Rate limit exceeded", "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        self._hits[key].append(now)
        response = await call_next(request)
        remaining = self._max - len(self._hits[key])
        response.headers["X-RateLimit-Limit"] = str(self._max)
        response.headers["X-RateLimit-Remaining"] = str(max(remaining, 0))
        return response
