# mypy: disable-error-code="misc"
"""Prometheus metrics middleware.

Enabled via ``FEATURES__ENABLE_METRICS=true``.
Automatically tracks request count, latency histogram, and active requests.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    Counter = cast(Any, None)
    Gauge = cast(Any, None)
    Histogram = cast(Any, None)
    generate_latest = cast(Any, None)
    CONTENT_TYPE_LATEST = cast(Any, None)

_SKIP_PATHS = {"/health", "/healthz", "/readiness", "/metrics"}


class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics collection per request."""

    def __init__(
        self, app: Any, *, prefix: str = "agentomatic"
    ) -> None:
        super().__init__(app)
        self._requests: Any | None = None
        self._duration: Any | None = None
        self._active: Any | None = None
        if HAS_PROMETHEUS:
            self._requests = Counter(
                f"{prefix}_http_requests_total",
                "Total HTTP requests",
                ["method", "path", "status"],
            )
            self._duration = Histogram(
                f"{prefix}_http_request_duration_seconds",
                "HTTP request duration",
                ["method", "path"],
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            self._active = Gauge(
                f"{prefix}_http_requests_active",
                "Active HTTP requests",
            )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in _SKIP_PATHS:
            # Serve /metrics endpoint
            if request.url.path == "/metrics" and HAS_PROMETHEUS:
                from starlette.responses import Response as StarletteResponse

                body = generate_latest()
                return StarletteResponse(
                    content=body,
                    media_type=CONTENT_TYPE_LATEST,
                )
            response: Response = await call_next(request)
            return response

        if self._active:
            self._active.inc()

        t0 = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - t0

        # Normalize path for cardinality control
        path = request.url.path
        # Collapse UUIDs and hex IDs to reduce cardinality
        parts = path.split("/")
        normalized = "/".join(
            "{id}" if (len(p) > 8 and any(c.isdigit() for c in p)) else p for p in parts
        )

        if self._requests:
            self._requests.labels(
                method=request.method,
                path=normalized,
                status=str(response.status_code),
            ).inc()
        if self._duration:
            self._duration.labels(
                method=request.method,
                path=normalized,
            ).observe(duration)
        if self._active:
            self._active.dec()

        return response
