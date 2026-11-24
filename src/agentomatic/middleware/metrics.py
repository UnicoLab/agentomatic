# mypy: disable-error-code="misc"
"""Prometheus metrics middleware.

Enabled via ``FEATURES__ENABLE_METRICS=true``.
Automatically tracks request count, latency histogram, and active requests.
"""

from __future__ import annotations

import time
from typing import Any, cast

from starlette.responses import Response as StarletteResponse
from starlette.types import Message, Receive, Scope, Send

from agentomatic.middleware.pathutils import OPERATIONAL_PATHS

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

#: Probe and scrape traffic is infrastructure noise, not user requests —
#: recording it would skew request counts and latency histograms.
_SKIP_PATHS = OPERATIONAL_PATHS

#: Collectors already registered, keyed by metric prefix.
#:
#: ``prometheus_client`` registers every collector into one process-wide
#: registry and raises ``ValueError: Duplicated timeseries`` when the same
#: metric name is registered twice. Building the collectors in
#: ``__init__`` therefore made a second ``AgentPlatform.build()`` in one
#: process a hard crash — which is an ordinary thing to do: uvicorn
#: ``--reload`` re-imports the app module, embedding hosts serve several
#: platforms side by side, and test suites build one per case. The metrics
#: are process-global anyway, so the instances are shared rather than
#: rebuilt.
_COLLECTOR_CACHE: dict[str, tuple[Any, Any, Any]] = {}


def _collectors(prefix: str) -> tuple[Any, Any, Any]:
    """Return the ``(requests, duration, active)`` collectors for *prefix*.

    Creates them on first use and reuses them afterwards, so constructing
    several :class:`MetricsMiddleware` instances in one process is safe.

    Args:
        prefix: Metric name prefix (e.g. ``agentomatic``).

    Returns:
        The counter, histogram and gauge registered for *prefix*.
    """
    cached = _COLLECTOR_CACHE.get(prefix)
    if cached is not None:
        return cached
    collectors = (
        Counter(
            f"{prefix}_http_requests_total",
            "Total HTTP requests",
            ["method", "path", "status"],
        ),
        Histogram(
            f"{prefix}_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "path"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        ),
        Gauge(
            f"{prefix}_http_requests_active",
            "Active HTTP requests",
        ),
    )
    _COLLECTOR_CACHE[prefix] = collectors
    return collectors


class MetricsMiddleware:
    """Pure ASGI Prometheus metrics collection per HTTP request."""

    def __init__(self, app: Any, *, prefix: str = "agentomatic") -> None:
        self.app = app
        self._requests: Any | None = None
        self._duration: Any | None = None
        self._active: Any | None = None
        if HAS_PROMETHEUS:
            self._requests, self._duration, self._active = _collectors(prefix)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        if path in _SKIP_PATHS:
            if path == "/metrics" and HAS_PROMETHEUS:
                resp = StarletteResponse(
                    content=generate_latest(),
                    media_type=CONTENT_TYPE_LATEST,
                )
                await resp(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method") or "GET").upper()
        if self._active:
            self._active.inc()

        status_code = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 0) or 0)
            await send(message)

        t0 = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - t0
            if self._active:
                self._active.dec()

            # Normalize path for cardinality control
            parts = path.split("/")
            normalized = "/".join(
                "{id}" if (len(p) > 8 and any(c.isdigit() for c in p)) else p for p in parts
            )
            resolved_status = status_code or 500
            if self._requests:
                self._requests.labels(
                    method=method,
                    path=normalized,
                    status=str(resolved_status),
                ).inc()
            if self._duration:
                self._duration.labels(
                    method=method,
                    path=normalized,
                ).observe(duration)
