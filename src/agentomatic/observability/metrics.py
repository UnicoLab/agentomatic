"""Prometheus metrics with graceful fallback."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class _DummyMetric:
    """No-op metric when prometheus_client is not installed."""

    def labels(self, **kw: Any) -> _DummyMetric:
        return self

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass

    def info(self, val: dict) -> None:
        pass


def _counter(name: str, doc: str, labels: list[str] | None = None) -> Any:
    """Create a Prometheus Counter or dummy fallback."""
    if HAS_PROMETHEUS:
        return Counter(name, doc, labels or [])
    return _DummyMetric()


def _histogram(name: str, doc: str, labels: list[str] | None = None, buckets: Any = None) -> Any:
    """Create a Prometheus Histogram or dummy fallback."""
    if HAS_PROMETHEUS:
        kw = {"buckets": buckets} if buckets else {}
        return Histogram(name, doc, labels or [], **kw)
    return _DummyMetric()


def _gauge(name: str, doc: str, labels: list[str] | None = None) -> Any:
    """Create a Prometheus Gauge or dummy fallback."""
    if HAS_PROMETHEUS:
        return Gauge(name, doc, labels or [])
    return _DummyMetric()


# Counters
REQUEST_COUNT = _counter(
    "agentomatic_requests_total", "Total requests", ["method", "endpoint", "status_code"]
)
AGENT_INVOCATION_COUNT = _counter(
    "agentomatic_agent_invocations_total", "Agent invocations", ["agent_name", "status"]
)
ERROR_COUNT = _counter("agentomatic_errors_total", "Errors", ["error_type", "agent_name"])

# Histograms
REQUEST_DURATION = _histogram(
    "agentomatic_request_duration_seconds", "Request duration", ["method", "endpoint"]
)
AGENT_DURATION = _histogram("agentomatic_agent_duration_seconds", "Agent duration", ["agent_name"])
LLM_DURATION = _histogram(
    "agentomatic_llm_duration_seconds", "LLM call duration", ["provider", "model"]
)

# Gauges
ACTIVE_REQUESTS = _gauge("agentomatic_active_requests", "Active requests")
ACTIVE_AGENTS = _gauge("agentomatic_active_agents", "Active agent invocations")
REGISTERED_AGENTS = _gauge("agentomatic_registered_agents", "Registered agents")


@asynccontextmanager
async def track_agent_invocation(agent_name: str):
    """Context manager to track agent invocation metrics."""
    import time

    ACTIVE_AGENTS.inc()
    AGENT_INVOCATION_COUNT.labels(agent_name=agent_name, status="started").inc()
    t0 = time.perf_counter()
    try:
        yield
        AGENT_INVOCATION_COUNT.labels(agent_name=agent_name, status="success").inc()
    except Exception:
        AGENT_INVOCATION_COUNT.labels(agent_name=agent_name, status="error").inc()
        raise
    finally:
        duration = time.perf_counter() - t0
        AGENT_DURATION.labels(agent_name=agent_name).observe(duration)
        ACTIVE_AGENTS.dec()
