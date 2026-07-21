"""Prometheus metrics with graceful fallback."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

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


def _histogram(
    name: str,
    doc: str,
    labels: list[str] | None = None,
    buckets: Any = None,
) -> Any:
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


_BUCKETS_LLM = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)

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
    "agentomatic_llm_duration_seconds",
    "LLM call total wall time",
    ["provider", "model", "profile"],
    buckets=_BUCKETS_LLM,
)

# Custom endpoints + upstream model calls
ENDPOINT_CALL_COUNT = _counter(
    "agentomatic_endpoint_calls_total", "Custom endpoint invocations", ["endpoint", "status"]
)
ENDPOINT_DURATION = _histogram(
    "agentomatic_endpoint_duration_seconds", "Custom endpoint duration", ["endpoint"]
)
UPSTREAM_CALL_COUNT = _counter(
    "agentomatic_upstream_calls_total", "Upstream model service calls", ["status"]
)
UPSTREAM_DURATION = _histogram(
    "agentomatic_upstream_duration_seconds", "Upstream model service call duration"
)

# Per-agent connections
CONNECTION_CALL_COUNT = _counter(
    "agentomatic_connection_calls_total", "Connection acquisitions", ["connection", "status"]
)

# ── Rich LLM model-fit telemetry (central invoke/stream path) ────────────────
LLM_CALLS = _counter(
    "agentomatic_llm_calls_total",
    "LLM invocations by outcome",
    ["provider", "model", "profile", "host", "outcome", "stream"],
)
LLM_TOKENS = _counter(
    "agentomatic_llm_tokens_total",
    "LLM token usage (prompt/completion/thinking)",
    ["direction", "provider", "model", "profile"],
)
LLM_TTFT = _histogram(
    "agentomatic_llm_ttft_seconds",
    "Time to first token (stream) or first usable content",
    ["provider", "model", "profile"],
    buckets=_BUCKETS_LLM,
)
LLM_THINKING = _histogram(
    "agentomatic_llm_thinking_seconds",
    "Time spent in thinking/reasoning phase",
    ["provider", "model", "profile"],
    buckets=_BUCKETS_LLM,
)
LLM_GENERATION = _histogram(
    "agentomatic_llm_generation_seconds",
    "Time spent generating answer tokens (excludes thinking when known)",
    ["provider", "model", "profile"],
    buckets=_BUCKETS_LLM,
)
LLM_RETRIES = _counter(
    "agentomatic_llm_retries_total",
    "LLM retries / format-repair attempts",
    ["provider", "model", "profile", "reason"],
)
LLM_STRUCTURE_ERRORS = _counter(
    "agentomatic_llm_structure_errors_total",
    "Structured output / schema parse failures",
    ["provider", "model", "profile", "agent"],
)
LLM_THROUGHPUT = _histogram(
    "agentomatic_llm_tokens_per_second",
    "Completion token throughput",
    ["provider", "model", "profile"],
    buckets=(1, 5, 10, 20, 40, 80, 160, 320, 640),
)

# Gauges
ACTIVE_REQUESTS = _gauge("agentomatic_active_requests", "Active requests")
ACTIVE_AGENTS = _gauge("agentomatic_active_agents", "Active agent invocations")
REGISTERED_AGENTS = _gauge("agentomatic_registered_agents", "Registered agents")
REGISTERED_ENDPOINTS = _gauge("agentomatic_registered_endpoints", "Registered custom endpoints")


def llm_identity(llm: Any) -> tuple[str, str, str, str]:
    """Return ``(provider, model, profile, host)`` labels for an LLM instance."""
    model = str(
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or getattr(llm, "deployment_name", None)
        or "unknown"
    )
    provider = str(
        getattr(llm, "_llm_type", None)
        or getattr(llm, "provider", None)
        or type(llm).__name__
        or "unknown"
    )
    profile = str(
        getattr(llm, "agentomatic_profile", None)
        or getattr(llm, "_agentomatic_profile", None)
        or "default"
    )
    base = (
        getattr(llm, "openai_api_base", None)
        or getattr(llm, "base_url", None)
        or getattr(llm, "api_base", None)
        or ""
    )
    host = "unknown"
    if base:
        try:
            host = urlparse(str(base)).hostname or str(base)[:64]
        except Exception:  # noqa: BLE001
            host = str(base)[:64]
    return provider, model, profile, host


def extract_token_usage(result: Any) -> tuple[int, int, int, int]:
    """Return ``(prompt, completion, total, thinking)`` token counts when present."""
    prompt = completion = total = thinking = 0
    usage_meta = getattr(result, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        prompt = int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0)
        completion = int(
            usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0
        )
        thinking = int(
            usage_meta.get("reasoning_tokens")
            or usage_meta.get("thinking_tokens")
            or usage_meta.get("output_reasoning_tokens")
            or 0
        )
        total = int(usage_meta.get("total_tokens") or (prompt + completion))
    meta = getattr(result, "response_metadata", None) or {}
    if isinstance(meta, dict):
        usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(usage, dict):
            prompt = prompt or int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = completion or int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
            if isinstance(details, dict):
                thinking = thinking or int(
                    details.get("reasoning_tokens") or details.get("thinking_tokens") or 0
                )
            total = total or int(usage.get("total_tokens") or (prompt + completion))
    return prompt, completion, total, thinking


def record_llm_call(
    *,
    llm: Any | None = None,
    provider: str = "unknown",
    model: str = "unknown",
    profile: str = "default",
    host: str = "unknown",
    outcome: str = "success",
    stream: bool = False,
    duration_s: float = 0.0,
    ttft_s: float | None = None,
    thinking_s: float | None = None,
    generation_s: float | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    thinking_tokens: int = 0,
    retries: int = 0,
    retry_reason: str = "invoke",
) -> None:
    """Record one LLM call for model-fit dashboards.

    ``outcome`` should be one of: ``success``, ``timeout``, ``http_error``,
    ``format_error``, ``error``.
    """
    if llm is not None:
        provider, model, profile, host = llm_identity(llm)

    stream_label = "1" if stream else "0"
    LLM_CALLS.labels(
        provider=provider,
        model=model,
        profile=profile,
        host=host,
        outcome=outcome,
        stream=stream_label,
    ).inc()
    if duration_s > 0:
        LLM_DURATION.labels(provider=provider, model=model, profile=profile).observe(duration_s)
        UPSTREAM_DURATION.observe(duration_s)
    UPSTREAM_CALL_COUNT.labels(status="ok" if outcome == "success" else "error").inc()

    if ttft_s is not None and ttft_s >= 0:
        LLM_TTFT.labels(provider=provider, model=model, profile=profile).observe(ttft_s)
    if thinking_s is not None and thinking_s > 0:
        LLM_THINKING.labels(provider=provider, model=model, profile=profile).observe(thinking_s)
    if generation_s is not None and generation_s > 0:
        LLM_GENERATION.labels(provider=provider, model=model, profile=profile).observe(
            generation_s
        )
    elif duration_s > 0 and thinking_s is not None and thinking_s >= 0:
        gen = max(0.0, duration_s - thinking_s)
        if gen > 0:
            LLM_GENERATION.labels(provider=provider, model=model, profile=profile).observe(gen)

    if prompt_tokens:
        LLM_TOKENS.labels(
            direction="prompt", provider=provider, model=model, profile=profile
        ).inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(
            direction="completion", provider=provider, model=model, profile=profile
        ).inc(completion_tokens)
    if thinking_tokens:
        LLM_TOKENS.labels(
            direction="thinking", provider=provider, model=model, profile=profile
        ).inc(thinking_tokens)

    if completion_tokens and duration_s > 0:
        LLM_THROUGHPUT.labels(provider=provider, model=model, profile=profile).observe(
            completion_tokens / duration_s
        )

    if retries > 0:
        LLM_RETRIES.labels(
            provider=provider, model=model, profile=profile, reason=retry_reason
        ).inc(retries)


def record_structure_error(
    *,
    llm: Any | None = None,
    provider: str = "unknown",
    model: str = "unknown",
    profile: str = "default",
    agent: str = "unknown",
) -> None:
    """Record a structured-output / JSON schema parse failure (model-fit signal)."""
    if llm is not None:
        provider, model, profile, _host = llm_identity(llm)
    LLM_STRUCTURE_ERRORS.labels(
        provider=provider, model=model, profile=profile, agent=agent
    ).inc()


@asynccontextmanager
async def track_agent_invocation(agent_name: str):
    """Context manager to track agent invocation metrics."""
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
