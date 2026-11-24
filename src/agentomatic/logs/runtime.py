"""Process-wide invocation log recorder binding.

Routers and in-process pipeline steps share one recorder bound at platform
lifespan once the store is ready. This avoids ContextVar propagation issues
across FastAPI request tasks while keeping step wrappers store-agnostic.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentomatic.logs.recorder import InvocationLogRecorder


@dataclass
class AgentLLMUsage:
    """Per-agent or per-step accumulated LLM usage for a single request/task."""

    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


@dataclass
class LLMUsageAccumulator:
    """Request/pipeline/task-scoped LLM usage accumulator."""

    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    by_agent: dict[str, AgentLLMUsage] = field(default_factory=dict)


_recorder: InvocationLogRecorder | None = None
_pipeline_name: ContextVar[str | None] = ContextVar(
    "agentomatic_pipeline_log_name",
    default=None,
)
_llm_usage: ContextVar[LLMUsageAccumulator | None] = ContextVar(
    "agentomatic_llm_usage",
    default=None,
)
_usage_agent_scope: ContextVar[str | None] = ContextVar(
    "agentomatic_llm_usage_scope",
    default=None,
)

RESOURCE_TYPES = frozenset({"agent", "plugin", "pipeline", "ingestion", "endpoint"})


def set_invocation_log_recorder(recorder: InvocationLogRecorder | None) -> None:
    """Bind the platform-wide invocation log recorder (or clear it)."""
    global _recorder
    _recorder = recorder


def get_invocation_log_recorder() -> InvocationLogRecorder | None:
    """Return the bound recorder, if any."""
    return _recorder


def bind_pipeline_log_name(name: str | None):
    """Set the active pipeline name for nested in-process step logs.

    Returns:
        A context token suitable for :func:`reset_pipeline_log_name`.
    """
    return _pipeline_name.set(name)


def reset_pipeline_log_name(token: Token[str | None]) -> None:
    """Reset the active pipeline name context."""
    _pipeline_name.reset(token)


def get_pipeline_log_name() -> str | None:
    """Return the active pipeline name for step metadata, if any."""
    return _pipeline_name.get()


def normalize_resource_type(resource_type: str | None) -> str:
    """Return a validated resource type (default ``agent``)."""
    value = (resource_type or "agent").strip().lower()
    if value not in RESOURCE_TYPES:
        raise ValueError(
            f"Invalid resource_type {resource_type!r}; expected one of {sorted(RESOURCE_TYPES)}"
        )
    return value


def ensure_llm_usage(
    *,
    force_new: bool = False,
) -> tuple[LLMUsageAccumulator, Token[LLMUsageAccumulator | None] | None]:
    """Ensure a mutable per-invocation LLM usage accumulator is bound."""
    if not force_new:
        existing = _llm_usage.get()
        if existing is not None:
            return existing, None
    fresh = LLMUsageAccumulator()
    token = _llm_usage.set(fresh)
    return fresh, token


def reset_llm_usage(token: Token[LLMUsageAccumulator | None] | None) -> None:
    """Reset the LLM usage accumulator ContextVar if a token was created."""
    if token is not None:
        _llm_usage.reset(token)


def get_llm_usage() -> LLMUsageAccumulator | None:
    """Return the active LLM usage accumulator, if any."""
    return _llm_usage.get()


def bind_llm_usage_scope(scope: str | None) -> Token[str | None]:
    """Bind the active agent/step usage scope for nested LLM calls."""
    return _usage_agent_scope.set(scope)


def reset_llm_usage_scope(token: Token[str | None] | None) -> None:
    """Reset the active agent/step usage scope."""
    if token is not None:
        _usage_agent_scope.reset(token)


def get_llm_usage_scope() -> str | None:
    """Return the active agent/step usage scope, if any."""
    return _usage_agent_scope.get()


def record_llm_usage(
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    *,
    agent_name: str | None = None,
) -> None:
    """Record one LLM call's model and token counts on the active accumulator."""
    acc = _llm_usage.get()
    if acc is None:
        return
    clean_model = str(model or "").strip()
    if clean_model.lower() in {"", "unknown", "none"}:
        clean_model = ""
    in_t = max(0, int(input_tokens or 0))
    out_t = max(0, int(output_tokens or 0))
    tot_t = max(0, int(total_tokens or 0)) or (in_t + out_t)

    acc.llm_calls += 1
    acc.input_tokens += in_t
    acc.output_tokens += out_t
    acc.total_tokens += tot_t
    if clean_model:
        acc.model = clean_model

    scope = (agent_name or _usage_agent_scope.get() or "").strip()
    if scope:
        stats = acc.by_agent.setdefault(scope, AgentLLMUsage())
        stats.llm_calls += 1
        stats.input_tokens += in_t
        stats.output_tokens += out_t
        stats.total_tokens += tot_t
        if clean_model:
            stats.model = clean_model


def resolve_llm_usage(
    resource_type: str,
    resource_name: str | None = None,
) -> tuple[str | None, int, int, int, int]:
    """Resolve ``(model, input_tokens, output_tokens, total_tokens, llm_calls)``.

    Isolates per-agent, per-plugin, and per-endpoint usage in multi-step
    pipelines so a deterministic fast-path step never inherits an earlier
    step's model or token counts.
    """
    acc = _llm_usage.get()
    if acc is None:
        return None, 0, 0, 0, 0
    rtype = (resource_type or "agent").strip().lower()
    if rtype == "ingestion":
        return None, 0, 0, 0, 0
    if rtype == "pipeline":
        return (
            acc.model,
            acc.input_tokens,
            acc.output_tokens,
            acc.total_tokens,
            acc.llm_calls,
        )
    name = (resource_name or "").strip()
    if rtype in {"plugin", "endpoint"}:
        stats = acc.by_agent.get(f"{rtype}:{name}") or acc.by_agent.get(name)
        if stats is not None:
            return (
                stats.model or acc.model,
                stats.input_tokens,
                stats.output_tokens,
                stats.total_tokens,
                stats.llm_calls,
            )
        if acc.llm_calls > 0 and not acc.by_agent:
            return (
                acc.model,
                acc.input_tokens,
                acc.output_tokens,
                acc.total_tokens,
                acc.llm_calls,
            )
        return None, 0, 0, 0, 0
    if rtype == "agent":
        stats = acc.by_agent.get(name)
        if stats is not None:
            return (
                stats.model or acc.model,
                stats.input_tokens,
                stats.output_tokens,
                stats.total_tokens,
                stats.llm_calls,
            )
        if acc.llm_calls > 0 and not acc.by_agent:
            return (
                acc.model,
                acc.input_tokens,
                acc.output_tokens,
                acc.total_tokens,
                acc.llm_calls,
            )
        return None, 0, 0, 0, 0
    if acc.llm_calls > 0:
        return (
            acc.model,
            acc.input_tokens,
            acc.output_tokens,
            acc.total_tokens,
            acc.llm_calls,
        )
    return None, 0, 0, 0, 0
