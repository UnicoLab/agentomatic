"""Dispatchers that execute a single platform resource by name.

Each dispatcher resolves a resource (agent / plugin / pipeline / endpoint) from
its registry and runs it against one input payload, returning a JSON-safe
result. Dispatchers are intentionally single-input; the :class:`TaskManager`
handles batching, concurrency, cancellation, and progress on top of them.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.core.registry import AgentRegistry
    from agentomatic.endpoints.registry import EndpointRegistry
    from agentomatic.ingestion.registry import IngestionRegistry
    from agentomatic.plugins.registry import PluginRegistry

    from .context import TaskContext


class Dispatcher(Protocol):
    """Callable that runs a single input against a named resource."""

    async def __call__(self, target: str, payload: Any, ctx: TaskContext) -> Any:
        """Execute ``target`` with ``payload`` and return a JSON-safe result."""
        ...


class TargetNotFoundError(LookupError):
    """Raised when a task references a resource that is not registered."""


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of a result to a JSON-serialisable value."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _build_agent_state(payload: Any) -> dict[str, Any]:
    """Build a minimal agent state dict from a raw input payload.

    Mirrors the standard fields produced by the synchronous invoke path so
    that agents behave identically whether called sync or as a task.
    """
    data: dict[str, Any] = payload if isinstance(payload, dict) else {"query": payload}
    query = data.get("query") or data.get("current_query") or ""
    if not query:
        for key, val in data.items():
            if key.endswith("_query") and isinstance(val, str):
                query = val
                break
    standard = {"query", "current_query", "user_id", "thread_id", "context", "metadata"}
    extra = {k: v for k, v in data.items() if k not in standard}
    return {
        "current_query": query,
        "user_id": data.get("user_id") or "default-user",
        "thread_id": data.get("thread_id") or f"thread_{uuid.uuid4().hex[:12]}",
        "messages": [],
        "context": data.get("context") or {},
        "metadata": {**(data.get("metadata") or {}), **extra},
        "steps_taken": [],
        "response": "",
        "suggestions": [],
        "citations": [],
        "prompt_version": data.get("prompt_version", "v1"),
    }


def make_agent_dispatcher(registry: AgentRegistry) -> Dispatcher:
    """Return a dispatcher that runs a registered agent as a task."""

    async def run(target: str, payload: Any, ctx: TaskContext) -> Any:
        agent = registry.get(target)
        if agent is None:
            raise TargetNotFoundError(
                f"Agent '{target}' not found. Available: {registry.list_names()}"
            )
        state = _build_agent_state(payload)
        await ctx.report(message=f"Invoking agent '{target}'")

        for hook in registry.before_node_hooks:
            try:
                hook(target, state)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"before_node hook error: {exc}")

        if agent.graph_fn:
            graph = agent.graph_fn()
            result = await graph.ainvoke(state)
        elif agent.node_fn:
            result = await agent.node_fn(state)
        else:
            raise RuntimeError(f"Agent '{target}' has no callable (node_fn/graph_fn)")

        for hook in registry.after_node_hooks:
            try:
                hook(target, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"after_node hook error: {exc}")

        return _to_jsonable(result)

    return run


def make_plugin_dispatcher(registry: PluginRegistry) -> Dispatcher:
    """Return a dispatcher that runs an ML plugin's ``predict`` as a task."""

    async def run(target: str, payload: Any, ctx: TaskContext) -> Any:
        plugin = registry.get_plugin(target)
        if plugin is None:
            raise TargetNotFoundError(
                f"Plugin '{target}' not found. Available: {registry.list_names()}"
            )
        if not plugin.is_loaded:
            raise RuntimeError(f"Plugin '{target}' is not loaded")
        input_schema = plugin.get_input_schema()
        data = payload if isinstance(payload, dict) else {"input": payload}
        try:
            inputs = input_schema(**data)
        except Exception:  # noqa: BLE001 - fall back to raw payload
            inputs = payload
        await ctx.report(message=f"Running plugin '{target}'")
        result = await plugin.predict(inputs)
        return _to_jsonable(result)

    return run


def make_endpoint_dispatcher(registry: EndpointRegistry) -> Dispatcher:
    """Return a dispatcher that runs a custom endpoint as a task."""

    async def run(target: str, payload: Any, ctx: TaskContext) -> Any:
        endpoint = registry.get(target)
        if endpoint is None:
            raise TargetNotFoundError(
                f"Endpoint '{target}' not found. Available: {registry.list_names()}"
            )
        await ctx.report(message=f"Calling endpoint '{target}'")
        result = await endpoint.call(payload)
        return _to_jsonable(result)

    return run


def make_ingestion_dispatcher(registry: IngestionRegistry) -> Dispatcher:
    """Return a dispatcher that runs an ingestor as a task."""

    async def run(target: str, payload: Any, ctx: TaskContext) -> Any:
        ingestor = registry.get(target)
        if ingestor is None:
            raise TargetNotFoundError(
                f"Ingestor '{target}' not found. Available: {registry.list_names()}"
            )
        await ctx.report(message=f"Ingesting via '{target}'")
        result = await ingestor.run(payload, ctx)
        return _to_jsonable(result)

    return run


def make_pipeline_dispatcher(
    pipelines: dict[str, Any],
    registry: AgentRegistry,
    endpoints: EndpointRegistry | None = None,
    ingestors: IngestionRegistry | None = None,
    plugins: PluginRegistry | None = None,
) -> Dispatcher:
    """Return a dispatcher that runs a pipeline as a task.

    Per-step progress is reported when the engine supports a progress
    callback; otherwise the task reports indeterminate progress.
    """

    async def run(target: str, payload: Any, ctx: TaskContext) -> Any:
        from agentomatic.pipelines.engine import PipelineEngine

        config = pipelines.get(target)
        if config is None:
            raise TargetNotFoundError(
                f"Pipeline '{target}' not found. Available: {sorted(pipelines)}"
            )
        sub = {name: cfg for name, cfg in pipelines.items() if name != target}
        engine = PipelineEngine(
            config,
            registry,
            sub_pipelines=sub,
            endpoints=endpoints,
            ingestors=ingestors,
            plugins=plugins,
        )
        errors = engine.validate()
        if errors:
            raise RuntimeError(f"Pipeline '{target}' invalid: {'; '.join(errors)}")

        total_steps = len(getattr(config, "steps", []) or [])
        await ctx.report(message=f"Running pipeline '{target}'", total=total_steps)
        result = await engine.run(payload if isinstance(payload, dict) else {})
        return _to_jsonable(result)

    return run
