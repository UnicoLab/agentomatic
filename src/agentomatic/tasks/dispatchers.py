"""Dispatchers that execute a single platform resource by name.

Each dispatcher resolves a resource (agent / plugin / pipeline / endpoint) from
its registry and runs it against one input payload, returning a JSON-safe
result. Dispatchers are intentionally single-input; the :class:`TaskManager`
handles batching, concurrency, cancellation, and progress on top of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from pydantic import ValidationError

from .manager import TaskInputValidationError

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

    Mirrors the synchronous invoke path so agents behave identically
    whether called sync or as a task. Preserves every top-level field.
    """
    from agentomatic.core.agent_invoke import build_invoke_state

    return build_invoke_state(payload)


def make_agent_dispatcher(registry: AgentRegistry) -> Dispatcher:
    """Return a dispatcher that runs a registered agent as a task.

    Uses :func:`~agentomatic.core.agent_invoke.invoke_registered_agent` so
    class agents (``BaseGraphAgent``) go through ``atransform`` /
    ``input_to_state`` — the same path as synchronous REST ``/invoke``.
    Preferring ``graph_fn().ainvoke(dict)`` would skip that conversion and
    break dataclass-typed states.
    """

    async def run(target: str, payload: Any, ctx: TaskContext) -> Any:
        from agentomatic.core.agent_invoke import invoke_registered_agent

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

        result = await invoke_registered_agent(agent, state)

        for hook in registry.after_node_hooks:
            try:
                hook(target, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"after_node hook error: {exc}")

        return _to_jsonable(result)

    return run


def make_agent_input_validator(
    registry: AgentRegistry,
) -> Callable[[str, Any], Any]:
    """Return the exact request validation used by an agent invoke route.

    The generic Task Board must not bypass an agent's published contract just
    because it queues execution through :class:`TaskManager`.  Prefer an
    agent's custom ``SchemaValidator`` and otherwise validate against the
    platform's standard ``AgentInvokeRequest`` model.
    """

    def validate(target: str, payload: Any) -> Any:
        from agentomatic.core.router_factory import AgentInvokeRequest

        agent = registry.get(target)
        if agent is None:
            raise TargetNotFoundError(
                f"Agent '{target}' not found. Available: {registry.list_names()}"
            )
        schema_validator = getattr(agent, "schema_validator", None)
        try:
            if schema_validator is not None and schema_validator.has_request_schema:
                # A custom RootModel owns its native scalar/array shape.  Do
                # not wrap it in the default chat ``query`` envelope.
                return schema_validator.validate_input(payload)
            data = payload if isinstance(payload, dict) else {"query": payload}
            return AgentInvokeRequest.model_validate(data).model_dump()
        except ValidationError as exc:
            raise TaskInputValidationError(f"Invalid input for agent '{target}': {exc}") from exc

    return validate


def _validate_model_input(
    *,
    resource_type: str,
    target: str,
    schema: Any,
    payload: Any,
) -> Any:
    """Validate a task payload with the exact Pydantic model served over HTTP.

    Dynamic plugin, endpoint, and ingestion routes expose their input models
    directly in FastAPI.  Reusing ``model_validate`` here keeps generic tasks
    from bypassing that public contract, including Pydantic root models.
    """
    try:
        validated = schema.model_validate(payload)
    except ValidationError as exc:
        raise TaskInputValidationError(
            f"Invalid input for {resource_type} '{target}': {exc}"
        ) from exc
    return validated.model_dump(mode="json")


def make_plugin_input_validator(registry: PluginRegistry) -> Callable[[str, Any], Any]:
    """Return the published request validator for a registered plugin."""

    def validate(target: str, payload: Any) -> Any:
        plugin = registry.get_plugin(target)
        if plugin is None:
            raise TargetNotFoundError(
                f"Plugin '{target}' not found. Available: {registry.list_names()}"
            )
        return _validate_model_input(
            resource_type="plugin",
            target=target,
            schema=plugin.get_input_schema(),
            payload=payload,
        )

    return validate


def make_endpoint_input_validator(registry: EndpointRegistry) -> Callable[[str, Any], Any]:
    """Return the published request validator for a registered endpoint."""

    def validate(target: str, payload: Any) -> Any:
        endpoint = registry.get(target)
        if endpoint is None:
            raise TargetNotFoundError(
                f"Endpoint '{target}' not found. Available: {registry.list_names()}"
            )
        return _validate_model_input(
            resource_type="endpoint",
            target=target,
            schema=endpoint.get_input_schema(),
            payload=payload,
        )

    return validate


def make_ingestion_input_validator(registry: IngestionRegistry) -> Callable[[str, Any], Any]:
    """Return the published request validator for a registered ingestor."""

    def validate(target: str, payload: Any) -> Any:
        ingestor = registry.get(target)
        if ingestor is None:
            raise TargetNotFoundError(
                f"Ingestor '{target}' not found. Available: {registry.list_names()}"
            )
        return _validate_model_input(
            resource_type="ingestor",
            target=target,
            schema=ingestor.get_input_schema(),
            payload=payload,
        )

    return validate


def make_pipeline_input_validator(
    pipelines: dict[str, Any],
) -> Callable[[str, Any], dict[str, Any]]:
    """Validate task input as a pipeline run would before it is queued.

    Pipeline schemas are intentionally advisory unless ``strict_schema`` is
    enabled.  Match that execution behaviour: every payload must be a JSON
    object, and only strict pipelines reject declared schema violations.
    """

    def validate(target: str, payload: Any) -> dict[str, Any]:
        from agentomatic.pipelines.validation import validate_against_schema

        config = pipelines.get(target)
        if config is None:
            raise TargetNotFoundError(
                f"Pipeline '{target}' not found. Available: {sorted(pipelines)}"
            )
        if not isinstance(payload, dict):
            raise TaskInputValidationError(
                f"Invalid input for pipeline '{target}': expected an object."
            )
        errors = validate_against_schema(payload, config.input_schema, label="input")
        if errors and config.strict_schema:
            raise TaskInputValidationError(
                f"Invalid input for pipeline '{target}': {'; '.join(errors)}"
            )
        return dict(payload)

    return validate


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
        try:
            inputs = input_schema.model_validate(payload)
        except Exception:  # noqa: BLE001 - fall back to raw payload
            inputs = payload
        await ctx.report(message=f"Running plugin '{target}'")
        result = await plugin.invoke(inputs)
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
        try:
            request = endpoint.get_input_schema().model_validate(payload)
        except Exception:  # noqa: BLE001 - preserve the programmatic task API fallback
            result = await endpoint.call(payload)
        else:
            result = await endpoint.handle(request)
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

    Threads the task context's progress reporter and a checkpoint hook
    through :meth:`PipelineEngine.run` so every step and every map item
    reports a percent + message that reaches SSE subscribers, and partial
    map-item results are persisted to the task record as they complete.
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

        partial: dict[str, dict[str, Any]] = {}

        async def progress_cb(
            *,
            current: int,
            total: int,
            message: str,
            stage: str,
            event: str,
        ) -> None:
            await ctx.report(
                message=message,
                current=current,
                total=total,
                stage=stage,
                event=event,
            )

        async def checkpoint_cb(step_name: str, index: int, sub_result: Any) -> None:
            partial.setdefault(step_name, {})[str(index)] = sub_result
            await ctx.report(
                message=f"Checkpoint '{step_name}' item {index}",
                stage=step_name,
                event="checkpoint",
                partial_step=step_name,
                partial_index=index,
                sub_result=sub_result,
            )

        input_payload = dict(payload) if isinstance(payload, dict) else {}
        completed_hint = input_payload.pop("__completed_indices", None)
        completed_indices: dict[str, set[int]] | None = None
        if isinstance(completed_hint, dict):
            completed_indices = {
                k: set(int(i) for i in v)
                for k, v in completed_hint.items()
                if isinstance(v, list | set | tuple)
            }

        result = await engine.run(
            input_payload,
            progress_cb=progress_cb,
            checkpoint_cb=checkpoint_cb,
            completed_indices=completed_indices,
        )
        jsonable = _to_jsonable(result)
        if isinstance(jsonable, dict) and partial:
            jsonable.setdefault("checkpoints", partial)
        return jsonable

    return run
