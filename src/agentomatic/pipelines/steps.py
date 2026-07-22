"""Pipeline step implementations.

Each step type is an async callable that takes a ``PipelineContext``,
a step configuration, and an ``AgentRegistry`` reference.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from .context import PipelineContext
from .models import (
    AgentStepConfig,
    EndpointStepConfig,
    ErrorPolicy,
    IngestionStepConfig,
    LoopStepConfig,
    MapStepConfig,
    ParallelStepConfig,
    ParallelStrategy,
    PluginStepConfig,
    StepResult,
    StepStatus,
    TransformStepConfig,
)

if TYPE_CHECKING:
    from agentomatic.core.registry import AgentRegistry
    from agentomatic.endpoints.registry import EndpointRegistry
    from agentomatic.ingestion.registry import IngestionRegistry
    from agentomatic.plugins.registry import PluginRegistry


# ---------------------------------------------------------------------------
# Agent Step
# ---------------------------------------------------------------------------


async def execute_agent_step(
    config: AgentStepConfig,
    ctx: PipelineContext,
    registry: AgentRegistry,
) -> StepResult:
    """Invoke a registered agent.

    Builds a ``BaseAgentState``-compatible dict from the resolved
    input mapping, invokes the agent via ``graph_fn`` or ``node_fn``,
    and returns a ``StepResult``.

    Args:
        config: Step configuration.
        ctx: Pipeline context for resolving ``$`` expressions.
        registry: The agent registry for agent lookup.

    Returns:
        Step result with output and timing.
    """
    t0 = time.perf_counter()
    agent_name = config.agent

    try:
        agent = registry.get(agent_name)
        if agent is None:
            return StepResult(
                name=config.name,
                status=StepStatus.FAILED,
                error=f"Agent '{agent_name}' not found in registry",
                agent_used=agent_name,
            )

        # Resolve input mapping → build state dict
        state = _build_agent_state(config, ctx)

        # Class agents need input_to_state — see invoke_registered_agent.
        from agentomatic.core.agent_invoke import invoke_registered_agent

        result = await asyncio.wait_for(
            invoke_registered_agent(agent, state),
            timeout=config.timeout,
        )

        duration = (time.perf_counter() - t0) * 1000
        output = _normalize_output(result)

        # Schema validation (advisory)
        if agent.schema_validator and agent.schema_validator.has_response_schema:
            agent.schema_validator.validate_output(output)

        await _record_step_invocation(
            resource_type="agent",
            resource_name=agent_name,
            step_name=config.name,
            input_data=state,
            output_data=output,
            duration_ms=duration,
            status="ok",
        )

        return StepResult(
            name=config.name,
            status=StepStatus.SUCCESS,
            output=output,
            duration_ms=duration,
            agent_used=agent_name,
        )

    except TimeoutError:
        duration = (time.perf_counter() - t0) * 1000
        await _record_step_invocation(
            resource_type="agent",
            resource_name=agent_name,
            step_name=config.name,
            input_data=None,
            error=f"Agent '{agent_name}' timed out after {config.timeout}s",
            duration_ms=duration,
            status="error",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=f"Agent '{agent_name}' timed out after {config.timeout}s",
            duration_ms=duration,
            agent_used=agent_name,
        )
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        logger.error(f"Agent step '{config.name}' failed: {exc}")
        await _record_step_invocation(
            resource_type="agent",
            resource_name=agent_name,
            step_name=config.name,
            input_data=None,
            error=str(exc),
            duration_ms=duration,
            status="error",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=str(exc),
            duration_ms=duration,
            agent_used=agent_name,
        )


async def _record_step_invocation(
    *,
    resource_type: str,
    resource_name: str,
    step_name: str,
    input_data: Any = None,
    output_data: Any = None,
    error: str | None = None,
    duration_ms: float | None = None,
    status: str = "ok",
) -> None:
    """Persist an in-process pipeline step invocation when logs_history is on."""
    from agentomatic.logs.helpers import record_invocation

    await record_invocation(
        resource_type=resource_type,
        resource_name=resource_name,
        endpoint="pipeline_step",
        input_data=input_data,
        output_data=output_data,
        metadata={"pipeline_step": step_name},
        error=error,
        duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
        status=status,
    )


def _build_agent_state(config: AgentStepConfig, ctx: PipelineContext) -> dict[str, Any]:
    """Build a BaseAgentState-compatible dict from step config + context."""
    state: dict[str, Any] = {}

    if config.input and config.input.mappings:
        # Explicit mapping provided
        resolved = ctx.resolve_mapping(config.input.mappings)
        state.update(resolved)
    else:
        # Auto-wiring: use current context's response as current_query
        if ctx.current and ctx.current.get("response"):
            state["current_query"] = ctx.current["response"]
        elif ctx.input.get("query"):
            state["current_query"] = ctx.input["query"]
        elif ctx.input.get("current_query"):
            state["current_query"] = ctx.input["current_query"]

    # Ensure current_query exists (agents expect it)
    if "current_query" not in state:
        # Try to find any string value in input
        for key in ("query", "current_query", "content", "text", "message"):
            if key in ctx.input:
                state["current_query"] = ctx.input[key]
                break

    # Copy pipeline input fields that agents commonly use
    for key in ("user_id", "thread_id", "metadata"):
        if key in ctx.input and key not in state:
            state[key] = ctx.input[key]

    return state


def _normalize_output(result: Any) -> dict[str, Any]:
    """Normalize agent output to a dict."""
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {"response": str(result)}


# ---------------------------------------------------------------------------
# Endpoint Step
# ---------------------------------------------------------------------------


async def execute_endpoint_step(
    config: EndpointStepConfig,
    ctx: PipelineContext,
    endpoints: EndpointRegistry | None,
) -> StepResult:
    """Invoke a registered custom endpoint.

    Resolves the input mapping into a payload, calls the endpoint (which
    usually fetches data from deployed model services), and returns the
    endpoint output as a ``StepResult`` for downstream steps/agents.

    Args:
        config: Endpoint step configuration.
        ctx: Pipeline context for resolving ``$`` expressions.
        endpoints: The endpoint registry for lookup.

    Returns:
        Step result with the endpoint's output.
    """
    t0 = time.perf_counter()
    name = config.endpoint

    if endpoints is None:
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error="No endpoint registry available to this pipeline",
        )

    endpoint = endpoints.get(name)
    if endpoint is None:
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=f"Endpoint '{name}' not found in registry",
        )

    try:
        if config.input and config.input.mappings:
            payload = ctx.resolve_mapping(config.input.mappings)
        else:
            payload = dict(ctx.current) if ctx.current else dict(ctx.input)

        kwargs: dict[str, Any] = {}
        if config.upstreams:
            kwargs["upstreams"] = config.upstreams

        result = await asyncio.wait_for(
            endpoint.call(payload, **kwargs),
            timeout=config.timeout,
        )
        output = _normalize_output(result)
        duration = (time.perf_counter() - t0) * 1000
        await _record_step_invocation(
            resource_type="endpoint",
            resource_name=name,
            step_name=config.name,
            input_data=payload,
            output_data=output,
            duration_ms=duration,
            status="ok",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.SUCCESS,
            output=output,
            duration_ms=duration,
            metadata={"endpoint": name},
        )
    except TimeoutError:
        duration = (time.perf_counter() - t0) * 1000
        await _record_step_invocation(
            resource_type="endpoint",
            resource_name=name,
            step_name=config.name,
            error=f"Endpoint '{name}' timed out after {config.timeout}s",
            duration_ms=duration,
            status="error",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=f"Endpoint '{name}' timed out after {config.timeout}s",
            duration_ms=duration,
        )
    except Exception as exc:  # noqa: BLE001
        duration = (time.perf_counter() - t0) * 1000
        logger.error(f"Endpoint step '{config.name}' failed: {exc}")
        await _record_step_invocation(
            resource_type="endpoint",
            resource_name=name,
            step_name=config.name,
            error=str(exc),
            duration_ms=duration,
            status="error",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=str(exc),
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Plugin Step
# ---------------------------------------------------------------------------


async def execute_plugin_step(
    config: PluginStepConfig,
    ctx: PipelineContext,
    plugins: PluginRegistry | None,
) -> StepResult:
    """Invoke a registered ML plugin's ``predict``.

    Resolves the input mapping into a payload, coerces it into the plugin's
    declared input schema, runs ``predict``, and stores the (normalized)
    prediction so downstream steps/agents can consume it.

    Args:
        config: Plugin step configuration.
        ctx: Pipeline context for resolving ``$`` expressions.
        plugins: The plugin registry for lookup.

    Returns:
        Step result with the plugin's prediction output.
    """
    t0 = time.perf_counter()
    name = config.plugin

    if plugins is None:
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error="No plugin registry available to this pipeline",
        )

    plugin = plugins.get_plugin(name)
    if plugin is None:
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=f"Plugin '{name}' not found in registry",
        )

    try:
        if config.input and config.input.mappings:
            payload = ctx.resolve_mapping(config.input.mappings)
        else:
            payload = dict(ctx.current) if ctx.current else dict(ctx.input)

        # Coerce the raw payload into the plugin's declared input schema.
        input_schema = plugin.get_input_schema()
        model_input: Any
        try:
            model_input = input_schema.model_validate(payload)
        except Exception:  # noqa: BLE001 - fall back to raw payload
            model_input = payload

        result = await asyncio.wait_for(
            plugin.predict(model_input),
            timeout=config.timeout,
        )
        output = _normalize_output(result)
        duration = (time.perf_counter() - t0) * 1000
        await _record_step_invocation(
            resource_type="plugin",
            resource_name=name,
            step_name=config.name,
            input_data=payload,
            output_data=output,
            duration_ms=duration,
            status="ok",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.SUCCESS,
            output=output,
            duration_ms=duration,
            metadata={"plugin": name},
        )
    except TimeoutError:
        duration = (time.perf_counter() - t0) * 1000
        await _record_step_invocation(
            resource_type="plugin",
            resource_name=name,
            step_name=config.name,
            error=f"Plugin '{name}' timed out after {config.timeout}s",
            duration_ms=duration,
            status="error",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=f"Plugin '{name}' timed out after {config.timeout}s",
            duration_ms=duration,
        )
    except Exception as exc:  # noqa: BLE001
        duration = (time.perf_counter() - t0) * 1000
        logger.error(f"Plugin step '{config.name}' failed: {exc}")
        await _record_step_invocation(
            resource_type="plugin",
            resource_name=name,
            step_name=config.name,
            error=str(exc),
            duration_ms=duration,
            status="error",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=str(exc),
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Ingestion Step
# ---------------------------------------------------------------------------


async def execute_ingestion_step(
    config: IngestionStepConfig,
    ctx: PipelineContext,
    ingestors: IngestionRegistry | None,
) -> StepResult:
    """Run a registered ingestor as a pipeline step.

    Resolves the input mapping into a payload, runs the ingestor, and stores
    its :class:`IngestionResult` (as a dict) so downstream steps can consume it.

    Args:
        config: Ingestion step configuration.
        ctx: Pipeline context for resolving ``$`` expressions.
        ingestors: The ingestion registry for lookup.

    Returns:
        Step result with the ingestor's output.
    """
    t0 = time.perf_counter()
    name = config.ingestor

    if ingestors is None:
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error="No ingestion registry available to this pipeline",
        )

    ingestor = ingestors.get(name)
    if ingestor is None:
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=f"Ingestor '{name}' not found in registry",
        )

    try:
        from agentomatic.ingestion.context import NullIngestionContext

        if config.input and config.input.mappings:
            payload = ctx.resolve_mapping(config.input.mappings)
        else:
            payload = dict(ctx.current) if ctx.current else dict(ctx.input)

        result = await asyncio.wait_for(
            ingestor.run(payload, NullIngestionContext()),
            timeout=config.timeout,
        )
        output = _normalize_output(result)
        duration = (time.perf_counter() - t0) * 1000
        await _record_step_invocation(
            resource_type="ingestion",
            resource_name=name,
            step_name=config.name,
            input_data=payload,
            output_data=output,
            duration_ms=duration,
            status="ok",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.SUCCESS,
            output=output,
            duration_ms=duration,
            metadata={"ingestor": name},
        )
    except TimeoutError:
        duration = (time.perf_counter() - t0) * 1000
        await _record_step_invocation(
            resource_type="ingestion",
            resource_name=name,
            step_name=config.name,
            error=f"Ingestor '{name}' timed out after {config.timeout}s",
            duration_ms=duration,
            status="error",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=f"Ingestor '{name}' timed out after {config.timeout}s",
            duration_ms=duration,
        )
    except Exception as exc:  # noqa: BLE001
        duration = (time.perf_counter() - t0) * 1000
        logger.error(f"Ingestion step '{config.name}' failed: {exc}")
        await _record_step_invocation(
            resource_type="ingestion",
            resource_name=name,
            step_name=config.name,
            error=str(exc),
            duration_ms=duration,
            status="error",
        )
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=str(exc),
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Transform Step
# ---------------------------------------------------------------------------


async def execute_transform_step(
    config: TransformStepConfig,
    ctx: PipelineContext,
) -> StepResult:
    """Execute a Python transform code block.

    The code runs with ``ctx`` in scope and must produce a dict
    via a ``return`` statement.

    Args:
        config: Transform step configuration.
        ctx: Pipeline context.

    Returns:
        Step result with transformed output.
    """
    t0 = time.perf_counter()

    try:
        # Build a restricted namespace
        namespace = ctx.to_eval_namespace()

        # Wrap the code in a function to support 'return'
        wrapped = "async def _transform(ctx):\n"
        for line in config.code.strip().splitlines():
            wrapped += f"    {line}\n"

        exec(wrapped, namespace)  # noqa: S102
        transform_fn = namespace["_transform"]
        result = await asyncio.wait_for(
            transform_fn(ctx),
            timeout=config.timeout,
        )

        if not isinstance(result, dict):
            result = {"result": result}

        duration = (time.perf_counter() - t0) * 1000
        return StepResult(
            name=config.name,
            status=StepStatus.SUCCESS,
            output=result,
            duration_ms=duration,
        )

    except TimeoutError:
        duration = (time.perf_counter() - t0) * 1000
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=f"Transform '{config.name}' timed out after {config.timeout}s",
            duration_ms=duration,
        )
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        logger.error(f"Transform step '{config.name}' failed: {exc}")
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=str(exc),
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Parallel Step
# ---------------------------------------------------------------------------


async def execute_parallel_step(
    config: ParallelStepConfig,
    ctx: PipelineContext,
    registry: AgentRegistry,
) -> StepResult:
    """Execute multiple agent steps in parallel.

    Supports three strategies:
    - ``all``: Wait for all steps, collect all results.
    - ``first``: Return first successful result.
    - ``majority``: Wait for >50% success.

    Args:
        config: Parallel step configuration.
        ctx: Pipeline context.
        registry: Agent registry.

    Returns:
        Step result with sub_results list.
    """
    t0 = time.perf_counter()

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(config.max_concurrency)

    async def _run_with_sem(step_cfg: AgentStepConfig) -> StepResult:
        async with sem:
            return await execute_agent_step(step_cfg, ctx, registry)

    try:
        tasks = [_run_with_sem(step_cfg) for step_cfg in config.steps]

        if config.strategy.value == "first":
            # Return first successful
            sub_results: list[StepResult] = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                sub_results.append(result)
                if result.status == StepStatus.SUCCESS:
                    break
        else:
            # all / majority: gather everything
            sub_results = list(await asyncio.gather(*tasks))

        duration = (time.perf_counter() - t0) * 1000

        # Determine overall status
        successes = sum(1 for r in sub_results if r.status == StepStatus.SUCCESS)
        total = len(config.steps)

        if config.strategy.value == "all":
            status = StepStatus.SUCCESS if successes == total else StepStatus.FAILED
        elif config.strategy.value == "first":
            status = StepStatus.SUCCESS if successes >= 1 else StepStatus.FAILED
        else:  # majority
            status = StepStatus.SUCCESS if successes > total / 2 else StepStatus.FAILED

        # Merge outputs from all successful sub-results
        merged: dict[str, Any] = {}
        for sr in sub_results:
            if sr.status == StepStatus.SUCCESS:
                merged.update(sr.output)

        return StepResult(
            name=config.name,
            status=status,
            output=merged,
            sub_results=sub_results,
            duration_ms=duration,
        )

    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        logger.error(f"Parallel step '{config.name}' failed: {exc}")
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=str(exc),
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Map Step (dynamic fan-out over a runtime list)
# ---------------------------------------------------------------------------


async def execute_map_step(
    config: MapStepConfig,
    ctx: PipelineContext,
    registry: AgentRegistry,
    *,
    progress_cb: Any | None = None,
    checkpoint_cb: Any | None = None,
) -> StepResult:
    """Fan out one agent over every element of a runtime list.

    The items list is resolved from ``config.items`` against ``ctx``
    (e.g. ``$.input.scopes``).  Each element is passed to the same agent
    under the configured ``item_key`` (and its index under ``index_key``).
    Any additional ``config.input`` mappings are merged into the per-item
    agent state.  Concurrency is capped by ``config.max_concurrency`` and
    each item can retry independently via ``config.retry``.

    Results are aggregated into a keyed fan-in dict so no keys are silently
    overwritten by :meth:`dict.update`::

        {
            "items": [<result_0>, <result_1>, ...],
            "by_key": {"0": <result_0>, ...},
            "count":  <n>,
            "succeeded": <k>,
        }

    Args:
        config: Map step configuration.
        ctx: Pipeline context (used to resolve ``items`` and mappings).
        registry: Agent registry (must contain ``config.agent``).
        progress_cb: Optional ``async def cb(current, total, message)``
            invoked whenever an item completes.
        checkpoint_cb: Optional ``async def cb(index, sub_result)``
            invoked as soon as each item finishes so callers can persist
            partial progress (used by task-level checkpointing).

    Returns:
        A :class:`StepResult` with ``sub_results`` for each item and a keyed
        ``output`` dictionary preserving order and per-index results.
    """
    t0 = time.perf_counter()

    raw_items = ctx.resolve(config.items)
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=(
                f"Map step '{config.name}': items expression '{config.items}' "
                f"did not resolve to a list (got {type(raw_items).__name__})."
            ),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    total = len(raw_items)
    if total == 0:
        return StepResult(
            name=config.name,
            status=StepStatus.SUCCESS,
            output={"items": [], "by_key": {}, "count": 0, "succeeded": 0},
            sub_results=[],
            duration_ms=(time.perf_counter() - t0) * 1000,
            metadata={"map_items": 0},
        )

    sem = asyncio.Semaphore(config.max_concurrency)

    extra_mappings = dict(config.input.mappings) if config.input else {}
    resolved_extras = ctx.resolve_mapping(extra_mappings) if extra_mappings else {}

    async def _run_one(index: int, item: Any) -> StepResult:
        """Run the single agent against one item, honouring retry."""
        step_name = f"{config.name}_item_{index}"

        input_mappings: dict[str, Any] = {}
        input_mappings[config.item_key] = item
        input_mappings[config.index_key] = index
        input_mappings.setdefault("current_query", item)
        for key, value in resolved_extras.items():
            input_mappings[key] = value

        sub_config = AgentStepConfig(
            name=step_name,
            agent=config.agent,
            input=type(config.input)(mappings=input_mappings),
            on_error=config.on_error,
            fallback_agent=config.fallback_agent,
            retry=config.retry,
            timeout=config.item_timeout,
        )

        async with sem:
            if config.retry:
                sub_result = await execute_with_retry(
                    execute_agent_step,
                    sub_config,
                    ctx,
                    registry,
                    retry_config=config.retry,
                    on_error=config.on_error,
                    fallback_agent=config.fallback_agent,
                    registry=registry,
                    ctx=ctx,
                    step_config=sub_config,
                )
            else:
                sub_result = await execute_agent_step(sub_config, ctx, registry)

        sub_result.metadata["map_index"] = index
        return sub_result

    try:
        pending = {asyncio.create_task(_run_one(i, item)): i for i, item in enumerate(raw_items)}
        sub_results: list[StepResult | None] = [None] * total
        completed = 0

        try:
            while pending:
                done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    index = pending.pop(task)
                    try:
                        sub_result = task.result()
                    except Exception as exc:  # noqa: BLE001 - map errors, keep others
                        sub_result = StepResult(
                            name=f"{config.name}_item_{index}",
                            status=StepStatus.FAILED,
                            error=str(exc),
                        )
                    sub_results[index] = sub_result
                    completed += 1

                    if progress_cb is not None:
                        try:
                            await progress_cb(
                                completed,
                                total,
                                f"Map '{config.name}': {completed}/{total} items done",
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(f"map progress_cb failed: {exc}")

                    if checkpoint_cb is not None:
                        try:
                            await checkpoint_cb(index, sub_result)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(f"map checkpoint_cb failed: {exc}")
        finally:
            for task in pending:
                task.cancel()

        finalized: list[StepResult] = [r for r in sub_results if r is not None]
        successes = sum(1 for r in finalized if r.status == StepStatus.SUCCESS)

        if config.strategy == ParallelStrategy.ALL:
            status = StepStatus.SUCCESS if successes == total else StepStatus.FAILED
        elif config.strategy == ParallelStrategy.FIRST:
            status = StepStatus.SUCCESS if successes >= 1 else StepStatus.FAILED
        else:  # majority
            status = StepStatus.SUCCESS if successes > total / 2 else StepStatus.FAILED

        # Keyed fan-in — no dict.update() overwriting.
        by_key: dict[str, dict[str, Any]] = {}
        items_out: list[dict[str, Any]] = []
        for i, sr in enumerate(finalized):
            entry: dict[str, Any] = {
                "index": i,
                "status": sr.status.value,
                "output": dict(sr.output),
                "error": sr.error,
                "duration_ms": sr.duration_ms,
                "retries": sr.retries,
            }
            items_out.append(entry)
            by_key[str(i)] = entry

        duration = (time.perf_counter() - t0) * 1000
        return StepResult(
            name=config.name,
            status=status,
            output={
                "items": items_out,
                "by_key": by_key,
                "count": total,
                "succeeded": successes,
            },
            sub_results=finalized,
            duration_ms=duration,
            metadata={
                "map_items": total,
                "map_succeeded": successes,
                "map_concurrency": config.max_concurrency,
            },
        )
    except Exception as exc:  # noqa: BLE001
        duration = (time.perf_counter() - t0) * 1000
        logger.error(f"Map step '{config.name}' failed: {exc}")
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=str(exc),
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Loop Step
# ---------------------------------------------------------------------------


async def execute_loop_step(
    config: LoopStepConfig,
    ctx: PipelineContext,
    registry: AgentRegistry,
) -> StepResult:
    """Execute a step iteratively until a condition is met.

    Args:
        config: Loop step configuration.
        ctx: Pipeline context.
        registry: Agent registry.

    Returns:
        Step result with iterations list.
    """
    t0 = time.perf_counter()
    iterations: list[StepResult] = []

    try:
        for i in range(config.max_iterations):
            # Execute the inner step
            iter_result = await execute_agent_step(config.step, ctx, registry)
            iter_result.name = f"{config.name}_iter_{i}"
            iterations.append(iter_result)

            if iter_result.status != StepStatus.SUCCESS:
                if config.on_error == ErrorPolicy.SKIP:
                    break
                elif config.on_error == ErrorPolicy.FAIL:
                    break

            # Update context.current for condition evaluation
            ctx.current = dict(iter_result.output)

            # Check until condition
            if config.until:
                try:
                    ns = ctx.to_eval_namespace()
                    if eval(config.until, {"__builtins__": {}}, ns):  # noqa: S307
                        break
                except Exception as eval_exc:
                    logger.warning(f"Loop condition eval failed: {eval_exc}")

        duration = (time.perf_counter() - t0) * 1000

        # Final output is from last successful iteration
        last_success = None
        for it in reversed(iterations):
            if it.status == StepStatus.SUCCESS:
                last_success = it
                break

        return StepResult(
            name=config.name,
            status=StepStatus.SUCCESS if last_success else StepStatus.FAILED,
            output=last_success.output if last_success else {},
            iterations=iterations,
            duration_ms=duration,
            metadata={"total_iterations": len(iterations)},
        )

    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        logger.error(f"Loop step '{config.name}' failed: {exc}")
        return StepResult(
            name=config.name,
            status=StepStatus.FAILED,
            error=str(exc),
            iterations=iterations,
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


async def execute_with_retry(
    step_fn,  # noqa: ANN001
    *args: Any,
    retry_config: Any | None = None,
    on_error: ErrorPolicy = ErrorPolicy.FAIL,
    fallback_agent: str | None = None,
    registry: AgentRegistry | None = None,
    ctx: PipelineContext | None = None,
    step_config: AgentStepConfig | None = None,
) -> StepResult:
    """Execute a step function with retry and fallback logic.

    Args:
        step_fn: The step execution function.
        *args: Arguments to pass to step_fn.
        retry_config: Retry configuration.
        on_error: Error policy.
        fallback_agent: Agent to use on failure.
        registry: Agent registry (for fallback).
        ctx: Pipeline context (for fallback).
        step_config: Original step config (for fallback).

    Returns:
        Step result (possibly from retry or fallback).
    """
    max_attempts = 1
    base_delay = 1.0
    backoff = "exponential"

    if retry_config:
        if hasattr(retry_config, "max_attempts"):
            max_attempts = retry_config.max_attempts
            base_delay = retry_config.base_delay
            backoff = retry_config.backoff
        elif isinstance(retry_config, dict):
            max_attempts = retry_config.get("max_attempts", 3)
            base_delay = retry_config.get("base_delay", 1.0)
            backoff = retry_config.get("backoff", "exponential")

    last_result = None
    for attempt in range(max_attempts):
        result = await step_fn(*args)
        last_result = result

        if result.status == StepStatus.SUCCESS:
            result.retries = attempt
            return result

        if attempt < max_attempts - 1:
            if backoff == "exponential":
                delay = base_delay * (2**attempt)
            elif backoff == "linear":
                delay = base_delay * (attempt + 1)
            else:
                delay = base_delay
            logger.info(
                f"Step '{result.name}' failed (attempt {attempt + 1}/"
                f"{max_attempts}), retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    # All retries exhausted
    if last_result:
        last_result.retries = max_attempts - 1

    # Try fallback agent
    if fallback_agent and on_error == ErrorPolicy.FALLBACK and registry and ctx and step_config:
        logger.info(f"Using fallback agent '{fallback_agent}' for step '{step_config.name}'")
        fallback_config = step_config.model_copy(update={"agent": fallback_agent})
        fallback_result = await execute_agent_step(fallback_config, ctx, registry)
        fallback_result.metadata["fallback_from"] = step_config.agent
        return fallback_result

    return last_result or StepResult(
        name="unknown",
        status=StepStatus.FAILED,
        error="No result produced",
    )
