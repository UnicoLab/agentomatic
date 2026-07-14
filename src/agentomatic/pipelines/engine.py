"""Pipeline execution engine.

Orchestrates step execution, manages context flow, handles errors,
and produces structured results.
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
    ParallelStepConfig,
    PipelineConfig,
    PipelineResult,
    PipelineStatus,
    PluginStepConfig,
    StepResult,
    StepStatus,
    SubPipelineStepConfig,
    TransformStepConfig,
)
from .steps import (
    execute_agent_step,
    execute_endpoint_step,
    execute_ingestion_step,
    execute_loop_step,
    execute_parallel_step,
    execute_plugin_step,
    execute_transform_step,
    execute_with_retry,
)
from .validation import validate_against_schema

if TYPE_CHECKING:
    from agentomatic.core.registry import AgentRegistry
    from agentomatic.endpoints.registry import EndpointRegistry
    from agentomatic.ingestion.registry import IngestionRegistry
    from agentomatic.plugins.registry import PluginRegistry


class PipelineEngine:
    """Execute a pipeline configuration against a registry of agents.

    The engine validates the pipeline, resolves agent references,
    executes steps in order, and returns a structured result.

    Args:
        config: The pipeline configuration.
        registry: The agent registry for resolving agent names.
        sub_pipelines: Optional dict of named sub-pipelines for
            ``sub_pipeline`` steps.

    Example::

        engine = PipelineEngine(config, registry)
        errors = engine.validate()
        if not errors:
            result = await engine.run({"query": "hello"})
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: AgentRegistry,
        sub_pipelines: dict[str, PipelineConfig] | None = None,
        endpoints: EndpointRegistry | None = None,
        ingestors: IngestionRegistry | None = None,
        plugins: PluginRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.sub_pipelines = sub_pipelines or {}
        self.endpoints = endpoints
        self.ingestors = ingestors
        self.plugins = plugins

    def validate(self) -> list[str]:
        """Pre-flight validation of the pipeline.

        Checks:
        - All referenced agents exist in the registry.
        - Step names are unique.
        - Sub-pipeline references are valid.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []

        # Check unique step names
        names = [s.name for s in self.config.steps]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                errors.append(f"Duplicate step name: '{name}'")
            seen.add(name)

        # Check agent references
        required_agents = self.config.get_agent_names()
        for agent_name in required_agents:
            if self.registry.get(agent_name) is None:
                errors.append(
                    f"Agent '{agent_name}' not found in registry. "
                    f"Available: {self.registry.list_names()}"
                )

        # Check sub-pipeline references
        for step in self.config.steps:
            if isinstance(step, SubPipelineStepConfig):
                if step.pipeline not in self.sub_pipelines:
                    errors.append(f"Sub-pipeline '{step.pipeline}' not found")

        # Check plugin references
        required_plugins = self.config.get_plugin_names()
        if required_plugins:
            available_plugins = set(self.plugins.list_names()) if self.plugins else set()
            for plugin_name in required_plugins:
                if plugin_name not in available_plugins:
                    errors.append(
                        f"Plugin '{plugin_name}' not found in registry. "
                        f"Available: {sorted(available_plugins)}"
                    )

        # Check endpoint references
        required_endpoints = self.config.get_endpoint_names()
        if required_endpoints:
            available = set(self.endpoints.list_names()) if self.endpoints else set()
            for endpoint_name in required_endpoints:
                if endpoint_name not in available:
                    errors.append(
                        f"Endpoint '{endpoint_name}' not found in registry. "
                        f"Available: {sorted(available)}"
                    )

        # Check ingestor references
        required_ingestors = self.config.get_ingestor_names()
        if required_ingestors:
            available_ing = set(self.ingestors.list_names()) if self.ingestors else set()
            for ingestor_name in required_ingestors:
                if ingestor_name not in available_ing:
                    errors.append(
                        f"Ingestor '{ingestor_name}' not found in registry. "
                        f"Available: {sorted(available_ing)}"
                    )

        return errors

    async def run(
        self,
        input_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Execute the pipeline.

        Args:
            input_data: Initial input data for the pipeline.
            **kwargs: Additional keyword arguments merged into input.

        Returns:
            A ``PipelineResult`` with per-step details.
        """
        t0 = time.perf_counter()
        merged_input = dict(input_data or {})
        merged_input.update(kwargs)

        pipeline_result = PipelineResult(
            pipeline_name=self.config.name,
            status=PipelineStatus.RUNNING,
            started_at=t0,
        )

        # Create context
        ctx = PipelineContext(
            input_data=merged_input,
            defaults=self.config.defaults,
        )

        logger.info(f"🚀 Pipeline '{self.config.name}' starting ({len(self.config.steps)} steps)")

        # Input-schema enforcement (advisory unless strict_schema=True)
        input_errors = validate_against_schema(
            merged_input, self.config.input_schema, label="input"
        )
        if input_errors:
            if self.config.strict_schema:
                pipeline_result.status = PipelineStatus.FAILED
                pipeline_result.error = "; ".join(input_errors)
                pipeline_result.duration_ms = (time.perf_counter() - t0) * 1000
                logger.error(f"❌ Pipeline '{self.config.name}' input invalid: {input_errors}")
                return pipeline_result
            for err in input_errors:
                logger.warning(f"  ⚠️ schema: {err}")

        try:
            await asyncio.wait_for(
                self._execute_steps(ctx, pipeline_result),
                timeout=self.config.timeout,
            )
        except TimeoutError:
            pipeline_result.status = PipelineStatus.FAILED
            pipeline_result.error = f"Pipeline timed out after {self.config.timeout}s"
            logger.error(pipeline_result.error)

        # Calculate final output
        duration = (time.perf_counter() - t0) * 1000
        pipeline_result.duration_ms = duration

        if pipeline_result.status == PipelineStatus.RUNNING:
            # Determine final status
            failed = sum(
                1 for s in pipeline_result.steps.values() if s.status == StepStatus.FAILED
            )
            if failed == 0:
                pipeline_result.status = PipelineStatus.SUCCESS
            elif failed < len(pipeline_result.steps):
                pipeline_result.status = PipelineStatus.PARTIAL
            else:
                pipeline_result.status = PipelineStatus.FAILED

        # Build final output from last successful step
        pipeline_result.output = self._build_final_output(ctx, pipeline_result)

        # Output-schema enforcement (skip if the pipeline already failed)
        if pipeline_result.status != PipelineStatus.FAILED:
            output_errors = validate_against_schema(
                pipeline_result.output, self.config.output_schema, label="output"
            )
            if output_errors:
                if self.config.strict_schema:
                    pipeline_result.status = PipelineStatus.FAILED
                    pipeline_result.error = "; ".join(output_errors)
                    logger.error(
                        f"❌ Pipeline '{self.config.name}' output invalid: {output_errors}"
                    )
                else:
                    for err in output_errors:
                        logger.warning(f"  ⚠️ schema: {err}")

        status_emoji = {
            PipelineStatus.SUCCESS: "✅",
            PipelineStatus.PARTIAL: "⚠️",
            PipelineStatus.FAILED: "❌",
        }
        logger.info(
            f"{status_emoji.get(pipeline_result.status, '❓')} Pipeline "
            f"'{self.config.name}' completed in {duration:.0f}ms "
            f"({pipeline_result.status.value})"
        )

        return pipeline_result

    async def _execute_steps(
        self,
        ctx: PipelineContext,
        pipeline_result: PipelineResult,
    ) -> None:
        """Execute pipeline steps sequentially."""
        for step_config in self.config.steps:
            # Evaluate condition if present
            condition = getattr(step_config, "condition", None)
            if condition and not self._evaluate_condition(condition, ctx):
                logger.info(f"  ⏭️ Skipping '{step_config.name}' (condition not met)")
                pipeline_result.steps[step_config.name] = StepResult(
                    name=step_config.name,
                    status=StepStatus.SKIPPED,
                )
                continue

            logger.info(f"  ▶️ Executing step '{step_config.name}'")

            # Execute based on step type
            result = await self._execute_single_step(step_config, ctx)

            # Store result in context and pipeline result
            ctx.set_step_result(step_config.name, result)
            pipeline_result.steps[step_config.name] = result

            # Apply output mapping if present
            output_map = getattr(step_config, "output", None)
            if output_map and hasattr(output_map, "mappings") and output_map.mappings:
                self._apply_output_mapping(output_map.mappings, result, ctx)

            # Handle failure
            if result.status == StepStatus.FAILED:
                on_error = getattr(step_config, "on_error", ErrorPolicy.FAIL)
                if on_error == ErrorPolicy.SKIP:
                    logger.warning(
                        f"  ⚠️ Step '{step_config.name}' failed (skipped): {result.error}"
                    )
                    result.status = StepStatus.SKIPPED
                elif on_error == ErrorPolicy.FAIL:
                    logger.error(f"  ❌ Step '{step_config.name}' failed: {result.error}")
                    if self.config.on_error == "rollback":
                        pipeline_result.status = PipelineStatus.FAILED
                        pipeline_result.error = f"Step '{step_config.name}' failed: {result.error}"
                        await self._run_rollbacks(ctx, pipeline_result)
                        return
                    if self.config.on_error == "fail_fast":
                        pipeline_result.status = PipelineStatus.FAILED
                        pipeline_result.error = f"Step '{step_config.name}' failed: {result.error}"
                        return
                else:
                    logger.warning(f"  ⚠️ Step '{step_config.name}' failed: {result.error}")
            else:
                logger.info(
                    f"  ✅ Step '{step_config.name}' completed in {result.duration_ms:.0f}ms"
                )

    async def _execute_single_step(
        self,
        step_config: Any,
        ctx: PipelineContext,
    ) -> StepResult:
        """Dispatch to the correct step executor."""
        if isinstance(step_config, AgentStepConfig):
            retry = step_config.retry
            if retry:
                return await execute_with_retry(
                    execute_agent_step,
                    step_config,
                    ctx,
                    self.registry,
                    retry_config=retry,
                    on_error=step_config.on_error,
                    fallback_agent=step_config.fallback_agent,
                    registry=self.registry,
                    ctx=ctx,
                    step_config=step_config,
                )
            return await execute_agent_step(step_config, ctx, self.registry)

        elif isinstance(step_config, PluginStepConfig):
            if step_config.retry:
                return await execute_with_retry(
                    execute_plugin_step,
                    step_config,
                    ctx,
                    self.plugins,
                    retry_config=step_config.retry,
                    on_error=step_config.on_error,
                )
            return await execute_plugin_step(step_config, ctx, self.plugins)

        elif isinstance(step_config, EndpointStepConfig):
            if step_config.retry:
                return await execute_with_retry(
                    execute_endpoint_step,
                    step_config,
                    ctx,
                    self.endpoints,
                    retry_config=step_config.retry,
                    on_error=step_config.on_error,
                )
            return await execute_endpoint_step(step_config, ctx, self.endpoints)

        elif isinstance(step_config, IngestionStepConfig):
            if step_config.retry:
                return await execute_with_retry(
                    execute_ingestion_step,
                    step_config,
                    ctx,
                    self.ingestors,
                    retry_config=step_config.retry,
                    on_error=step_config.on_error,
                )
            return await execute_ingestion_step(step_config, ctx, self.ingestors)

        elif isinstance(step_config, TransformStepConfig):
            return await execute_transform_step(step_config, ctx)

        elif isinstance(step_config, ParallelStepConfig):
            return await execute_parallel_step(step_config, ctx, self.registry)

        elif isinstance(step_config, LoopStepConfig):
            return await execute_loop_step(step_config, ctx, self.registry)

        elif isinstance(step_config, SubPipelineStepConfig):
            return await self._execute_sub_pipeline(step_config, ctx)

        else:
            return StepResult(
                name=getattr(step_config, "name", "unknown"),
                status=StepStatus.FAILED,
                error=f"Unknown step type: {type(step_config).__name__}",
            )

    async def _execute_sub_pipeline(
        self,
        config: SubPipelineStepConfig,
        ctx: PipelineContext,
    ) -> StepResult:
        """Execute a nested sub-pipeline."""
        t0 = time.perf_counter()

        sub_config = self.sub_pipelines.get(config.pipeline)
        if sub_config is None:
            return StepResult(
                name=config.name,
                status=StepStatus.FAILED,
                error=f"Sub-pipeline '{config.pipeline}' not found",
            )

        # Resolve input mapping
        sub_input: dict[str, Any] = {}
        if config.input and config.input.mappings:
            sub_input = ctx.resolve_mapping(config.input.mappings)
        else:
            sub_input = dict(ctx.current)

        # Create and run sub-engine
        sub_engine = PipelineEngine(
            sub_config,
            self.registry,
            self.sub_pipelines,
            endpoints=self.endpoints,
            ingestors=self.ingestors,
            plugins=self.plugins,
        )
        sub_result = await asyncio.wait_for(
            sub_engine.run(sub_input),
            timeout=config.timeout,
        )

        duration = (time.perf_counter() - t0) * 1000
        return StepResult(
            name=config.name,
            status=(StepStatus.SUCCESS if sub_result.succeeded else StepStatus.FAILED),
            output=sub_result.output,
            duration_ms=duration,
            metadata={"sub_pipeline": config.pipeline},
        )

    async def _run_rollbacks(
        self,
        ctx: PipelineContext,
        pipeline_result: PipelineResult,
    ) -> None:
        """Run compensation code for completed steps in reverse order.

        Only invoked when the pipeline's ``on_error`` policy is ``rollback``.
        Each step config may define a ``rollback`` Python block executed with
        ``ctx`` and ``output`` (the step's output) in scope.  Rollback failures
        are logged but never mask the original error.
        """
        succeeded = [
            name for name, res in pipeline_result.steps.items() if res.status == StepStatus.SUCCESS
        ]
        if not succeeded:
            return

        logger.warning(f"↩️  Rolling back {len(succeeded)} completed step(s)")
        compensated: list[str] = []

        for name in reversed(succeeded):
            step_config = self.config.get_step(name)
            code = getattr(step_config, "rollback", None)
            if not code:
                continue
            try:
                namespace = ctx.to_eval_namespace()
                namespace["output"] = dict(pipeline_result.steps[name].output)
                wrapped = "async def _rollback(ctx, output):\n"
                for line in code.strip().splitlines():
                    wrapped += f"    {line}\n"
                exec(wrapped, namespace)  # noqa: S102
                await namespace["_rollback"](ctx, namespace["output"])
                compensated.append(name)
                pipeline_result.steps[name].metadata["rolled_back"] = True
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Rollback for step '{name}' failed: {exc}")

        pipeline_result.metadata["rolled_back_steps"] = compensated

    def _evaluate_condition(self, condition: str, ctx: PipelineContext) -> bool:
        """Safely evaluate a condition expression."""
        try:
            ns = ctx.to_eval_namespace()
            return bool(eval(condition, {"__builtins__": {}}, ns))  # noqa: S307
        except Exception as exc:
            logger.warning(f"Condition eval failed: {exc}")
            return False

    def _apply_output_mapping(
        self,
        mapping: dict[str, Any],
        result: StepResult,
        ctx: PipelineContext,
    ) -> None:
        """Apply output mapping to store named values in context."""
        for target_key, source_expr in mapping.items():
            if isinstance(source_expr, str) and source_expr.startswith("$"):
                # Resolve from the step's output
                path = source_expr.lstrip("$").lstrip(".")
                parts = path.split(".")
                value: Any = result.output
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = None
                        break
                ctx.shared[target_key] = value
            else:
                ctx.shared[target_key] = source_expr

    def _build_final_output(
        self,
        ctx: PipelineContext,
        pipeline_result: PipelineResult,
    ) -> dict[str, Any]:
        """Build the final pipeline output."""
        output: dict[str, Any] = {}

        # Include shared context values
        output.update(ctx.shared)

        # Include last successful step's output
        last_output: dict[str, Any] = {}
        for step in reversed(self.config.steps):
            step_result = pipeline_result.steps.get(step.name)
            if step_result and step_result.status == StepStatus.SUCCESS:
                last_output = dict(step_result.output)
                break

        # Merge — explicit output mapping takes precedence
        for k, v in last_output.items():
            if k not in output:
                output[k] = v

        return output

    def visualize(self) -> str:
        """Generate a Mermaid diagram of the pipeline.

        Returns:
            Mermaid graph definition string.
        """
        lines = ["graph TD"]
        lines.append(f'    START(["🚀 {self.config.name}"])')

        prev = "START"
        for step in self.config.steps:
            node_id = step.name.replace(" ", "_")

            if isinstance(step, ParallelStepConfig):
                # Fork node
                fork = f"{node_id}_fork"
                join = f"{node_id}_join"
                lines.append(f"    {fork}{{{{{step.name}}}}}")
                lines.append(f"    {prev} --> {fork}")
                for sub in step.steps:
                    sub_id = f"{node_id}_{sub.name}"
                    lines.append(f'    {sub_id}["{sub.agent}"]')
                    lines.append(f"    {fork} --> {sub_id}")
                    lines.append(f"    {sub_id} --> {join}")
                join_label = f"{step.name} done"
                lines.append(f"    {join}{{{{" + join_label + "}}}}")
                prev = join

            elif isinstance(step, LoopStepConfig):
                lines.append(f'    {node_id}[["🔄 {step.name} (max {step.max_iterations})"]]')
                lines.append(f"    {prev} --> {node_id}")
                lines.append(f"    {node_id} -. loop .-> {node_id}")
                prev = node_id

            elif isinstance(step, PluginStepConfig):
                lines.append(f'    {node_id}[/"🧮 {step.name} ({step.plugin})"/]')
                lines.append(f"    {prev} --> {node_id}")
                prev = node_id

            elif isinstance(step, EndpointStepConfig):
                lines.append(f'    {node_id}[/"🌐 {step.name} ({step.endpoint})"/]')
                lines.append(f"    {prev} --> {node_id}")
                prev = node_id

            elif isinstance(step, IngestionStepConfig):
                lines.append(f'    {node_id}[("📥 {step.name} ({step.ingestor})")]')
                lines.append(f"    {prev} --> {node_id}")
                prev = node_id

            elif isinstance(step, TransformStepConfig):
                lines.append(f'    {node_id}[/"⚡ {step.name}"/]')
                lines.append(f"    {prev} --> {node_id}")
                prev = node_id

            elif isinstance(step, SubPipelineStepConfig):
                lines.append(f'    {node_id}[["📦 {step.name} ({step.pipeline})"]]')
                lines.append(f"    {prev} --> {node_id}")
                prev = node_id

            else:
                # Agent step
                agent_name = getattr(step, "agent", step.name)
                condition = getattr(step, "condition", None)
                if condition:
                    lines.append(f'    {node_id}{{"{step.name}\\n({agent_name})"}}')
                else:
                    lines.append(f'    {node_id}["{step.name}\\n({agent_name})"]')
                lines.append(f"    {prev} --> {node_id}")
                prev = node_id

        lines.append('    END(["✅ Done"])')
        lines.append(f"    {prev} --> END")

        return "\n".join(lines)
