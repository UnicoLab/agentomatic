"""Comprehensive tests for the pipeline composition DSL.

Covers: models, context, steps, engine, builder, loader, and integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentomatic.pipelines.context import PipelineContext
from agentomatic.pipelines.models import (
    AgentStepConfig,
    ErrorPolicy,
    InputMapping,
    LoopStepConfig,
    OutputMapping,
    ParallelStepConfig,
    PipelineConfig,
    PipelineResult,
    PipelineStatus,
    RetryConfig,
    StepResult,
    StepStatus,
    TransformStepConfig,
)

# =====================================================================
# Models
# =====================================================================


class TestModels:
    """Test Pydantic model creation and validation."""

    def test_retry_config_defaults(self):
        rc = RetryConfig()
        assert rc.max_attempts == 3
        assert rc.backoff == "exponential"
        assert rc.base_delay == 1.0

    def test_agent_step_config_minimal(self):
        step = AgentStepConfig(name="plan", agent="planner")
        assert step.name == "plan"
        assert step.agent == "planner"
        assert step.on_error == ErrorPolicy.FAIL
        assert step.timeout == 30.0

    def test_agent_step_config_full(self):
        step = AgentStepConfig(
            name="verify",
            agent="fact_checker",
            input=InputMapping(mappings={"query": "$.input.query"}),
            output=OutputMapping(mappings={"result": "$.response"}),
            condition="len(ctx.input.query) > 0",
            on_error=ErrorPolicy.SKIP,
            retry=RetryConfig(max_attempts=5),
            timeout=60.0,
        )
        assert step.condition is not None
        assert step.retry.max_attempts == 5

    def test_transform_step_config(self):
        step = TransformStepConfig(
            name="merge",
            code="return {'merged': True}",
        )
        assert step.step_type.value == "transform"

    def test_parallel_step_config(self):
        step = ParallelStepConfig(
            name="research",
            steps=[
                AgentStepConfig(name="web", agent="web_researcher"),
                AgentStepConfig(name="kb", agent="knowledge_base"),
            ],
            max_concurrency=3,
        )
        assert len(step.steps) == 2
        assert step.max_concurrency == 3

    def test_loop_step_config(self):
        step = LoopStepConfig(
            name="refine",
            step=AgentStepConfig(name="refiner", agent="refiner"),
            max_iterations=5,
            until="ctx.current.confidence > 0.9",
        )
        assert step.max_iterations == 5

    def test_pipeline_config(self):
        config = PipelineConfig(
            name="test_pipeline",
            steps=[
                AgentStepConfig(name="step1", agent="agent1"),
                AgentStepConfig(name="step2", agent="agent2"),
            ],
        )
        assert config.step_names == ["step1", "step2"]
        assert config.get_agent_names() == {"agent1", "agent2"}

    def test_pipeline_config_get_step(self):
        config = PipelineConfig(
            name="test",
            steps=[
                AgentStepConfig(name="a", agent="x"),
                AgentStepConfig(name="b", agent="y"),
            ],
        )
        assert config.get_step("a") is not None
        assert config.get_step("a").agent == "x"
        assert config.get_step("z") is None

    def test_pipeline_result(self):
        result = PipelineResult(pipeline_name="test")
        assert result.status == PipelineStatus.PENDING
        assert result.succeeded is False

    def test_step_result(self):
        result = StepResult(
            name="step1",
            status=StepStatus.SUCCESS,
            output={"response": "hello"},
            duration_ms=100.0,
        )
        assert result.status == StepStatus.SUCCESS

    def test_input_mapping(self):
        m = InputMapping(mappings={"query": "$.input.query"})
        assert bool(m) is True
        assert m["query"] == "$.input.query"
        assert len(list(m.items())) == 1

    def test_empty_input_mapping(self):
        m = InputMapping()
        assert bool(m) is False


# =====================================================================
# Context
# =====================================================================


class TestContext:
    """Test PipelineContext and $ expression resolver."""

    def test_create_context(self):
        ctx = PipelineContext(
            input_data={"query": "hello"},
            defaults={"language": "en"},
        )
        assert ctx.input["query"] == "hello"
        assert ctx.defaults["language"] == "en"

    def test_resolve_input(self):
        ctx = PipelineContext(input_data={"query": "hello", "depth": 3})
        assert ctx.resolve("$.input.query") == "hello"
        assert ctx.resolve("$.input.depth") == 3

    def test_resolve_input_wildcard(self):
        ctx = PipelineContext(input_data={"a": 1, "b": 2})
        result = ctx.resolve("$.input.*")
        assert result == {"a": 1, "b": 2}

    def test_resolve_defaults(self):
        ctx = PipelineContext(defaults={"lang": "en"})
        assert ctx.resolve("$.defaults.lang") == "en"

    def test_resolve_step_output(self):
        ctx = PipelineContext()
        ctx.set_step_result(
            "plan",
            StepResult(
                name="plan",
                status=StepStatus.SUCCESS,
                output={"response": "my plan", "metadata": {"key": "val"}},
            ),
        )
        assert ctx.resolve("$.steps.plan.response") == "my plan"
        assert ctx.resolve("$.steps.plan.metadata.key") == "val"

    def test_resolve_step_wildcard(self):
        ctx = PipelineContext()
        ctx.set_step_result(
            "s1",
            StepResult(name="s1", output={"a": 1, "b": 2}),
        )
        result = ctx.resolve("$.steps.s1.*")
        assert result == {"a": 1, "b": 2}

    def test_resolve_parallel_step(self):
        ctx = PipelineContext()
        ctx.set_step_result(
            "research",
            StepResult(
                name="research",
                sub_results=[
                    StepResult(name="r1", output={"text": "result 1"}),
                    StepResult(name="r2", output={"text": "result 2"}),
                ],
            ),
        )
        # Without further path → returns list of outputs
        results = ctx.resolve("$.steps.research")
        assert len(results) == 2
        assert results[0]["text"] == "result 1"

    def test_resolve_parallel_index(self):
        ctx = PipelineContext()
        ctx.set_step_result(
            "research",
            StepResult(
                name="research",
                sub_results=[
                    StepResult(name="r1", output={"text": "first"}),
                    StepResult(name="r2", output={"text": "second"}),
                ],
            ),
        )
        assert ctx.resolve("$.steps.research[0].text") == "first"
        assert ctx.resolve("$.steps.research[1].text") == "second"

    def test_resolve_context_shared(self):
        ctx = PipelineContext()
        ctx.shared["custom_key"] = "custom_value"
        assert ctx.resolve("$.context.custom_key") == "custom_value"

    def test_resolve_current(self):
        ctx = PipelineContext()
        ctx.current = {"response": "latest"}
        assert ctx.resolve("$.current.response") == "latest"

    def test_resolve_literal_passthrough(self):
        ctx = PipelineContext()
        assert ctx.resolve("plain string") == "plain string"
        assert ctx.resolve(42) == 42
        assert ctx.resolve(None) is None

    def test_resolve_nonexistent(self):
        ctx = PipelineContext()
        assert ctx.resolve("$.input.nonexistent") is None

    def test_resolve_mapping(self):
        ctx = PipelineContext(input_data={"query": "hello", "depth": 3})
        result = ctx.resolve_mapping(
            {
                "q": "$.input.query",
                "d": "$.input.depth",
                "literal": "plain text",
            }
        )
        assert result == {"q": "hello", "d": 3, "literal": "plain text"}

    def test_set_step_result_updates_current(self):
        ctx = PipelineContext()
        ctx.set_step_result(
            "s1",
            StepResult(name="s1", output={"response": "hello"}),
        )
        assert ctx.current == {"response": "hello"}

    def test_to_eval_namespace(self):
        ctx = PipelineContext(input_data={"query": "test"})
        ns = ctx.to_eval_namespace()
        assert "ctx" in ns
        assert "len" in ns
        assert "any" in ns
        assert ns["ctx"] is ctx


# =====================================================================
# Steps
# =====================================================================


def _make_registry_with_agent(
    agent_name: str,
    node_fn: AsyncMock | None = None,
    graph_fn: MagicMock | None = None,
) -> MagicMock:
    """Create a mock registry with a single agent."""
    registry = MagicMock()
    agent = MagicMock()
    agent.node_fn = node_fn
    agent.graph_fn = graph_fn
    agent.schema_validator = None
    registry.get.return_value = agent
    return registry


class TestSteps:
    """Test step execution functions."""

    @pytest.mark.asyncio
    async def test_agent_step_with_node_fn(self):
        from agentomatic.pipelines.steps import execute_agent_step

        node_fn = AsyncMock(return_value={"response": "hello"})
        registry = _make_registry_with_agent("test_agent", node_fn=node_fn)
        ctx = PipelineContext(input_data={"query": "test"})

        config = AgentStepConfig(name="test", agent="test_agent")
        result = await execute_agent_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        assert result.output["response"] == "hello"
        assert result.agent_used == "test_agent"
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_agent_step_with_graph_fn(self):
        from agentomatic.pipelines.steps import execute_agent_step

        graph = AsyncMock()
        graph.ainvoke = AsyncMock(return_value={"response": "graph result"})
        graph_fn = MagicMock(return_value=graph)
        registry = _make_registry_with_agent("test_agent", graph_fn=graph_fn)
        ctx = PipelineContext(input_data={"query": "test"})

        config = AgentStepConfig(name="test", agent="test_agent")
        result = await execute_agent_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        assert result.output["response"] == "graph result"

    @pytest.mark.asyncio
    async def test_agent_step_not_found(self):
        from agentomatic.pipelines.steps import execute_agent_step

        registry = MagicMock()
        registry.get.return_value = None
        ctx = PipelineContext()

        config = AgentStepConfig(name="test", agent="missing_agent")
        result = await execute_agent_step(config, ctx, registry)

        assert result.status == StepStatus.FAILED
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_agent_step_with_input_mapping(self):
        from agentomatic.pipelines.steps import execute_agent_step

        node_fn = AsyncMock(return_value={"response": "mapped"})
        registry = _make_registry_with_agent("agent1", node_fn=node_fn)
        ctx = PipelineContext(input_data={"query": "hello world"})

        config = AgentStepConfig(
            name="test",
            agent="agent1",
            input=InputMapping(mappings={"current_query": "$.input.query"}),
        )
        result = await execute_agent_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        # Verify the node_fn was called with mapped state
        call_args = node_fn.call_args[0][0]
        assert call_args["current_query"] == "hello world"

    @pytest.mark.asyncio
    async def test_agent_step_auto_wiring(self):
        from agentomatic.pipelines.steps import execute_agent_step

        node_fn = AsyncMock(return_value={"response": "auto-wired"})
        registry = _make_registry_with_agent("agent1", node_fn=node_fn)
        ctx = PipelineContext(input_data={"query": "test"})
        ctx.current = {"response": "previous output"}

        config = AgentStepConfig(name="test", agent="agent1")
        result = await execute_agent_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        call_args = node_fn.call_args[0][0]
        assert call_args["current_query"] == "previous output"

    @pytest.mark.asyncio
    async def test_transform_step(self):
        from agentomatic.pipelines.steps import execute_transform_step

        ctx = PipelineContext(input_data={"items": [1, 2, 3]})
        config = TransformStepConfig(
            name="transform",
            code='return {"count": len(ctx.input["items"]), "doubled": True}',
        )
        result = await execute_transform_step(config, ctx)

        assert result.status == StepStatus.SUCCESS
        assert result.output["count"] == 3
        assert result.output["doubled"] is True

    @pytest.mark.asyncio
    async def test_transform_step_error(self):
        from agentomatic.pipelines.steps import execute_transform_step

        ctx = PipelineContext()
        config = TransformStepConfig(
            name="bad_transform",
            code="raise ValueError('oops')",
        )
        result = await execute_transform_step(config, ctx)

        assert result.status == StepStatus.FAILED
        assert "oops" in result.error

    @pytest.mark.asyncio
    async def test_parallel_step_all_strategy(self):
        from agentomatic.pipelines.steps import execute_parallel_step

        node_fn1 = AsyncMock(return_value={"response": "result1"})
        node_fn2 = AsyncMock(return_value={"response": "result2"})

        registry = MagicMock()

        def get_agent(name):
            agent = MagicMock()
            agent.graph_fn = None
            agent.schema_validator = None
            if name == "agent1":
                agent.node_fn = node_fn1
            else:
                agent.node_fn = node_fn2
            return agent

        registry.get.side_effect = get_agent

        ctx = PipelineContext(input_data={"query": "test"})
        config = ParallelStepConfig(
            name="parallel",
            steps=[
                AgentStepConfig(name="s1", agent="agent1"),
                AgentStepConfig(name="s2", agent="agent2"),
            ],
        )
        result = await execute_parallel_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        assert result.sub_results is not None
        assert len(result.sub_results) == 2

    @pytest.mark.asyncio
    async def test_loop_step(self):
        from agentomatic.pipelines.steps import execute_loop_step

        call_count = 0

        async def counting_fn(state):
            nonlocal call_count
            call_count += 1
            return {"response": "ok", "confidence": 0.5 + (call_count * 0.2)}

        registry = _make_registry_with_agent("refiner", node_fn=AsyncMock(side_effect=counting_fn))
        ctx = PipelineContext()

        config = LoopStepConfig(
            name="refine",
            step=AgentStepConfig(name="refiner", agent="refiner"),
            max_iterations=5,
            until="ctx.current.get('confidence', 0) > 0.9",
        )
        result = await execute_loop_step(config, ctx, registry)

        assert result.status == StepStatus.SUCCESS
        assert result.iterations is not None
        assert len(result.iterations) <= 5

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        from agentomatic.pipelines.steps import execute_with_retry

        call_count = 0

        async def flaky_fn(*args):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return StepResult(name="test", status=StepStatus.FAILED, error="transient")
            return StepResult(
                name="test",
                status=StepStatus.SUCCESS,
                output={"response": "finally"},
            )

        result = await execute_with_retry(
            flaky_fn,
            retry_config=RetryConfig(max_attempts=3, backoff="fixed", base_delay=0.1),
        )

        assert result.status == StepStatus.SUCCESS
        assert call_count == 3


# =====================================================================
# Engine
# =====================================================================


class TestEngine:
    """Test PipelineEngine execution."""

    @pytest.mark.asyncio
    async def test_simple_pipeline(self):
        from agentomatic.pipelines.engine import PipelineEngine

        node_fn1 = AsyncMock(return_value={"response": "planned", "agent_type": "planner"})
        node_fn2 = AsyncMock(return_value={"response": "executed", "agent_type": "executor"})

        registry = MagicMock()

        def get_agent(name):
            agent = MagicMock()
            agent.graph_fn = None
            agent.schema_validator = None
            if name == "planner":
                agent.node_fn = node_fn1
            else:
                agent.node_fn = node_fn2
            return agent

        registry.get.side_effect = get_agent
        registry.list_names.return_value = ["planner", "executor"]

        config = PipelineConfig(
            name="simple",
            steps=[
                AgentStepConfig(name="plan", agent="planner"),
                AgentStepConfig(name="execute", agent="executor"),
            ],
        )

        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "hello"})

        assert result.status == PipelineStatus.SUCCESS
        assert "plan" in result.steps
        assert "execute" in result.steps
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_condition(self):
        from agentomatic.pipelines.engine import PipelineEngine

        node_fn = AsyncMock(return_value={"response": "done"})
        registry = MagicMock()
        agent = MagicMock()
        agent.node_fn = node_fn
        agent.graph_fn = None
        agent.schema_validator = None
        registry.get.return_value = agent
        registry.list_names.return_value = ["agent1"]

        config = PipelineConfig(
            name="conditional",
            steps=[
                AgentStepConfig(name="step1", agent="agent1"),
                AgentStepConfig(
                    name="step2",
                    agent="agent1",
                    condition="False",
                ),
            ],
        )

        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "test"})

        assert result.status == PipelineStatus.SUCCESS
        assert result.steps["step2"].status == StepStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_pipeline_fail_fast(self):
        from agentomatic.pipelines.engine import PipelineEngine

        failing_fn = AsyncMock(side_effect=Exception("boom"))
        ok_fn = AsyncMock(return_value={"response": "ok"})

        registry = MagicMock()

        def get_agent(name):
            agent = MagicMock()
            agent.graph_fn = None
            agent.schema_validator = None
            if name == "bad_agent":
                agent.node_fn = failing_fn
            else:
                agent.node_fn = ok_fn
            return agent

        registry.get.side_effect = get_agent
        registry.list_names.return_value = ["bad_agent", "good_agent"]

        config = PipelineConfig(
            name="failing",
            steps=[
                AgentStepConfig(name="bad", agent="bad_agent"),
                AgentStepConfig(name="good", agent="good_agent"),
            ],
            on_error="fail_fast",
        )

        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "test"})

        assert result.status == PipelineStatus.FAILED
        # Second step should NOT have been executed
        assert "good" not in result.steps

    @pytest.mark.asyncio
    async def test_pipeline_continue_on_error(self):
        from agentomatic.pipelines.engine import PipelineEngine

        failing_fn = AsyncMock(side_effect=Exception("boom"))
        ok_fn = AsyncMock(return_value={"response": "ok"})

        registry = MagicMock()

        def get_agent(name):
            agent = MagicMock()
            agent.graph_fn = None
            agent.schema_validator = None
            if name == "bad_agent":
                agent.node_fn = failing_fn
            else:
                agent.node_fn = ok_fn
            return agent

        registry.get.side_effect = get_agent
        registry.list_names.return_value = ["bad_agent", "good_agent"]

        config = PipelineConfig(
            name="resilient",
            steps=[
                AgentStepConfig(name="bad", agent="bad_agent", on_error=ErrorPolicy.SKIP),
                AgentStepConfig(name="good", agent="good_agent"),
            ],
            on_error="continue",
        )

        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "test"})

        # Pipeline should still succeed since bad step was skipped
        assert result.status in (PipelineStatus.SUCCESS, PipelineStatus.PARTIAL)
        assert "good" in result.steps

    def test_validate_missing_agents(self):
        from agentomatic.pipelines.engine import PipelineEngine

        registry = MagicMock()
        registry.get.return_value = None
        registry.list_names.return_value = []

        config = PipelineConfig(
            name="invalid",
            steps=[
                AgentStepConfig(name="s1", agent="nonexistent"),
            ],
        )

        engine = PipelineEngine(config, registry)
        errors = engine.validate()
        assert len(errors) > 0
        assert "nonexistent" in errors[0]

    def test_validate_duplicate_names(self):
        from agentomatic.pipelines.engine import PipelineEngine

        registry = MagicMock()
        registry.get.return_value = MagicMock()
        registry.list_names.return_value = ["agent1"]

        config = PipelineConfig(
            name="dupes",
            steps=[
                AgentStepConfig(name="same", agent="agent1"),
                AgentStepConfig(name="same", agent="agent1"),
            ],
        )

        engine = PipelineEngine(config, registry)
        errors = engine.validate()
        assert any("Duplicate" in e for e in errors)

    def test_visualize(self):
        from agentomatic.pipelines.engine import PipelineEngine

        registry = MagicMock()
        config = PipelineConfig(
            name="viz_test",
            steps=[
                AgentStepConfig(name="plan", agent="planner"),
                AgentStepConfig(name="execute", agent="executor"),
            ],
        )

        engine = PipelineEngine(config, registry)
        mermaid = engine.visualize()

        assert "graph TD" in mermaid
        assert "plan" in mermaid
        assert "execute" in mermaid
        assert "planner" in mermaid
        assert "Done" in mermaid


# =====================================================================
# Builder
# =====================================================================


class TestBuilder:
    """Test the fluent Pipeline builder API."""

    def test_minimal_pipeline(self):
        from agentomatic.pipelines.builder import Pipeline

        config = Pipeline("qa").step("researcher").step("writer").to_config()

        assert config.name == "qa"
        assert len(config.steps) == 2
        assert config.steps[0].agent == "researcher"
        assert config.steps[1].agent == "writer"

    def test_full_pipeline(self):
        from agentomatic.pipelines.builder import Pipeline

        config = (
            Pipeline("research")
            .description("Research pipeline")
            .version("2.0.0")
            .step("plan", agent="query_planner")
            .step(
                "verify",
                agent="fact_checker",
                on_error="skip",
                timeout=60.0,
            )
            .on_error("continue")
            .timeout(300.0)
            .to_config()
        )

        assert config.name == "research"
        assert config.description == "Research pipeline"
        assert config.version == "2.0.0"
        assert len(config.steps) == 2
        assert config.on_error == "continue"
        assert config.timeout == 300.0

    def test_parallel_step(self):
        from agentomatic.pipelines.builder import Pipeline

        config = (
            Pipeline("test")
            .parallel(
                "research",
                [
                    Pipeline.agent("web_researcher"),
                    Pipeline.agent("knowledge_base"),
                ],
                strategy="all",
            )
            .to_config()
        )

        assert len(config.steps) == 1
        step = config.steps[0]
        assert isinstance(step, ParallelStepConfig)
        assert len(step.steps) == 2

    def test_transform_step(self):
        from agentomatic.pipelines.builder import Pipeline

        config = Pipeline("test").transform("merge", "return {'count': 42}").to_config()

        assert len(config.steps) == 1
        step = config.steps[0]
        assert isinstance(step, TransformStepConfig)
        assert "42" in step.code

    def test_loop_step(self):
        from agentomatic.pipelines.builder import Pipeline

        config = (
            Pipeline("test")
            .loop(
                "refine",
                agent="refiner",
                max_iterations=5,
                until="ctx.current.confidence > 0.9",
            )
            .to_config()
        )

        assert len(config.steps) == 1
        step = config.steps[0]
        assert isinstance(step, LoopStepConfig)
        assert step.max_iterations == 5

    def test_smart_defaults_agent_name(self):
        from agentomatic.pipelines.builder import Pipeline

        config = Pipeline("test").step("my_agent").to_config()
        assert config.steps[0].agent == "my_agent"
        assert config.steps[0].name == "my_agent"

    def test_repr(self):
        from agentomatic.pipelines.builder import Pipeline

        p = Pipeline("demo").step("a").step("b")
        assert "demo" in repr(p)


# =====================================================================
# Loader
# =====================================================================


class TestLoader:
    """Test YAML pipeline loader."""

    def test_from_dict_minimal(self):
        from agentomatic.pipelines.loader import PipelineLoader

        data = {
            "name": "simple",
            "steps": [
                {"agent": "researcher"},
                {"agent": "writer"},
            ],
        }
        config = PipelineLoader.from_dict(data)

        assert config.name == "simple"
        assert len(config.steps) == 2
        assert config.steps[0].agent == "researcher"

    def test_from_dict_full(self):
        from agentomatic.pipelines.loader import PipelineLoader

        data = {
            "name": "full",
            "description": "A full pipeline",
            "version": "2.0.0",
            "steps": [
                {
                    "name": "plan",
                    "agent": "planner",
                    "input": {"current_query": "$.input.query"},
                    "output": {"plan": "$.response"},
                    "on_error": "skip",
                    "timeout": 60.0,
                },
                {
                    "name": "merge",
                    "transform": "return {'merged': True}",
                },
                {
                    "name": "research",
                    "parallel": {
                        "strategy": "all",
                        "steps": [
                            {"agent": "web_researcher"},
                            {"agent": "kb_researcher"},
                        ],
                    },
                },
                {
                    "name": "refine",
                    "loop": {
                        "max_iterations": 3,
                        "step": {"agent": "refiner"},
                    },
                },
            ],
            "on_error": "continue",
            "timeout": 120.0,
        }
        config = PipelineLoader.from_dict(data)

        assert config.name == "full"
        assert len(config.steps) == 4
        assert isinstance(config.steps[0], AgentStepConfig)
        assert isinstance(config.steps[1], TransformStepConfig)
        assert isinstance(config.steps[2], ParallelStepConfig)
        assert isinstance(config.steps[3], LoopStepConfig)

    def test_from_dict_shorthand(self):
        from agentomatic.pipelines.loader import PipelineLoader

        data = {
            "name": "shorthand",
            "steps": [
                {"agent": "planner"},
            ],
        }
        config = PipelineLoader.from_dict(data)
        assert config.steps[0].name == "planner"
        assert config.steps[0].agent == "planner"

    def test_from_yaml_string(self):
        from agentomatic.pipelines.loader import PipelineLoader

        yaml_str = """
name: yaml_test
steps:
  - agent: agent1
  - agent: agent2
"""
        config = PipelineLoader.from_yaml_string(yaml_str)
        assert config.name == "yaml_test"
        assert len(config.steps) == 2

    def test_discover_pipelines(self, tmp_path):
        from agentomatic.pipelines.loader import PipelineLoader

        # Create a pipeline.yaml
        pipeline_file = tmp_path / "pipeline.yaml"
        pipeline_file.write_text("name: test_pipeline\nsteps:\n  - agent: agent1\n")

        pipelines = PipelineLoader.discover_pipelines(tmp_path)
        assert "test_pipeline" in pipelines

    def test_discover_pipelines_in_subdir(self, tmp_path):
        from agentomatic.pipelines.loader import PipelineLoader

        # Create pipelines/ subdir
        pipelines_dir = tmp_path / "pipelines"
        pipelines_dir.mkdir()
        (pipelines_dir / "research.yaml").write_text(
            "name: research\nsteps:\n  - agent: researcher\n"
        )

        pipelines = PipelineLoader.discover_pipelines(tmp_path)
        assert "research" in pipelines


# =====================================================================
# Integration
# =====================================================================


class TestIntegration:
    """Integration tests with mocked agents through registry."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Full pipeline: plan → execute → verify."""
        from agentomatic.pipelines.engine import PipelineEngine

        plan_fn = AsyncMock(return_value={"response": "Step 1: research, Step 2: write"})
        exec_fn = AsyncMock(return_value={"response": "Executed the plan successfully"})
        verify_fn = AsyncMock(return_value={"response": "Verified", "metadata": {"score": 0.95}})

        registry = MagicMock()

        def get_agent(name):
            agent = MagicMock()
            agent.graph_fn = None
            agent.schema_validator = None
            agents = {
                "planner": plan_fn,
                "executor": exec_fn,
                "verifier": verify_fn,
            }
            agent.node_fn = agents.get(name)
            return agent

        registry.get.side_effect = get_agent
        registry.list_names.return_value = ["planner", "executor", "verifier"]

        config = PipelineConfig(
            name="e2e",
            steps=[
                AgentStepConfig(
                    name="plan",
                    agent="planner",
                    input=InputMapping(mappings={"current_query": "$.input.query"}),
                ),
                AgentStepConfig(
                    name="execute",
                    agent="executor",
                    input=InputMapping(mappings={"current_query": "$.steps.plan.response"}),
                ),
                AgentStepConfig(
                    name="verify",
                    agent="verifier",
                ),
            ],
        )

        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "Build a report"})

        assert result.succeeded
        assert len(result.steps) == 3
        assert result.steps["plan"].status == StepStatus.SUCCESS
        assert result.steps["execute"].status == StepStatus.SUCCESS
        assert result.steps["verify"].status == StepStatus.SUCCESS
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_transform_and_parallel(self):
        """Pipeline with parallel fan-out and transform merge."""
        from agentomatic.pipelines.engine import PipelineEngine

        web_fn = AsyncMock(return_value={"response": "web result", "citations": ["web1"]})
        kb_fn = AsyncMock(return_value={"response": "kb result", "citations": ["kb1"]})

        registry = MagicMock()

        def get_agent(name):
            agent = MagicMock()
            agent.graph_fn = None
            agent.schema_validator = None
            agents = {"web_researcher": web_fn, "kb_researcher": kb_fn}
            agent.node_fn = agents.get(name)
            return agent

        registry.get.side_effect = get_agent
        registry.list_names.return_value = ["web_researcher", "kb_researcher"]

        config = PipelineConfig(
            name="parallel_test",
            steps=[
                ParallelStepConfig(
                    name="research",
                    steps=[
                        AgentStepConfig(name="web", agent="web_researcher"),
                        AgentStepConfig(name="kb", agent="kb_researcher"),
                    ],
                ),
                TransformStepConfig(
                    name="merge",
                    code="return {'total': 2, 'merged': True}",
                ),
            ],
        )

        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "research"})

        assert result.succeeded
        assert result.steps["research"].sub_results is not None
        assert result.steps["merge"].output["merged"] is True

    @pytest.mark.asyncio
    async def test_builder_to_engine_flow(self):
        """Test building a pipeline with Builder and executing it."""
        from agentomatic.pipelines.builder import Pipeline
        from agentomatic.pipelines.engine import PipelineEngine

        node_fn = AsyncMock(return_value={"response": "done"})
        registry = MagicMock()
        agent = MagicMock()
        agent.node_fn = node_fn
        agent.graph_fn = None
        agent.schema_validator = None
        registry.get.return_value = agent
        registry.list_names.return_value = ["my_agent"]

        config = Pipeline("builder_test").step("my_agent").to_config()

        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "test"})

        assert result.succeeded
        assert node_fn.called


# =====================================================================
# Endpoint Steps
# =====================================================================


class TestEndpointStep:
    """Test the custom-endpoint pipeline step type."""

    @pytest.mark.asyncio
    async def test_execute_endpoint_step_success(self):
        from agentomatic.endpoints import BaseEndpoint, EndpointRegistry
        from agentomatic.pipelines.models import EndpointStepConfig
        from agentomatic.pipelines.steps import execute_endpoint_step

        class _Ep(BaseEndpoint):
            endpoint_name = "scorer"

            async def handle(self, request):  # type: ignore[override]
                return {"score": 0.9, "payload": request.payload}

        reg = EndpointRegistry()
        reg.register(_Ep())

        ctx = PipelineContext(input_data={"text": "hello"})
        cfg = EndpointStepConfig(
            name="score",
            endpoint="scorer",
            input=InputMapping(mappings={"text": "$.input.text"}),
        )
        result = await execute_endpoint_step(cfg, ctx, reg)
        assert result.status == StepStatus.SUCCESS
        assert result.output["score"] == 0.9

    @pytest.mark.asyncio
    async def test_execute_endpoint_step_missing_endpoint(self):
        from agentomatic.endpoints import EndpointRegistry
        from agentomatic.pipelines.models import EndpointStepConfig
        from agentomatic.pipelines.steps import execute_endpoint_step

        ctx = PipelineContext(input_data={})
        cfg = EndpointStepConfig(name="score", endpoint="nope")
        result = await execute_endpoint_step(cfg, ctx, EndpointRegistry())
        assert result.status == StepStatus.FAILED
        assert "not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_endpoint_step_no_registry(self):
        from agentomatic.pipelines.models import EndpointStepConfig
        from agentomatic.pipelines.steps import execute_endpoint_step

        ctx = PipelineContext(input_data={})
        cfg = EndpointStepConfig(name="score", endpoint="scorer")
        result = await execute_endpoint_step(cfg, ctx, None)
        assert result.status == StepStatus.FAILED

    def test_pipeline_config_get_endpoint_names(self):
        from agentomatic.pipelines.models import EndpointStepConfig

        config = PipelineConfig(
            name="with_endpoint",
            steps=[
                EndpointStepConfig(name="fetch", endpoint="scorer"),
                AgentStepConfig(name="respond", agent="responder"),
            ],
        )
        assert "scorer" in config.get_endpoint_names()
