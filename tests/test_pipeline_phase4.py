"""Tests for Phase 4 pipeline hardening.

Covers the plugin step type, rollback/compensation, and optional
input/output schema enforcement.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agentomatic.pipelines.engine import PipelineEngine
from agentomatic.pipelines.loader import PipelineLoader
from agentomatic.pipelines.models import (
    AgentStepConfig,
    InputMapping,
    PipelineConfig,
    PipelineStatus,
    PluginStepConfig,
    TransformStepConfig,
)
from agentomatic.pipelines.validation import validate_against_schema
from agentomatic.plugins.ml import BaseMLPlugin

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class _DoublerIn(BaseModel):
    x: int = 0


class _DoublerOut(BaseModel):
    y: int = 0


class DoublerPlugin(BaseMLPlugin[_DoublerIn, _DoublerOut]):
    plugin_name = "doubler"

    async def predict(self, inputs: _DoublerIn) -> _DoublerOut:
        return _DoublerOut(y=inputs.x * 2)


def _plugin_registry(name: str = "doubler") -> MagicMock:
    registry = MagicMock()
    registry.get_plugin.return_value = DoublerPlugin()
    registry.list_names.return_value = [name]
    return registry


def _multi_agent_registry(agents: dict) -> MagicMock:
    registry = MagicMock()
    registry.get.side_effect = lambda n: agents.get(n)
    registry.list_names.return_value = list(agents)
    return registry


def _agent(node_fn: AsyncMock) -> MagicMock:
    agent = MagicMock()
    agent.node_fn = node_fn
    agent.graph_fn = None
    agent.schema_validator = None
    return agent


# ---------------------------------------------------------------------------
# Plugin step
# ---------------------------------------------------------------------------


class TestPluginStep:
    """The `plugin:` step type calls a registered ML plugin's predict()."""

    @pytest.mark.asyncio
    async def test_execute_plugin_step(self):
        from agentomatic.pipelines.context import PipelineContext
        from agentomatic.pipelines.steps import execute_plugin_step

        ctx = PipelineContext(input_data={"x": 21})
        config = PluginStepConfig(
            name="double",
            plugin="doubler",
            input=InputMapping(mappings={"x": "$.input.x"}),
        )
        result = await execute_plugin_step(config, ctx, _plugin_registry())

        assert result.status.value == "success"
        assert result.output["y"] == 42
        assert result.metadata["plugin"] == "doubler"

    @pytest.mark.asyncio
    async def test_plugin_step_missing_registry(self):
        from agentomatic.pipelines.context import PipelineContext
        from agentomatic.pipelines.steps import execute_plugin_step

        ctx = PipelineContext(input_data={"x": 1})
        config = PluginStepConfig(name="double", plugin="doubler")
        result = await execute_plugin_step(config, ctx, None)

        assert result.status.value == "failed"
        assert "registry" in (result.error or "")

    @pytest.mark.asyncio
    async def test_plugin_step_in_pipeline(self):
        config = PipelineConfig(
            name="ml",
            steps=[
                PluginStepConfig(
                    name="double",
                    plugin="doubler",
                    input=InputMapping(mappings={"x": "$.input.x"}),
                )
            ],
        )
        engine = PipelineEngine(config, MagicMock(), plugins=_plugin_registry())
        assert engine.validate() == []

        result = await engine.run({"x": 5})
        assert result.status == PipelineStatus.SUCCESS
        assert result.output["y"] == 10

    def test_plugin_step_validation_missing_plugin(self):
        config = PipelineConfig(
            name="ml",
            steps=[PluginStepConfig(name="double", plugin="ghost")],
        )
        empty = MagicMock()
        empty.list_names.return_value = []
        engine = PipelineEngine(config, MagicMock(), plugins=empty)
        errors = engine.validate()
        assert any("ghost" in e for e in errors)

    def test_get_plugin_names(self):
        config = PipelineConfig(
            name="ml",
            steps=[
                PluginStepConfig(name="a", plugin="p1"),
                PluginStepConfig(name="b", plugin="p2"),
            ],
        )
        assert config.get_plugin_names() == {"p1", "p2"}

    def test_loader_parses_plugin_step(self):
        config = PipelineLoader.from_dict(
            {
                "name": "ml",
                "steps": [{"plugin": "doubler", "input": {"x": "$.input.x"}}],
            }
        )
        step = config.steps[0]
        assert isinstance(step, PluginStepConfig)
        assert step.plugin == "doubler"
        assert step.name == "doubler"


# ---------------------------------------------------------------------------
# Rollback / compensation
# ---------------------------------------------------------------------------


class TestRollback:
    """`on_error: rollback` runs step compensation in reverse order."""

    @pytest.mark.asyncio
    async def test_rollback_runs_for_completed_steps(self):
        good = _agent(AsyncMock(return_value={"response": "ok"}))
        bad = _agent(AsyncMock(side_effect=RuntimeError("boom")))
        registry = _multi_agent_registry({"good": good, "bad": bad})

        config = PipelineConfig(
            name="tx",
            on_error="rollback",
            steps=[
                AgentStepConfig(
                    name="s1",
                    agent="good",
                    rollback="ctx.shared['undo_s1'] = True",
                ),
                AgentStepConfig(name="s2", agent="bad"),
            ],
        )
        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "hi"})

        assert result.status == PipelineStatus.FAILED
        assert result.metadata["rolled_back_steps"] == ["s1"]
        assert result.steps["s1"].metadata.get("rolled_back") is True
        # Compensation side effects surface via shared context in the output.
        assert result.output.get("undo_s1") is True

    @pytest.mark.asyncio
    async def test_rollback_skips_steps_without_compensation(self):
        good = _agent(AsyncMock(return_value={"response": "ok"}))
        bad = _agent(AsyncMock(side_effect=RuntimeError("boom")))
        registry = _multi_agent_registry({"good": good, "bad": bad})

        config = PipelineConfig(
            name="tx",
            on_error="rollback",
            steps=[
                AgentStepConfig(name="s1", agent="good"),  # no rollback
                AgentStepConfig(name="s2", agent="bad"),
            ],
        )
        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "hi"})

        assert result.status == PipelineStatus.FAILED
        assert result.metadata["rolled_back_steps"] == []

    @pytest.mark.asyncio
    async def test_fail_fast_does_not_rollback(self):
        good = _agent(AsyncMock(return_value={"response": "ok"}))
        bad = _agent(AsyncMock(side_effect=RuntimeError("boom")))
        registry = _multi_agent_registry({"good": good, "bad": bad})

        config = PipelineConfig(
            name="tx",
            on_error="fail_fast",
            steps=[
                AgentStepConfig(name="s1", agent="good", rollback="ctx.shared['undo'] = True"),
                AgentStepConfig(name="s2", agent="bad"),
            ],
        )
        engine = PipelineEngine(config, registry)
        result = await engine.run({"query": "hi"})

        assert result.status == PipelineStatus.FAILED
        assert "rolled_back_steps" not in result.metadata


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------


class TestSchemaValidationHelper:
    def test_missing_required_field(self):
        errors = validate_against_schema({}, {"query": "str"}, label="input")
        assert len(errors) == 1
        assert "query" in errors[0]

    def test_optional_field_allowed_missing(self):
        errors = validate_against_schema({}, {"query": {"type": "str", "required": False}})
        assert errors == []

    def test_type_mismatch(self):
        errors = validate_against_schema({"n": "x"}, {"n": "int"})
        assert len(errors) == 1
        assert "int" in errors[0]

    def test_bool_not_accepted_as_int(self):
        errors = validate_against_schema({"n": True}, {"n": "int"})
        assert len(errors) == 1

    def test_valid_payload(self):
        errors = validate_against_schema(
            {"q": "hi", "k": 3, "flag": True},
            {"q": "str", "k": "int", "flag": "bool"},
        )
        assert errors == []

    def test_empty_schema_is_noop(self):
        assert validate_against_schema({"anything": 1}, None) == []


class TestPipelineSchemaEnforcement:
    @pytest.mark.asyncio
    async def test_strict_input_schema_fails(self):
        config = PipelineConfig(
            name="p",
            input_schema={"query": "str"},
            strict_schema=True,
            steps=[TransformStepConfig(name="t", code="return {'ok': True}")],
        )
        engine = PipelineEngine(config, MagicMock())
        result = await engine.run({})  # missing required 'query'

        assert result.status == PipelineStatus.FAILED
        assert "query" in (result.error or "")

    @pytest.mark.asyncio
    async def test_advisory_input_schema_passes(self):
        config = PipelineConfig(
            name="p",
            input_schema={"query": "str"},
            strict_schema=False,
            steps=[TransformStepConfig(name="t", code="return {'ok': True}")],
        )
        engine = PipelineEngine(config, MagicMock())
        result = await engine.run({})  # missing but advisory

        assert result.status == PipelineStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_strict_output_schema_fails(self):
        config = PipelineConfig(
            name="p",
            output_schema={"missing": "str"},
            strict_schema=True,
            steps=[TransformStepConfig(name="t", code="return {'answer': 'hi'}")],
        )
        engine = PipelineEngine(config, MagicMock())
        result = await engine.run({})

        assert result.status == PipelineStatus.FAILED
        assert "missing" in (result.error or "")

    @pytest.mark.asyncio
    async def test_strict_output_schema_passes(self):
        config = PipelineConfig(
            name="p",
            output_schema={"answer": "str"},
            strict_schema=True,
            steps=[TransformStepConfig(name="t", code="return {'answer': 'hi'}")],
        )
        engine = PipelineEngine(config, MagicMock())
        result = await engine.run({})

        assert result.status == PipelineStatus.SUCCESS
        assert result.output["answer"] == "hi"

    def test_loader_parses_strict_schema(self):
        config = PipelineLoader.from_dict(
            {
                "name": "p",
                "strict_schema": True,
                "input_schema": {"query": "str"},
                "steps": [{"transform": "return {}", "name": "t"}],
            }
        )
        assert config.strict_schema is True
        assert config.input_schema == {"query": "str"}
