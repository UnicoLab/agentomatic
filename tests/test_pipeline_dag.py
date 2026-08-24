# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for DAG scheduling via ``upstreams`` (Phase 2).

Covers the ordering algorithm (topological order with list-index
tie-breaking), engine execution under DAG scheduling, validation
(unknown refs, self-refs, cycles), loader round-trips for ``upstreams``
on every step type, and the Mermaid upstream edges.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentomatic.pipelines.engine import PipelineEngine
from agentomatic.pipelines.loader import PipelineLoader, pipeline_to_dict
from agentomatic.pipelines.models import (
    PipelineConfig,
    PipelineStatus,
    TransformStepConfig,
)
from agentomatic.pipelines.ordering import compute_execution_order
from agentomatic.pipelines.validation import validate_pipeline_draft

# =====================================================================
# Ordering algorithm
# =====================================================================


def _steps(*upstream_groups: list[str] | None) -> list[object]:
    """Build simple stub steps with the given upstream lists."""
    return [
        type(
            "_Stub",
            (),
            {"name": f"s{i}", "upstreams": group},
        )()
        for i, group in enumerate(upstream_groups)
    ]


class TestExecutionOrder:
    def test_no_upstreams_is_list_order(self) -> None:
        steps = _steps(None, None, None)
        assert compute_execution_order(steps) == [0, 1, 2]

    def test_simple_dependency_runs_upstream_first(self) -> None:
        # s1 depends on s2 which is later in the list → s2 must run first.
        steps = _steps(None, ["s2"], None)
        order = compute_execution_order(steps)
        assert order.index(1) > order.index(2)
        assert order == [0, 2, 1]

    def test_tie_break_by_list_index(self) -> None:
        steps = _steps(None, ["s3"], ["s3"], None)
        # s1 and s2 both depend on s3; s1 (lower index) runs first.
        assert compute_execution_order(steps) == [0, 3, 1, 2]

    def test_transitive_dependencies(self) -> None:
        steps = _steps(None, None, ["s1"], ["s2"])
        order = compute_execution_order(steps)
        assert order.index(1) < order.index(2) < order.index(3)

    def test_unknown_upstream_raises(self) -> None:
        steps = _steps(None, ["ghost"])
        with pytest.raises(ValueError, match="ghost"):
            compute_execution_order(steps)

    def test_self_reference_raises(self) -> None:
        steps = _steps(["s0"])
        with pytest.raises(ValueError, match="itself"):
            compute_execution_order(steps)

    def test_direct_cycle_raises(self) -> None:
        steps = _steps(["s1"], ["s0"])
        with pytest.raises(ValueError, match="[Cc]ycle"):
            compute_execution_order(steps)

    def test_indirect_cycle_raises(self) -> None:
        steps = _steps(["s2"], ["s0"], ["s1"])
        with pytest.raises(ValueError, match="[Cc]ycle"):
            compute_execution_order(steps)


# =====================================================================
# Engine execution under DAG scheduling
# =====================================================================


def _dag_config(steps: list[TransformStepConfig]) -> PipelineConfig:
    return PipelineConfig(name="dag", steps=steps)


class TestEngineDagExecution:
    @pytest.mark.asyncio
    async def test_upstream_runs_before_dependent(self) -> None:
        config = _dag_config(
            [
                TransformStepConfig(name="a", code="return {'v': 1}"),
                TransformStepConfig(
                    name="b",
                    code="return {'total': ctx.steps['a'].output['v'] + ctx.steps['c'].output['v']}",
                    upstreams=["a", "c"],
                ),
                TransformStepConfig(name="c", code="return {'v': 10}"),
            ]
        )
        engine = PipelineEngine(config, MagicMock())
        result = await engine.run({})
        assert result.status == PipelineStatus.SUCCESS
        # b ran after both a and c despite being listed second.
        assert result.output["total"] == 11
        assert result.steps["a"].status.value == "success"
        assert result.steps["b"].status.value == "success"
        assert result.steps["c"].status.value == "success"

    @pytest.mark.asyncio
    async def test_conditions_still_skip_in_dag_order(self) -> None:
        config = _dag_config(
            [
                TransformStepConfig(name="a", code="return {'v': 1}"),
                TransformStepConfig(
                    name="b",
                    code="return {'v': 99}",
                    upstreams=["c"],
                ),
                TransformStepConfig(
                    name="c",
                    code="return {'v': 3}",
                    condition="len(ctx.input.query) > 100",  # never true
                ),
            ]
        )
        engine = PipelineEngine(config, MagicMock())
        result = await engine.run({"query": "hi"})
        assert result.status == PipelineStatus.SUCCESS
        assert result.steps["c"].status.value == "skipped"
        assert result.steps["b"].status.value == "success"

    @pytest.mark.asyncio
    async def test_downstream_step_skipped_when_upstream_fails_and_pipeline_continues(
        self,
    ) -> None:
        """A step whose declared upstream FAILED must not run against a
        missing/stale output just because the pipeline-level policy is
        "continue" — it should be skipped, and that skip should cascade to
        its own dependents too.
        """
        config = PipelineConfig(
            name="dag",
            on_error="continue",
            steps=[
                TransformStepConfig(name="a", code="raise ValueError('boom')"),
                TransformStepConfig(
                    name="b",
                    code="return {'v': ctx.steps['a'].output['v'] + 1}",
                    upstreams=["a"],
                ),
                TransformStepConfig(
                    name="c",
                    code="return {'v': ctx.steps['b'].output['v'] + 1}",
                    upstreams=["b"],
                ),
                TransformStepConfig(name="d", code="return {'v': 100}"),
            ],
        )
        engine = PipelineEngine(config, MagicMock())
        result = await engine.run({})

        assert result.steps["a"].status.value == "failed"
        # b depends directly on the failed step a -> skipped, not run.
        assert result.steps["b"].status.value == "skipped"
        assert "a" in result.steps["b"].error
        # c depends on b, which is now unsuccessful too -> cascaded skip.
        assert result.steps["c"].status.value == "skipped"
        # d has no dependency on the failed branch -> runs normally.
        assert result.steps["d"].status.value == "success"

    @pytest.mark.asyncio
    async def test_cycle_fails_pipeline_without_validate(self) -> None:
        config = _dag_config(
            [
                TransformStepConfig(name="a", code="return {'v': 1}", upstreams=["b"]),
                TransformStepConfig(name="b", code="return {'v': 2}", upstreams=["a"]),
            ]
        )
        engine = PipelineEngine(config, MagicMock())
        result = await engine.run({})
        assert result.status == PipelineStatus.FAILED
        assert "cycle" in (result.error or "").lower()

    def test_validate_reports_cycle(self) -> None:
        config = _dag_config(
            [
                TransformStepConfig(name="a", code="return {'v': 1}", upstreams=["b"]),
                TransformStepConfig(name="b", code="return {'v': 2}", upstreams=["a"]),
            ]
        )
        errors = PipelineEngine(config, MagicMock()).validate()
        assert any("cycle" in e.lower() for e in errors)

    def test_validate_reports_unknown_upstream(self) -> None:
        config = _dag_config(
            [
                TransformStepConfig(name="a", code="return {'v': 1}", upstreams=["ghost"]),
            ]
        )
        errors = PipelineEngine(config, MagicMock()).validate()
        assert any("ghost" in e for e in errors)

    def test_validate_passes_valid_upstreams(self) -> None:
        config = _dag_config(
            [
                TransformStepConfig(name="a", code="return {'v': 1}"),
                TransformStepConfig(name="b", code="return {'v': 2}", upstreams=["a"]),
            ]
        )
        assert PipelineEngine(config, MagicMock()).validate() == []


# =====================================================================
# Draft validation: mapping rules + upstreams
# =====================================================================


class TestDagDraftValidation:
    def test_later_step_reference_requires_upstreams(self) -> None:
        raw = {
            "name": "p",
            "steps": [
                {
                    "name": "first",
                    "agent": "planner",
                    "input": {"q": "$.steps.second.result"},
                },
                {"name": "second", "agent": "planner"},
            ],
        }
        config = PipelineLoader.from_dict(raw)
        errors, _ = validate_pipeline_draft(config)
        assert any("runs after" in e for e in errors)
        assert any("upstreams" in e for e in errors)

    def test_later_step_reference_ok_with_upstreams(self) -> None:
        raw = {
            "name": "p",
            "steps": [
                {
                    "name": "first",
                    "agent": "planner",
                    "upstreams": ["second"],
                    "input": {"q": "$.steps.second.result"},
                },
                {"name": "second", "agent": "planner"},
            ],
        }
        config = PipelineLoader.from_dict(raw)
        errors, _ = validate_pipeline_draft(config)
        assert not any("runs after" in e for e in errors)

    def test_draft_cycle_is_error(self) -> None:
        raw = {
            "name": "p",
            "steps": [
                {"name": "a", "agent": "planner", "upstreams": ["b"]},
                {"name": "b", "agent": "planner", "upstreams": ["a"]},
            ],
        }
        config = PipelineLoader.from_dict(raw)
        errors, _ = validate_pipeline_draft(config)
        assert any("cycle" in e.lower() for e in errors)

    def test_nested_upstreams_are_warning(self) -> None:
        raw = {
            "name": "p",
            "steps": [
                {
                    "name": "par",
                    "parallel": {
                        "steps": [
                            {"name": "w", "agent": "web", "upstreams": ["x"]},
                            {"name": "x", "agent": "kb"},
                        ]
                    },
                }
            ],
        }
        config = PipelineLoader.from_dict(raw)
        errors, warnings = validate_pipeline_draft(config)
        assert not errors
        assert any("not independently scheduled" in w for w in warnings)


# =====================================================================
# Loader round-trips with upstreams on every step type
# =====================================================================


class TestUpstreamsRoundTrip:
    def test_all_step_types_round_trip(self) -> None:
        raw = {
            "name": "dag_kitchen_sink",
            "steps": [
                {"name": "a", "agent": "planner", "upstreams": ["ing"]},
                {"name": "pl", "plugin": "sentiment", "upstreams": ["a"]},
                {"name": "ep", "endpoint": "ensemble", "upstreams": ["pl"]},
                {"name": "ing", "ingestion": "docs", "upstreams": ["t"]},
                {"name": "t", "transform": "return {'x': 1}", "upstreams": ["sub"]},
                {
                    "name": "par",
                    "parallel": {"steps": [{"name": "w", "agent": "web"}]},
                    "upstreams": ["ep"],
                },
                {
                    "name": "m",
                    "map": {"agent": "scorer", "items": "$.steps.a.results"},
                    "upstreams": ["par"],
                },
                {
                    "name": "lp",
                    "loop": {"step": {"name": "ref", "agent": "refiner"}},
                    "upstreams": ["m"],
                },
                {"name": "sub", "sub_pipeline": "other", "upstreams": ["lp"]},
            ],
        }
        config = PipelineLoader.from_dict(raw)
        assert config.get_step("a").upstreams == ["ing"]  # type: ignore[union-attr]
        assert config.get_step("par").upstreams == ["ep"]  # type: ignore[union-attr]
        dumped = pipeline_to_dict(config)
        assert PipelineLoader.from_dict(dumped).model_dump() == config.model_dump()

    def test_empty_upstreams_omitted_on_serialize(self) -> None:
        config = PipelineLoader.from_dict(
            {"name": "p", "steps": [{"name": "a", "agent": "planner", "upstreams": []}]}
        )
        dumped = pipeline_to_dict(config)
        assert "upstreams" not in dumped["steps"][0]


# =====================================================================
# Visualization
# =====================================================================


class TestDagVisualize:
    def test_upstream_edges_rendered(self) -> None:
        config = _dag_config(
            [
                TransformStepConfig(name="a", code="return {'v': 1}"),
                TransformStepConfig(name="b", code="return {'v': 2}", upstreams=["a"]),
            ]
        )
        mermaid = PipelineEngine(config, MagicMock()).visualize()
        assert "a -.->|upstream| b" in mermaid
