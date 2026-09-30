# pyright: reportMissingParameterType=none
"""Every metric must work in every slot, whichever protocol it implements.

Two metric protocols coexist:

* ``agentomatic.agents`` — sync ``score(example, prediction) -> float``
  (``compile(metrics=...)``, ``evaluate``, ``agents.WeightedMetric``,
  ``MetricLoss``);
* ``agentomatic.optimize`` — async ``evaluate(query, response, expected,
  context) -> EvalResult`` (``PromptFitter``, ``CompositeMetric``).

Handing ``PromptFitterBridge(metric=OptimizeMetricAdapter(judge))`` to fit
crashed mid-run with ``'OptimizeMetricAdapter' object has no attribute
'evaluate'`` — logged as a warning, so training silently continued without
optimizing. These tests pin the bridge in both directions.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentomatic.agents import (
    AgentExample,
    CallableMetric,
    ContainsTermsMetric,
    MetricLoss,
    OptimizeMetricAdapter,
    PromptFitterBridge,
    as_agent_metric,
    resolve_loss,
)
from agentomatic.agents import WeightedMetric as AgentWeightedMetric
from agentomatic.optimize import (
    BaseMetric,
    CompositeMetric,
    ExactMatchMetric,
    ScoreMetricAdapter,
    as_optimize_metric,
    resolve_metrics,
)
from agentomatic.optimize.metrics import EvalResult, WeightedMetric, scoring_run
from agentomatic.optimize.runner import RunResult


class _StubJudge(BaseMetric):
    """Optimize metric returning a fixed score (or failing on demand)."""

    name = "stub_judge"

    def __init__(self, score: float = 0.8, *, fail: bool = False) -> None:
        self._score = score
        self._fail = fail
        self.calls: list[tuple[str, str, str | None]] = []

    async def evaluate(self, query, response, expected=None, context=None) -> EvalResult:
        self.calls.append((query, response, expected))
        if self._fail:
            raise ConnectionError("judge unreachable")
        return EvalResult(metric_name=self.name, score=self._score, reason="stub")


def _example(**metadata: Any) -> AgentExample:
    return AgentExample(
        id="ex1",
        input={"current_query": "What is the refund window?"},
        expected_output={"response": "30 days"},
        metadata=metadata,
    )


class TestOptimizeMetricAdapter:
    def test_speaks_both_protocols(self) -> None:
        adapter = OptimizeMetricAdapter(_StubJudge(0.7), name="judge")
        assert adapter.score(_example(), {"response": "30 days"}) == pytest.approx(0.7)
        result = asyncio.run(adapter.evaluate("q", "r", "e"))
        assert result.score == pytest.approx(0.7)

    def test_failures_score_zero_and_are_counted_not_faked(self) -> None:
        adapter = OptimizeMetricAdapter(_StubJudge(fail=True), name="judge")
        assert adapter.score(_example(), {"response": "x"}) == 0.0
        assert adapter.failures == 1

    def test_rejects_a_non_optimize_metric_with_guidance(self) -> None:
        with pytest.raises(TypeError, match="score\\(example, prediction\\)"):
            OptimizeMetricAdapter(ContainsTermsMetric(["x"]))

    def test_unwrapped_for_the_fitter(self) -> None:
        judge = _StubJudge()
        assert as_optimize_metric(OptimizeMetricAdapter(judge)) is judge


class TestPromptFitterBridgeMetric:
    def test_accepts_the_adapter_from_the_user_script(self) -> None:
        judge = _StubJudge()
        bridge = PromptFitterBridge(metric=OptimizeMetricAdapter(judge, name="judge"))
        assert bridge.metric is judge

    def test_wraps_class_agent_metrics(self) -> None:
        bridge = PromptFitterBridge(metric=ContainsTermsMetric(["30 days"]))
        assert isinstance(bridge.metric, ScoreMetricAdapter)

    def test_fails_fast_on_an_unusable_metric(self) -> None:
        with pytest.raises(TypeError, match="not a usable metric"):
            PromptFitterBridge(metric=object())


class TestScoreMetricAdapter:
    def test_class_agent_metric_scores_through_evaluate(self) -> None:
        adapter = as_optimize_metric(ContainsTermsMetric(["30 days"]))
        result = asyncio.run(adapter.evaluate("q", "Refunds within 30 days.", "30 days"))
        assert result.score == 1.0

    def test_rebuilds_expected_output_from_a_rich_reference(self) -> None:
        seen: dict[str, Any] = {}

        def fn(example: AgentExample, prediction: dict[str, Any]) -> float:
            seen["expected"] = example.expected_output
            return 1.0

        reference = _example().to_datapoint().expected_answer
        asyncio.run(as_optimize_metric(CallableMetric("c", fn)).evaluate("q", "r", reference))
        assert seen["expected"] == {"response": "30 days"}

    def test_sees_example_metadata_and_structured_output_inside_the_fitter(self) -> None:
        seen: dict[str, Any] = {}

        def fn(example: AgentExample, prediction: dict[str, Any]) -> float:
            seen["metadata"] = example.metadata
            seen["prediction"] = prediction
            return 0.5

        run = RunResult(
            query="q",
            response="30 days",
            metadata={
                "example": {"must_include": ["30 days"]},
                "output": {"response": "30 days", "sources": ["kb"]},
            },
        )
        adapter = as_optimize_metric(CallableMetric("c", fn))

        async def score() -> EvalResult:
            with scoring_run(run):
                return await adapter.evaluate("q", "30 days", "30 days")

        assert asyncio.run(score()).score == 0.5
        assert seen["metadata"] == {"must_include": ["30 days"]}
        assert seen["prediction"] == {"response": "30 days", "sources": ["kb"]}


class TestCompositionAcceptsEitherProtocol:
    def test_composite_metric_with_a_class_agent_component(self) -> None:
        composite = CompositeMetric(
            [
                WeightedMetric(name="judge", metric=_StubJudge(1.0), weight=1),
                WeightedMetric(name="terms", metric=ContainsTermsMetric(["30 days"]), weight=1),
            ]
        )
        result = asyncio.run(composite.evaluate("q", "within 30 days", "30 days"))
        assert result.metadata["dimensions"] == {"judge": 1.0, "terms": 1.0}
        assert result.score == pytest.approx(1.0)

    def test_metadata_metrics_inside_a_composite_score_the_same_outside_fit(self) -> None:
        """``evaluate()``/``compile(metrics=[composite])`` scored a nested
        metadata-reading metric 0: only the fitter exposed the example."""
        facts = CallableMetric(
            "facts",
            lambda ex, pred: float(
                all(t in pred["response"] for t in ex.metadata["must_include"])
            ),
        )
        composite = CompositeMetric([WeightedMetric(name="facts", metric=facts, weight=1)])
        example = _example(must_include=["30 days"])
        assert composite.score(example, {"response": "within 30 days"}) == pytest.approx(1.0)
        assert OptimizeMetricAdapter(composite).score(
            example, {"response": "no idea"}
        ) == pytest.approx(0.0)

    def test_agents_weighted_metric_with_an_optimize_component(self) -> None:
        weighted = AgentWeightedMetric(
            [("judge", _StubJudge(0.4), 1), ("em", ExactMatchMetric(), 1)]
        )
        score = weighted.score(_example(), {"response": "30 days"})
        assert weighted.last_component_scores["judge"] == pytest.approx(0.4)
        assert weighted.last_component_scores["em"] == pytest.approx(1.0)
        assert score == pytest.approx(0.7)

    def test_metric_loss_and_resolve_loss_accept_optimize_metrics(self) -> None:
        loss = MetricLoss(_StubJudge(0.75))
        assert loss.compute(_example(), {"response": "x"}) == pytest.approx(0.25)
        assert resolve_loss(_StubJudge(0.75)) is not None

    def test_resolve_metrics_accepts_instances_of_both_protocols(self) -> None:
        resolved = resolve_metrics(["exact_match", ContainsTermsMetric(["a"]), _StubJudge()])
        assert all(callable(getattr(m, "evaluate", None)) for m in resolved)

    def test_compile_and_evaluate_accept_optimize_metrics(self) -> None:
        from dataclasses import dataclass

        from agentomatic.agents import AgentDataset, BaseGraphAgent

        @dataclass
        class S:
            q: str = ""
            out: str = ""

        class Echo(BaseGraphAgent[S]):
            agent_name = "echo"

            def build_graph(self) -> Any:
                g = self.new_graph()
                g.add_node("n", self.n)
                g.set_entry_point("n")
                g.set_finish_point("n")
                return g.compile()

            def n(self, s: S) -> S:
                s.out = "30 days"
                return s

            def input_to_state(self, d: dict[str, Any]) -> S:
                return S(q=d.get("current_query", ""))

            def state_to_output(self, s: S) -> dict[str, Any]:
                return {"response": s.out}

        agent = Echo()
        agent.compile(AgentDataset(examples=[_example()]), metrics=[ExactMatchMetric()])
        report = agent.evaluate([_example()])
        assert report.scores["exact_match"] == pytest.approx(1.0)

    def test_as_agent_metric_rejects_unusable_objects(self) -> None:
        with pytest.raises(TypeError, match="not a usable metric"):
            as_agent_metric(42)
