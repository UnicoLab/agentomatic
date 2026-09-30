# pyright: reportMissingParameterType=none
# pyright: reportAttributeAccessIssue=none
"""Scores must mean what they say — no clamped scales, no score *rising* on failure.

Covers the judge-parsing, aggregation and training-loop defects found by
auditing the optimization metrics: 0–10 judge scores clamped to a perfect 1.0,
``{"score": …}`` replies rejected, a failed judge raising a composite score,
panel weights attached to the wrong judge, lower-is-better DeepEval metrics
maximised, Keras-incompatible EarlyStopping, crashed examples "passing", and
the same judge being called several times per example.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentomatic.agents import (
    AgentExample,
    CallableMetric,
    EarlyStopping,
    OptimizeMetricAdapter,
)
from agentomatic.agents.types import ExampleResult
from agentomatic.optimize import (
    BaseMetric,
    CompositeMetric,
    CustomMetric,
    LatencyMetric,
    LLMJudgeMetric,
    LocalJudgeMetric,
    MultiJudgePanel,
)
from agentomatic.optimize.llm_caller import LLMCaller
from agentomatic.optimize.metrics import (
    EvalResult,
    WeightedMetric,
    coerce_judge_score,
    scoring_run,
)
from agentomatic.optimize.runner import RunResult


class TestCoerceJudgeScore:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0.8, 0.8),
            (1, 1.0),
            (7, 0.7),
            (85, 0.85),
            ("0.6", 0.6),
            ("8/10", 0.8),
            ("80%", 0.8),
            ({"score": 0.4}, 0.4),
        ],
    )
    def test_scales_are_normalised(self, raw: Any, expected: float) -> None:
        assert coerce_judge_score(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, True, -0.1, 250, "great", float("nan"), {}])
    def test_unusable_values(self, raw: Any) -> None:
        assert coerce_judge_score(raw) is None


def _judge_reply(monkeypatch, reply: dict[str, Any]) -> None:
    async def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return reply

    monkeypatch.setattr(LLMCaller, "call_with_json", staticmethod(fake))


class TestLocalJudgeParsing:
    def _run(self, monkeypatch, reply: dict[str, Any]) -> EvalResult:
        _judge_reply(monkeypatch, reply)
        judge = LocalJudgeMetric(name="j", model="omlx/x", dimensions=["correctness", "relevance"])
        return asyncio.run(judge.evaluate("q", "r", "e"))

    def test_score_key_is_accepted(self, monkeypatch) -> None:
        result = self._run(monkeypatch, {"score": 0.8})
        assert result.score == pytest.approx(0.8)
        assert not result.metadata["evaluation_failed"]

    def test_ten_point_scale_is_not_a_perfect_score(self, monkeypatch) -> None:
        assert self._run(monkeypatch, {"overall_score": 2}).score == pytest.approx(0.2)

    def test_dimension_only_reply_uses_their_mean(self, monkeypatch) -> None:
        result = self._run(monkeypatch, {"dimensions": {"correctness": 0.9, "relevance": 0.7}})
        assert result.score == pytest.approx(0.8)
        assert result.metadata["score_source"] == "dimension_mean"

    def test_nested_dimension_does_not_discard_the_overall(self, monkeypatch) -> None:
        result = self._run(
            monkeypatch,
            {"overall_score": 0.7, "dimensions": {"correctness": {"score": 0.9, "why": "…"}}},
        )
        assert result.score == pytest.approx(0.7)
        assert result.metadata["dimensions"]["correctness"] == pytest.approx(0.9)

    def test_string_fields_are_not_exploded_into_characters(self, monkeypatch) -> None:
        result = self._run(monkeypatch, {"score": 0.5, "what_worked": "clear answer"})
        assert result.metadata["what_worked"] == ["clear answer"]

    def test_unusable_reply_is_a_failed_evaluation(self, monkeypatch) -> None:
        result = self._run(monkeypatch, {"verdict": "good"})
        assert result.score == 0.0
        assert result.metadata["evaluation_failed"] is True

    def test_feedback_text_does_not_decide_failure(self, monkeypatch) -> None:
        result = self._run(
            monkeypatch, {"score": 0.9, "feedback": "Judge evaluation failed: not really"}
        )
        assert result.metadata["evaluation_failed"] is False

    def test_per_judge_endpoint_is_forwarded(self, monkeypatch) -> None:
        seen: dict[str, Any] = {}

        async def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {"score": 1}

        monkeypatch.setattr(LLMCaller, "call_with_json", staticmethod(fake))
        judge = LocalJudgeMetric(model="omlx/x", base_url="http://judge:9/v1", api_key="k")
        asyncio.run(judge.evaluate("q", "r"))
        assert seen["base_url"] == "http://judge:9/v1"
        assert seen["api_key"] == "k"

    def test_llm_judge_accepts_overall_score(self, monkeypatch) -> None:
        _judge_reply(monkeypatch, {"overall_score": 9})
        judge = LLMJudgeMetric(criteria="c", model="omlx/x")
        assert asyncio.run(judge.evaluate("q", "r")).score == pytest.approx(0.9)


class _Fixed(BaseMetric):
    def __init__(self, name: str, score: float | None) -> None:
        self.name = name
        self._score = score

    async def evaluate(self, query, response, expected=None, context=None) -> EvalResult:
        if self._score is None:
            return EvalResult(self.name, 0.0, "down", {"evaluation_failed": True})
        return EvalResult(self.name, self._score, "ok")


class TestCompositeMetric:
    def test_a_failed_judge_never_raises_the_score(self) -> None:
        working = CompositeMetric(
            [
                WeightedMetric("judge", _Fixed("judge", 0.8), 0.3),
                WeightedMetric("format", _Fixed("format", 1.0), 0.7),
            ]
        )
        failing = CompositeMetric(
            [
                WeightedMetric("judge", _Fixed("judge", None), 0.3),
                WeightedMetric("format", _Fixed("format", 1.0), 0.7),
            ]
        )
        ok = asyncio.run(working.evaluate("q", "r"))
        down = asyncio.run(failing.evaluate("q", "r"))
        assert down.score < ok.score
        assert down.metadata["failed_components"] == ["judge"]

    def test_negative_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="higher-is-better"):
            CompositeMetric(
                [
                    WeightedMetric("quality", _Fixed("q", 1.0), 0.9),
                    WeightedMetric("latency", LatencyMetric(), -0.1),
                ]
            )


class TestMultiJudgePanel:
    def _panel(self, scores: list[float | None], **kw: Any) -> MultiJudgePanel:
        class _J(LocalJudgeMetric):
            def __init__(self, idx: int, score: float | None) -> None:
                super().__init__(name=f"j{idx}")
                self._s = score

            async def evaluate_rich(self, *a: Any, **k: Any) -> Any:
                from agentomatic.optimize.metrics import MetricResult

                result = MetricResult(score=self._s or 0.0, feedback="f", dimensions={})
                result._judge_failed = self._s is None
                return result

        return MultiJudgePanel(judges=[_J(i, s) for i, s in enumerate(scores)], **kw)

    def test_weights_stay_with_their_judge_when_one_fails(self) -> None:
        panel = self._panel([None, 0.5, 1.0], weights=[5, 1, 3])
        assert asyncio.run(panel.evaluate("q", "r")).score == pytest.approx(0.875)

    def test_zero_weight_survivors_fail_cleanly(self) -> None:
        panel = self._panel([0.9, None], weights=[0, 1])
        assert asyncio.run(panel.evaluate("q", "r")).metadata["evaluation_failed"]

    def test_median_aggregation(self) -> None:
        panel = self._panel([0.1, 0.8, 0.9], aggregation="median")
        assert asyncio.run(panel.evaluate("q", "r")).score == pytest.approx(0.8)

    def test_unknown_aggregation_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregation"):
            self._panel([0.5], aggregation="vibes")


class TestDeepEvalDirection:
    def test_lower_is_better_metrics_are_inverted(self) -> None:
        from agentomatic.optimize.metrics import DeepEvalMetric

        class ToxicityMetric:  # name matches deepeval's class
            score = 0.9
            reason = "very toxic"

            def measure(self, test_case: Any) -> None:
                pass

        pytest.importorskip("deepeval")
        result = asyncio.run(DeepEvalMetric(ToxicityMetric()).evaluate("q", "r"))
        assert result.score == pytest.approx(0.1)


class TestOtherMetrics:
    def test_custom_metric_clamps_and_rejects_nan(self) -> None:
        assert asyncio.run(CustomMetric(lambda *a: 7.0).evaluate("q", "r")).score == 1.0
        nan = asyncio.run(CustomMetric(lambda *a: float("nan")).evaluate("q", "r"))
        assert nan.metadata["evaluation_failed"]

    def test_latency_is_read_from_the_scored_run(self) -> None:
        metric = LatencyMetric(target_seconds=1.0, max_seconds=3.0)

        async def score() -> EvalResult:
            with scoring_run(RunResult(query="q", response="r", duration_ms=2000.0)):
                return await metric.evaluate("q", "r")

        assert asyncio.run(score()).score == pytest.approx(0.5)


class TestTrainingLoopSemantics:
    def _agent(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            stop_training=False, compiled_config={}, invalidate_graph=lambda: None
        )

    def test_early_stopping_uses_keras_patience(self) -> None:
        cb = EarlyStopping(monitor="val_loss", patience=1)
        cb.set_agent(self._agent())
        cb.on_train_begin()
        cb.on_epoch_end(0, {"val_loss": 0.5})
        cb.on_epoch_end(1, {"val_loss": 0.6})
        assert cb.agent.stop_training is True
        assert cb.stopped_epoch == 1

    def test_restore_best_restores_the_best_epoch_config(self) -> None:
        agent = self._agent()
        cb = EarlyStopping(monitor="val_loss", patience=5, restore_best=True)
        cb.set_agent(agent)
        cb.on_train_begin()
        agent.compiled_config = {"system_prompt": "good"}
        cb.on_epoch_end(0, {"val_loss": 0.2})
        agent.compiled_config = {"system_prompt": "worse"}
        cb.on_epoch_end(1, {"val_loss": 0.4})
        cb.on_train_end()
        assert agent.compiled_config == {"system_prompt": "good"}

    def test_auto_mode_checks_the_suffix(self) -> None:
        assert EarlyStopping(monitor="gloss_quality").mode == "max"
        assert EarlyStopping(monitor="val_loss").mode == "min"

    def test_optimize_callbacks_are_rejected_by_agent_fit(self) -> None:
        from dataclasses import dataclass

        from agentomatic.agents import BaseGraphAgent
        from agentomatic.optimize import OptimizeEarlyStopping

        @dataclass
        class S:
            q: str = ""

        class A(BaseGraphAgent[S]):
            agent_name = "a"

            def build_graph(self) -> Any:
                g = self.new_graph()
                g.add_node("n", lambda s: s)
                g.set_entry_point("n")
                g.set_finish_point("n")
                return g.compile()

            def input_to_state(self, d: dict[str, Any]) -> S:
                return S()

            def state_to_output(self, s: S) -> dict[str, Any]:
                return {"response": "x"}

        agent = A()
        agent.compile(metrics=[CallableMetric("m", lambda e, p: 1.0)])
        with pytest.raises(TypeError, match="PromptFitter"):
            agent.fit(
                [AgentExample(input={"current_query": "q"})], callbacks=[OptimizeEarlyStopping()]
            )

    def test_crashed_examples_do_not_pass(self) -> None:
        assert not ExampleResult(example_id="x", error="boom").passed
        assert ExampleResult(example_id="x", scores={"m": 0.9}).passed


class TestJudgeCallsPerExample:
    def test_one_judge_call_per_example_across_wrappers(self) -> None:
        calls: list[str] = []

        class _Judge(BaseMetric):
            name = "judge"

            async def evaluate(self, query, response, expected=None, context=None):
                calls.append(response)
                return EvalResult("judge", 0.5, "ok")

        from agentomatic.agents import MetricLoss
        from agentomatic.agents import WeightedMetric as AgentWeighted

        judge = _Judge()
        example = AgentExample(id="e", input={"current_query": "q"})
        prediction = {"response": "same answer"}
        OptimizeMetricAdapter(judge).score(example, prediction)
        AgentWeighted([("judge", judge, 1.0)]).score(example, prediction)
        MetricLoss(judge).compute(example, prediction)
        assert calls == ["same answer"]


def test_duplicate_metric_names_are_rejected() -> None:
    from dataclasses import dataclass

    from agentomatic.agents import BaseGraphAgent

    @dataclass
    class S:
        q: str = ""

    class A(BaseGraphAgent[S]):
        agent_name = "a"

        def build_graph(self) -> Any:
            g = self.new_graph()
            g.add_node("n", lambda s: s)
            g.set_entry_point("n")
            g.set_finish_point("n")
            return g.compile()

        def input_to_state(self, d: dict[str, Any]) -> S:
            return S()

        def state_to_output(self, s: S) -> dict[str, Any]:
            return {}

    metric = CallableMetric("quality", lambda e, p: 1.0)
    with pytest.raises(ValueError, match="unique"):
        A().compile(metrics=[metric, CallableMetric("quality", lambda e, p: 0.5)])


def test_run_sync_reuses_one_loop_inside_a_running_loop() -> None:
    from agentomatic.async_utils import run_sync

    loops: set[int] = set()

    async def probe() -> int:
        loops.add(id(asyncio.get_running_loop()))
        return 1

    async def main() -> list[int]:
        return [run_sync(probe()) for _ in range(3)]

    assert asyncio.run(main()) == [1, 1, 1]
    assert len(loops) == 1


def test_rewrite_text_keeps_fenced_blocks() -> None:
    from agentomatic.optimize.briefing import extract_prompt_text
    from agentomatic.optimize.llm_caller import _normalize_llm_text

    raw = "<think>hmm</think>Old:\n```\nold prompt\n```\n---\nNew prompt with:\n```json\n{}\n```"
    text = _normalize_llm_text(raw)
    assert "old prompt" in text and "```json" in text
    assert extract_prompt_text(text) == "New prompt with:\n```json\n{}\n```"
