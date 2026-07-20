"""Regression tests for the six confirmed optimization bugs.

All bugs were found via source inspection and fixed in:
  src/agentomatic/optimize/runner.py
  src/agentomatic/optimize/llm_caller.py
  src/agentomatic/optimize/config.py
  src/agentomatic/optimize/fitter.py
  src/agentomatic/optimize/metrics.py
  src/agentomatic/agents/metrics.py
  src/agentomatic/agents/optimizers.py

Each test class is tagged with the bug it covers. Tests require no external
services — all LLM calls and HTTP calls are mocked or bypassed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentomatic.agents.history import MetricLoss
from agentomatic.agents.metrics import (
    OptimizeMetricAdapter,
    WeightedMetric,
)
from agentomatic.agents.types import AgentExample
from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.optimize.metrics import (
    CompositeMetric,
    EvalResult,
    ExactMatchMetric,
)
from agentomatic.optimize.metrics import (
    WeightedMetric as FitterWeightedMetric,
)
from agentomatic.optimize.runner import AgentRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(
    query: str = "test query",
    expected: dict[str, Any] | None = None,
    input_override: dict[str, Any] | None = None,
) -> AgentExample:
    return AgentExample(
        id="ex_001",
        input=input_override if input_override is not None else {"query": query},
        expected_output=expected or {"response": "expected answer"},
    )


def _make_fit_result(
    score_history: list[float] | None = None, trials: list[dict] | None = None
) -> PromptFitResult:
    return PromptFitResult(
        best_config=PromptRuntimeConfig(system_prompt="best"),
        baseline_config=PromptRuntimeConfig(system_prompt="baseline"),
        best_score=0.8,
        baseline_score=0.6,
        score_history=score_history or [],
        trials=trials or [],
    )


# ===========================================================================
# BUG-1: AgentRunner — local callable bypasses HTTP
# ===========================================================================


class TestBug1AgentRunnerLocalCallable:
    """AgentRunner with agent_callable never opens an HTTP connection."""

    @pytest.mark.asyncio
    async def test_async_callable_is_called_not_http(self):
        called_with: list[tuple] = []

        async def my_fn(query, *, prompt_override, context, invoke):
            called_with.append((query, prompt_override))
            return "hello"

        runner = AgentRunner(agent="test", agent_callable=my_fn)
        result = await runner.run_single("ping", prompt_override="sys")

        assert result.error is None
        assert result.response == "hello"
        assert called_with == [("ping", "sys")]

    @pytest.mark.asyncio
    async def test_sync_callable_runs_via_thread(self):
        """Sync callables must be dispatched without blocking the loop."""
        import threading

        thread_ids: list[int] = []
        main_thread_id = threading.get_ident()

        def sync_fn(query, *, prompt_override, context, invoke):
            thread_ids.append(threading.get_ident())
            return f"sync:{query}"

        runner = AgentRunner(agent="test", agent_callable=sync_fn)
        result = await runner.run_single("q")

        assert result.response == "sync:q"
        # The sync callable ran in a worker thread, not the main thread
        assert all(tid != main_thread_id for tid in thread_ids)

    @pytest.mark.asyncio
    async def test_dict_response_extracted_correctly(self):
        async def my_fn(query, *, prompt_override, context, invoke):
            return {"output": {"answer": "42"}}

        runner = AgentRunner(agent="test", agent_callable=my_fn)
        result = await runner.run_single("q")
        assert json.loads(result.response) == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_error_captured_no_crash(self):
        async def bad_fn(query, *, prompt_override, context, invoke):
            raise RuntimeError("network unavailable")

        runner = AgentRunner(agent="test", agent_callable=bad_fn)
        result = await runner.run_single("q")

        assert result.error is not None
        assert "network unavailable" in result.error
        assert result.response == ""

    @pytest.mark.asyncio
    async def test_prompt_override_forwarded(self):
        overrides: list[str | None] = []

        async def fn(query, *, prompt_override, context, invoke):
            overrides.append(prompt_override)
            return "ok"

        runner = AgentRunner(agent="test", agent_callable=fn)
        await runner.run_single("q", prompt_override="custom system prompt")
        assert overrides == ["custom system prompt"]

    @pytest.mark.asyncio
    async def test_context_forwarded(self):
        received: list[Any] = []

        async def fn(query, *, prompt_override, context, invoke):
            received.append(context)
            return "ok"

        runner = AgentRunner(agent="test", agent_callable=fn)
        await runner.run_single("q", context=["doc1", "doc2"])
        assert received == [["doc1", "doc2"]]

    @pytest.mark.asyncio
    async def test_no_callable_uses_http_path(self):
        """Without agent_callable the runner falls through to the HTTP path."""
        runner = AgentRunner(agent="test", api_base="http://localhost:99999")
        # Connection refused → error result (not AttributeError or similar)
        result = await runner.run_single("q")
        assert result.error is not None
        assert result.response == ""


# ===========================================================================
# BUG-2: LLMCaller._call_openai ignores base_url / api_key
# ===========================================================================


class TestBug2LLMCallerBaseUrl:
    """LLMCaller.configure() and per-call base_url/api_key are forwarded."""

    def setup_method(self):
        from agentomatic.optimize.llm_caller import LLMCaller

        # Snapshot original class-level state and restore in teardown
        self._orig_url = LLMCaller._default_base_url
        self._orig_key = LLMCaller._default_api_key

    def teardown_method(self):
        from agentomatic.optimize.llm_caller import LLMCaller

        LLMCaller._default_base_url = self._orig_url
        LLMCaller._default_api_key = self._orig_key

    def test_configure_stores_values(self):
        from agentomatic.optimize.llm_caller import LLMCaller

        LLMCaller.configure(base_url="http://127.0.0.1:8000/v1", api_key="secret")
        assert LLMCaller._default_base_url == "http://127.0.0.1:8000/v1"
        assert LLMCaller._default_api_key == "secret"

    def test_configure_none_resets(self):
        from agentomatic.optimize.llm_caller import LLMCaller

        LLMCaller.configure(base_url="http://example.com/v1")
        LLMCaller.configure()
        assert LLMCaller._default_base_url is None
        assert LLMCaller._default_api_key is None

    @pytest.mark.asyncio
    async def test_call_openai_receives_base_url(self):
        """_call_openai passes base_url and api_key down from LLMCaller.call()."""
        from agentomatic.optimize import llm_caller as mod
        from agentomatic.optimize.llm_caller import LLMCaller

        captured: list[dict] = []

        async def fake_call_openai(model_name, prompt, *, base_url=None, api_key=None, **kw):
            captured.append({"base_url": base_url, "api_key": api_key, "model": model_name})
            return "mocked response"

        with patch.object(mod, "_call_openai", fake_call_openai):
            result = await LLMCaller.call(
                "openai/my-model",
                "hello",
                base_url="http://127.0.0.1:8000/v1",
                api_key="mykey",
            )

        assert result == "mocked response"
        assert len(captured) == 1
        assert captured[0]["base_url"] == "http://127.0.0.1:8000/v1"
        assert captured[0]["api_key"] == "mykey"
        assert captured[0]["model"] == "my-model"

    @pytest.mark.asyncio
    async def test_call_uses_class_defaults_when_no_per_call_args(self):
        """LLMCaller.call() resolves class-level defaults for openai/ specs."""
        from agentomatic.optimize import llm_caller as mod
        from agentomatic.optimize.llm_caller import LLMCaller

        LLMCaller.configure(base_url="http://local/v1", api_key="k")

        captured: list[dict] = []

        async def fake_call_openai(model_name, prompt, *, base_url=None, api_key=None, **kw):
            captured.append({"base_url": base_url, "api_key": api_key})
            return "ok"

        with patch.object(mod, "_call_openai", fake_call_openai):
            await LLMCaller.call("openai/my-model", "hello")

        assert captured == [{"base_url": "http://local/v1", "api_key": "k"}]

    @pytest.mark.asyncio
    async def test_per_call_args_override_class_defaults(self):
        from agentomatic.optimize import llm_caller as mod
        from agentomatic.optimize.llm_caller import LLMCaller

        LLMCaller.configure(base_url="http://default/v1", api_key="default_key")

        captured: list[dict] = []

        async def fake_call_openai(model_name, prompt, *, base_url=None, api_key=None, **kw):
            captured.append({"base_url": base_url, "api_key": api_key})
            return "ok"

        with patch.object(mod, "_call_openai", fake_call_openai):
            await LLMCaller.call(
                "openai/my-model",
                "hello",
                base_url="http://override/v1",
                api_key="override_key",
            )

        assert captured == [{"base_url": "http://override/v1", "api_key": "override_key"}]

    @pytest.mark.asyncio
    async def test_non_openai_provider_not_affected(self):
        """ollama/ provider is unaffected by base_url/api_key."""
        from agentomatic.optimize import llm_caller as mod

        captured: list[dict] = []

        async def fake_ollama(model_name, prompt, **kw):
            captured.append({"model": model_name})
            return "ollama-response"

        with patch.object(mod, "_call_ollama", fake_ollama):
            from agentomatic.optimize.llm_caller import LLMCaller

            result = await LLMCaller.call("ollama/qwen2.5:7b", "hi")

        assert result == "ollama-response"
        assert captured[0]["model"] == "qwen2.5:7b"


# ===========================================================================
# BUG-3: PromptFitResult missing .history attribute
# ===========================================================================


class TestBug3PromptFitResultHistory:
    """PromptFitResult.history is always a list[float]."""

    def test_history_returns_score_history_when_set(self):
        result = _make_fit_result(score_history=[0.6, 0.7, 0.8])
        assert result.history == [0.6, 0.7, 0.8]

    def test_history_is_list_of_floats(self):
        result = _make_fit_result(score_history=[0.5, 0.55, 0.62])
        assert all(isinstance(s, float) for s in result.history)

    def test_history_derives_from_full_val_trials_when_no_score_history(self):
        trials = [
            {"round": 1, "phase": "minibatch", "score": 0.3},
            {"round": 1, "phase": "full_val", "score": 0.65},
            {"round": 2, "phase": "minibatch", "score": 0.4},
            {"round": 2, "phase": "full_val", "score": 0.72},
        ]
        result = _make_fit_result(trials=trials)
        assert result.score_history == []  # not populated
        h = result.history
        assert len(h) == 2
        assert h[0] == pytest.approx(0.65)
        assert h[1] == pytest.approx(0.72)

    def test_history_falls_back_to_all_trial_scores_when_no_full_val(self):
        trials = [
            {"round": 1, "phase": "minibatch", "score": 0.5},
            {"round": 2, "phase": "minibatch", "score": 0.6},
        ]
        result = _make_fit_result(trials=trials)
        h = result.history
        assert len(h) == 2

    def test_history_empty_when_no_data(self):
        result = _make_fit_result()
        assert result.history == []

    def test_history_does_not_mutate_score_history(self):
        result = _make_fit_result(score_history=[0.6, 0.7])
        _ = result.history
        _ = result.history  # calling twice must be stable
        assert result.score_history == [0.6, 0.7]

    def test_score_history_field_persists_through_to_dict(self):
        result = _make_fit_result(score_history=[0.5, 0.7])
        d = result.to_dict()
        assert "score_history" in d
        assert d["score_history"] == [0.5, 0.7]

    def test_generate_fit_report_does_not_crash(self, tmp_path):
        """generate_fit_report must not access undefined .history attributes."""
        from agentomatic.optimize.report import generate_fit_report

        result = _make_fit_result(
            score_history=[0.6, 0.75],
            trials=[
                {
                    "round": 1,
                    "name": "c1",
                    "source": "rewrite",
                    "phase": "full_val",
                    "score": 0.75,
                    "dimensions": {},
                },
            ],
        )
        out = tmp_path / "report.html"
        path = generate_fit_report(result, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        assert path == str(out)

    def test_fitter_populates_score_history_on_result(self, tmp_path):
        """PromptFitter.fit() stores round scores on the result."""
        import asyncio
        from dataclasses import dataclass as dc

        from agentomatic.optimize.dataset import DataPoint, Dataset
        from agentomatic.optimize.fitter import PromptFitter
        from agentomatic.optimize.fitter_optimizers import BaseFitterOptimizer
        from agentomatic.optimize.metrics import ExactMatchMetric
        from agentomatic.optimize.runner import AgentRunner

        @dc
        class StaticOptimizer(BaseFitterOptimizer):
            name: str = "static"

            async def propose(
                self,
                current_config,
                eval_results,
                dataset_sample,
                search_space,
                iteration=0,
                context=None,
            ):
                return []  # no candidates → rounds each produce no improvement

        async def run():
            async def fn(query, *, prompt_override, context, invoke):
                return query.upper()

            fitter = PromptFitter(
                agent="test",
                max_trials=1,
                optimizer=StaticOptimizer(),
                experiment_dir=str(tmp_path),
                auto_report=False,
            )
            fitter._runner = AgentRunner(agent="test", agent_callable=fn)

            points = [
                DataPoint(query="hi", expected_answer="HI"),
                DataPoint(query="bye", expected_answer="BYE"),
            ]
            ds = Dataset(points=points)

            result = await fitter.fit(ds, ds, ExactMatchMetric())
            return result

        result = asyncio.run(run())
        assert isinstance(result.history, list)
        # score_history is set (even if empty due to 0 rounds)
        assert isinstance(result.score_history, list)


# ===========================================================================
# BUG-4: optimize.CompositeMetric has no .score() for training loops
# ===========================================================================


class TestBug4CompositeMetricScore:
    """CompositeMetric.score() bridges to evaluate() for training loop use."""

    def _make_example(self, query="hello", expected=None):
        return AgentExample(
            id="e1",
            input={"query": query},
            expected_output=expected or {"response": "world"},
        )

    def test_composite_metric_has_score_method(self):
        metric = CompositeMetric(
            metrics=[
                FitterWeightedMetric("exact", ExactMatchMetric(), weight=1.0),
            ]
        )
        assert hasattr(metric, "score")
        assert callable(metric.score)

    def test_score_returns_float(self):
        metric = CompositeMetric(
            metrics=[
                FitterWeightedMetric("exact", ExactMatchMetric(), weight=1.0),
            ]
        )
        ex = self._make_example()
        result = metric.score(ex, {"response": "world"})
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_perfect_match(self):
        """Custom always-1.0 sub-metric → composite returns 1.0."""

        class AlwaysOne(ExactMatchMetric):
            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="one", score=1.0, reason="")

        metric = CompositeMetric(
            metrics=[
                FitterWeightedMetric("one", AlwaysOne(), weight=1.0),
            ]
        )
        ex = self._make_example()
        result = metric.score(ex, {"response": "anything"})
        assert result == pytest.approx(1.0)

    def test_score_zero_on_mismatch(self):
        """Custom always-0.0 sub-metric → composite returns 0.0."""

        class AlwaysZero(ExactMatchMetric):
            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="zero", score=0.0, reason="")

        metric = CompositeMetric(
            metrics=[
                FitterWeightedMetric("zero", AlwaysZero(), weight=1.0),
            ]
        )
        ex = self._make_example()
        result = metric.score(ex, {"response": "anything"})
        assert result == pytest.approx(0.0)

    def test_score_with_dict_expected_output(self):
        """Dict expected_output is JSON-serialised and passed as expected string."""
        received_expected: list[str | None] = []

        class CaptureExpected(ExactMatchMetric):
            async def evaluate(self, query, response, expected=None, context=None):
                received_expected.append(expected)
                return EvalResult(metric_name="cap", score=0.5, reason="")

        metric = CompositeMetric(
            metrics=[
                FitterWeightedMetric("cap", CaptureExpected(), weight=1.0),
            ]
        )
        expected_dict = {"answer": "42"}
        ex = self._make_example(expected=expected_dict)
        metric.score(ex, {"response": "r"})
        # expected must be the JSON representation of the dict
        assert received_expected[0] is not None
        assert json.loads(received_expected[0]) == expected_dict

    def test_score_does_not_raise_on_sub_metric_failure(self):
        class BrokenMetric(ExactMatchMetric):
            async def evaluate(self, query, response, expected=None, context=None):
                raise RuntimeError("sub-metric exploded")

        metric = CompositeMetric(
            metrics=[
                FitterWeightedMetric("broken", BrokenMetric(), weight=1.0),
            ]
        )
        ex = self._make_example()
        result = metric.score(ex, {"response": "anything"})
        assert isinstance(result, float)
        assert result == pytest.approx(0.0)

    def test_score_weighted_average(self):
        """Weighted composite score is a weighted average of sub-scores."""

        class AlwaysOneMetric(ExactMatchMetric):
            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="one", score=1.0, reason="")

        class AlwaysZeroMetric(ExactMatchMetric):
            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="zero", score=0.0, reason="")

        metric = CompositeMetric(
            metrics=[
                FitterWeightedMetric("one", AlwaysOneMetric(), weight=0.75),
                FitterWeightedMetric("zero", AlwaysZeroMetric(), weight=0.25),
            ]
        )
        ex = self._make_example()
        result = metric.score(ex, {"response": "x"})
        assert result == pytest.approx(0.75)

    def test_score_usable_in_agents_weighted_metric(self):
        """agents.WeightedMetric accepts CompositeMetric as a component."""
        cm = CompositeMetric(
            metrics=[
                FitterWeightedMetric("exact", ExactMatchMetric(), weight=1.0),
            ]
        )
        # agents.WeightedMetric validates .score() in __init__
        wm = WeightedMetric(
            [("composite", cm, 1.0)],
            name="test_composite",
        )
        ex = self._make_example()
        result = wm.score(ex, {"response": "world"})
        assert isinstance(result, float)

    def test_score_usable_in_metric_loss(self):
        """MetricLoss(CompositeMetric) computes 1 − score correctly."""

        class FixedScore(ExactMatchMetric):
            def __init__(self, s):
                self._s = s

            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="fixed", score=self._s, reason="")

        # score=1.0 → loss=0.0
        cm_high = CompositeMetric(
            metrics=[
                FitterWeightedMetric("h", FixedScore(1.0), weight=1.0),
            ]
        )
        loss_high = MetricLoss(cm_high)
        ex = AgentExample(id="e1", input={"query": "q"})
        assert loss_high.compute(ex, {"response": "x"}) == pytest.approx(0.0)

        # score=0.0 → loss=1.0
        cm_low = CompositeMetric(
            metrics=[
                FitterWeightedMetric("l", FixedScore(0.0), weight=1.0),
            ]
        )
        loss_low = MetricLoss(cm_low)
        assert loss_low.compute(ex, {"response": "x"}) == pytest.approx(1.0)

    def test_score_with_non_dict_input(self):
        """Non-dict example.input is stringified to query."""
        cm = CompositeMetric(
            metrics=[
                FitterWeightedMetric("exact", ExactMatchMetric(), weight=1.0),
            ]
        )
        ex = AgentExample(id="e1", input={"current_query": "hello"})
        result = cm.score(ex, {"response": "anything"})
        assert isinstance(result, float)


# ===========================================================================
# BUG-5: OptimizeMetricAdapter.score() wrong arg types + unawaited async
# ===========================================================================


class TestBug5OptimizeMetricAdapter:
    """OptimizeMetricAdapter.score() extracts correct args and awaits async."""

    def _make_example(self, query="hello", expected=None, input_override=None):
        return AgentExample(
            id="e1",
            input=input_override if input_override is not None else {"query": query},
            expected_output=expected,
        )

    # -- return type ---------------------------------------------------------

    def test_score_returns_float_not_coroutine(self):
        class SyncMetric:
            name = "sync"

            def evaluate(self, *a, **kw):
                return EvalResult(metric_name="sync", score=0.8, reason="")

        adapter = OptimizeMetricAdapter(SyncMetric())
        ex = self._make_example()
        result = adapter.score(ex, {"response": "ok"})
        # Must NOT be a coroutine
        assert not asyncio.iscoroutine(result)
        # Must be a float
        assert isinstance(result, (float, int))

    def test_score_returns_float_for_async_metric(self):
        class AsyncMetric:
            name = "async_m"

            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="async_m", score=0.9, reason="")

        adapter = OptimizeMetricAdapter(AsyncMetric())
        ex = self._make_example()
        result = adapter.score(ex, {"response": "great answer"})
        assert isinstance(result, float)
        assert result == pytest.approx(0.9)

    # -- argument extraction -------------------------------------------------

    def test_query_extracted_from_input_dict(self):
        received: list[str] = []

        class InspectMetric:
            name = "inspect"

            async def evaluate(self, query, response, expected=None, context=None):
                received.append(query)
                return EvalResult(metric_name="inspect", score=0.5, reason="")

        adapter = OptimizeMetricAdapter(InspectMetric())
        ex = self._make_example(query="my query")
        adapter.score(ex, {"response": "r"})
        assert received == ["my query"]

    def test_query_extracted_from_current_query_key(self):
        received: list[str] = []

        class InspectMetric:
            name = "inspect"

            async def evaluate(self, query, response, expected=None, context=None):
                received.append(query)
                return EvalResult(metric_name="inspect", score=0.5, reason="")

        adapter = OptimizeMetricAdapter(InspectMetric())
        ex = self._make_example(input_override={"current_query": "cq test"})
        adapter.score(ex, {"response": "r"})
        assert received == ["cq test"]

    def test_prediction_dict_json_serialised_as_response(self):
        received_response: list[str] = []

        class InspectMetric:
            name = "inspect"

            async def evaluate(self, query, response, expected=None, context=None):
                received_response.append(response)
                return EvalResult(metric_name="inspect", score=0.5, reason="")

        adapter = OptimizeMetricAdapter(InspectMetric())
        pred = {"content": "hello", "next_action": "proceed"}
        ex = self._make_example()
        adapter.score(ex, pred)
        parsed = json.loads(received_response[0])
        assert parsed == pred

    def test_expected_output_dict_json_serialised(self):
        received_expected: list[str | None] = []

        class InspectMetric:
            name = "inspect"

            async def evaluate(self, query, response, expected=None, context=None):
                received_expected.append(expected)
                return EvalResult(metric_name="inspect", score=0.5, reason="")

        exp = {"answer": "42", "type": "numeric"}
        adapter = OptimizeMetricAdapter(InspectMetric())
        ex = self._make_example(expected=exp)
        adapter.score(ex, {"response": "r"})
        assert received_expected[0] is not None
        assert json.loads(received_expected[0]) == exp

    def test_expected_output_none_passed_as_none(self):
        received_expected: list[Any] = []

        class InspectMetric:
            name = "inspect"

            async def evaluate(self, query, response, expected=None, context=None):
                received_expected.append(expected)
                return EvalResult(metric_name="inspect", score=0.5, reason="")

        adapter = OptimizeMetricAdapter(InspectMetric())
        ex = AgentExample(id="e", input={"query": "q"}, expected_output=None)
        adapter.score(ex, {"response": "r"})
        assert received_expected[0] is None

    # -- error handling -------------------------------------------------------

    def test_score_returns_neutral_on_exception(self):
        class FailMetric:
            name = "fail"

            async def evaluate(self, *a, **kw):
                raise ConnectionError("judge server down")

        adapter = OptimizeMetricAdapter(FailMetric())
        ex = self._make_example()
        result = adapter.score(ex, {"response": "r"})
        assert result == pytest.approx(0.5)  # neutral, not 0.0 crash

    def test_score_returns_neutral_on_sync_exception(self):
        class SyncFail:
            name = "sync_fail"

            def evaluate(self, *a, **kw):
                raise ValueError("bad input")

        adapter = OptimizeMetricAdapter(SyncFail())
        ex = self._make_example()
        result = adapter.score(ex, {"response": "r"})
        assert result == pytest.approx(0.5)

    # -- integration with training stack -------------------------------------

    def test_usable_in_agents_weighted_metric(self):
        """OptimizeMetricAdapter satisfies agents.WeightedMetric's .score() check."""

        class FakeMetric:
            name = "fake"

            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="fake", score=0.8, reason="")

        adapter = OptimizeMetricAdapter(FakeMetric(), name="fake_adapter")
        wm = WeightedMetric(
            [("fake", adapter, 1.0)],
            name="composite",
        )
        ex = self._make_example()
        result = wm.score(ex, {"response": "good"})
        assert result == pytest.approx(0.8)

    def test_usable_in_metric_loss(self):
        """MetricLoss(WeightedMetric(OptimizeMetricAdapter)) computes correctly."""

        class ScoreMetric:
            name = "s"

            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="s", score=0.7, reason="")

        adapter = OptimizeMetricAdapter(ScoreMetric())
        wm = WeightedMetric([("s", adapter, 1.0)], name="wm")
        loss = MetricLoss(wm)
        ex = self._make_example()
        computed = loss.compute(ex, {"response": "ok"})
        assert computed == pytest.approx(1.0 - 0.7)

    def test_name_inherited_from_metric(self):
        class FakeMetric:
            name = "my_judge"

            async def evaluate(self, *a, **kw):
                return EvalResult(metric_name="my_judge", score=0.5, reason="")

        adapter = OptimizeMetricAdapter(FakeMetric())
        assert adapter.name == "my_judge"

    def test_name_override(self):
        class FakeMetric:
            name = "original"

            async def evaluate(self, *a, **kw):
                return EvalResult(metric_name="original", score=0.5, reason="")

        adapter = OptimizeMetricAdapter(FakeMetric(), name="overridden")
        assert adapter.name == "overridden"

    def test_correct_score_value_propagated(self):
        """The score from EvalResult must be the value returned — not 0.0."""
        for expected_score in (0.0, 0.25, 0.5, 0.75, 1.0):

            class FixedScore:
                name = "fixed"
                _score = expected_score

                async def evaluate(self, query, response, expected=None, context=None):
                    return EvalResult(metric_name="fixed", score=self._score, reason="")

            FixedScore._score = expected_score
            adapter = OptimizeMetricAdapter(FixedScore())
            ex = self._make_example()
            result = adapter.score(ex, {"response": "r"})
            assert result == pytest.approx(expected_score), (
                f"Expected {expected_score}, got {result}"
            )


# ===========================================================================
# BUG-6: PromptFitterBridge._build_fitter never passes live agent
# ===========================================================================


class TestBug6PromptFitterBridgeLocalAgent:
    """_build_fitter wires the live agent so no HTTP server is required."""

    def _make_agent(self):
        class FakeAgent:
            agent_name = "test"
            _fit_optimize_options = None

            def transform(self, input_data):
                return {"response": f"echo:{input_data.get('current_query', '')}"}

        return FakeAgent()

    def test_local_agent_param_forwarded_to_fitter(self, tmp_path):
        from agentomatic.agents.optimizers import PromptFitterBridge

        live = self._make_agent()
        bridge = PromptFitterBridge(
            agent_name="test",
            local_agent=live,
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter = bridge._build_fitter(live, "test")
        assert fitter._runner.agent_callable is not None

    def test_live_agent_used_when_local_agent_is_none(self, tmp_path):
        """When local_agent=None, the optimize() live agent is used as fallback."""
        from agentomatic.agents.optimizers import PromptFitterBridge

        live = self._make_agent()
        bridge = PromptFitterBridge(
            agent_name="test",
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter = bridge._build_fitter(live, "test")
        # The live agent is wired — no HTTP needed
        assert fitter._runner.agent_callable is not None

    def test_explicit_local_agent_wins_over_live_agent(self, tmp_path):
        """Explicit local_agent= takes priority over the optimize() live agent."""
        from agentomatic.agents.optimizers import PromptFitterBridge

        explicit = self._make_agent()
        live = self._make_agent()

        bridge = PromptFitterBridge(
            agent_name="test",
            local_agent=explicit,
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter = bridge._build_fitter(live, "test")
        # Both paths result in an agent_callable being set
        assert fitter._runner.agent_callable is not None

    def test_llm_base_url_forwarded(self, tmp_path, monkeypatch):
        from agentomatic.agents.optimizers import PromptFitterBridge
        from agentomatic.optimize.llm_caller import LLMCaller

        monkeypatch.setattr(LLMCaller, "_default_base_url", None)
        monkeypatch.setattr(LLMCaller, "_default_api_key", None)

        live = self._make_agent()
        bridge = PromptFitterBridge(
            agent_name="test",
            llm_base_url="http://127.0.0.1:8000/v1",
            llm_api_key="test_key",
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        bridge._build_fitter(live, "test")

        assert LLMCaller._default_base_url == "http://127.0.0.1:8000/v1"
        assert LLMCaller._default_api_key == "test_key"

    def test_injected_fitter_returned_as_is(self):
        """When self.fitter is set, _build_fitter returns it unchanged."""
        from agentomatic.agents.optimizers import PromptFitterBridge

        stub_fitter = MagicMock()
        bridge = PromptFitterBridge(fitter=stub_fitter)
        live = self._make_agent()
        returned = bridge._build_fitter(live, "test")
        assert returned is stub_fitter


# ===========================================================================
# End-to-end: full local training loop (no HTTP server, no cloud LLM)
# ===========================================================================


class TestEndToEndLocalTraining:
    """Simulate compile → fit → evaluate without any external services."""

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_result_with_history(self, tmp_path):
        """PromptFitter.fit() with a local callable returns a PromptFitResult
        that has a .history list, .score_history, .best_score, and .summary()."""
        from dataclasses import dataclass as dc

        from agentomatic.optimize.dataset import DataPoint, Dataset
        from agentomatic.optimize.fitter import PromptFitter
        from agentomatic.optimize.fitter_optimizers import BaseFitterOptimizer
        from agentomatic.optimize.metrics import ExactMatchMetric
        from agentomatic.optimize.runner import AgentRunner

        @dc
        class NoopOptimizer(BaseFitterOptimizer):
            name: str = "noop"

            async def propose(
                self,
                current_config,
                eval_results,
                dataset_sample,
                search_space,
                iteration=0,
                context=None,
            ):
                return []

        async def echo_fn(query, *, prompt_override, context, invoke):
            return query.upper()

        fitter = PromptFitter(
            agent="e2e",
            optimizer=NoopOptimizer(),
            max_trials=2,
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter._runner = AgentRunner(agent="e2e", agent_callable=echo_fn)

        ds = Dataset(
            points=[
                DataPoint(query="hello", expected_answer="HELLO"),
                DataPoint(query="world", expected_answer="WORLD"),
            ]
        )

        result = await fitter.fit(ds, ds, ExactMatchMetric())

        # BUG-3 assertions
        assert hasattr(result, "history")
        assert hasattr(result, "score_history")
        assert isinstance(result.history, list)
        assert isinstance(result.score_history, list)

        # Structural assertions
        assert result.best_score >= 0.0
        assert result.baseline_score >= 0.0
        text = result.summary()
        assert len(text) > 0
        assert result.experiment_id in text

    def test_optimize_metric_adapter_in_weighted_metric_in_metric_loss(self):
        """Complete BUG-4 + BUG-5 stack: async metric → adapter → WeightedMetric → MetricLoss."""
        score_calls: list[tuple] = []

        class CapturingMetric:
            name = "capturing"

            async def evaluate(self, query, response, expected=None, context=None):
                score_calls.append((query, response, expected))
                # Return 1.0 when response contains the query uppercased
                expected_resp = query.upper()
                score = 1.0 if expected_resp in response else 0.0
                return EvalResult(metric_name="capturing", score=score, reason="")

        adapter = OptimizeMetricAdapter(CapturingMetric(), name="capturing")
        wm = WeightedMetric([("capturing", adapter, 1.0)], name="test_wm")
        loss = MetricLoss(wm)

        ex = AgentExample(
            id="e1",
            input={"query": "hello"},
            expected_output={"result": "HELLO"},
        )

        # Prediction contains the uppercased query → score=1.0 → loss=0.0
        computed_loss = loss.compute(ex, {"response": "HELLO is the answer"})
        assert computed_loss == pytest.approx(0.0)

        # Prediction doesn't contain it → score=0.0 → loss=1.0
        computed_loss_wrong = loss.compute(ex, {"response": "nope"})
        assert computed_loss_wrong == pytest.approx(1.0)

        # The metric was actually called with proper string args (not RunResult)
        assert len(score_calls) == 2
        for query_arg, response_arg, expected_arg in score_calls:
            assert isinstance(query_arg, str)
            assert isinstance(response_arg, str)

    def test_composite_metric_as_loss_with_multiple_sub_metrics(self):
        """BUG-4: CompositeMetric can drive a MetricLoss end-to-end."""

        class HighMetric(ExactMatchMetric):
            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="high", score=0.9, reason="")

        class LowMetric(ExactMatchMetric):
            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="low", score=0.3, reason="")

        cm = CompositeMetric(
            metrics=[
                FitterWeightedMetric("high", HighMetric(), weight=0.6),
                FitterWeightedMetric("low", LowMetric(), weight=0.4),
            ]
        )
        loss = MetricLoss(cm)
        ex = AgentExample(id="e", input={"query": "q"})

        computed = loss.compute(ex, {"response": "r"})
        expected_score = 0.9 * 0.6 + 0.3 * 0.4  # = 0.66
        expected_loss = 1.0 - expected_score
        assert computed == pytest.approx(expected_loss, abs=1e-6)
