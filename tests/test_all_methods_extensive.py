"""Extensive coverage of train / optimize / execution mechanics.

Exercises every public method the scaffold ``train.py`` / ``optimize.py`` /
platform APIs rely on: compile/fit/evaluate/save/load, GridSearch,
PromptFitterBridge, AgentRunner (local), metrics, judges, search space,
resolve_system_prompt, and sync/async/batch/stream/task routes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform
from agentomatic.agents import (
    AgentDataset,
    AgentExample,
    BaseGraphAgent,
    EarlyStopping,
    History,
)
from agentomatic.agents.history import CallableLoss, MetricLoss, resolve_loss
from agentomatic.agents.metrics import (
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
    OptimizeMetricAdapter,
    ResponseSimilarityMetric,
    WeightedMetric,
)
from agentomatic.agents.optimizers import (
    GridSearchOptimizer,
    NoOpOptimizer,
    PromptFitterBridge,
)
from agentomatic.cli.templates import get_template_files
from agentomatic.core.manifest import AgentManifest
from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.optimize.dataset import Dataset
from agentomatic.optimize.fitter import PromptFitter
from agentomatic.optimize.judges import LocalJudgeMetric, MultiJudgePanel
from agentomatic.optimize.metrics import (
    CompositeMetric,
    ContainsMetric,
    ExactMatchMetric,
    LLMJudgeMetric,
    resolve_metrics,
)
from agentomatic.optimize.metrics import (
    WeightedMetric as OptWeightedMetric,
)
from agentomatic.optimize.runner import AgentRunner
from agentomatic.optimize.search_space import PromptSearchSpace, load_search_space

# =====================================================================
# Shared demo agent
# =====================================================================


@dataclass
class _State:
    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class _DemoAgent(BaseGraphAgent[_State]):
    """Multi-node agent covering stream + prompt resolution."""

    agent_name = "demo"
    agent_description = "Extensive mechanics demo"
    agent_framework = "graph_agent"

    def __init__(self) -> None:
        super().__init__()
        self.prompt_manager = SimpleNamespace(
            get_prompt=lambda *a, **k: "MANAGER_PROMPT",
        )
        self.temperature = 0.0

    def _system_prompt(self) -> str:
        return self.resolve_system_prompt(default="DEFAULT_PROMPT")

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("prep", self.prep)
        g.add_node("respond", self.respond)
        g.set_entry_point("prep")
        g.add_edge("prep", "respond")
        g.set_finish_point("respond")
        return g.compile()

    def prep(self, state: _State) -> _State:
        state.request = state.request.strip()
        return state

    def respond(self, state: _State) -> _State:
        prompt = self._system_prompt()
        marker = "OPT" if "precise" in prompt else "BASE"
        state.output = {
            "response": f"Result for: {state.request} [{marker}]",
            "used_prompt": prompt,
            "temperature": self.temperature,
        }
        return state

    def input_to_state(self, data: dict[str, Any]) -> _State:
        return _State(request=data.get("current_query") or data.get("query") or "")

    def state_to_output(self, state: _State) -> dict[str, Any]:
        return state.output


def _ds() -> AgentDataset:
    examples = [
        AgentExample(
            id=f"e{i}",
            input={"current_query": f"q{i}"},
            expected_output={"response": f"Result for: q{i} [OPT]"},
            split="train" if i < 4 else ("validation" if i == 4 else "test"),
            metadata={"split": "train" if i < 4 else ("validation" if i == 4 else "test")},
        )
        for i in range(6)
    ]
    return AgentDataset(name="ds", examples=examples)


def _tmpl_metrics() -> list:
    rows = [
        ("exact_response", ExactKeyMatchMetric(["response"]), 0.5),
        ("contains_terms", ContainsTermsMetric(["Result"]), 0.3),
        (
            "has_output",
            CallableMetric("has_output", lambda ex, pred: 1.0 if pred.get("response") else 0.0),
            0.2,
        ),
    ]
    return [m for _, m, _ in rows] + [WeightedMetric(rows, name="composite")]


def _poll(client: TestClient, task_id: str, tries: int = 80) -> dict:
    got: dict = {}
    for _ in range(tries):
        got = client.get(f"/api/v1/tasks/{task_id}").json()
        if got["status"] in {"succeeded", "failed", "cancelled"}:
            return got
        time.sleep(0.02)
    return got


# =====================================================================
# resolve_system_prompt priority
# =====================================================================


class TestResolveSystemPrompt:
    def test_override_top_level(self) -> None:
        a = _DemoAgent()
        assert a.resolve_system_prompt({"system_prompt_override": "OVR"}) == "OVR"

    def test_override_metadata_and_context(self) -> None:
        a = _DemoAgent()
        assert a.resolve_system_prompt({"metadata": {"system_prompt_override": "M"}}) == "M"
        assert a.resolve_system_prompt({"context": {"system_prompt_override": "C"}}) == "C"

    def test_request_stash_during_transform(self) -> None:
        a = _DemoAgent()
        out = a.transform({"query": "x", "system_prompt_override": "STASHED"})
        assert out["used_prompt"] == "STASHED"
        assert a._request_system_prompt is None  # cleared after run  # noqa: SLF001

    def test_compiled_config_then_manager_then_default(self) -> None:
        a = _DemoAgent()
        assert a.resolve_system_prompt() == "MANAGER_PROMPT"
        a.compiled_config["system_prompt"] = "FITTED"
        assert a.resolve_system_prompt() == "FITTED"
        a.prompt_manager = None
        a.compiled_config.clear()
        assert a.resolve_system_prompt(default="FALLBACK") == "FALLBACK"

    @pytest.mark.asyncio
    async def test_atransform_honours_override(self) -> None:
        a = _DemoAgent()
        out = await a.atransform({"query": "x", "system_prompt_override": "ASYNC_OVR"})
        assert out["used_prompt"] == "ASYNC_OVR"


# =====================================================================
# BaseGraphAgent lifecycle methods
# =====================================================================


class TestBaseGraphAgentLifecycle:
    def test_invoke_alias_and_traces(self) -> None:
        a = _DemoAgent()
        out = a.invoke({"query": "hi"})
        assert out["response"].startswith("Result for: hi")
        assert a.get_last_trace()
        assert len(a.get_trace_history()) >= 1

    def test_compile_fit_evaluate_save_load_roundtrip(self, tmp_path: Path) -> None:
        a = _DemoAgent()
        ds = _ds()
        metrics = _tmpl_metrics()
        composite = next(m for m in metrics if m.name == "composite")

        a.compile(ds, metrics, optimizer=NoOpOptimizer(), loss=MetricLoss(composite))
        assert a.compiled_metadata["loss"] == "composite_loss"

        history = a.fit(
            ds,
            epochs=2,
            verbose=0,
            validation_data=ds.validation,
            callbacks=[EarlyStopping(monitor="loss", patience=2)],
        )
        assert isinstance(history, History)
        assert "loss" in history.history
        assert "val_loss" in history.history or history.history.get("loss")

        report = a.evaluate(ds.test, metrics)
        assert report.num_examples == 1
        assert 0.0 <= report.pass_rate <= 1.0
        assert report.summary()

        out = tmp_path / "compiled"
        a.compiled_config["system_prompt"] = "You are a precise assistant."
        a.save(str(out))
        assert (out / "config.json").exists()
        assert (out / "metadata.json").exists()
        assert (out / "fit_history.json").exists()

        b = _DemoAgent()
        b.load(str(out))
        assert b.compiled_config["system_prompt"] == "You are a precise assistant."
        assert "precise" in b.transform({"query": "z"})["used_prompt"]

    def test_fit_optimize_knobs_build_prompt_fitter(self) -> None:
        a = _DemoAgent()
        ds = _ds()
        a.compile(ds, _tmpl_metrics())

        class _StubFitter:
            name = "stub"

            async def fit(self, *a, **k):
                return PromptFitResult(
                    best_config=PromptRuntimeConfig(system_prompt="You are precise."),
                    baseline_config=PromptRuntimeConfig(system_prompt="BASE"),
                    best_score=0.8,
                    baseline_score=0.3,
                )

        bridge = PromptFitterBridge(agent_name="demo", fitter=_StubFitter())
        history = a.fit(
            ds,
            epochs=1,
            verbose=0,
            optimizer=bridge,
            optimize_mode="rewrite",
            optimize_prompt=True,
            optimize_params=False,
            max_trials=3,
            search_space=PromptSearchSpace(optimize_system_prompt=True),
        )
        assert history.params["optimize"]["optimizer"] == "rewrite"
        assert a.compiled_config.get("system_prompt") == "You are precise."
        assert a._last_optimize_status == "ok"  # noqa: SLF001

    def test_invalidate_graph_and_visualize_and_manifest(self) -> None:
        a = _DemoAgent()
        g1 = a.graph
        a.invalidate_graph()
        g2 = a.graph
        assert g1 is not g2
        mermaid = a.visualize()
        assert "prep" in mermaid and "respond" in mermaid
        manifest = a.to_manifest()
        assert manifest.name == "demo"
        reg = a.as_registered_agent()
        assert reg.class_instance is a

    def test_load_dataset_helper(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "1",
                    "split": "train",
                    "input": {"current_query": "q"},
                    "expected_output": {"response": "r"},
                }
            )
            + "\n"
        )
        a = _DemoAgent()
        ds = a.load_dataset(str(path))
        assert len(ds) == 1


# =====================================================================
# History / Loss helpers
# =====================================================================


class TestHistoryAndLoss:
    def test_history_best_final_summary(self) -> None:
        h = History(params={"epochs": 2})
        h.record(0, {"loss": 0.8, "acc": 0.2})
        h.record(1, {"loss": 0.3, "acc": 0.9})
        assert h.final("loss") == pytest.approx(0.3)
        assert h.best("loss", mode="min") == (1, pytest.approx(0.3))
        assert h.best("acc", mode="max") == (1, pytest.approx(0.9))
        assert "loss" in h.summary()
        assert h.to_dict()["history"]["loss"] == [0.8, 0.3]

    def test_resolve_loss_variants(self) -> None:
        m = ExactKeyMatchMetric(["response"])
        assert resolve_loss(None) is None
        assert isinstance(resolve_loss(MetricLoss(m)), MetricLoss)
        assert isinstance(resolve_loss(m), MetricLoss)
        assert isinstance(resolve_loss(lambda ex, pred: 0.1), CallableLoss)

    def test_metric_loss_is_one_minus_score(self) -> None:
        m = ExactKeyMatchMetric(["response"])
        loss = MetricLoss(m)
        ex = AgentExample(id="1", input={}, expected_output={})
        assert loss.compute(ex, {"response": "x"}) == pytest.approx(0.0)


# =====================================================================
# GridSearchOptimizer
# =====================================================================


class TestGridSearchOptimizer:
    def test_empty_grid_and_empty_train(self) -> None:
        opt = GridSearchOptimizer({})
        assert opt.optimize(_DemoAgent(), _ds(), []) == {}
        opt2 = GridSearchOptimizer({"temperature": [0.0, 0.5]})
        empty = AgentDataset(name="e", examples=[])
        assert opt2.optimize(_DemoAgent(), empty, []) == {}

    def test_searches_temperature_attribute_and_restores(self) -> None:
        a = _DemoAgent()
        a.temperature = 0.1
        # Higher temperature marker won't change response; use system_prompt.
        opt = GridSearchOptimizer(
            {
                "system_prompt": [
                    "plain",
                    "You are a precise assistant.",
                ]
            },
            max_examples=4,
        )
        metrics = [ResponseSimilarityMetric()]
        best = opt.optimize(a, _ds(), metrics)
        assert "precise" in best["system_prompt"]
        # Original compiled_config must be restored after search.
        assert a.compiled_config.get("system_prompt") != best["system_prompt"] or (
            a.compiled_config.get("system_prompt") is None
        )

    def test_apply_config_sets_attr_and_compiled(self) -> None:
        a = _DemoAgent()
        GridSearchOptimizer._apply_config(a, {"temperature": 0.7, "system_prompt": "X"})
        assert a.temperature == 0.7
        assert a.compiled_config["system_prompt"] == "X"


# =====================================================================
# PromptFitterBridge
# =====================================================================


class TestPromptFitterBridgeMethods:
    def test_skip_empty_dataset(self) -> None:
        bridge = PromptFitterBridge(agent_name="demo", fitter=object())
        a = _DemoAgent()
        cfg = bridge.optimize(a, AgentDataset(name="e"), [])
        assert cfg == {}
        assert a._last_optimize_status.startswith("skipped:")  # noqa: SLF001

    def test_extract_config_without_hasattr(self) -> None:
        a = _DemoAgent()
        result = PromptFitResult(
            best_config=PromptRuntimeConfig(
                system_prompt="BEST",
                user_template="{q}",
                model_choice="m",
            ),
            baseline_config=PromptRuntimeConfig(system_prompt="B"),
            best_score=1.0,
            baseline_score=0.0,
        )
        cfg = PromptFitterBridge._extract_config(a, result)
        assert cfg["system_prompt"] == "BEST"

    def test_split_prefers_metadata_labels(self) -> None:
        opt_ds = _ds().to_optimize_dataset()
        train, val = PromptFitterBridge._split(opt_ds)
        assert len(train) >= 1
        assert len(val) >= 1

    def test_build_fitter_wires_local_agent(self) -> None:
        a = _DemoAgent()
        bridge = PromptFitterBridge(agent_name="demo", max_trials=2)
        a._fit_optimize_options = {  # noqa: SLF001
            "optimizer": "rewrite",
            "max_trials": 2,
            "auto_report": False,
        }
        fitter = bridge._build_fitter(a, "demo")  # noqa: SLF001
        assert isinstance(fitter, PromptFitter)
        assert fitter._runner.agent_callable is not None  # noqa: SLF001

    def test_run_async_inside_running_loop(self) -> None:
        async def _coro():
            return 42

        # Simulate being inside a loop via ThreadPool path.
        import asyncio

        async def _outer():
            return PromptFitterBridge._run_async(_coro())

        assert asyncio.run(_outer()) == 42


# =====================================================================
# AgentRunner
# =====================================================================


class TestAgentRunnerMethods:
    @pytest.mark.asyncio
    async def test_local_run_single_and_dataset(self) -> None:
        a = _DemoAgent()
        from agentomatic.optimize.fitter import _wrap_local_agent

        runner = AgentRunner(agent="demo", agent_callable=_wrap_local_agent(a))
        one = await runner.run_single(
            "q0",
            prompt_override="You are a precise assistant.",
            context=["doc"],
            invoke={"extra_field": 1},
        )
        assert one.error is None
        assert "[OPT]" in one.response

        results = await runner.run_dataset(
            [
                {
                    "query": "q0",
                    "expected_answer": "Result for: q0 [OPT]",
                    "metadata": {"invoke": {}},
                },
                {"query": "q1", "expected": "alt"},  # alias
            ],
            prompt_override="You are a precise assistant.",
            concurrency=2,
        )
        assert len(results) == 2
        assert results[0].expected == "Result for: q0 [OPT]"
        assert results[1].expected == "alt"

    @pytest.mark.asyncio
    async def test_response_text_prefers_output_dict(self) -> None:
        from agentomatic.optimize.runner import _response_text

        assert "a" in _response_text({"output": {"a": 1}})
        assert _response_text({"response": "hi"}) == "hi"
        assert _response_text({"response": {"x": 1}}).startswith("{")


# =====================================================================
# Agent-side metrics
# =====================================================================


class TestAgentMetricsMethods:
    def test_response_similarity_variants(self) -> None:
        m = ResponseSimilarityMetric()
        ex = AgentExample(
            id="1",
            input={},
            expected_output={"response": "Hello World"},
        )
        assert m.score(ex, {"response": "Hello World"}) == pytest.approx(1.0)
        assert ResponseSimilarityMetric(fuzzy=False).score(
            ex, {"response": "hello world"}
        ) == pytest.approx(1.0)
        assert m.score(ex, {"response": ""}) == 0.0
        assert m.score(AgentExample(id="2", input={}), {"response": "x"}) == 0.0
        # expected as plain string / answer key
        ex2 = AgentExample(id="3", input={}, expected_output="abc")
        assert m.score(ex2, {"response": "abc"}) == pytest.approx(1.0)
        ex3 = AgentExample(id="4", input={}, expected_output={"answer": "xyz"})
        assert m.score(ex3, {"response": "xyz"}) == pytest.approx(1.0)

    def test_exact_key_contains_callable_weighted(self) -> None:
        ex = AgentExample(id="1", input={}, expected_output={})
        assert ExactKeyMatchMetric(["response", "x"]).score(ex, {"response": 1}) == 0.5
        assert ExactKeyMatchMetric([]).score(ex, {}) == 1.0
        assert ContainsTermsMetric(["Result", "for"]).score(
            ex, {"response": "Result for: q"}
        ) == pytest.approx(1.0)
        assert ContainsTermsMetric([]).score(ex, {}) == 1.0
        cm = CallableMetric("c", lambda e, p: 0.25)
        assert cm.score(ex, {}) == 0.25
        wm = WeightedMetric(
            [
                ("a", ExactKeyMatchMetric(["response"]), 1.0),
                ("b", CallableMetric("b", lambda e, p: 0.0), 1.0),
            ]
        )
        assert wm.score(ex, {"response": "x"}) == pytest.approx(0.5)
        assert set(wm.last_component_scores) == {"a", "b"}

    def test_optimize_metric_adapter_async_bridge(self, monkeypatch) -> None:
        class _M:
            name = "m"

            async def evaluate(self, query, response, expected=None, context=None):
                from agentomatic.optimize.metrics import EvalResult

                return EvalResult(metric_name="m", score=0.75, reason="ok")

        adapter = OptimizeMetricAdapter(_M())
        ex = AgentExample(
            id="1",
            input={"query": "q"},
            expected_output={"response": "r"},
        )
        assert adapter.score(ex, {"response": "r"}) == pytest.approx(0.75)

        class _Fail:
            name = "f"

            async def evaluate(self, *a, **k):
                from agentomatic.optimize.metrics import EvalResult

                return EvalResult(
                    metric_name="f",
                    score=0.0,
                    reason="down",
                    metadata={"evaluation_failed": True},
                )

        assert OptimizeMetricAdapter(_Fail()).score(ex, {"response": "r"}) == 0.0


# =====================================================================
# Optimize metrics / judges / resolve
# =====================================================================


class TestOptimizeMetricsAndJudges:
    @pytest.mark.asyncio
    async def test_exact_contains_custom(self) -> None:
        assert (await ExactMatchMetric().evaluate("q", "a", expected=None)).score == 0.0
        assert (await ExactMatchMetric(fuzzy=False).evaluate("q", "A", expected="a")).score == 1.0
        assert (
            await ContainsMetric().evaluate("q", "cats and dogs", expected="cats, dogs")
        ).score == 1.0

        from agentomatic.optimize.metrics import CustomMetric

        async def _async_fn(q, r, e, c):
            return 0.6

        assert (await CustomMetric(_async_fn).evaluate("q", "r")).score == 0.6

    @pytest.mark.asyncio
    async def test_llm_judge_and_geval_failure_honest(self, monkeypatch) -> None:
        async def _empty(*a, **k):
            return {}

        monkeypatch.setattr(
            "agentomatic.optimize.llm_types.call_llm_json",
            _empty,
        )
        j = LLMJudgeMetric(criteria="good?")
        r = await j.evaluate("q", "r", expected="e")
        assert r.score == 0.0
        assert r.metadata.get("evaluation_failed") is True

        from agentomatic.optimize.metrics import EvalResult, GEvalMetric

        async def _empty_json(*a, **k):
            return {}

        monkeypatch.setattr(
            "agentomatic.optimize.llm_types.call_llm_json",
            _empty_json,
        )
        g = GEvalMetric(criteria="good?")
        # Honest empty-JSON fallback (used when deepeval is missing/fails).
        gr = await g._fallback_eval("q", "r", None)  # noqa: SLF001
        assert gr.score == 0.0
        assert gr.metadata.get("evaluation_failed") is True

        # evaluate() falls through to fallback on non-ImportError deepeval failures.
        monkeypatch.setattr(
            g,
            "_fallback_eval",
            AsyncMock(
                return_value=EvalResult(
                    metric_name="geval",
                    score=0.0,
                    reason="forced",
                    metadata={"evaluation_failed": True},
                )
            ),
        )

        class _BadGEval:
            def __init__(self, *a, **k):
                raise RuntimeError("deepeval boom")

        monkeypatch.setitem(
            __import__("sys").modules,
            "deepeval.metrics",
            type("Mod", (), {"GEval": _BadGEval})(),
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "deepeval.test_case",
            type(
                "TC",
                (),
                {
                    "LLMTestCase": object,
                    "LLMTestCaseParams": type(
                        "P", (), {"ACTUAL_OUTPUT": "a", "EXPECTED_OUTPUT": "e"}
                    ),
                },
            )(),
        )
        gr2 = await g.evaluate("q", "r")
        assert gr2.metadata.get("evaluation_failed") is True

    @pytest.mark.asyncio
    async def test_composite_partial_and_total_failure(self, monkeypatch) -> None:
        ok = ExactMatchMetric(fuzzy=False)

        class _Fail(LLMJudgeMetric):
            async def evaluate(self, *a, **k):
                from agentomatic.optimize.metrics import EvalResult

                return EvalResult(
                    metric_name="f",
                    score=0.0,
                    reason="fail",
                    metadata={"evaluation_failed": True},
                )

        comp = CompositeMetric(
            metrics=[
                OptWeightedMetric("ok", ok, weight=0.5),
                OptWeightedMetric("bad", _Fail(criteria="x"), weight=0.5),
            ]
        )
        r = await comp.evaluate("q", "yes", expected="yes")
        assert r.score == pytest.approx(1.0)
        assert not r.metadata.get("evaluation_failed")

        all_bad = CompositeMetric(
            metrics=[
                OptWeightedMetric("b1", _Fail(criteria="x"), weight=0.5),
                OptWeightedMetric("b2", _Fail(criteria="y"), weight=0.5),
            ]
        )
        r2 = await all_bad.evaluate("q", "r")
        assert r2.metadata.get("evaluation_failed") is True
        assert hasattr(all_bad, "metrics")
        assert len(all_bad.metrics) == 2

    @pytest.mark.asyncio
    async def test_local_judge_and_panel(self, monkeypatch) -> None:
        async def _json(*a, **k):
            return {
                "overall_score": 0.8,
                "feedback": "good",
                "dimensions": {"correctness": 0.9},
            }

        monkeypatch.setattr(
            "agentomatic.optimize.llm_caller.LLMCaller.call_with_json",
            _json,
        )
        judge = LocalJudgeMetric(dimensions=["correctness", "completeness"])
        r = await judge.evaluate("q", "r", "e")
        assert r.score == 0.8
        assert r.metadata["dimensions"]["completeness"] == 0.8  # inherits overall

        async def _boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(
            "agentomatic.optimize.llm_caller.LLMCaller.call_with_json",
            _boom,
        )
        bad = await LocalJudgeMetric().evaluate_rich("q", "r")
        assert bad.score == 0.0
        assert bad.feedback.startswith("Judge evaluation failed:")

        panel = MultiJudgePanel(judges=[LocalJudgeMetric(name="j1")])
        # all fail
        pr = await panel.evaluate("q", "r")
        assert pr.score == 0.0
        assert pr.metadata.get("evaluation_failed") is True

    def test_resolve_metrics_shorthands(self) -> None:
        ms = resolve_metrics(["exact_match", "contains", "llm_judge", "llm_judge:polite?"])
        assert ms[0].name == "exact_match"
        assert ms[2].name == "llm_judge"
        assert "polite" in ms[3].criteria
        with pytest.raises(ValueError):
            resolve_metrics(["not_a_real_metric"])


# =====================================================================
# PromptFitter helpers
# =====================================================================


class TestPromptFitterHelpers:
    def test_augment_local_judges_wraps_and_extends(self) -> None:
        f = PromptFitter(
            agent="demo",
            local_judges=["ollama/a", "ollama/b"],
            auto_report=False,
        )
        base = ExactMatchMetric()
        wrapped = f._augment_metric_with_local_judges(base)  # noqa: SLF001
        assert isinstance(wrapped, CompositeMetric)
        assert any(wm.name == "local_judges" for wm in wrapped.metrics)

        already = CompositeMetric(
            metrics=[OptWeightedMetric("base", ExactMatchMetric(), weight=1.0)]
        )
        aug = f._augment_metric_with_local_judges(already)  # noqa: SLF001
        assert any(wm.name == "local_judges" for wm in aug.metrics)

    @pytest.mark.asyncio
    async def test_evaluate_config_skips_failed_and_averages(self, monkeypatch) -> None:
        a = _DemoAgent()
        f = PromptFitter(
            agent="demo",
            auto_report=False,
            max_trials=1,
            local_agent=a,
        )

        call = {"n": 0}

        async def _eval(query, response, expected=None, context=None):
            from agentomatic.optimize.metrics import EvalResult

            call["n"] += 1
            if call["n"] == 1:
                return EvalResult(
                    metric_name="m",
                    score=0.0,
                    reason="fail",
                    metadata={"evaluation_failed": True},
                )
            return EvalResult(metric_name="m", score=1.0, reason="ok")

        metric = ExactMatchMetric()
        monkeypatch.setattr(metric, "evaluate", _eval)
        avg, dims, details = await f._evaluate_config(  # noqa: SLF001
            PromptRuntimeConfig(system_prompt="You are precise."),
            Dataset.from_list(
                [
                    {"query": "q0", "expected_answer": "x"},
                    {"query": "q1", "expected_answer": "y"},
                ]
            ),
            metric,
        )
        assert avg == pytest.approx(1.0)
        assert any(d.get("error") for d in details)

    def test_param_suggestions_and_baseline_load(self, tmp_path: Path, monkeypatch) -> None:
        prompts = tmp_path / "agents" / "demo"
        prompts.mkdir(parents=True)
        (prompts / "prompts.json").write_text(
            json.dumps({"v1": {"system": "FROM_FILE", "user_template": "{query}"}})
        )
        monkeypatch.chdir(tmp_path)
        f = PromptFitter(agent="demo", auto_report=False)
        cfg = f._load_baseline_config()  # noqa: SLF001
        assert cfg.system_prompt == "FROM_FILE"

        base = PromptRuntimeConfig(system_prompt="old", model_params={"temperature": 0.0})
        best = PromptRuntimeConfig(
            system_prompt="new longer prompt here",
            model_params={"temperature": 0.5},
            few_shot_examples=[{"q": "a", "a": "b"}],
        )
        suggestions, texts = f._build_param_suggestions(base, best)  # noqa: SLF001
        assert "system_prompt" in suggestions
        assert any("temperature" in t for t in texts)


# =====================================================================
# Search space
# =====================================================================


class TestSearchSpaceMethods:
    def test_sample_active_total_yaml_roundtrip(self, tmp_path: Path) -> None:
        space = PromptSearchSpace(
            optimize_model_params=True,
            model_param_space={"temperature": [0.0, 0.5], "top_p": [1.0]},
        )
        assert "model" in space.active_spaces()
        assert space.total_search_size() >= 2
        assert space.n_combinations("model") == 2
        combos = space.param_combinations("model")
        assert len(combos) == 2
        sample = space.sample_params(1, "model")
        assert len(sample) == 1
        path = tmp_path / "ss.yaml"
        space.to_yaml(path)
        restored = load_search_space(path)
        assert restored.model_param_space["temperature"] == [0.0, 0.5]


# =====================================================================
# Platform APIs: sync / async / batch / stream / optimize / tasks
# =====================================================================


class TestPlatformExecutionSurfaces:
    @pytest.fixture
    def client(self, tmp_path: Path):
        platform = AgentPlatform(
            agents_dir=tmp_path / "agents",
            plugins_dir=tmp_path / "plugins",
            endpoints_dir=tmp_path / "endpoints",
            enable_studio=False,
        )
        reg = _DemoAgent().as_registered_agent()
        platform.register_agent(
            manifest=reg.manifest,
            node_fn=reg.node_fn,
            graph_fn=reg.graph_fn,
            class_instance=reg.class_instance,
        )

        # Also a simple node agent for comparison
        async def _echo(state: dict) -> dict:
            return {"response": f"echo:{state.get('current_query', '')}"}

        platform.register_agent(
            AgentManifest(name="echo", slug="echo", description="e", version="1.0.0"),
            node_fn=_echo,
        )
        with TestClient(platform.build()) as c:
            yield c

    def test_invoke_sync(self, client: TestClient) -> None:
        r = client.post("/api/v1/demo/invoke", json={"query": "hi"})
        assert r.status_code == 200
        assert "Result for: hi" in r.json()["response"]

    def test_invoke_stream_per_node(self, client: TestClient) -> None:
        r = client.post("/api/v1/demo/invoke/stream", json={"query": "stream"})
        assert r.status_code == 200
        body = r.text
        assert "prep" in body
        assert "respond" in body
        assert "Result for: stream" in body
        assert "[DONE]" in body

    def test_optimize_invoke_override(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/demo/optimize/invoke",
            json={
                "query": "opt",
                "system_prompt_override": "You are a precise assistant.",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "[OPT]" in body["response"]
        assert body["metadata"].get("used_prompt") == "You are a precise assistant." or (
            "[OPT]" in body["response"]
        )

    def test_invoke_async_and_events_and_result(self, client: TestClient) -> None:
        r = client.post("/api/v1/demo/invoke/async", json={"query": "async"})
        assert r.status_code == 202
        tid = r.json()["id"]
        assert "links" in r.json()
        done = _poll(client, tid)
        assert done["status"] == "succeeded"
        # events endpoint
        ev = client.get(f"/api/v1/tasks/{tid}/events")
        assert ev.status_code == 200
        result = client.get(f"/api/v1/tasks/{tid}/result").json()
        assert "Result for: async" in result["result"]["response"]

    def test_invoke_batch(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/demo/invoke/batch",
            json={"inputs": [{"query": "a"}, {"query": "b"}]},
        )
        assert r.status_code == 202
        done = _poll(client, r.json()["id"])
        assert done["status"] == "succeeded"
        result = client.get(f"/api/v1/tasks/{r.json()['id']}/result").json()["result"]
        assert len(result) == 2

    def test_chat(self, client: TestClient) -> None:
        r = client.post("/api/v1/demo/chat", json={"content": "chatty"})
        assert r.status_code == 200
        assert "Result for: chatty" in r.json()["response"]


# =====================================================================
# Generated scaffold train/optimize scripts execute
# =====================================================================


class TestScaffoldScriptsExecute:
    def test_generated_train_and_optimize_contracts(self, tmp_path: Path, monkeypatch) -> None:
        files = get_template_files("full", "scaffold_bot")
        assert "TrainConfig" in files["train.py"]
        assert "train_and_report" in files["train.py"]
        assert "augment" in files["train.py"]
        assert "prompt_only" in files["optimize.py"]
        assert "GridSearchOptimizer" in files["optimize.py"]

        agent_dir = tmp_path / "agents" / "scaffold_bot"
        agent_dir.mkdir(parents=True)
        for rel, content in files.items():
            path = agent_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        (tmp_path / "agents" / "__init__.py").write_text("")
        # Minimal stack so optimize prompt_only can resolve models without a live LLM.
        stacks = tmp_path / "stacks"
        stacks.mkdir()
        (stacks / "local.yaml").write_text(
            "name: local\n"
            "llm:\n"
            "  default:\n"
            "    provider: openai\n"
            "    model: stub-model\n"
            "    base_url: http://127.0.0.1:9\n"
            "    api_key: test\n"
        )
        monkeypatch.chdir(tmp_path)
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            from agents.scaffold_bot.optimize import main as opt_main

            monkeypatch.setattr(
                "sys.argv",
                [
                    "optimize",
                    "--strategy",
                    "prompt_only",
                    "--dataset",
                    str(agent_dir / "dataset.jsonl"),
                ],
            )
            opt_main()
            compiled = tmp_path / "compiled" / "scaffold_bot"
            assert (compiled / "config.json").exists()
            cfg = json.loads((compiled / "config.json").read_text())
            assert isinstance(cfg, dict)
        finally:
            sys.path.pop(0)
