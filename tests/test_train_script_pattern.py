"""Tests for every API surface used by the train_next.py script pattern.

Covers all imports, helper functions, metric composition, PromptFitterBridge
construction, compile/fit kwargs, result accessors, and the full end-to-end
training loop — all without a running HTTP server or cloud LLM.

Mirrors the script structure::

    load stack → AgentDataset → metrics → PromptFitterBridge (local_agent)
    → compile → fit → evaluate → generate_fit_report → fit_result.history

"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Re-export surface verified by import tests
# ---------------------------------------------------------------------------
from agentomatic.agents import (
    AgentDataset,
    CallableMetric,
    EarlyStopping,
    ExactKeyMatchMetric,
    MetricLoss,
    OptimizeMetricAdapter,
    PromptFitterBridge,
    WeightedMetric,
)
from agentomatic.agents.types import AgentExample
from agentomatic.optimize import (
    CustomMetric,
    LocalJudgeMetric,
    PromptSearchSpace,
    generate_fit_report,
)
from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.optimize.metrics import EvalResult

# ===========================================================================
# Minimal test agent (mirrors BaseGraphAgent subclass in user scripts)
# ===========================================================================


class _EchoAgent:
    """Minimal agent: returns 'current_query' uppercased as JSON content."""

    agent_name = "test_echo"
    _fit_optimize_options = None
    system_prompt = "You are a helpful assistant."

    def transform(self, input_data: dict[str, Any]) -> dict[str, Any]:
        query = input_data.get("current_query", input_data.get("query", ""))
        override = input_data.get("metadata", {}).get("system_prompt_override")
        if override:
            self.system_prompt = override
        return {
            "content": query.upper(),
            "next_action": "done",
        }


# ===========================================================================
# Script helper functions
# ===========================================================================

REQUIRED_KEYS = ["content", "next_action"]


def _keyword_score(example: Any, prediction: dict[str, Any]) -> float:
    expected = example.expected_output or {}
    text = json.dumps(prediction, ensure_ascii=False).lower()
    terms: list[str] = []
    for key, val in expected.items():
        if isinstance(val, str) and val.strip() and val.lower() not in {"true", "false"}:
            terms.append(val)
        elif val is True:
            terms.append(str(key))
    if not terms:
        return 1.0
    return sum(1 for t in terms if t.lower() in text) / len(terms)


def _json_valid(example: Any, prediction: dict[str, Any]) -> float:
    return 1.0 if isinstance(prediction, dict) and prediction else 0.0


def _token_sets(example: Any, prediction: dict[str, Any]) -> tuple[dict, dict]:
    expected = example.expected_output or {}
    exp_tok = {k: set(str(expected.get(k, "")).lower().split()) for k in REQUIRED_KEYS}
    pred_tok = {k: set(str(prediction.get(k, "")).lower().split()) for k in REQUIRED_KEYS}
    return exp_tok, pred_tok


def _precision(example: Any, prediction: dict[str, Any]) -> float:
    exp_tok, pred_tok = _token_sets(example, prediction)
    scores = []
    for k in REQUIRED_KEYS:
        if not pred_tok[k]:
            scores.append(0.0)
        elif not exp_tok[k]:
            scores.append(1.0)
        else:
            scores.append(len(exp_tok[k] & pred_tok[k]) / len(pred_tok[k]))
    return sum(scores) / max(len(scores), 1)


def _recall(example: Any, prediction: dict[str, Any]) -> float:
    exp_tok, pred_tok = _token_sets(example, prediction)
    scores = []
    for k in REQUIRED_KEYS:
        if not exp_tok[k]:
            scores.append(1.0)
        elif not pred_tok[k]:
            scores.append(0.0)
        else:
            scores.append(len(exp_tok[k] & pred_tok[k]) / len(exp_tok[k]))
    return sum(scores) / max(len(scores), 1)


def _f1(example: Any, prediction: dict[str, Any]) -> float:
    p, r = _precision(example, prediction), _recall(example, prediction)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _opt_composite(
    query: str,
    response: str,
    expected: str | None = None,
    context: Any = None,
) -> float:
    if not response:
        return 0.0
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        start, end = response.find("{"), response.rfind("}")
        if start < 0 or end <= start:
            return 0.0
        try:
            data = json.loads(response[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return 0.0
    if not isinstance(data, dict):
        return 0.0
    key_score = sum(1 for k in REQUIRED_KEYS if k in data) / max(len(REQUIRED_KEYS), 1)
    try:
        exp_data = json.loads(expected) if isinstance(expected, str) else (expected or {})
    except (json.JSONDecodeError, TypeError):
        exp_data = {}
    text = json.dumps(data, ensure_ascii=False).lower()
    terms = [
        val
        for val in (exp_data.values() if isinstance(exp_data, dict) else [])
        if isinstance(val, str) and val.strip() and val.lower() not in {"true", "false"}
    ]
    kw_score = (
        sum(1 for t in terms if t.lower() in text) / len(terms) if terms else key_score
    )
    return 0.5 * key_score + 0.5 * kw_score


def _model_spec(entry: Any) -> str:
    return f"{entry.provider}/{entry.model}"


# ===========================================================================
# Tests: imports
# ===========================================================================


class TestImports:
    """Every import the script uses must resolve."""

    def test_agents_imports(self):
        from agentomatic.agents import (  # noqa: F401
            AgentDataset,
            CallableMetric,
            EarlyStopping,
            ExactKeyMatchMetric,
            MetricLoss,
            OptimizeMetricAdapter,
            PromptFitterBridge,
            WeightedMetric,
        )

    def test_optimize_imports(self):
        from agentomatic.optimize import (  # noqa: F401
            CustomMetric,
            LocalJudgeMetric,
            PromptSearchSpace,
            generate_fit_report,
        )

    def test_stack_imports(self):
        from agentomatic.stacks.manager import StackManager  # noqa: F401

    def test_providers_imports(self):
        from agentomatic.providers import (  # noqa: F401
            apply_stack_defaults,
            get_llm_for_agent,
        )

    def test_config_settings_import(self):
        from agentomatic.config.settings import load_environment  # noqa: F401


# ===========================================================================
# Tests: helper functions
# ===========================================================================


class TestHelperFunctions:
    """All helpers used in the script behave correctly."""

    def _ex(self, expected: dict | None = None) -> AgentExample:
        return AgentExample(
            id="e1",
            input={"query": "test"},
            expected_output=expected or {"content": "hello world", "next_action": "done"},
        )

    # --- _json_valid --------------------------------------------------------

    def test_json_valid_with_dict(self):
        assert _json_valid(self._ex(), {"content": "x"}) == 1.0

    def test_json_valid_empty_dict(self):
        assert _json_valid(self._ex(), {}) == 0.0

    def test_json_valid_non_dict(self):
        assert _json_valid(self._ex(), "string") == 0.0  # type: ignore[arg-type]

    # --- _keyword_score -----------------------------------------------------

    def test_keyword_score_all_found(self):
        ex = self._ex({"content": "hello", "next_action": "proceed"})
        pred = {"content": "hello world", "next_action": "proceed with caution"}
        assert _keyword_score(ex, pred) == pytest.approx(1.0)

    def test_keyword_score_none_found(self):
        ex = self._ex({"content": "foobar"})
        pred = {"content": "completely different"}
        assert _keyword_score(ex, pred) == pytest.approx(0.0)

    def test_keyword_score_no_terms_returns_1(self):
        ex = AgentExample(id="e", input={}, expected_output={"flag": True})
        # "True" values add the key name as a term
        result = _keyword_score(ex, {"flag": True})
        assert result == pytest.approx(1.0)

    def test_keyword_score_no_expected_returns_1(self):
        ex = AgentExample(id="e", input={}, expected_output=None)
        assert _keyword_score(ex, {"content": "x"}) == pytest.approx(1.0)

    # --- _precision / _recall / _f1 -----------------------------------------

    def test_precision_perfect(self):
        ex = self._ex({"content": "hello world", "next_action": "done"})
        pred = {"content": "hello world", "next_action": "done"}
        assert _precision(ex, pred) == pytest.approx(1.0)

    def test_recall_zero_when_empty_prediction(self):
        ex = self._ex({"content": "hello world", "next_action": "done"})
        pred = {"content": "", "next_action": ""}
        # Empty pred_tok → 0 for each key
        assert _recall(ex, pred) == pytest.approx(0.0)

    def test_f1_harmonic_mean(self):
        ex = self._ex({"content": "hello world", "next_action": "go"})
        pred = {"content": "hello world", "next_action": "go right now"}
        p = _precision(ex, pred)
        r = _recall(ex, pred)
        f1 = _f1(ex, pred)
        if (p + r) > 0:
            assert f1 == pytest.approx(2 * p * r / (p + r))
        else:
            assert f1 == pytest.approx(0.0)

    def test_f1_zero_when_no_overlap(self):
        ex = self._ex({"content": "xyz abc", "next_action": "stop"})
        pred = {"content": "completely different", "next_action": "go"}
        assert _f1(ex, pred) == pytest.approx(0.0)

    # --- _opt_composite ------------------------------------------------------

    def test_opt_composite_empty_response(self):
        assert _opt_composite("q", "") == pytest.approx(0.0)

    def test_opt_composite_valid_json_all_keys(self):
        response = json.dumps({"content": "hello", "next_action": "done"})
        result = _opt_composite("q", response)
        assert result == pytest.approx(1.0)  # key_score=1.0, kw_score falls back to 1.0

    def test_opt_composite_missing_keys(self):
        response = json.dumps({"content": "hello"})   # missing next_action
        result = _opt_composite("q", response)
        assert result == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)  # key_score=0.5, kw=0.5

    def test_opt_composite_non_json_response(self):
        assert _opt_composite("q", "not json at all") == pytest.approx(0.0)

    def test_opt_composite_json_embedded_in_text(self):
        response = 'Here is the answer: {"content": "hi", "next_action": "done"} — done.'
        result = _opt_composite("q", response)
        assert result == pytest.approx(1.0)

    def test_opt_composite_with_expected_kw_scoring(self):
        response = json.dumps({"content": "paris", "next_action": "travel"})
        expected = json.dumps({"content": "paris", "next_action": "travel"})
        result = _opt_composite("What is the capital?", response, expected)
        assert result > 0.5

    def test_opt_composite_not_a_dict(self):
        assert _opt_composite("q", json.dumps([1, 2, 3])) == pytest.approx(0.0)

    # --- _model_spec ---------------------------------------------------------

    def test_model_spec_formats_correctly(self):
        from types import SimpleNamespace
        entry = SimpleNamespace(provider="openai", model="LFM2.5-8B")
        assert _model_spec(entry) == "openai/LFM2.5-8B"


# ===========================================================================
# Tests: StackManager.get_llm_config_or_default
# ===========================================================================


class TestStackManagerGetLlmConfigOrDefault:
    """get_llm_config_or_default falls back gracefully."""

    def _make_stack(self, profiles: list[str]):
        from unittest.mock import MagicMock

        from agentomatic.stacks.manager import LLMStackEntry, StackManager

        entry = LLMStackEntry(provider="ollama", model="qwen2.5:7b")
        llm_dict = {p: entry for p in profiles}
        stack = MagicMock()
        stack.name = "test"
        stack.llm = llm_dict

        sm = StackManager.__new__(StackManager)
        sm._active_stack = stack
        sm._stacks = {}

        # minimal interpolate_env
        def interpolate_env(v):
            return v

        sm.interpolate_env = interpolate_env
        return sm

    def test_returns_named_profile_when_present(self):
        sm = self._make_stack(["default", "rewrite"])
        result = sm.get_llm_config_or_default("rewrite")
        assert result is not None
        assert result.provider == "ollama"

    def test_falls_back_to_default_when_missing(self):
        sm = self._make_stack(["default"])   # no "rewrite"
        result = sm.get_llm_config_or_default("rewrite")
        assert result is not None
        assert result.provider == "ollama"

    def test_raises_when_fallback_also_missing(self):
        from unittest.mock import MagicMock

        from agentomatic.stacks.manager import StackManager

        sm = StackManager.__new__(StackManager)
        sm._stacks = {}
        stack = MagicMock()
        stack.name = "test"
        stack.llm = {}
        sm._active_stack = stack
        sm.interpolate_env = lambda v: v

        with pytest.raises(ValueError, match="not found"):
            sm.get_llm_config_or_default("rewrite", default="default")


# ===========================================================================
# Tests: Metric construction — identical to the script
# ===========================================================================


class TestMetricConstruction:
    """All metrics from the script construct without error."""

    def _make_judge(self) -> LocalJudgeMetric:
        return LocalJudgeMetric(
            name="pertinence",
            model="openai/LFM2.5-8B-A1B-MLX-4bit",
            criteria="Is the response pertinent and grounded?",
            dimensions=["pertinence", "groundedness", "actionability"],
        )

    def test_local_judge_metric_construction(self):
        judge = self._make_judge()
        assert judge.name == "pertinence"
        assert "pertinence" in judge.dimensions

    def test_optimize_metric_adapter_with_name_kwarg(self):
        judge = self._make_judge()
        adapter = OptimizeMetricAdapter(judge, name="judge")
        assert adapter.name == "judge"
        assert hasattr(adapter, "score")

    def test_callable_metrics_have_score(self):
        for name, fn in [
            ("json_valid", _json_valid),
            ("precision", _precision),
            ("recall", _recall),
            ("f1", _f1),
            ("keywords", _keyword_score),
        ]:
            m = CallableMetric(name, fn)
            assert hasattr(m, "score"), f"{name} missing .score()"

    def test_exact_key_match_metric(self):
        m = ExactKeyMatchMetric(REQUIRED_KEYS)
        assert hasattr(m, "score")

    def test_agents_weighted_metric_accepts_all_metric_types(self):
        """agents.WeightedMetric validates .score() on every component — all must pass."""
        judge = self._make_judge()
        judge_m = OptimizeMetricAdapter(judge, name="judge")
        key_m = ExactKeyMatchMetric(REQUIRED_KEYS)
        json_m = CallableMetric("json_valid", _json_valid)
        f1_m = CallableMetric("f1", _f1)
        keyword_m = CallableMetric("keywords", _keyword_score)

        # Should not raise
        loss_metric = WeightedMetric(
            [
                ("judge",      judge_m,   0.30),
                ("key_match",  key_m,     0.25),
                ("f1",         f1_m,      0.25),
                ("json_valid", json_m,    0.10),
                ("keywords",   keyword_m, 0.10),
            ],
            name="composite_loss",
        )
        assert loss_metric.name == "composite_loss"

    def test_metric_loss_wraps_weighted_metric(self):
        key_m = ExactKeyMatchMetric(REQUIRED_KEYS)
        wm = WeightedMetric([("key", key_m, 1.0)], name="wm")
        loss = MetricLoss(wm)
        assert loss.name == "wm_loss"

    def test_custom_metric_construction(self):
        m = CustomMetric(fn=_opt_composite, name="composite")
        assert m.name == "composite"

    def test_prompt_search_space_construction(self):
        space = PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_few_shot=True,
            optimize_model_choice=False,
            optimize_model_params=False,
            model_param_space={
                "temperature": [0.0, 0.1, 0.2, 0.4, 0.7],
                "top_p": [0.7, 0.9, 1.0],
            },
            optimize_rag_params=False,
        )
        assert space.optimize_system_prompt is True
        assert space.optimize_model_params is False


# ===========================================================================
# Tests: PromptFitterBridge construction with all script kwargs
# ===========================================================================


class TestPromptFitterBridgeConstruction:
    """PromptFitterBridge accepts every kwarg the script passes."""

    def test_full_construction_does_not_raise(self, tmp_path):
        space = PromptSearchSpace(optimize_system_prompt=True)
        bridge = PromptFitterBridge(
            agent_name="assistant",
            task_model="openai/LFM2.5-8B",
            rewrite_model="openai/qwen2.5:7b",
            local_agent=_EchoAgent(),
            llm_base_url="http://127.0.0.1:8000/v1",
            llm_api_key="local",
            max_trials=8,
            metric=CustomMetric(fn=_opt_composite, name="composite"),
            base_prompt_version="v1",
            search_space=space,
            optimizer="gepa_like",
            min_absolute_improvement=0.02,
            concurrency=2,
            experiment_dir=str(tmp_path / ".fit"),
            auto_report=True,
        )
        assert bridge.agent_name == "assistant"
        assert bridge.llm_base_url == "http://127.0.0.1:8000/v1"
        assert bridge.llm_api_key == "local"
        assert bridge.local_agent is not None

    def test_local_agent_forwarded_to_fitter(self, tmp_path):
        agent = _EchoAgent()
        bridge = PromptFitterBridge(
            agent_name="test",
            local_agent=agent,
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter = bridge._build_fitter(agent, "test")
        assert fitter._runner.agent_callable is not None

    def test_llm_base_url_configures_llm_caller(self, tmp_path, monkeypatch):
        from agentomatic.optimize.llm_caller import LLMCaller
        monkeypatch.setattr(LLMCaller, "_default_base_url", None)
        monkeypatch.setattr(LLMCaller, "_default_api_key", None)

        agent = _EchoAgent()
        bridge = PromptFitterBridge(
            agent_name="test",
            llm_base_url="http://127.0.0.1:9999/v1",
            llm_api_key="test-key",
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        bridge._build_fitter(agent, "test")

        assert LLMCaller._default_base_url == "http://127.0.0.1:9999/v1"
        assert LLMCaller._default_api_key == "test-key"

    def test_rewrite_model_forwarded(self, tmp_path):
        agent = _EchoAgent()
        bridge = PromptFitterBridge(
            agent_name="test",
            task_model="openai/task-model",
            rewrite_model="openai/rewrite-model",
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter = bridge._build_fitter(agent, "test")
        assert fitter.rewrite_model == "openai/rewrite-model"

    def test_no_local_agent_still_works_with_live_agent(self, tmp_path):
        """When local_agent= omitted, the live agent from optimize() is used."""
        agent = _EchoAgent()
        bridge = PromptFitterBridge(
            agent_name="test",
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter = bridge._build_fitter(agent, "test")
        # live agent wired automatically
        assert fitter._runner.agent_callable is not None


# ===========================================================================
# Tests: BaseGraphAgent.compile() / fit() kwargs matching the script
# ===========================================================================


class TestBaseGraphAgentFitInterface:
    """compile() and fit() accept exactly the kwargs the script uses."""

    def _make_agent(self):
        from agentomatic.agents.base import BaseGraphAgent

        class _Agent(BaseGraphAgent):
            def input_to_state(self, d):
                return d

            def state_to_output(self, s):
                return {"content": s.get("current_query", "").upper(),
                        "next_action": "done"}

        return _Agent()

    def test_compile_accepts_loss_kwarg(self, tmp_path):
        agent = self._make_agent()
        key_m = ExactKeyMatchMetric(REQUIRED_KEYS)
        wm = WeightedMetric([("key", key_m, 1.0)], name="wm")
        loss = MetricLoss(wm)

        examples = [
            AgentExample(id="e1", input={"current_query": "hi"},
                         expected_output={"content": "HI", "next_action": "done"},
                         split="train"),
        ]
        ds = AgentDataset(examples=examples, name="test")
        # Must not raise
        agent.compile(ds, metrics=[key_m], loss=loss)

    def test_fit_accepts_all_script_kwargs(self, tmp_path):
        """fit() must accept optimize_mode, optimize_prompt, optimize_params."""
        agent = self._make_agent()
        key_m = ExactKeyMatchMetric(REQUIRED_KEYS)

        examples = [
            AgentExample(id="e1", input={"current_query": "hi"},
                         expected_output={"content": "HI", "next_action": "done"},
                         split="train"),
        ]
        ds = AgentDataset(examples=examples, name="test")
        agent.compile(ds, metrics=[key_m])
        space = PromptSearchSpace(optimize_system_prompt=False, optimize_model_params=False)

        # Must not raise — all kwargs from the script
        history = agent.fit(
            ds,
            epochs=1,
            validation_data=ds,
            callbacks=[EarlyStopping(monitor="val_loss", patience=1, mode="min")],
            max_trials=2,
            search_space=space,
            optimize_mode="rewrite",
            optimize_prompt=True,
            optimize_params=True,
            verbose=0,
        )
        assert history is not None
        assert hasattr(history, "history")

    def test_history_history_is_dict(self, tmp_path):
        """history.history must be a dict (script does f'history={history.history}')."""
        agent = self._make_agent()
        key_m = ExactKeyMatchMetric(REQUIRED_KEYS)
        examples = [
            AgentExample(id="e1", input={"current_query": "hi"},
                         expected_output={"content": "HI", "next_action": "done"},
                         split="train"),
        ]
        ds = AgentDataset(examples=examples, name="test")
        agent.compile(ds, metrics=[key_m])
        history = agent.fit(ds, epochs=1, verbose=0)
        assert isinstance(history.history, dict)

    def test_evaluate_returns_report_with_scores(self):
        agent = self._make_agent()
        key_m = ExactKeyMatchMetric(REQUIRED_KEYS)
        examples = [
            AgentExample(
                id="e1",
                input={"current_query": "hi"},
                expected_output={"content": "HI", "next_action": "done"},
            ),
        ]
        report = agent.evaluate(examples, [key_m])
        assert hasattr(report, "scores")
        assert isinstance(report.scores, dict)


# ===========================================================================
# Tests: result.history and generate_fit_report after fit
# ===========================================================================


class TestFitResultAccessors:
    """fit_result.history is list[float]; generate_fit_report doesn't crash."""

    def _make_result(self, score_history: list[float] | None = None) -> PromptFitResult:
        return PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="best"),
            baseline_config=PromptRuntimeConfig(system_prompt="base"),
            best_score=0.80,
            baseline_score=0.65,
            score_history=score_history or [0.65, 0.72, 0.80],
        )

    def test_fit_result_history_is_list_of_floats(self):
        result = self._make_result([0.6, 0.7, 0.8])
        h = result.history
        assert isinstance(h, list)
        assert all(isinstance(v, float) for v in h)
        assert h == [0.6, 0.7, 0.8]

    def test_generate_fit_report_does_not_crash(self, tmp_path):
        result = self._make_result()
        out = tmp_path / "fit.html"
        path = generate_fit_report(result, output_path=out)
        assert out.exists()
        content = out.read_text()
        assert len(content) > 100
        assert path == str(out)

    def test_fit_result_summary_does_not_crash(self):
        result = self._make_result()
        text = result.summary()
        assert len(text) > 0
        assert "0.8" in text or "0.80" in text

    def test_fit_result_apply_writes_prompts_json(self, tmp_path):
        (tmp_path / "prompts.json").write_text('{"v1": {"system": "old"}}')
        result = self._make_result()
        result.best_config = PromptRuntimeConfig(system_prompt="new optimized prompt")
        version = result.apply(version="v2_fit", agent_dir=str(tmp_path))
        assert version == "v2_fit"
        data = json.loads((tmp_path / "prompts.json").read_text())
        assert "v2_fit" in data
        assert data["v2_fit"]["system_prompt"] == "new optimized prompt"

    def test_fit_result_to_dict_includes_score_history(self):
        result = self._make_result([0.6, 0.7, 0.8])
        d = result.to_dict()
        assert "score_history" in d
        assert d["score_history"] == [0.6, 0.7, 0.8]


# ===========================================================================
# Tests: Full end-to-end compile → fit → report — mirrors script exactly
# ===========================================================================


class TestFullScriptPatternEndToEnd:
    """Full pipeline matching the train_next.py script without any HTTP server."""

    def _build_metrics_and_loss(self):
        """Build the exact metric stack from the script."""
        # Use a fake judge that always returns 0.7 (no LLM needed)
        class FakeJudge:
            name = "pertinence"
            async def evaluate(self, query, response, expected=None, context=None):
                return EvalResult(metric_name="pertinence", score=0.7, reason="ok")

        judge_metric     = OptimizeMetricAdapter(FakeJudge(), name="judge")
        key_metric       = ExactKeyMatchMetric(REQUIRED_KEYS)
        json_metric      = CallableMetric("json_valid",  _json_valid)
        precision_metric = CallableMetric("precision",   _precision)
        recall_metric    = CallableMetric("recall",      _recall)
        f1_metric        = CallableMetric("f1",          _f1)
        keyword_metric   = CallableMetric("keywords",    _keyword_score)

        metrics = [
            judge_metric, key_metric, json_metric,
            precision_metric, recall_metric, f1_metric, keyword_metric,
        ]

        loss_metric = WeightedMetric(
            [
                ("judge",      judge_metric,   0.30),
                ("key_match",  key_metric,     0.25),
                ("f1",         f1_metric,      0.25),
                ("json_valid", json_metric,    0.10),
                ("keywords",   keyword_metric, 0.10),
            ],
            name="composite_loss",
        )
        return metrics, loss_metric

    def _make_dataset(self) -> AgentDataset:
        examples = [
            AgentExample(
                id=f"e{i}",
                input={"current_query": f"query {i}"},
                expected_output={"content": f"QUERY {i}", "next_action": "done"},
                split="train" if i < 3 else "validation",
            )
            for i in range(4)
        ]
        return AgentDataset(examples=examples, name="test")

    def test_metric_loss_computes_without_crash(self):
        metrics, loss_metric = self._build_metrics_and_loss()
        loss = MetricLoss(loss_metric)
        ex = AgentExample(
            id="e1",
            input={"current_query": "hello"},
            expected_output={"content": "HELLO", "next_action": "done"},
        )
        computed = loss.compute(ex, {"content": "HELLO", "next_action": "done"})
        assert isinstance(computed, float)
        assert 0.0 <= computed <= 1.0

    def test_compile_fit_evaluate_full_cycle(self, tmp_path):
        """Full script pattern: compile → fit → evaluate."""
        from agentomatic.agents.base import BaseGraphAgent

        class EchoAgent(BaseGraphAgent):
            def input_to_state(self, d):
                return d

            def state_to_output(self, s):
                return {
                    "content": s.get("current_query", "").upper(),
                    "next_action": "done",
                }

        agent = EchoAgent()
        metrics, loss_metric = self._build_metrics_and_loss()
        dataset = self._make_dataset()
        space = PromptSearchSpace(optimize_system_prompt=False, optimize_model_params=False)

        agent.compile(
            dataset,
            metrics=metrics,
            optimizer=PromptFitterBridge(
                agent_name="test",
                experiment_dir=str(tmp_path / ".fit"),
                auto_report=False,
            ),
            loss=MetricLoss(loss_metric),
        )

        history = agent.fit(
            dataset,
            epochs=1,
            validation_data=dataset,
            callbacks=[EarlyStopping(monitor="val_loss", patience=1, mode="min")],
            max_trials=1,
            search_space=space,
            optimize_mode="rewrite",
            optimize_prompt=True,
            optimize_params=True,
            verbose=0,
        )

        # Script assertions
        assert isinstance(history.history, dict)
        status = getattr(agent, "_last_optimize_status", "")
        assert isinstance(status, str)

        held_out = dataset.test or dataset.validation or dataset.train
        report = agent.evaluate(held_out, metrics)
        assert isinstance(report.scores, dict)

    def test_fit_result_accessible_and_usable(self, tmp_path):
        """After fit, _last_fit_result.history is list[float] and report writes."""
        from dataclasses import dataclass as dc

        from agentomatic.optimize.dataset import DataPoint, Dataset
        from agentomatic.optimize.fitter import PromptFitter
        from agentomatic.optimize.fitter_optimizers import BaseFitterOptimizer
        from agentomatic.optimize.metrics import ExactMatchMetric
        from agentomatic.optimize.runner import AgentRunner

        @dc
        class Noop(BaseFitterOptimizer):
            name: str = "noop"
            async def propose(self, *a, **kw): return []

        async def echo(query, *, prompt_override, context, invoke):
            return json.dumps({"content": query.upper(), "next_action": "done"})

        fitter = PromptFitter(
            agent="test", optimizer=Noop(), max_trials=1,
            experiment_dir=str(tmp_path / ".fit"), auto_report=False,
        )
        fitter._runner = AgentRunner(agent="test", agent_callable=echo)

        ds = Dataset(points=[
            DataPoint(query="hello", expected_answer="HELLO"),
            DataPoint(query="world", expected_answer="WORLD"),
        ])
        result = asyncio.run(fitter.fit(ds, ds, ExactMatchMetric()))

        # BUG-3: history must be accessible
        assert hasattr(result, "history")
        assert isinstance(result.history, list)
        assert hasattr(result, "score_history")

        # generate_fit_report must not crash
        out = tmp_path / "report.html"
        generate_fit_report(result, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_weighted_metric_score_called_correctly_in_loss(self):
        """MetricLoss(WeightedMetric(OptimizeMetricAdapter)) flows correctly."""
        call_log: list[tuple[str, str]] = []

        class LoggingJudge:
            name = "judge"
            async def evaluate(self, query, response, expected=None, context=None):
                call_log.append((query, response))
                # Score 1.0 when response contains the expected keys
                score = 1.0 if "content" in response and "next_action" in response else 0.5
                return EvalResult(metric_name="judge", score=score, reason="")

        judge_m = OptimizeMetricAdapter(LoggingJudge(), name="judge")
        wm = WeightedMetric([("judge", judge_m, 1.0)], name="wm")
        loss = MetricLoss(wm)

        ex = AgentExample(
            id="e1",
            input={"current_query": "what is 2+2?"},
            expected_output={"content": "4", "next_action": "done"},
        )
        pred = {"content": "The answer is 4", "next_action": "done"}

        computed = loss.compute(ex, pred)
        assert isinstance(computed, float)
        assert 0.0 <= computed <= 1.0

        # Verify the judge was actually called with string args, not RunResult
        assert len(call_log) == 1
        query_arg, response_arg = call_log[0]
        assert isinstance(query_arg, str)
        assert isinstance(response_arg, str)
        # Response must be the JSON-serialised prediction
        parsed = json.loads(response_arg)
        assert parsed == pred

    def test_early_stopping_callback_construction(self):
        cb = EarlyStopping(monitor="val_loss", patience=1, mode="min")
        assert cb.monitor == "val_loss"
        assert cb.patience == 1
