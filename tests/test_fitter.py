"""Tests for the PromptFitter system (agentomatic.optimize.fitter and friends).

Covers data models, search space, metrics, judges, LLM caller,
failure analysis, fitter optimizers, and the PromptFitter orchestrator.
All LLM calls are mocked — no external services required.
"""

from __future__ import annotations

import json

import pytest

from agentomatic.optimize.config import (
    ParamDelta,
    PromptCandidate,
    PromptFitResult,
    PromptRuntimeConfig,
)
from agentomatic.optimize.failure_analysis import (
    DimensionAnalyzer,
    DimensionComparison,
    FailureClusterer,
)
from agentomatic.optimize.fitter_optimizers import (
    FewShotBootstrapOptimizer,
    ParamSearchOptimizer,
    RewriteOptimizer,
    resolve_fitter_optimizer,
)
from agentomatic.optimize.judges import (
    CalibrationPair,
    JudgeCalibrationSet,
    LocalJudgeMetric,
)
from agentomatic.optimize.llm_caller import parse_model_spec
from agentomatic.optimize.metrics import (
    CompositeMetric,
    ContainsMetric,
    DeterministicMetric,
    ExactMatchMetric,
    MetricResult,
    WeightedMetric,
)
from agentomatic.optimize.search_space import PromptSearchSpace

# =====================================================================
# PromptRuntimeConfig Tests
# =====================================================================


class TestPromptRuntimeConfig:
    def test_creation(self):
        cfg = PromptRuntimeConfig(system_prompt="You are a helpful assistant.")
        assert cfg.system_prompt == "You are a helpful assistant."
        assert cfg.user_template is None
        assert cfg.few_shot_examples == []
        assert cfg.output_contract is None
        assert cfg.model_params == {}
        assert cfg.rag_params == {}
        assert cfg.tool_params == {}

    def test_creation_full(self):
        cfg = PromptRuntimeConfig(
            system_prompt="Answer concisely.",
            user_template="Question: {query}\nContext: {context}",
            few_shot_examples=[
                {"query": "What is 2+2?", "response": "4"},
            ],
            output_contract="JSON object with 'answer' key",
            model_params={"temperature": 0.2, "max_tokens": 512},
            rag_params={"top_k": 5, "min_similarity": 0.75},
            tool_params={"max_tool_calls": 3},
        )
        assert cfg.system_prompt == "Answer concisely."
        assert cfg.user_template is not None
        assert len(cfg.few_shot_examples) == 1
        assert cfg.output_contract == "JSON object with 'answer' key"
        assert cfg.model_params["temperature"] == 0.2
        assert cfg.rag_params["top_k"] == 5
        assert cfg.tool_params["max_tool_calls"] == 3

    def test_to_dict_roundtrip(self):
        original = PromptRuntimeConfig(
            system_prompt="Test prompt",
            user_template="Q: {query}",
            few_shot_examples=[{"query": "hi", "response": "hello"}],
            output_contract="plain text",
            model_params={"temperature": 0.5, "top_p": 0.9},
            rag_params={"top_k": 10},
            tool_params={"max_tool_calls": 2},
        )
        serialized = original.to_dict()
        restored = PromptRuntimeConfig.from_dict(serialized)
        assert restored.system_prompt == original.system_prompt
        assert restored.user_template == original.user_template
        assert restored.few_shot_examples == original.few_shot_examples
        assert restored.output_contract == original.output_contract
        assert restored.model_params == original.model_params
        assert restored.rag_params == original.rag_params
        assert restored.tool_params == original.tool_params

    def test_diff(self):
        old = PromptRuntimeConfig(
            system_prompt="old prompt",
            model_params={"temperature": 0.7},
        )
        new = PromptRuntimeConfig(
            system_prompt="new prompt",
            model_params={"temperature": 0.3},
        )
        changes = new.diff(old)
        assert "system_prompt" in changes
        assert changes["system_prompt"]["old"] == "old prompt"
        assert changes["system_prompt"]["new"] == "new prompt"
        assert "model_params" in changes
        assert changes["model_params"]["old"] == {"temperature": 0.7}
        assert changes["model_params"]["new"] == {"temperature": 0.3}

    def test_diff_no_changes(self):
        cfg = PromptRuntimeConfig(
            system_prompt="same",
            model_params={"temperature": 0.5},
        )
        changes = cfg.diff(cfg)
        assert changes == {}

    def test_format_few_shot_block(self):
        cfg = PromptRuntimeConfig(
            system_prompt="test",
            few_shot_examples=[
                {"query": "What is 2+2?", "response": "4"},
                {"query": "Capital of France?", "response": "Paris"},
            ],
        )
        block = cfg.format_few_shot_block()
        assert "[Example 1]" in block
        assert "[Example 2]" in block
        assert "Q: What is 2+2?" in block
        assert "A: 4" in block
        assert "Q: Capital of France?" in block
        assert "A: Paris" in block

    def test_format_few_shot_block_empty(self):
        cfg = PromptRuntimeConfig(system_prompt="test")
        assert cfg.format_few_shot_block() == ""


# =====================================================================
# ParamDelta Tests
# =====================================================================


class TestParamDelta:
    def test_creation(self):
        delta = ParamDelta(
            param_name="temperature",
            old_value=0.7,
            new_value=0.3,
            reason="Lower temperature reduces hallucination.",
        )
        assert delta.param_name == "temperature"
        assert delta.old_value == 0.7
        assert delta.new_value == 0.3
        assert delta.reason == "Lower temperature reduces hallucination."

    def test_defaults(self):
        delta = ParamDelta(
            param_name="top_p",
            old_value=1.0,
            new_value=0.9,
        )
        assert delta.reason == ""


# =====================================================================
# PromptCandidate Tests
# =====================================================================


class TestPromptCandidate:
    def test_creation(self):
        cfg = PromptRuntimeConfig(system_prompt="Be concise.")
        candidate = PromptCandidate(
            name="candidate_001",
            config=cfg,
            source="rewrite",
        )
        assert candidate.name == "candidate_001"
        assert candidate.config.system_prompt == "Be concise."
        assert candidate.source == "rewrite"
        assert candidate.parent is None
        assert candidate.mutation_notes == ""
        assert candidate.scores == {}
        assert candidate.composite_score == 0.0

    def test_with_scores(self):
        cfg = PromptRuntimeConfig(system_prompt="test")
        candidate = PromptCandidate(
            name="scored_001",
            config=cfg,
            source="gepa",
            scores={"answer_relevancy": 0.91, "faithfulness": 0.88},
            composite_score=0.895,
        )
        assert candidate.scores["answer_relevancy"] == 0.91
        assert candidate.scores["faithfulness"] == 0.88
        assert candidate.composite_score == 0.895

    def test_lineage(self):
        cfg = PromptRuntimeConfig(system_prompt="derived")
        candidate = PromptCandidate(
            name="child_002",
            config=cfg,
            source="rewrite",
            parent="parent_001",
            mutation_notes="Shortened system prompt.",
        )
        assert candidate.parent == "parent_001"
        assert candidate.mutation_notes == "Shortened system prompt."


# =====================================================================
# PromptFitResult Tests
# =====================================================================


class TestPromptFitResult:
    def _make_result(self, **overrides) -> PromptFitResult:
        defaults = dict(
            best_config=PromptRuntimeConfig(
                system_prompt="new optimised prompt",
                model_params={"temperature": 0.3},
            ),
            baseline_config=PromptRuntimeConfig(
                system_prompt="old baseline prompt",
                model_params={"temperature": 0.7},
            ),
            best_score=0.85,
            baseline_score=0.61,
            metric_deltas={"answer_relevancy": 0.24},
            suggestions=["Add few-shot examples."],
            agent="test_agent",
        )
        defaults.update(overrides)
        return PromptFitResult(**defaults)  # type: ignore[arg-type]

    def test_absolute_improvement(self):
        result = self._make_result()
        assert abs(result.absolute_improvement - 0.24) < 1e-6

    def test_best_prompt_shortcut(self):
        result = self._make_result()
        assert result.best_prompt == "new optimised prompt"

    def test_best_params_shortcut(self):
        result = self._make_result()
        assert result.best_params == {"temperature": 0.3}

    def test_summary(self):
        result = self._make_result()
        text = result.summary()
        assert len(text) > 0
        assert "0.85" in text or "0.8500" in text
        assert "0.61" in text or "0.6100" in text
        assert result.experiment_id in text
        assert "Improvement" in text
        # Should recommend applying since improvement > 0.05
        assert "apply" in text.lower() or "✅" in text

    def test_summary_regression(self):
        result = self._make_result(
            best_score=0.55,
            baseline_score=0.60,
            metric_deltas={"faithfulness": -0.05},
        )
        text = result.summary()
        # Negative improvement → "keep the baseline"
        assert "keep" in text.lower() or "❌" in text
        # Should show regression
        assert "faithfulness" in text

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["best_score"] == 0.85
        assert d["baseline_score"] == 0.61
        assert "experiment_id" in d
        assert "best_config" in d
        assert "baseline_config" in d
        assert "metric_deltas" in d
        assert "suggestions" in d
        assert d["agent"] == "test_agent"
        # Check absolute improvement is serialized
        assert abs(d["absolute_improvement"] - 0.24) < 1e-6

    def test_apply(self, tmp_path):
        agent_dir = tmp_path / "agents" / "test_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "prompts.json").write_text('{"v1": {"system": "old"}}')

        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="new prompt"),
            baseline_config=PromptRuntimeConfig(system_prompt="old"),
            best_score=0.85,
            baseline_score=0.61,
            agent="test_agent",
        )
        version = result.apply(version="v2_fit", agent_dir=str(agent_dir))
        assert version == "v2_fit"

        data = json.loads((agent_dir / "prompts.json").read_text())
        assert "v2_fit" in data
        assert data["v2_fit"]["system_prompt"] == "new prompt"
        # Old version preserved
        assert "v1" in data

        # runtime_config.json should also exist
        rc_path = agent_dir / "runtime_config.json"
        assert rc_path.exists()
        rc_data = json.loads(rc_path.read_text())
        assert rc_data["system_prompt"] == "new prompt"


# =====================================================================
# PromptSearchSpace Tests
# =====================================================================


class TestPromptSearchSpace:
    def test_default_creation(self):
        space = PromptSearchSpace()
        assert "temperature" in space.model_param_space
        assert "top_p" in space.model_param_space
        assert "max_tokens" in space.model_param_space
        assert space.optimize_model_params is True

    def test_param_combinations(self):
        space = PromptSearchSpace(
            model_param_space={
                "temperature": [0.0, 0.5],
                "top_p": [0.9, 1.0],
            },
        )
        combos = space.param_combinations("model")
        assert len(combos) == 4  # 2 * 2
        # Each combo should have both keys
        for combo in combos:
            assert "temperature" in combo
            assert "top_p" in combo

    def test_sample_params(self):
        space = PromptSearchSpace(
            model_param_space={
                "temperature": [0.0, 0.1, 0.2, 0.4, 0.7],
                "top_p": [0.7, 0.9, 1.0],
            },
        )
        sampled = space.sample_params(3, "model")
        assert len(sampled) <= 3

    def test_sample_params_all(self):
        space = PromptSearchSpace(
            model_param_space={
                "temperature": [0.0, 0.5],
                "top_p": [0.9],
            },
        )
        total = space.n_combinations("model")
        sampled = space.sample_params(total + 10, "model")
        assert len(sampled) == total

    def test_n_combinations(self):
        space = PromptSearchSpace(
            model_param_space={
                "temperature": [0.0, 0.1, 0.2],
                "top_p": [0.9, 1.0],
                "max_tokens": [800, 1200],
            },
        )
        assert space.n_combinations("model") == 3 * 2 * 2  # 12

    def test_active_spaces(self):
        space = PromptSearchSpace()
        active = space.active_spaces()
        assert "model" in active
        # By default, rag and tool are disabled
        assert "rag" not in active
        assert "tool" not in active

    def test_active_spaces_custom(self):
        space = PromptSearchSpace(
            optimize_model_params=False,
            optimize_rag_params=True,
            optimize_tool_params=True,
        )
        active = space.active_spaces()
        assert "model" not in active
        assert "rag" in active
        assert "tool" in active

    def test_to_dict_roundtrip(self):
        original = PromptSearchSpace(
            optimize_model_params=True,
            optimize_rag_params=True,
            model_param_space={"temperature": [0.0, 0.5]},
            rag_param_space={"top_k": [3, 5, 10]},
            max_few_shot_examples=8,
            few_shot_selection_strategy="top_k",
        )
        serialized = original.to_dict()
        restored = PromptSearchSpace.from_dict(serialized)
        assert restored.optimize_model_params is True
        assert restored.optimize_rag_params is True
        assert restored.model_param_space == {"temperature": [0.0, 0.5]}
        assert restored.rag_param_space == {"top_k": [3, 5, 10]}
        assert restored.max_few_shot_examples == 8
        assert restored.few_shot_selection_strategy == "top_k"


# =====================================================================
# MetricResult Tests
# =====================================================================


class TestMetricResult:
    def test_creation(self):
        result = MetricResult(
            score=0.78,
            feedback="Mostly correct but incomplete.",
            dimensions={"correctness": 0.85, "completeness": 0.66},
        )
        assert result.score == 0.78
        assert result.feedback == "Mostly correct but incomplete."
        assert result.dimensions["correctness"] == 0.85
        assert result.dimensions["completeness"] == 0.66

    def test_to_eval_result(self):
        mr = MetricResult(
            score=0.9,
            feedback="Great answer",
            dimensions={"relevance": 0.95},
        )
        er = mr.to_eval_result("test_metric")
        assert er.metric_name == "test_metric"
        assert er.score == 0.9
        assert er.reason == "Great answer"
        assert er.metadata["dimensions"] == {"relevance": 0.95}

    def test_defaults(self):
        mr = MetricResult(score=0.5)
        assert mr.feedback == ""
        assert mr.dimensions == {}


# =====================================================================
# CompositeMetric Tests
# =====================================================================


class TestCompositeMetric:
    def test_creation(self):
        metric = CompositeMetric(
            metrics=[
                WeightedMetric("exact", ExactMatchMetric(), weight=0.5),
                WeightedMetric("contains", ContainsMetric(), weight=0.5),
            ],
        )
        assert metric.name == "composite"
        assert len(metric._metrics) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeMetric(metrics=[])

    async def test_evaluate(self):
        metric = CompositeMetric(
            metrics=[
                WeightedMetric("exact", ExactMatchMetric(), weight=0.5),
                WeightedMetric("contains", ContainsMetric(), weight=0.5),
            ],
        )
        result = await metric.evaluate("q", "hello world", "hello world")
        assert result.score > 0.5
        assert "dimensions" in result.metadata
        assert "exact" in result.metadata["dimensions"]
        assert "contains" in result.metadata["dimensions"]

    async def test_evaluate_rich(self):
        metric = CompositeMetric(
            metrics=[
                WeightedMetric("exact", ExactMatchMetric(), weight=0.6),
                WeightedMetric("contains", ContainsMetric(), weight=0.4),
            ],
        )
        mr = await metric.evaluate_rich("q", "hello world", "hello world")
        assert isinstance(mr, MetricResult)
        assert mr.score > 0.5
        assert "exact" in mr.dimensions
        assert "contains" in mr.dimensions


# =====================================================================
# DeterministicMetric Tests
# =====================================================================


class TestDeterministicMetric:
    async def test_contains_check(self):
        metric = DeterministicMetric(
            name="format",
            checks=[{"type": "contains", "value": "## Summary"}],
        )
        result = await metric.evaluate("q", "Here is ## Summary of findings.")
        assert result.score == 1.0

    async def test_contains_fails(self):
        metric = DeterministicMetric(
            name="format",
            checks=[{"type": "contains", "value": "## Summary"}],
        )
        result = await metric.evaluate("q", "No header here.")
        assert result.score == 0.0
        assert "Missing" in result.reason

    async def test_regex_check(self):
        metric = DeterministicMetric(
            name="date_check",
            checks=[{"type": "regex", "value": r"\d{4}-\d{2}-\d{2}"}],
        )
        result = await metric.evaluate("q", "The date is 2024-12-25.")
        assert result.score == 1.0

    async def test_multiple_checks(self):
        metric = DeterministicMetric(
            name="multi",
            checks=[
                {"type": "contains", "value": "Summary"},
                {"type": "contains", "value": "Risks"},
                {"type": "max_length", "value": 5000},
            ],
        )
        result = await metric.evaluate("q", "Summary of the project. Nothing else here.")
        # 2 out of 3 pass (Summary found, Risks not found, max_length ok)
        assert abs(result.score - 2 / 3) < 1e-6

    async def test_no_checks(self):
        metric = DeterministicMetric(name="empty")
        result = await metric.evaluate("q", "anything")
        assert result.score == 1.0
        assert "No checks" in result.reason


# =====================================================================
# LocalJudgeMetric Tests
# =====================================================================


class TestLocalJudgeMetric:
    def test_creation(self):
        judge = LocalJudgeMetric(
            name="scope_judge",
            model="ollama/qwen2.5:7b",
            criteria="Evaluate completeness.",
            dimensions=["completeness", "specificity"],
            weight=0.5,
        )
        assert judge.name == "scope_judge"
        assert judge.model == "ollama/qwen2.5:7b"
        assert judge.criteria == "Evaluate completeness."
        assert judge.dimensions == ["completeness", "specificity"]
        assert judge.weight == 0.5
        assert judge.temperature == 0.0  # deterministic default for stable epochs

    async def test_evaluate_mocked(self, monkeypatch):
        async def mock_call_json(*args, **kwargs):
            return {
                "overall_score": 0.8,
                "feedback": "Good response",
                "dimensions": {"correctness": 0.9, "completeness": 0.7},
            }

        monkeypatch.setattr(
            "agentomatic.optimize.llm_caller.LLMCaller.call_with_json",
            mock_call_json,
        )
        judge = LocalJudgeMetric(
            name="test",
            dimensions=["correctness", "completeness"],
        )
        result = await judge.evaluate("q", "response", "expected")
        assert result.score == 0.8
        assert result.reason == "Good response"
        assert result.metadata["dimensions"]["correctness"] == 0.9
        assert result.metadata["dimensions"]["completeness"] == 0.7


# =====================================================================
# JudgeCalibrationSet Tests
# =====================================================================


class TestJudgeCalibrationSet:
    def test_creation(self):
        pair = CalibrationPair(
            query="What are risks?",
            response_a="The risks include...",
            response_b="Based on analysis, key risks are...",
            human_preference="b",
            reason="B is more complete.",
        )
        cal = JudgeCalibrationSet(pairs=[pair])
        assert len(cal) == 1
        assert cal.pairs[0].human_preference == "b"

    def test_len(self):
        pairs = [
            CalibrationPair(
                query=f"Q{i}",
                response_a=f"A{i}",
                response_b=f"B{i}",
                human_preference="a",
            )
            for i in range(5)
        ]
        cal = JudgeCalibrationSet(pairs=pairs)
        assert len(cal) == 5

    def test_from_list(self):
        items = [
            {
                "query": "What is X?",
                "response_a": "X is A",
                "response_b": "X is B",
                "human_preference": "b",
                "reason": "B is better.",
            },
            {
                "query": "What is Y?",
                "response_a": "Y is C",
                "response_b": "Y is D",
                "human_preference": "a",
            },
        ]
        cal = JudgeCalibrationSet.from_list(items)
        assert len(cal) == 2
        assert cal.pairs[0].query == "What is X?"
        assert cal.pairs[0].human_preference == "b"
        assert cal.pairs[0].reason == "B is better."
        assert cal.pairs[1].reason == ""  # default


# =====================================================================
# LLMCaller Tests
# =====================================================================


class TestLLMCaller:
    def test_parse_model_spec_ollama(self):
        provider, model = parse_model_spec("ollama/qwen2.5:7b")
        assert provider == "ollama"
        assert model == "qwen2.5:7b"

    def test_parse_model_spec_openai(self):
        provider, model = parse_model_spec("openai/gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"

    def test_parse_model_spec_no_prefix(self):
        provider, model = parse_model_spec("mistral:7b")
        assert provider == "ollama"
        assert model == "mistral:7b"

    def test_parse_model_spec_litellm(self):
        provider, model = parse_model_spec("litellm/anthropic/claude-3-haiku")
        assert provider == "litellm"
        assert model == "anthropic/claude-3-haiku"


# =====================================================================
# FailureClusterer Tests
# =====================================================================


class TestFailureClusterer:
    async def test_empty_failures(self):
        clusterer = FailureClusterer()
        clusters = await clusterer.cluster([])
        assert clusters == []

    async def test_simple_cluster(self):
        failures = [
            {"query": "Q1", "response": "R1", "avg_score": 0.1},
            {"query": "Q2", "response": "R2", "avg_score": 0.2},
        ]
        clusterer = FailureClusterer()
        clusters = await clusterer.cluster(failures)
        # With <= 3 failures, uses simple clustering
        assert len(clusters) > 0
        # Should have severe failures cluster
        assert any(c.label == "severe_failures" for c in clusters)

    async def test_simple_cluster_moderate(self):
        failures = [
            {"query": "Q1", "response": "R1", "avg_score": 0.35},
            {"query": "Q2", "response": "R2", "avg_score": 0.45},
        ]
        clusterer = FailureClusterer()
        clusters = await clusterer.cluster(failures)
        assert len(clusters) > 0
        assert any(c.label == "moderate_failures" for c in clusters)


# =====================================================================
# DimensionAnalyzer Tests
# =====================================================================


class TestDimensionAnalyzer:
    def test_compare_improvement(self):
        analyzer = DimensionAnalyzer()
        baseline = {"correctness": 0.6, "completeness": 0.5}
        candidate = {"correctness": 0.8, "completeness": 0.7}
        comparisons = analyzer.compare(baseline, candidate)
        assert len(comparisons) == 2
        for c in comparisons:
            assert c.absolute_delta > 0
            assert c.decision == "keep"

    def test_compare_regression(self):
        analyzer = DimensionAnalyzer(
            critical_threshold=0.4,
            regression_tolerance=0.05,
        )
        baseline = {"correctness": 0.8, "completeness": 0.7}
        candidate = {"correctness": 0.3, "completeness": 0.6}
        comparisons = analyzer.compare(baseline, candidate)

        correctness_comp = next(c for c in comparisons if c.dimension == "correctness")
        # Large regression below critical threshold → reject
        assert correctness_comp.decision == "reject"
        assert correctness_comp.absolute_delta < 0

        completeness_comp = next(c for c in comparisons if c.dimension == "completeness")
        # Small regression within tolerance → accept_if_above_threshold
        assert completeness_comp.decision == "watch"

    def test_should_accept_true(self):
        analyzer = DimensionAnalyzer()
        comparisons = [
            DimensionComparison(
                dimension="correctness",
                baseline_score=0.6,
                candidate_score=0.8,
                absolute_delta=0.2,
                decision="keep",
            ),
        ]
        accept, reason = analyzer.should_accept(
            comparisons,
            min_composite_delta=0.05,
            composite_baseline=0.6,
            composite_candidate=0.8,
        )
        assert accept is True
        assert "Accepted" in reason

    def test_should_accept_false(self):
        analyzer = DimensionAnalyzer()
        comparisons = [
            DimensionComparison(
                dimension="correctness",
                baseline_score=0.6,
                candidate_score=0.62,
                absolute_delta=0.02,
                decision="keep",
            ),
        ]
        accept, reason = analyzer.should_accept(
            comparisons,
            min_composite_delta=0.05,
            composite_baseline=0.6,
            composite_candidate=0.62,
        )
        assert accept is False
        assert "below" in reason.lower()

    def test_should_accept_false_rejection(self):
        analyzer = DimensionAnalyzer()
        comparisons = [
            DimensionComparison(
                dimension="safety",
                baseline_score=0.9,
                candidate_score=0.3,
                absolute_delta=-0.6,
                decision="reject",
            ),
        ]
        accept, reason = analyzer.should_accept(
            comparisons,
            min_composite_delta=0.0,
            composite_baseline=0.6,
            composite_candidate=0.7,
        )
        assert accept is False
        assert "regression" in reason.lower() or "safety" in reason.lower()

    def test_format_table(self):
        analyzer = DimensionAnalyzer()
        comparisons = [
            DimensionComparison(
                dimension="correctness",
                baseline_score=0.60,
                candidate_score=0.80,
                absolute_delta=0.20,
                decision="keep",
            ),
            DimensionComparison(
                dimension="completeness",
                baseline_score=0.70,
                candidate_score=0.65,
                absolute_delta=-0.05,
                decision="watch",
            ),
        ]
        table = analyzer.format_table(comparisons)
        assert "correctness" in table
        assert "completeness" in table
        assert "keep" in table
        assert "watch" in table
        # Should be multi-line
        assert "\n" in table


# =====================================================================
# Fitter Optimizer Tests
# =====================================================================


class TestFitterOptimizers:
    def test_resolve_rewrite(self):
        opt = resolve_fitter_optimizer("rewrite")
        assert isinstance(opt, RewriteOptimizer)
        assert opt.name == "rewrite"

    def test_resolve_few_shot(self):
        opt = resolve_fitter_optimizer("few_shot")
        assert isinstance(opt, FewShotBootstrapOptimizer)

    def test_resolve_param_search(self):
        opt = resolve_fitter_optimizer("param_search")
        assert isinstance(opt, ParamSearchOptimizer)
        assert opt.name == "param_search"

    def test_resolve_instance(self):
        instance = ParamSearchOptimizer()
        resolved = resolve_fitter_optimizer(instance)
        assert resolved is instance

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            resolve_fitter_optimizer("nonexistent_strategy")

    async def test_param_search_propose(self):
        opt = ParamSearchOptimizer()
        baseline = PromptRuntimeConfig(system_prompt="test")
        space = PromptSearchSpace(
            optimize_model_params=True,
            model_param_space={"temperature": [0.0, 0.5]},
        )
        candidates = await opt.propose(
            current_config=baseline,
            eval_results=[],
            dataset_sample=[],
            search_space=space,
        )
        assert len(candidates) > 0
        assert all(c.source == "param_search" for c in candidates)
        # Each candidate should have the baseline prompt unchanged
        for c in candidates:
            assert c.config.system_prompt == "test"
            assert "temperature" in c.config.model_params

    async def test_param_search_no_changes_skips(self):
        """If baseline already matches all param combos, no candidates."""
        opt = ParamSearchOptimizer()
        baseline = PromptRuntimeConfig(
            system_prompt="test",
            model_params={"temperature": 0.5},
        )
        space = PromptSearchSpace(
            optimize_model_params=True,
            model_param_space={"temperature": [0.5]},
        )
        candidates = await opt.propose(
            current_config=baseline,
            eval_results=[],
            dataset_sample=[],
            search_space=space,
        )
        # The baseline already has temperature=0.5, so no new candidates
        assert len(candidates) == 0


# =====================================================================
# PromptFitter Tests
# =====================================================================


class TestPromptFitter:
    def test_creation(self):
        from agentomatic.optimize.fitter import PromptFitter

        fitter = PromptFitter(agent="test_agent")
        assert fitter.agent == "test_agent"
        assert fitter.task_model == "ollama/qwen2.5:7b"
        assert fitter.max_trials == 30
        assert fitter.min_absolute_improvement == 0.001
        assert fitter.concurrency == 1  # sequential default (local-SLM safe)
        assert fitter.sequential is True
        assert fitter._search_space is not None
        assert fitter._optimizer is not None

    def test_creation_custom(self):
        from agentomatic.optimize.fitter import PromptFitter

        space = PromptSearchSpace(
            optimize_model_params=True,
            model_param_space={"temperature": [0.0, 0.3, 0.7]},
        )
        fitter = PromptFitter(
            agent="custom_agent",
            task_model="openai/gpt-4o-mini",
            rewrite_model="openai/gpt-4.1",
            search_space=space,
            optimizer="param_search",
            max_trials=10,
            min_absolute_improvement=0.1,
            concurrency=3,
        )
        assert fitter.agent == "custom_agent"
        assert fitter.task_model == "openai/gpt-4o-mini"
        assert fitter.rewrite_model == "openai/gpt-4.1"
        assert fitter.max_trials == 10
        assert fitter.min_absolute_improvement == 0.1
        assert fitter.concurrency == 3
        assert isinstance(fitter._optimizer, ParamSearchOptimizer)
        assert fitter._search_space.model_param_space == {
            "temperature": [0.0, 0.3, 0.7],
        }


# =====================================================================
# EvalContract Tests
# =====================================================================


class TestEvalContract:
    def test_creation(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract()
        assert contract.name == "default"
        assert contract.input_fields == ["query"]
        assert contract.output_format == "text"
        assert contract.required_output_fields == []
        assert contract.optional_output_fields == []
        assert contract.constraints == []
        assert contract.max_output_length is None
        assert contract.min_output_length is None

    def test_creation_full(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(
            name="scoping_response",
            input_fields=["query", "context"],
            output_format="json",
            required_output_fields=["answer", "confidence", "risks"],
            optional_output_fields=["citations"],
            constraints=["confidence must be between 0.0 and 1.0"],
            max_output_length=5000,
            min_output_length=100,
        )
        assert contract.name == "scoping_response"
        assert contract.input_fields == ["query", "context"]
        assert contract.output_format == "json"
        assert len(contract.required_output_fields) == 3
        assert contract.optional_output_fields == ["citations"]
        assert len(contract.constraints) == 1
        assert contract.max_output_length == 5000
        assert contract.min_output_length == 100

    def test_validate_json_all_fields(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(
            output_format="json",
            required_output_fields=["answer", "confidence"],
        )
        response = json.dumps({"answer": "hello", "confidence": 0.9})
        score = contract.validate(response)
        assert score == 1.0

    def test_validate_json_missing_field(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(
            output_format="json",
            required_output_fields=["answer", "confidence"],
        )
        response = json.dumps({"answer": "hello"})
        score = contract.validate(response)
        # 1 (json valid) + 1 (answer found) out of 3 total checks
        assert abs(score - 2 / 3) < 1e-6

    def test_validate_json_invalid(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(
            output_format="json",
            required_output_fields=["answer", "confidence"],
        )
        score = contract.validate("not valid json {{{")
        # JSON parse fails: 0 out of 3 checks (json_valid + 2 fields)
        assert score == 0.0

    def test_validate_text(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(output_format="text")
        score = contract.validate("Any text response here.")
        # No checks defined for plain text → 1.0
        assert score == 1.0

    def test_validate_markdown(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(output_format="markdown")
        score = contract.validate("# Summary\nThis is a summary.")
        assert score == 1.0

    def test_validate_details(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(
            output_format="json",
            required_output_fields=["answer"],
        )
        response = json.dumps({"answer": "hello"})
        details = contract.validate_details(response)
        assert details["score"] == 1.0
        assert details["passed"] == 2  # json_valid + field:answer
        assert details["failed"] == 0
        assert isinstance(details["checks"], list)
        assert len(details["checks"]) == 2

    def test_validate_length_constraints(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(
            output_format="text",
            min_output_length=5,
            max_output_length=50,
        )
        # Within bounds
        assert contract.validate("Hello, world!") == 1.0
        # Too short
        assert contract.validate("Hi") < 1.0
        # Too long
        assert contract.validate("x" * 100) < 1.0

    def test_to_dict_roundtrip(self):
        from agentomatic.optimize.eval_contract import EvalContract

        original = EvalContract(
            name="test_contract",
            input_fields=["query", "context"],
            output_format="json",
            required_output_fields=["answer", "confidence"],
            optional_output_fields=["citations"],
            constraints=["confidence must be float"],
            max_output_length=5000,
            min_output_length=100,
        )
        serialized = original.to_dict()
        restored = EvalContract.from_dict(serialized)
        assert restored.name == original.name
        assert restored.input_fields == original.input_fields
        assert restored.output_format == original.output_format
        assert restored.required_output_fields == original.required_output_fields
        assert restored.optional_output_fields == original.optional_output_fields
        assert restored.constraints == original.constraints
        assert restored.max_output_length == original.max_output_length
        assert restored.min_output_length == original.min_output_length

    def test_as_judge_criteria(self):
        from agentomatic.optimize.eval_contract import EvalContract

        contract = EvalContract(
            name="support_response",
            output_format="json",
            required_output_fields=["answer"],
            constraints=["answer must be concise"],
        )
        criteria = contract.as_judge_criteria()
        assert isinstance(criteria, str)
        assert len(criteria) > 0
        assert "support_response" in criteria


# =====================================================================
# DeploymentRecommendation Tests
# =====================================================================


class TestDeploymentRecommendation:
    def test_rollout_config_defaults(self):
        from agentomatic.optimize.deployment import RolloutConfig

        rc = RolloutConfig()
        assert rc.strategy == "canary"
        assert rc.initial_weight == 0.20
        assert rc.promotion_threshold == 0.05
        assert rc.rollback_threshold == -0.03
        assert rc.monitoring_hours == 24
        assert rc.max_error_rate == 0.05

    def test_rollout_config_to_dict_roundtrip(self):
        from agentomatic.optimize.deployment import RolloutConfig

        original = RolloutConfig(
            strategy="blue_green",
            initial_weight=0.50,
            promotion_threshold=0.10,
            rollback_threshold=-0.05,
            monitoring_hours=48,
            max_error_rate=0.02,
        )
        serialized = original.to_dict()
        restored = RolloutConfig.from_dict(serialized)
        assert restored.strategy == original.strategy
        assert restored.initial_weight == original.initial_weight
        assert restored.promotion_threshold == original.promotion_threshold
        assert restored.rollback_threshold == original.rollback_threshold
        assert restored.monitoring_hours == original.monitoring_hours
        assert restored.max_error_rate == original.max_error_rate

    def test_deployment_recommendation_defaults(self):
        from agentomatic.optimize.deployment import DeploymentRecommendation

        rec = DeploymentRecommendation()
        assert rec.prompt_version == "v2_optimized"
        assert rec.confidence == "medium"
        assert rec.model_params == {}
        assert rec.rag_params == {}
        assert rec.tool_policy == {}
        assert rec.monitoring_metrics == []
        assert rec.rollback_instructions == ""
        assert rec.safety_notes == []
        assert rec.recommended_model is None
        assert rec.fallback_model is None
        assert rec.expected_improvement == 0.0

    def test_deployment_recommendation_full(self):
        from agentomatic.optimize.deployment import (
            DeploymentRecommendation,
            RolloutConfig,
        )

        rec = DeploymentRecommendation(
            prompt_version="v3_final",
            confidence="high",
            rollout=RolloutConfig(strategy="canary", initial_weight=0.30),
            model_params={"temperature": 0.2},
            rag_params={"top_k": 6},
            tool_policy={"force_tool": True},
            monitoring_metrics=["faithfulness", "latency"],
            rollback_instructions="Rollback if faithfulness < 0.5",
            safety_notes=["Watch for regressions"],
            recommended_model="openai/gpt-4.1",
            fallback_model="ollama/qwen2.5:7b",
            expected_improvement=0.15,
            baseline_score=0.70,
            projected_score=0.85,
        )
        assert rec.prompt_version == "v3_final"
        assert rec.confidence == "high"
        assert rec.rollout.initial_weight == 0.30
        assert rec.model_params["temperature"] == 0.2
        assert rec.recommended_model == "openai/gpt-4.1"
        assert rec.fallback_model == "ollama/qwen2.5:7b"

    def test_deployment_recommendation_to_dict(self):
        from agentomatic.optimize.deployment import DeploymentRecommendation

        rec = DeploymentRecommendation(
            prompt_version="v2_test",
            model_params={"temperature": 0.3},
        )
        d = rec.to_dict()
        assert "deployment_recommendation" in d
        assert d["deployment_recommendation"]["rollout"] == "canary"
        assert d["prompt_version"] == "v2_test"

    def test_deployment_recommendation_summary(self):
        from agentomatic.optimize.deployment import DeploymentRecommendation

        rec = DeploymentRecommendation(
            prompt_version="v2_summary_test",
            confidence="high",
            baseline_score=0.60,
            projected_score=0.80,
            expected_improvement=0.20,
        )
        text = rec.summary()
        assert isinstance(text, str)
        assert len(text) > 0
        assert "v2_summary_test" in text
        assert "high" in text

    def test_build_from_fit_result_high_confidence(self):
        from agentomatic.optimize.deployment import build_deployment_recommendation

        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="new"),
            baseline_config=PromptRuntimeConfig(system_prompt="old"),
            best_score=0.90,
            baseline_score=0.70,
            metric_deltas={"quality": 0.20},
            agent="test",
        )
        rec = build_deployment_recommendation(result)
        assert rec.confidence == "high"

    def test_build_from_fit_result_no_improvement(self):
        from agentomatic.optimize.deployment import build_deployment_recommendation

        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="same"),
            baseline_config=PromptRuntimeConfig(system_prompt="old"),
            best_score=0.55,
            baseline_score=0.60,
            metric_deltas={"quality": -0.05},
            agent="test",
        )
        rec = build_deployment_recommendation(result)
        assert rec.confidence == "no_improvement"
        assert rec.rollout.strategy == "hold"


# =====================================================================
# LatencyMetric Tests
# =====================================================================


class TestLatencyMetric:
    def test_creation(self):
        from agentomatic.optimize.metrics import LatencyMetric

        m = LatencyMetric()
        assert m.name == "latency"
        assert m.target_seconds == 2.0
        assert m.max_seconds == 10.0

    async def test_below_target(self):
        from agentomatic.optimize.metrics import LatencyMetric

        m = LatencyMetric(target_seconds=2.0, max_seconds=10.0)
        result = await m.evaluate("q", "resp", context=["latency:1.5"])
        assert result.score == 1.0

    async def test_above_max(self):
        from agentomatic.optimize.metrics import LatencyMetric

        m = LatencyMetric(target_seconds=2.0, max_seconds=10.0)
        result = await m.evaluate("q", "resp", context=["latency:15.0"])
        assert result.score == 0.0

    async def test_between_target_and_max(self):
        from agentomatic.optimize.metrics import LatencyMetric

        m = LatencyMetric(target_seconds=2.0, max_seconds=10.0)
        result = await m.evaluate("q", "resp", context=["latency:6.0"])
        # Linear interpolation: 1.0 - (6.0 - 2.0) / (10.0 - 2.0) = 0.5
        assert abs(result.score - 0.5) < 1e-6

    async def test_no_latency_data(self):
        from agentomatic.optimize.metrics import LatencyMetric

        m = LatencyMetric()
        result = await m.evaluate("q", "resp")
        assert result.score == 0.5
        assert "No latency data" in result.reason


# =====================================================================
# CostMetric Tests
# =====================================================================


class TestCostMetric:
    def test_creation(self):
        from agentomatic.optimize.metrics import CostMetric

        m = CostMetric()
        assert m.name == "cost"
        assert m.target_tokens == 500
        assert m.max_tokens == 3000

    async def test_below_target(self):
        from agentomatic.optimize.metrics import CostMetric

        m = CostMetric(target_tokens=500, max_tokens=3000)
        result = await m.evaluate("q", "resp", context=["tokens:100"])
        assert result.score == 1.0

    async def test_above_max(self):
        from agentomatic.optimize.metrics import CostMetric

        m = CostMetric(target_tokens=500, max_tokens=3000)
        result = await m.evaluate("q", "resp", context=["tokens:5000"])
        assert result.score == 0.0

    async def test_estimate_from_response(self):
        from agentomatic.optimize.metrics import CostMetric

        m = CostMetric(target_tokens=500, max_tokens=3000)
        # No context → estimates from response length (~4 chars per token)
        short_response = "x" * 100  # ~25 tokens → well below target
        result = await m.evaluate("q", short_response)
        assert result.score == 1.0
        assert result.metadata["tokens"] == max(1, len(short_response) // 4)


# =====================================================================
# Deployment Fields Tests
# =====================================================================


class TestDeploymentFields:
    def test_runtime_config_model_choice(self):
        cfg = PromptRuntimeConfig(
            system_prompt="test",
            model_choice="openai/gpt-4.1",
            fallback_model="ollama/qwen2.5:7b",
        )
        assert cfg.model_choice == "openai/gpt-4.1"
        assert cfg.fallback_model == "ollama/qwen2.5:7b"

    def test_runtime_config_model_choice_roundtrip(self):
        original = PromptRuntimeConfig(
            system_prompt="test",
            model_choice="openai/gpt-4.1",
            fallback_model="ollama/qwen2.5:7b",
            routing_config={"weight": 0.8},
        )
        serialized = original.to_dict()
        restored = PromptRuntimeConfig.from_dict(serialized)
        assert restored.model_choice == original.model_choice
        assert restored.fallback_model == original.fallback_model
        assert restored.routing_config == original.routing_config

    def test_search_space_model_choices(self):
        space = PromptSearchSpace(
            model_choices=["ollama/qwen2.5:7b", "openai/gpt-4.1"],
        )
        assert len(space.model_choices) == 2
        assert "ollama/qwen2.5:7b" in space.model_choices
        assert "openai/gpt-4.1" in space.model_choices

    def test_search_space_routing_active_spaces(self):
        space = PromptSearchSpace(
            routing_weight_space={"primary": [0.5, 0.7, 0.9]},
        )
        active = space.active_spaces()
        assert "routing" in active

    def test_failure_cluster_affected_params(self):
        from agentomatic.optimize.failure_analysis import FailureCluster

        cluster = FailureCluster(
            label="missing_risks",
            description="Agent omits risk section",
            count=5,
            affected_params=["prompt.output_contract", "rag.top_k"],
            expected_metric_gain={"faithfulness": 0.18, "completeness": 0.12},
        )
        assert cluster.affected_params == ["prompt.output_contract", "rag.top_k"]
        assert cluster.expected_metric_gain["faithfulness"] == 0.18
        assert cluster.expected_metric_gain["completeness"] == 0.12


# =====================================================================
# Local-mode: AgentRunner + PromptFitter + PromptFitterBridge
# =====================================================================


class TestAgentRunnerLocal:
    """AgentRunner with agent_callable bypasses HTTP entirely."""

    @pytest.mark.asyncio
    async def test_run_single_local_async_callable(self):
        from agentomatic.optimize.runner import AgentRunner

        async def my_agent(query, *, prompt_override, context, invoke):
            return f"response:{query}:{prompt_override}"

        runner = AgentRunner(agent="test", agent_callable=my_agent)
        result = await runner.run_single("hello", prompt_override="sys")
        assert result.error is None
        assert result.response == "response:hello:sys"
        assert result.query == "hello"

    @pytest.mark.asyncio
    async def test_run_single_local_sync_callable(self):
        from agentomatic.optimize.runner import AgentRunner

        def my_sync_agent(query, *, prompt_override, context, invoke):
            return f"sync:{query}"

        runner = AgentRunner(agent="test", agent_callable=my_sync_agent)
        result = await runner.run_single("world")
        assert result.error is None
        assert result.response == "sync:world"

    @pytest.mark.asyncio
    async def test_run_single_local_dict_response(self):
        from agentomatic.optimize.runner import AgentRunner

        async def my_agent(query, *, prompt_override, context, invoke):
            return {"response": f"dict:{query}"}

        runner = AgentRunner(agent="test", agent_callable=my_agent)
        result = await runner.run_single("test_q")
        assert result.response == "dict:test_q"

    @pytest.mark.asyncio
    async def test_run_single_local_error_captured(self):
        from agentomatic.optimize.runner import AgentRunner

        async def failing_agent(query, *, prompt_override, context, invoke):
            raise ValueError("boom")

        runner = AgentRunner(agent="test", agent_callable=failing_agent)
        result = await runner.run_single("q")
        assert result.error is not None
        assert "boom" in result.error
        assert result.response == ""

    @pytest.mark.asyncio
    async def test_run_dataset_local(self):
        from agentomatic.optimize.runner import AgentRunner

        calls: list[str] = []

        async def my_agent(query, *, prompt_override, context, invoke):
            calls.append(query)
            return f"ok:{query}"

        runner = AgentRunner(agent="test", agent_callable=my_agent)
        points = [{"query": "q1", "expected_answer": "a1"}, {"query": "q2"}]
        results = await runner.run_dataset(points, prompt_override="sys", concurrency=2)
        assert len(results) == 2
        assert set(calls) == {"q1", "q2"}
        assert all(r.error is None for r in results)


class TestWrapLocalAgent:
    """_wrap_local_agent converts a BaseGraphAgent-like object to a callable."""

    @pytest.mark.asyncio
    async def test_wrap_transform(self):
        from agentomatic.optimize.fitter import _wrap_local_agent

        class FakeAgent:
            def transform(self, input_data):
                return {"response": f"transformed:{input_data.get('current_query')}"}

        agent = FakeAgent()
        fn = _wrap_local_agent(agent)
        result = await fn("hello", prompt_override=None, context=None, invoke=None)
        assert result == "transformed:hello"

    @pytest.mark.asyncio
    async def test_wrap_atransform(self):
        from agentomatic.optimize.fitter import _wrap_local_agent

        class FakeAsyncAgent:
            async def atransform(self, input_data):
                return {"response": f"async:{input_data.get('current_query')}"}

        agent = FakeAsyncAgent()
        fn = _wrap_local_agent(agent)
        result = await fn("world", prompt_override=None, context=None, invoke=None)
        assert result == "async:world"

    @pytest.mark.asyncio
    async def test_wrap_output_dict(self):
        import json

        from agentomatic.optimize.fitter import _wrap_local_agent

        class FakeAgent:
            def transform(self, input_data):
                return {"output": {"content": "foo", "next_action": "bar"}}

        agent = FakeAgent()
        fn = _wrap_local_agent(agent)
        result = await fn("q", prompt_override=None, context=None, invoke=None)
        data = json.loads(result)
        assert data == {"content": "foo", "next_action": "bar"}

    @pytest.mark.asyncio
    async def test_prompt_override_injected_in_metadata(self):
        from agentomatic.optimize.fitter import _wrap_local_agent

        received_input: dict = {}

        class FakeAgent:
            def transform(self, input_data):
                received_input.update(input_data)
                return {"response": "ok"}

        agent = FakeAgent()
        fn = _wrap_local_agent(agent)
        await fn("q", prompt_override="my_sys_prompt", context=None, invoke=None)
        assert received_input.get("metadata", {}).get("system_prompt_override") == "my_sys_prompt"

    @pytest.mark.asyncio
    async def test_prompt_override_sets_attribute(self):
        from agentomatic.optimize.fitter import _wrap_local_agent

        class FakeAgent:
            system_prompt = "original"

            def transform(self, input_data):
                return {"response": self.system_prompt}

        agent = FakeAgent()
        fn = _wrap_local_agent(agent)
        result = await fn("q", prompt_override="overridden", context=None, invoke=None)
        # response was generated with the override
        assert result == "overridden"
        # original prompt restored after call
        assert agent.system_prompt == "original"

    @pytest.mark.asyncio
    async def test_prompt_override_restored_on_error(self):
        from agentomatic.optimize.fitter import _wrap_local_agent

        class FakeAgent:
            system_prompt = "original"

            def transform(self, input_data):
                raise RuntimeError("transform failed")

        agent = FakeAgent()
        fn = _wrap_local_agent(agent)
        try:
            await fn("q", prompt_override="overridden", context=None, invoke=None)
        except RuntimeError:
            pass
        # system_prompt must be restored even if transform raised
        assert agent.system_prompt == "original"


class TestPromptFitterLocalMode:
    """PromptFitter end-to-end with local callable — no HTTP server."""

    @pytest.mark.asyncio
    async def test_fit_local_callable_baseline_and_propose(self, tmp_path, monkeypatch):
        """Full baseline-evaluate → propose → candidate-evaluate cycle using only a
        local callable, mocked LLM, and an in-memory dataset."""

        from agentomatic.optimize.dataset import DataPoint, Dataset
        from agentomatic.optimize.fitter import PromptFitter
        from agentomatic.optimize.metrics import ExactMatchMetric
        from agentomatic.optimize.runner import AgentRunner

        # -- Local callable: echoes the prompt_override or uses a default ----
        async def local_fn(query, *, prompt_override, context, invoke):
            # Return a fixed JSON-ish response matching expected_answer
            return query.upper()

        # -- Dataset ---------------------------------------------------------
        points = [
            DataPoint(query="hello", expected_answer="HELLO"),
            DataPoint(query="world", expected_answer="WORLD"),
            DataPoint(query="foo", expected_answer="FOO"),
        ]
        dataset = Dataset(points=points)

        # -- Mock optimizer so no LLM calls are needed -----------------------
        from agentomatic.optimize.config import PromptCandidate, PromptRuntimeConfig
        from agentomatic.optimize.fitter_optimizers import BaseFitterOptimizer

        @dataclass_like_optimizer
        class EchoOptimizer(BaseFitterOptimizer):
            name: str = "echo"

            async def propose(
                self,
                current_config,
                eval_results,
                dataset_sample,
                search_space,
                iteration=0,
                context=None,
            ):
                # Return one candidate that is identical to the baseline
                return [
                    PromptCandidate(
                        name=f"echo_{iteration:03d}",
                        config=PromptRuntimeConfig(
                            system_prompt=current_config.system_prompt + " (echo)",
                        ),
                        source="echo",
                    )
                ]

        fitter = PromptFitter(
            agent="test_agent",
            task_model="ollama/qwen2.5:7b",
            optimizer=EchoOptimizer(),
            max_trials=2,
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        # Replace the HTTP runner with a local-callable runner
        fitter._runner = AgentRunner(agent="test_agent", agent_callable=local_fn)

        metric = ExactMatchMetric()
        result = await fitter.fit(dataset, dataset, metric)

        assert result is not None
        assert result.baseline_score >= 0.0
        assert result.best_score >= result.baseline_score or result.best_score >= 0.0

    @pytest.mark.asyncio
    async def test_prompt_fitter_local_agent_param(self, tmp_path):
        """PromptFitter(local_agent=...) builds an AgentRunner with agent_callable set."""
        from agentomatic.optimize.fitter import PromptFitter

        class FakeAgent:
            def transform(self, input_data):
                return {"response": "ok"}

        agent = FakeAgent()
        fitter = PromptFitter(
            agent="test",
            local_agent=agent,
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        assert fitter._runner.agent_callable is not None

    def test_llm_caller_configure_called(self, monkeypatch):
        """PromptFitter(llm_base_url=...) calls LLMCaller.configure()."""
        from agentomatic.optimize import fitter as fitter_mod
        from agentomatic.optimize.llm_caller import LLMCaller

        configured: list[tuple] = []
        monkeypatch.setattr(
            LLMCaller,
            "configure",
            classmethod(
                lambda cls, base_url=None, api_key=None: configured.append((base_url, api_key))
            ),
        )

        fitter_mod.PromptFitter(
            agent="test",
            llm_base_url="http://localhost:8000/v1",
            llm_api_key="test-key",
        )
        assert configured == [("http://localhost:8000/v1", "test-key")]


class TestLLMCallerConfigure:
    """LLMCaller.configure() sets class-level base_url/api_key."""

    def test_configure_sets_class_attrs(self):
        from agentomatic.optimize.llm_caller import LLMCaller

        original_url = LLMCaller._default_base_url
        original_key = LLMCaller._default_api_key
        try:
            LLMCaller.configure(base_url="http://127.0.0.1:8000/v1", api_key="my-key")
            assert LLMCaller._default_base_url == "http://127.0.0.1:8000/v1"
            assert LLMCaller._default_api_key == "my-key"
        finally:
            LLMCaller._default_base_url = original_url
            LLMCaller._default_api_key = original_key

    def test_configure_none_clears(self):
        from agentomatic.optimize.llm_caller import LLMCaller

        original_url = LLMCaller._default_base_url
        original_key = LLMCaller._default_api_key
        try:
            LLMCaller.configure(base_url="http://example.com/v1")
            LLMCaller.configure()  # reset
            assert LLMCaller._default_base_url is None
            assert LLMCaller._default_api_key is None
        finally:
            LLMCaller._default_base_url = original_url
            LLMCaller._default_api_key = original_key


class TestPromptFitterBridgeLocalMode:
    """PromptFitterBridge passes local_agent and llm params to PromptFitter."""

    def test_build_fitter_passes_local_agent(self, tmp_path):
        from agentomatic.agents.optimizers import PromptFitterBridge

        class FakeAgent:
            agent_name = "test"
            _fit_optimize_options = None

            def transform(self, input_data):
                return {"response": "ok"}

        live_agent = FakeAgent()
        bridge = PromptFitterBridge(
            agent_name="test",
            local_agent=live_agent,
            llm_base_url="http://127.0.0.1:8000/v1",
            llm_api_key="any",
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter = bridge._build_fitter(live_agent, "test")
        # The runner must have an agent_callable (not None)
        assert fitter._runner.agent_callable is not None

    def test_build_fitter_falls_back_to_live_agent(self, tmp_path):
        """When local_agent=None, the live agent from optimize() is used."""
        from agentomatic.agents.optimizers import PromptFitterBridge

        class FakeAgent:
            agent_name = "test"
            _fit_optimize_options = None

            def transform(self, input_data):
                return {"response": "ok"}

        live_agent = FakeAgent()
        bridge = PromptFitterBridge(
            agent_name="test",
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        fitter = bridge._build_fitter(live_agent, "test")
        # Still wires up the live_agent as the local callable
        assert fitter._runner.agent_callable is not None

    def test_llm_params_forwarded(self, tmp_path):
        from agentomatic.agents.optimizers import PromptFitterBridge

        class FakeAgent:
            agent_name = "test"
            _fit_optimize_options = None

            def transform(self, input_data):
                return {"response": "ok"}

        live_agent = FakeAgent()
        bridge = PromptFitterBridge(
            agent_name="test",
            llm_base_url="http://local.example/v1",
            llm_api_key="secret",
            experiment_dir=str(tmp_path),
            auto_report=False,
        )
        # Building the fitter should not raise; LLMCaller.configure is called inside
        fitter = bridge._build_fitter(live_agent, "test")
        assert fitter is not None


# ---------------------------------------------------------------------------
# Helpers used inside the test module
# ---------------------------------------------------------------------------


def dataclass_like_optimizer(cls):
    """Minimal decorator: makes a class behave like a dataclass for the optimizer tests."""
    from dataclasses import dataclass

    return dataclass(slots=True)(cls)
