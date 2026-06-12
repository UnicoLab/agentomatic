"""Tests for agentomatic.optimize module."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentomatic.optimize.dataset import DataPoint, Dataset
from agentomatic.optimize.metrics import (
    BaseMetric,
    ContainsMetric,
    CustomMetric,
    EvalResult,
    ExactMatchMetric,
    resolve_metrics,
)
from agentomatic.optimize.optimizer import (
    ExperimentLog,
    OptimizationResult,
    PromptOptimizer,
)
from agentomatic.optimize.report import generate_html_report
from agentomatic.optimize.strategies import (
    ChainOfThought,
    FewShotBootstrap,
    IterativeRewrite,
    IterationResult,
    resolve_strategy,
)
from agentomatic.optimize.runner import AgentRunner, RunResult


# =====================================================================
# Dataset Tests
# =====================================================================


class TestDataPoint:
    def test_create_minimal(self):
        dp = DataPoint(query="Hello")
        assert dp.query == "Hello"
        assert dp.expected_answer is None
        assert dp.context == []
        assert dp.metadata == {}

    def test_create_full(self):
        dp = DataPoint(
            query="What is X?",
            expected_answer="X is Y",
            context=["doc1", "doc2"],
            metadata={"category": "test"},
        )
        assert dp.expected_answer == "X is Y"
        assert len(dp.context) == 2

    def test_to_dict(self):
        dp = DataPoint(query="Q", expected_answer="A")
        d = dp.to_dict()
        assert d["query"] == "Q"
        assert d["expected_answer"] == "A"
        assert "context" not in d  # empty context not included

    def test_to_dict_with_context(self):
        dp = DataPoint(query="Q", context=["c1"])
        d = dp.to_dict()
        assert d["context"] == ["c1"]


class TestDataset:
    def test_from_list(self):
        ds = Dataset.from_list([
            {"query": "Q1", "expected_answer": "A1"},
            {"query": "Q2"},
        ])
        assert len(ds) == 2
        assert ds[0].query == "Q1"
        assert ds[1].expected_answer is None

    def test_iter(self):
        ds = Dataset.from_list([{"query": "Q1"}, {"query": "Q2"}])
        queries = [p.query for p in ds]
        assert queries == ["Q1", "Q2"]

    def test_split(self):
        items = [{"query": f"Q{i}"} for i in range(10)]
        ds = Dataset.from_list(items)
        train, test = ds.split(0.8)
        assert len(train) == 8
        assert len(test) == 2

    def test_add(self):
        ds = Dataset()
        ds.add(DataPoint(query="Q1"))
        assert len(ds) == 1

    def test_from_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"query": "Q1", "expected_answer": "A1"}\n'
            '{"query": "Q2", "context": ["c1"]}\n'
        )
        ds = Dataset.from_jsonl(str(path))
        assert len(ds) == 2
        assert ds[0].expected_answer == "A1"
        assert ds[1].context == ["c1"]

    def test_to_jsonl(self, tmp_path):
        ds = Dataset.from_list([
            {"query": "Q1", "expected_answer": "A1"},
        ])
        path = tmp_path / "out.jsonl"
        ds.to_jsonl(str(path))
        loaded = Dataset.from_jsonl(str(path))
        assert len(loaded) == 1
        assert loaded[0].query == "Q1"

    def test_from_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("query,expected_answer\nQ1,A1\nQ2,A2\n")
        ds = Dataset.from_csv(str(path))
        assert len(ds) == 2
        assert ds[0].expected_answer == "A1"

    def test_empty_lines_in_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"query": "Q1"}\n\n{"query": "Q2"}\n\n')
        ds = Dataset.from_jsonl(str(path))
        assert len(ds) == 2


# =====================================================================
# Metrics Tests
# =====================================================================


class TestExactMatchMetric:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        m = ExactMatchMetric(fuzzy=False)
        result = await m.evaluate("q", "hello", expected="hello")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_exact_no_match(self):
        m = ExactMatchMetric(fuzzy=False)
        result = await m.evaluate("q", "hi", expected="hello")
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_fuzzy_match(self):
        m = ExactMatchMetric(fuzzy=True)
        result = await m.evaluate("q", "hello world", expected="hello worl")
        assert result.score > 0.8

    @pytest.mark.asyncio
    async def test_no_expected(self):
        m = ExactMatchMetric()
        result = await m.evaluate("q", "answer", expected=None)
        assert result.score == 0.0


class TestContainsMetric:
    @pytest.mark.asyncio
    async def test_all_keywords(self):
        m = ContainsMetric()
        result = await m.evaluate("q", "I like cats and dogs", expected="cats, dogs")
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_partial_keywords(self):
        m = ContainsMetric()
        result = await m.evaluate("q", "I like cats", expected="cats, dogs")
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_no_expected(self):
        m = ContainsMetric()
        result = await m.evaluate("q", "answer", expected=None)
        assert result.score == 0.0


class TestCustomMetric:
    @pytest.mark.asyncio
    async def test_sync_callable(self):
        def has_greeting(q, r, e, c) -> float:
            return 1.0 if "hello" in r.lower() else 0.0

        m = CustomMetric(has_greeting, name="greeting")
        result = await m.evaluate("q", "Hello there!")
        assert result.score == 1.0
        assert result.metric_name == "greeting"

    @pytest.mark.asyncio
    async def test_async_callable(self):
        async def async_check(q, r, e, c) -> float:
            return 0.75

        m = CustomMetric(async_check, name="async_test")
        result = await m.evaluate("q", "response")
        assert result.score == 0.75


class TestResolveMetrics:
    def test_resolve_builtin(self):
        metrics = resolve_metrics(["exact_match", "contains"])
        assert len(metrics) == 2
        assert metrics[0].name == "exact_match"
        assert metrics[1].name == "contains"

    def test_resolve_instance(self):
        custom = ExactMatchMetric(fuzzy=True)
        metrics = resolve_metrics([custom])
        assert metrics[0] is custom

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            resolve_metrics(["nonexistent_metric"])

    def test_resolve_wrong_type(self):
        with pytest.raises(TypeError):
            resolve_metrics([123])  # type: ignore


# =====================================================================
# Strategy Tests
# =====================================================================


class TestResolveStrategy:
    def test_resolve_iterative_rewrite(self):
        s = resolve_strategy("iterative_rewrite")
        assert isinstance(s, IterativeRewrite)

    def test_resolve_few_shot(self):
        s = resolve_strategy("few_shot")
        assert isinstance(s, FewShotBootstrap)

    def test_resolve_cot(self):
        s = resolve_strategy("cot")
        assert isinstance(s, ChainOfThought)

    def test_resolve_instance(self):
        strategy = IterativeRewrite(model="test")
        s = resolve_strategy(strategy)
        assert s is strategy

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolve_strategy("nonexistent")


class TestFewShotBootstrap:
    @pytest.mark.asyncio
    async def test_step_adds_examples(self):
        s = FewShotBootstrap(n_examples=2)
        eval_results = [
            {"query": "Q1", "response": "R1", "avg_score": 0.9},
            {"query": "Q2", "response": "R2", "avg_score": 0.3},
            {"query": "Q3", "response": "R3", "avg_score": 0.8},
        ]
        result = await s.step("Original prompt", eval_results, [], 1)
        assert "Examples" in result
        assert "Q1" in result  # top scored
        assert "Original prompt" in result


class TestChainOfThought:
    @pytest.mark.asyncio
    async def test_step_adds_reasoning(self):
        s = ChainOfThought()
        eval_results = [
            {"query": "Why is the sky blue?", "avg_score": 0.3},
            {"query": "Compare X vs Y", "avg_score": 0.4},
        ]
        result = await s.step("Base prompt", eval_results, [], 1)
        assert "Reasoning Instructions" in result
        assert "Base prompt" in result


# =====================================================================
# Runner Tests
# =====================================================================


class TestRunResult:
    def test_create(self):
        r = RunResult(query="Q", response="R", duration_ms=50.0)
        assert r.error is None


class TestAgentRunner:
    @pytest.mark.asyncio
    async def test_run_single_success(self):
        runner = AgentRunner(agent="test_agent")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello!"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=mock_response)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = instance

            result = await runner.run_single("Hello")
            assert result.response == "Hello!"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_run_single_error(self):
        runner = AgentRunner(agent="test_agent")
        with patch("httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.post = AsyncMock(side_effect=Exception("Connection refused"))
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = instance

            result = await runner.run_single("Hello")
            assert result.error is not None
            assert "Connection" in result.error


# =====================================================================
# Optimizer Tests
# =====================================================================


class TestPromptOptimizer:
    def test_constructor_defaults(self):
        opt = PromptOptimizer(agent="test", auto_report=False)
        assert opt.agent == "test"
        assert opt.rewrite_llm == "ollama/mistral:7b"
        assert opt.eval_llm == "ollama/mistral:7b"

    def test_constructor_dual_models(self):
        opt = PromptOptimizer(
            agent="test",
            rewrite_llm="ollama/llama3:70b",
            eval_llm="ollama/mistral:7b",
            auto_report=False,
        )
        assert opt.rewrite_llm == "ollama/llama3:70b"
        assert opt.eval_llm == "ollama/mistral:7b"

    def test_constructor_single_llm_propagates(self):
        opt = PromptOptimizer(
            agent="test", llm="openai/gpt-4", auto_report=False
        )
        assert opt.rewrite_llm == "openai/gpt-4"
        assert opt.eval_llm == "openai/gpt-4"


# =====================================================================
# OptimizationResult Tests
# =====================================================================


class TestOptimizationResult:
    def _make_result(self) -> OptimizationResult:
        return OptimizationResult(
            best_prompt="You are great",
            best_score=0.85,
            best_iteration=3,
            baseline_prompt="You are ok",
            baseline_score=0.60,
            history=[
                IterationResult(iteration=0, prompt="You are ok", avg_score=0.60,
                                per_metric_scores={"exact_match": 0.55, "contains": 0.65}),
                IterationResult(iteration=1, prompt="You are good", avg_score=0.70,
                                per_metric_scores={"exact_match": 0.65, "contains": 0.75}),
                IterationResult(iteration=2, prompt="You are better", avg_score=0.75,
                                per_metric_scores={"exact_match": 0.70, "contains": 0.80}),
                IterationResult(iteration=3, prompt="You are great", avg_score=0.85,
                                per_metric_scores={"exact_match": 0.80, "contains": 0.90}),
            ],
            duration_seconds=12.5,
            experiment_id="test123",
            agent="test_agent",
        )

    def test_improvement(self):
        r = self._make_result()
        assert r.improvement == pytest.approx(41.67, abs=0.1)

    def test_improvement_zero_baseline(self):
        r = OptimizationResult(
            best_prompt="p", best_score=0.5, best_iteration=1,
            baseline_prompt="p", baseline_score=0.0,
        )
        assert r.improvement == float("inf")

    def test_to_dict(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["agent"] == "test_agent"
        assert d["best_score"] == 0.85
        assert d["improvement_pct"] == pytest.approx(41.67, abs=0.1)
        assert len(d["history"]) == 4

    def test_compare(self):
        r = self._make_result()
        text = r.compare()
        assert "0.8500" in text
        assert "0.6000" in text

    def test_report_plain(self):
        r = self._make_result()
        text = r._plain_report()
        assert "test_agent" in text
        assert "0.8500" in text

    def test_apply(self, tmp_path):
        r = self._make_result()
        agent_dir = tmp_path / "test_agent"
        agent_dir.mkdir()
        version = r.apply(agent_dir=str(agent_dir), version="v1_opt")
        assert version == "v1_opt"
        prompts = json.loads((agent_dir / "prompts.json").read_text())
        assert "v1_opt" in prompts
        assert prompts["v1_opt"]["system"] == "You are great"

    def test_apply_appends(self, tmp_path):
        r = self._make_result()
        agent_dir = tmp_path / "test_agent"
        agent_dir.mkdir()
        (agent_dir / "prompts.json").write_text(json.dumps({
            "v1": {"system": "original", "user_template": "{query}"},
        }))
        version = r.apply(agent_dir=str(agent_dir))
        prompts = json.loads((agent_dir / "prompts.json").read_text())
        assert "v1" in prompts  # original preserved
        assert version in prompts  # new version added


# =====================================================================
# Experiment Log Tests
# =====================================================================


class TestExperimentLog:
    def test_log_iteration(self):
        log = ExperimentLog(agent="test")
        it = IterationResult(iteration=1, prompt="p", avg_score=0.8)
        log.log_iteration(it)
        assert len(log.iterations) == 1
        assert log.best_score == 0.8
        assert log.best_iteration == 1

    def test_best_tracking(self):
        log = ExperimentLog(agent="test")
        log.log_iteration(IterationResult(iteration=1, prompt="p", avg_score=0.5))
        log.log_iteration(IterationResult(iteration=2, prompt="p", avg_score=0.9))
        log.log_iteration(IterationResult(iteration=3, prompt="p", avg_score=0.7))
        assert log.best_iteration == 2
        assert log.best_score == 0.9

    def test_save_and_load(self, tmp_path):
        log = ExperimentLog(agent="test")
        log.log_iteration(IterationResult(iteration=1, prompt="p", avg_score=0.8))

        path = tmp_path / "experiments.json"
        log.save(path)

        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["agent"] == "test"
        assert data[0]["best_score"] == 0.8

    def test_save_appends(self, tmp_path):
        path = tmp_path / "experiments.json"

        log1 = ExperimentLog(agent="test1")
        log1.log_iteration(IterationResult(iteration=1, prompt="p", avg_score=0.5))
        log1.save(path)

        log2 = ExperimentLog(agent="test2")
        log2.log_iteration(IterationResult(iteration=1, prompt="p", avg_score=0.9))
        log2.save(path)

        data = json.loads(path.read_text())
        assert len(data) == 2


# =====================================================================
# Report Tests
# =====================================================================


class TestReport:
    def _make_result(self) -> OptimizationResult:
        return OptimizationResult(
            best_prompt="You are great",
            best_score=0.85,
            best_iteration=2,
            baseline_prompt="You are ok",
            baseline_score=0.60,
            history=[
                IterationResult(iteration=0, prompt="You are ok", avg_score=0.60,
                                per_metric_scores={"exact_match": 0.55}),
                IterationResult(iteration=1, prompt="You are good", avg_score=0.70,
                                per_metric_scores={"exact_match": 0.65}),
                IterationResult(iteration=2, prompt="You are great", avg_score=0.85,
                                per_metric_scores={"exact_match": 0.80}),
            ],
            duration_seconds=5.0,
            experiment_id="rpt123",
            agent="test_agent",
        )

    def test_generate_html_report(self, tmp_path):
        r = self._make_result()
        path = tmp_path / "report.html"
        result_path = generate_html_report(r, output_path=path)
        assert Path(result_path).exists()
        html = Path(result_path).read_text()
        assert "test_agent" in html
        assert "0.8500" in html
        assert "svg" in html.lower()  # chart generated
        assert "diff" in html.lower()  # prompt diff present

    def test_report_auto_path(self, tmp_path):
        r = self._make_result()
        # Monkey-patch to write to tmp
        path = tmp_path / ".optimize" / "test_agent" / "report.html"
        result_path = generate_html_report(r, output_path=path)
        assert Path(result_path).exists()
