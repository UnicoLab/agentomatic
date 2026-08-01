"""Tests for Agent Lightning–inspired optimize foundations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentomatic.optimize.algorithm import FitterAlgorithm, as_algorithm
from agentomatic.optimize.algorithms.apo import APOOptimizer, _strip_fences
from agentomatic.optimize.config import PromptCandidate, PromptRuntimeConfig
from agentomatic.optimize.dataset import DataPoint, Dataset
from agentomatic.optimize.feedback_dataset import (
    dataset_from_feedback_jsonl,
    feedback_records_to_dataset,
)
from agentomatic.optimize.fitter_optimizers import (
    ParamSearchOptimizer,
    resolve_fitter_optimizer,
)
from agentomatic.optimize.metrics import ExactMatchMetric
from agentomatic.optimize.resources import ResourceRegistry
from agentomatic.optimize.reward import (
    FeedbackRewardAdapter,
    MetricRewardAdapter,
    resolve_reward_adapter,
)
from agentomatic.optimize.rollout import (
    RewardSignal,
    RolloutTraceStore,
    rollout_from_run_result,
)
from agentomatic.optimize.runner import RunResult
from agentomatic.optimize.search_space import PromptSearchSpace
from agentomatic.optimize.trace_adapter import TraceToCritiqueContext, TraceToMessages


def test_resource_registry_versions() -> None:
    """Publishing configs yields monotonic resource ids."""
    registry = ResourceRegistry(prefix="t")
    cfg = PromptRuntimeConfig(system_prompt="A")
    b0 = registry.publish(cfg, label="seed")
    b1 = registry.publish(
        PromptRuntimeConfig(system_prompt="B"),
        parent_id=b0.resource_id,
    )
    assert b0.resource_id == "t0"
    assert b1.resource_id == "t1"
    assert registry.latest() is b1
    registry.update_score(b1.resource_id, 0.9)
    assert registry.get("t1") is not None
    assert registry.get("t1").score == 0.9  # type: ignore[union-attr]


def test_rollout_trace_store_memory_and_sqlite(tmp_path: Path) -> None:
    """Rollouts persist in memory and optional SQLite."""
    rr = RunResult(
        query="q",
        response="a",
        expected="a",
        steps_taken=["respond"],
        tool_calls=[{"name": "search", "output": "x"}],
        reasoning="think",
    )
    rollout = rollout_from_run_result(rr, resource_id="r0", reward=0.75)
    assert rollout.reward is not None
    assert rollout.reward.value == 0.75
    assert any(s.kind == "tool" for s in rollout.spans)

    store = RolloutTraceStore(path=tmp_path / "rollouts.db")
    store.add(rollout)
    assert len(store) == 1
    assert store.list(resource_id="r0")[0].query == "q"

    reloaded = RolloutTraceStore(path=tmp_path / "rollouts.db")
    assert len(reloaded) == 1


def test_trace_to_messages_and_node_match() -> None:
    """Trace adapter builds messages and filters by node_match."""
    rr = RunResult(
        query="hello",
        response="world",
        steps_taken=["plan", "respond"],
        tool_calls=[{"name": "lookup", "output": "doc"}],
        reasoning="r",
    )
    all_msgs = TraceToMessages().adapt_run_result(rr)
    assert any(m.get("role") == "user" for m in all_msgs)
    filtered = TraceToMessages(node_match=r"respond|lookup").adapt_run_result(rr)
    contents = " ".join(str(m.get("content")) for m in filtered)
    assert "plan" not in contents or "[step:plan]" not in contents
    assert "lookup" in contents or any(m.get("name") == "lookup" for m in filtered)


def test_critique_context_ranks_failures() -> None:
    """Critique context prefers low-scoring eval rows."""
    details = [
        {"query": "ok", "response": "y", "avg_score": 0.9, "feedback": "good"},
        {"query": "bad", "response": "n", "avg_score": 0.1, "feedback": "miss"},
    ]
    experiments = TraceToCritiqueContext().from_eval_details(details, max_items=1)
    assert len(experiments) == 1
    assert experiments[0].query == "bad"


def test_reward_adapters() -> None:
    """Metric and feedback reward adapters map onto [0, 1]."""
    from agentomatic.optimize.metrics import EvalResult

    metric_reward = MetricRewardAdapter().reward_from_eval(
        EvalResult(metric_name="x", score=0.8, reason="ok", metadata={"dimensions": {"a": 0.7}})
    )
    assert metric_reward.value == 0.8
    assert metric_reward.dimensions["a"] == 0.7

    fb = FeedbackRewardAdapter()
    assert fb.reward_from_rating(5).value == 1.0
    assert fb.reward_from_rating(1).value == 0.0
    assert resolve_reward_adapter("feedback").reward_from_rating(3).value == 0.5  # type: ignore[attr-defined]


def test_feedback_records_to_dataset() -> None:
    """Feedback export becomes an optimisation dataset."""
    records = [
        {"query": "Q1", "correction": "A1", "rating": 1, "comment": "wrong"},
        {"query": "Q2", "response": "A2", "rating": 5},
        {"query": "", "correction": "ignored"},
    ]
    ds = feedback_records_to_dataset(records, only_corrections=True)
    assert len(ds) == 1
    assert ds.points[0].expected_answer == "A1"


def test_dataset_from_feedback_jsonl(tmp_path: Path) -> None:
    """JSONL feedback files load into Dataset."""
    path = tmp_path / "fb.jsonl"
    path.write_text(
        '{"query": "hi", "correction": "hello", "rating": 1}\n',
        encoding="utf-8",
    )
    ds = dataset_from_feedback_jsonl(path)
    assert len(ds) == 1


def test_search_space_tpe_and_node_fields() -> None:
    """Search space supports TPE sampling and node scope fields."""
    space = PromptSearchSpace(
        optimize_model_params=True,
        model_param_space={"temperature": [0.0, 0.3, 0.7], "top_p": [0.9, 1.0]},
        search_method="tpe",
        optimize_nodes=["respond"],
        node_match=r"respond",
    )
    observed = [
        ({"temperature": 0.0, "top_p": 0.9}, 0.2),
        ({"temperature": 0.7, "top_p": 1.0}, 0.9),
        ({"temperature": 0.7, "top_p": 0.9}, 0.85),
        ({"temperature": 0.3, "top_p": 1.0}, 0.4),
        ({"temperature": 0.7, "top_p": 1.0}, 0.95),
    ]
    samples = space.sample_params_tpe(3, "model", observed=observed)
    assert len(samples) == 3
    assert all("temperature" in s for s in samples)
    dumped = space.to_dict()
    assert dumped["search_method"] == "tpe"
    assert dumped["optimize_nodes"] == ["respond"]


@pytest.mark.asyncio
async def test_param_search_tpe_observe() -> None:
    """ParamSearchOptimizer records observations and proposes via TPE."""
    space = PromptSearchSpace(
        optimize_model_params=True,
        optimize_few_shot=False,
        optimize_system_prompt=False,
        model_param_space={"temperature": [0.0, 0.5, 1.0]},
        search_method="tpe",
    )
    opt = ParamSearchOptimizer(n_samples=2)
    opt.observe({"temperature": 0.0}, 0.1)
    opt.observe({"temperature": 1.0}, 0.9)
    opt.observe({"temperature": 1.0}, 0.85)
    opt.observe({"temperature": 0.5}, 0.4)
    cfg = PromptRuntimeConfig(system_prompt="x", model_params={"temperature": 0.0})
    cands = await opt.propose(cfg, [], [], space, iteration=0)
    assert cands
    assert all(c.source == "param_search" for c in cands)


def test_resolve_fitter_optimizer_apo() -> None:
    """``apo`` resolves to APOOptimizer."""
    opt = resolve_fitter_optimizer("apo", model="ollama/x")
    assert isinstance(opt, APOOptimizer)
    assert opt.name == "apo"


def test_strip_fences() -> None:
    """APO edit post-processor removes markdown fences."""
    assert _strip_fences("```\nHello\n```") == "Hello"


@pytest.mark.asyncio
async def test_apo_propose_with_mocked_llm() -> None:
    """APO propose yields candidates from textual gradient + edit."""
    opt = APOOptimizer(
        gradient_model="ollama/mock",
        apply_edit_model="ollama/mock",
        branch_factor=2,
        beam_width=2,
    )
    cfg = PromptRuntimeConfig(system_prompt="You are helpful.")
    space = PromptSearchSpace(optimize_system_prompt=True)
    eval_results = [
        {
            "query": "2+2?",
            "response": "maybe 5",
            "expected": "4",
            "avg_score": 0.0,
            "feedback": "wrong",
            "steps_taken": ["respond"],
        }
    ]

    async def _fake_call(model: object, prompt: str, **kwargs: object) -> str:
        lowered = prompt.lower()
        if "return a bullet list" in lowered or "produce a brief critique" in lowered:
            return "- Add an explicit arithmetic instruction"
        return "You are a precise math assistant. Answer with the number only."

    with patch(
        "agentomatic.optimize.algorithms.apo.LLMCaller.call",
        new=AsyncMock(side_effect=_fake_call),
    ):
        cands = await opt.propose(cfg, eval_results, [], space, iteration=0)

    assert cands
    assert all(isinstance(c, PromptCandidate) for c in cands)
    assert all(c.source == "apo" for c in cands)
    assert any("critique" in (c.metadata or {}) for c in cands)


@pytest.mark.asyncio
async def test_fitter_algorithm_wrapper() -> None:
    """FitterAlgorithm delegates to PromptFitter.fit."""
    from agentomatic.optimize.fitter import PromptFitter

    fitter = PromptFitter(
        agent="demo",
        optimizer="param_search",
        max_trials=2,
        local_agent=_DummyAgent(),
        auto_report=False,
        sequential=True,
        search_space=PromptSearchSpace(
            optimize_system_prompt=False,
            optimize_few_shot=False,
            optimize_model_params=True,
            model_param_space={"temperature": [0.0, 0.2]},
        ),
    )
    algo = as_algorithm(fitter)
    assert isinstance(algo, FitterAlgorithm)
    ds = Dataset(
        points=[
            DataPoint(query="ping", expected_answer="pong"),
            DataPoint(query="ping", expected_answer="pong"),
            DataPoint(query="ping", expected_answer="pong"),
            DataPoint(query="ping", expected_answer="pong"),
        ]
    )
    result = await algo.run(ds, ds, metric=ExactMatchMetric())
    assert result.best_config is not None
    assert len(fitter._trace_store) >= 1  # noqa: SLF001


class _DummyAgent:
    """Minimal local agent for fitter tests."""

    agent_name = "demo"
    system_prompt = "echo"

    def transform(self, data: dict) -> dict:
        query = data.get("current_query") or data.get("query") or ""
        return {"response": "pong" if "ping" in str(query) else str(query)}


def test_reward_signal_roundtrip() -> None:
    """RewardSignal serialises cleanly."""
    signal = RewardSignal(value=0.5, dimensions={"x": 0.5}, reason="ok")
    restored = RewardSignal.from_dict(signal.to_dict())
    assert restored.value == 0.5
    assert restored.dimensions["x"] == 0.5


def test_invalid_search_method_raises() -> None:
    """Unknown search_method is rejected."""
    with pytest.raises(ValueError, match="search_method"):
        PromptSearchSpace(search_method="bayes")


# Keep asyncio import used when running under pytest-asyncio auto mode.
_ = asyncio
