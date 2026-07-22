"""Matrix coverage for fitter optimizers + LLM-as-judge metrics.

Proves every registered PromptFitter optimizer name resolves and can run a
minimal ``PromptFitter.fit`` loop (mocked LLM rewrite / judge). Also covers
HolySheet fit reports and the ``run_train`` / ``train_and_report`` aliases.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.optimize.dataset import DataPoint, Dataset
from agentomatic.optimize.fitter_optimizers import resolve_fitter_optimizer
from agentomatic.optimize.judges import LocalJudgeMetric
from agentomatic.optimize.metrics import CustomMetric, ExactMatchMetric
from agentomatic.optimize.report import generate_fit_report
from agentomatic.optimize.train_api import run_train, run_training, train_and_report

# Canonical names exposed on the fit path (aliases collapsed).
FIT_OPTIMIZERS = [
    "rewrite",
    "gepa_like",
    "mipro_like",
    "few_shot_bootstrap",
    "param_search",
]
# Aliases that should resolve to the same classes.
FIT_OPTIMIZER_ALIASES = {
    "gepa": "gepa_like",
    "mipro": "mipro_like",
    "few_shot": "few_shot_bootstrap",
}


def _tiny_dataset() -> tuple[Dataset, Dataset]:
    train = Dataset(
        points=[
            DataPoint(
                query="What is the budget?",
                expected_answer='{"content":"Budget is 1M","next_action":"Confirm with PM"}',
                metadata={"split": "train"},
            ),
            DataPoint(
                query="List next steps",
                expected_answer='{"content":"Draft plan","next_action":"Share with team"}',
                metadata={"split": "train"},
            ),
            DataPoint(
                query="Who owns delivery?",
                expected_answer='{"content":"Delivery owned by ops","next_action":"Ping ops lead"}',
                metadata={"split": "train"},
            ),
        ]
    )
    val = Dataset(
        points=[
            DataPoint(
                query="What is the budget?",
                expected_answer='{"content":"Budget is 1M","next_action":"Confirm with PM"}',
                metadata={"split": "validation"},
            ),
            DataPoint(
                query="List next steps",
                expected_answer='{"content":"Draft plan","next_action":"Share with team"}',
                metadata={"split": "validation"},
            ),
        ]
    )
    return train, val


class _LocalAgent:
    """Deterministic local agent: improves slightly when prompt mentions IMPROVED."""

    agent_name = "matrix_agent"
    system_prompt = "BASE PROMPT"

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        q = str(data.get("current_query") or data.get("query") or "")
        override = ""
        meta = data.get("metadata") or {}
        if isinstance(meta, dict):
            override = str(meta.get("system_prompt_override") or "")
        boosted = "IMPROVED" in (override or self.system_prompt)
        content = f"{'Boosted ' if boosted else ''}Answer about {q}"
        return {
            "content": content,
            "next_action": "Follow up with stakeholder",
        }

    async def atransform(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.transform(data)


@pytest.mark.parametrize("name", FIT_OPTIMIZERS)
def test_resolve_each_fitter_optimizer(name: str) -> None:
    """Every fit-path optimizer name resolves to a propose()-capable instance."""
    opt = resolve_fitter_optimizer(name, model="openai/mock-model")
    assert hasattr(opt, "propose")
    assert getattr(opt, "name", "")


@pytest.mark.parametrize("alias,canonical", list(FIT_OPTIMIZER_ALIASES.items()))
def test_resolve_optimizer_aliases(alias: str, canonical: str) -> None:
    """Short aliases map onto the same optimizer family."""
    a = resolve_fitter_optimizer(alias, model="openai/mock-model")
    b = resolve_fitter_optimizer(canonical, model="openai/mock-model")
    assert type(a) is type(b)


@pytest.mark.parametrize("name", FIT_OPTIMIZERS)
def test_prompt_fitter_runs_each_optimizer(name: str, tmp_path) -> None:
    """PromptFitter.fit completes for each optimizer with mocked rewrite LLM."""
    from agentomatic.optimize.fitter import PromptFitter

    train, val = _tiny_dataset()
    agent = _LocalAgent()

    async def _fake_rewrite(*_a: Any, **_k: Any) -> str:
        return "IMPROVED system prompt with clearer grounding instructions."

    async def _run() -> PromptFitResult:
        fitter = PromptFitter(
            agent="matrix_agent",
            task_model="openai/mock-task",
            rewrite_model="openai/mock-rewrite",
            optimizer=name,
            local_agent=agent,
            max_trials=4,
            concurrency=1,
            sequential=True,
            auto_report=False,
            experiment_dir=str(tmp_path / ".fit"),
            min_absolute_improvement=0.0,
            drain_seconds=0.0,
        )
        with patch(
            "agentomatic.optimize.llm_caller.LLMCaller.call",
            new=AsyncMock(side_effect=_fake_rewrite),
        ):
            return await fitter.fit(train, val, ExactMatchMetric())

    result = asyncio.run(_run())
    assert isinstance(result, PromptFitResult)
    assert result.optimizer_name
    assert result.dataset_sizes.get("train", 0) >= 1
    assert result.early_stop_reason
    assert isinstance(result.score_history, list)
    assert len(result.score_history) >= 1


def test_local_judge_metric_score_path() -> None:
    """LLM-as-judge metric evaluates via async path and returns graded score."""

    async def _fake_json(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {
            "overall_score": 0.72,
            "feedback": "Solid answer",
            "motivation": "Grounded and actionable next step.",
            "what_worked": ["clear content"],
            "what_failed": [],
            "improvement_hints": ["keep grounding"],
            "dimensions": {
                "pertinence": 0.7,
                "groundedness": 0.8,
                "actionability": 0.65,
            },
        }

    judge = LocalJudgeMetric(
        name="pertinence",
        model="openai/mock-judge",
        criteria="Score pertinence, groundedness, actionability.",
        dimensions=["pertinence", "groundedness", "actionability"],
    )

    async def _run() -> Any:
        with patch(
            "agentomatic.optimize.llm_caller.LLMCaller.call_with_json",
            new=AsyncMock(side_effect=_fake_json),
        ):
            return await judge.evaluate(
                query="What is the budget?",
                response='{"content":"Budget is 1M","next_action":"Confirm"}',
                expected='{"content":"Budget is 1M"}',
            )

    out = asyncio.run(_run())
    assert float(out.score) > 0.0
    assert "Motivation" in (out.reason or "") or "Grounded" in (out.reason or "")


def test_fit_with_judge_composite_metric(tmp_path) -> None:
    """PromptFitter accepts a CustomMetric+judge-style objective without crash."""
    from agentomatic.optimize.fitter import PromptFitter

    train, val = _tiny_dataset()
    agent = _LocalAgent()

    def _score(query: str, response: str, expected: str | None = None, context=None) -> float:
        return 0.55 if "Answer" in (response or "") else 0.1

    metric = CustomMetric(fn=_score, name="composite")

    async def _fake_rewrite(*_a: Any, **_k: Any) -> str:
        return "IMPROVED prompt"

    async def _run() -> PromptFitResult:
        fitter = PromptFitter(
            agent="matrix_agent",
            task_model="openai/mock-task",
            rewrite_model="openai/mock-rewrite",
            optimizer="rewrite",
            local_agent=agent,
            max_trials=3,
            concurrency=1,
            sequential=True,
            auto_report=True,
            experiment_dir=str(tmp_path / ".fit"),
            min_absolute_improvement=0.0,
            drain_seconds=0.0,
        )
        with patch(
            "agentomatic.optimize.llm_caller.LLMCaller.call",
            new=AsyncMock(side_effect=_fake_rewrite),
        ):
            return await fitter.fit(train, val, metric)

    result = asyncio.run(_run())
    assert result.best_score >= 0.0
    # auto_report should have written something under experiment_dir
    htmls = list((tmp_path / ".fit").rglob("*.html"))
    assert htmls, "expected auto_report HTML artefact"


def test_holysheet_fit_report_contains_spec(tmp_path) -> None:
    """generate_fit_report emits a HolySheet interactive HTML when installed."""
    pytest.importorskip("holysheet")
    result = PromptFitResult(
        best_config=PromptRuntimeConfig(system_prompt="BEST PROMPT with detail"),
        baseline_config=PromptRuntimeConfig(system_prompt="BASE PROMPT"),
        best_score=0.6,
        baseline_score=0.4,
        trials=[
            {
                "round": 0,
                "name": "cand_a",
                "phase": "full_val",
                "score": 0.6,
                "mutation_notes": "rewrote grounding",
            }
        ],
        score_history=[0.4, 0.5, 0.6],
        prompt_history=[
            {
                "round_idx": 0,
                "score": 0.5,
                "accepted": True,
                "next_focus": ["add grounding"],
                "what_failed": ["vague next_action"],
                "what_worked": ["clear content"],
                "judge_insights": ["Groundedness was weak on budget query."],
                "prompt_snapshot": "BEST PROMPT with detail",
                "candidate_name": "cand_a",
            }
        ],
        suggestions=["Keep the improved grounding clause."],
        duration_seconds=12.5,
        experiment_id="testexp01",
        agent="matrix_agent",
        optimizer_name="rewrite",
        early_stop_reason="completed all 1 optimize round(s) (max_trials=4)",
        dataset_sizes={"train": 3, "fit_val": 2, "holdout": 1, "test": 0},
    )
    out = tmp_path / "fit.html"
    path = generate_fit_report(
        result,
        output_path=out,
        keras_history={"loss": [0.5, 0.4], "val_loss": [0.55, 0.42], "judge": [0.4, 0.5]},
        eval_scores={"judge": 0.51, "f1": 0.44},
        dataset_sizes={"train": 3, "validation": 2, "test": 1},
        optimizer_name="rewrite",
        stack_name="gemini",
        model_name="google/gemini-2.5-flash",
    )
    text = __import__("pathlib").Path(path).read_text(encoding="utf-8")
    assert "__HOLYSHEET_SPEC__" in text
    assert "PromptFitter Report" in text
    assert "rewrite" in text
    assert "Judge samples" in text or "motivation" in text.lower()


def test_train_api_aliases() -> None:
    """Public aliases point at the same run_train implementation."""
    assert train_and_report is run_train
    assert run_training is run_train


def test_train_template_uses_run_train() -> None:
    """Scaffold train.py template is the thin train_and_report pattern."""
    from agentomatic.cli.templates import _train_py

    src = _train_py("assistant")
    assert "TrainConfig" in src
    assert "train_and_report" in src
    assert "print_train_result" in src
    assert "augment" in src
    assert "n_examples" in src
    assert "persist" in src
    assert "persist_fit_store" in src
    assert "from agents.assistant.agent import" in src
    assert "PromptFitterBridge" not in src
    assert "generate_fit_report" not in src


def test_eval_template_uses_evaluate_and_report() -> None:
    """Scaffold eval.py template is the thin evaluate_and_report pattern."""
    from agentomatic.cli.templates import _eval_py

    src = _eval_py("assistant")
    assert "EvalConfig" in src
    assert "evaluate_and_report" in src
    assert "prefer_augmented" in src
    assert "report_path" in src
    assert "from agents.assistant.agent import" in src
    assert "PromptFitterBridge" not in src


def test_prepare_dataset_persist_without_augment(tmp_path) -> None:
    """persist=True writes JSONL even when augment is off."""
    from agentomatic.agents.types import AgentDataset, AgentExample
    from agentomatic.optimize.train_api import prepare_dataset

    ds = AgentDataset(
        name="tiny",
        examples=[
            AgentExample(
                id="t1",
                input={"current_query": "hello"},
                expected_output={"content": "hi", "next_action": "done"},
                split="train",
            )
        ],
    )
    out_path = tmp_path / "out.jsonl"
    out_ds, written = prepare_dataset(
        ds,
        augment=False,
        persist=True,
        persist_path=out_path,
    )
    assert written == out_path
    assert out_path.exists()
    assert len(out_ds.examples) == 1


def test_nr_examples_alias() -> None:
    """nr_examples is accepted as an alias for n_examples."""
    from agentomatic.optimize import TrainConfig

    cfg = TrainConfig(
        agent_name="x",
        agent_dir=Path("."),
        nr_examples=42,
    )
    assert cfg.n_examples == 42
