"""Tests for fit learning history, generalization safety, and apply guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentomatic.agents.types import AgentExample
from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.optimize.learning import (
    check_generalization,
    format_learnings_history,
    split_holdout,
    synthesize_epoch_learning,
)


class TestJudgeDeterminismDefaults:
    """Judge metrics must default to temperature=0.0 for stable epochs."""

    def test_llm_judge_metric_defaults_to_zero_temp(self) -> None:
        from agentomatic.optimize.metrics import GEvalMetric, LLMJudgeMetric

        assert LLMJudgeMetric(criteria="x").temperature == 0.0
        assert GEvalMetric().temperature == 0.0


class TestRichExpectedReference:
    """Boolean-flag expected_output must yield a quality contract, not a key list."""

    def test_flag_only_includes_quality_contract(self) -> None:
        ex = AgentExample(
            id="e1",
            input={"query": "What should we do next?", "question": "Next steps?"},
            expected_output={"next_action": True},
        )
        dp = ex.to_datapoint()
        assert dp.expected_answer is not None
        assert "Response must include: 'next_action'" not in dp.expected_answer
        assert "quality contract" in dp.expected_answer.lower()
        assert "next_action" in dp.expected_answer
        assert "Next steps?" in dp.expected_answer or "What should we do" in dp.expected_answer

    def test_judge_expected_metadata_wins(self) -> None:
        ex = AgentExample(
            id="e2",
            input={"query": "hi"},
            expected_output={"next_action": True},
            metadata={"judge_expected": "Propose a concrete next step with rationale."},
        )
        dp = ex.to_datapoint()
        assert dp.expected_answer == "Propose a concrete next step with rationale."

    def test_rich_expected_serialises_structured_output(self) -> None:
        ex = AgentExample(
            id="e3",
            input={"query": "q"},
            expected_output={"response": "Paris", "confidence": 0.9},
        )
        dp = ex.to_datapoint()
        assert "Paris" in (dp.expected_answer or "")
        assert "confidence" in (dp.expected_answer or "")


class TestGeneralizationCheck:
    def test_rejects_large_gap(self) -> None:
        result = check_generalization(
            fit_score=0.95,
            holdout_score=0.60,
            max_gap=0.15,
        )
        assert result.ok is False
        assert "Overfit" in result.reason

    def test_accepts_small_gap(self) -> None:
        result = check_generalization(
            fit_score=0.80,
            holdout_score=0.75,
            max_gap=0.15,
        )
        assert result.ok is True

    def test_rejects_holdout_regression(self) -> None:
        result = check_generalization(
            fit_score=0.85,
            holdout_score=0.50,
            baseline_holdout=0.70,
            max_gap=0.5,
        )
        assert result.ok is False
        assert "regression" in result.reason.lower()

    def test_skips_when_no_holdout(self) -> None:
        result = check_generalization(fit_score=0.9, holdout_score=None)
        assert result.ok is True


class TestEpochLearning:
    def test_synthesize_extracts_failures_and_focus(self) -> None:
        details = [
            {
                "query": "bad case",
                "expected": "good",
                "response": "wrong",
                "avg_score": 0.2,
                "feedback": "Missing next step",
                "motivation": "No actionable plan",
                "improvement_hints": ["Require a next_action field with concrete text"],
            },
            {
                "query": "good case",
                "response": "ok",
                "avg_score": 0.9,
                "feedback": "Clear and complete",
            },
        ]
        learning = synthesize_epoch_learning(
            round_idx=0,
            prompt_snapshot="You are helpful.",
            score=0.55,
            dims={"correctness": 0.4, "relevance": 0.7},
            eval_details=details,
            accepted=True,
            candidate_name="rewrite_000",
            holdout_score=0.50,
            train_score=0.70,
        )
        assert learning.accepted is True
        assert learning.what_failed
        assert learning.what_worked
        assert learning.next_focus
        assert any("overfit" in x.lower() or "gap" in x.lower() for x in learning.next_focus)
        assert "Missing next step" in format_learnings_history([learning])

    def test_split_holdout_reserves_points(self) -> None:
        pts = [{"query": f"q{i}"} for i in range(20)]
        fit, hold = split_holdout(pts, fraction=0.2, min_size=2)
        assert len(hold) >= 2
        assert len(fit) + len(hold) == 20
        assert not set(map(id, fit)) & set(map(id, hold))

    def test_split_holdout_tiny_dataset(self) -> None:
        pts = [{"query": "a"}, {"query": "b"}]
        fit, hold = split_holdout(pts, fraction=0.25, min_size=1)
        assert len(hold) == 1
        assert len(fit) == 1


class TestApplyGuards:
    def _result(self, best: float, baseline: float, gap: float | None = None) -> PromptFitResult:
        return PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="best"),
            baseline_config=PromptRuntimeConfig(system_prompt="base"),
            best_score=best,
            baseline_score=baseline,
            agent="demo",
            holdout_score=(best - gap) if gap is not None else None,
            generalization_gap=gap,
            prompt_history=[
                {
                    "round_idx": 0,
                    "score": baseline,
                    "accepted": False,
                    "prompt_snapshot": "base",
                },
                {
                    "round_idx": 1,
                    "score": best,
                    "accepted": True,
                    "prompt_snapshot": "best",
                },
            ],
        )

    def test_apply_refuses_zero_improvement(self, tmp_path: Path) -> None:
        result = self._result(0.5, 0.5)
        written = result.apply(version="v2", agent_dir=str(tmp_path))
        assert written is None
        assert not (tmp_path / "prompts.json").exists()
        assert result.applied is False

    def test_apply_force_allows_zero_improvement(self, tmp_path: Path) -> None:
        result = self._result(0.5, 0.5)
        written = result.apply(version="v2", agent_dir=str(tmp_path), force=True)
        assert written == "v2"
        assert (tmp_path / "prompts.json").exists()
        assert (tmp_path / "fit_history.jsonl").exists()

    def test_apply_refuses_large_generalization_gap(self, tmp_path: Path) -> None:
        result = self._result(0.9, 0.5, gap=0.35)
        written = result.apply(version="v2", agent_dir=str(tmp_path))
        assert written is None

    def test_apply_writes_when_improved(self, tmp_path: Path) -> None:
        result = self._result(0.8, 0.5, gap=0.05)
        written = result.apply(version="v2_fit", agent_dir=str(tmp_path))
        assert written == "v2_fit"
        data = json.loads((tmp_path / "prompts.json").read_text(encoding="utf-8"))
        assert data["v2_fit"]["metadata"]["absolute_improvement"] == pytest.approx(0.3)
        history = (tmp_path / "fit_history.jsonl").read_text(encoding="utf-8").strip()
        assert "prompt_history" in history

    def test_summary_includes_curve_and_holdout(self) -> None:
        result = self._result(0.8, 0.5, gap=0.05)
        text = result.summary()
        assert "Holdout" in text
        assert "Score curve" in text or "Epoch learnings" in text


@pytest.mark.asyncio
async def test_fitter_records_prompt_history_and_holdout(tmp_path: Path) -> None:
    """End-to-end fitter with deterministic metric records learnings + holdout."""
    from agentomatic.optimize.dataset import Dataset
    from agentomatic.optimize.fitter import PromptFitter
    from agentomatic.optimize.metrics import ExactMatchMetric

    class TinyAgent:
        system_prompt = "Answer with the expected token."

        def transform(self, data: dict) -> dict:
            q = data.get("current_query", "")
            # Echo a stable answer so ExactMatch can improve with better prompts
            # only when the prompt contains the magic word "EXACT".
            prompt = data.get("system_prompt_override") or self.system_prompt
            if "EXACT" in prompt:
                return {"response": q.replace("Q:", "").strip()}
            return {"response": "wrong"}

    points = [{"query": f"Q: ans{i}", "expected_answer": f"ans{i}"} for i in range(10)]
    train = Dataset.from_list(points[:6])
    val = Dataset.from_list(points)

    class FixedRewrite:
        name = "fixed_rewrite"

        async def propose(
            self,
            current_config,
            eval_results,
            dataset_sample,
            search_space,
            iteration=0,
            context=None,
        ):
            from agentomatic.optimize.config import PromptCandidate

            return [
                PromptCandidate(
                    name=f"rewrite_{iteration:03d}",
                    config=PromptRuntimeConfig(
                        system_prompt="EXACT match the expected answer token.",
                        model_params=dict(current_config.model_params),
                    ),
                    source="rewrite",
                    mutation_notes="inject EXACT",
                )
            ]

    fitter = PromptFitter(
        agent="tiny",
        optimizer=FixedRewrite(),
        local_agent=TinyAgent(),
        max_trials=4,
        min_absolute_improvement=0.01,
        max_generalization_gap=0.5,
        holdout_fraction=0.3,
        drain_seconds=0.0,
        sequential=True,
        auto_report=False,
        experiment_dir=str(tmp_path),
        baseline_system_prompt="Be vague.",
    )
    result = await fitter.fit(train, val, ExactMatchMetric())

    assert result.prompt_history, "epoch learnings must be recorded"
    assert result.holdout_score is not None
    assert (tmp_path / "tiny" / "retrain_history.jsonl").exists()
    # Improvement path should beat the all-wrong baseline
    assert result.best_score >= result.baseline_score
