# pyright: reportMissingParameterType=none
# pyright: reportAttributeAccessIssue=none
"""Prompt optimization must not reward prompts that overfit.

The mechanism, end to end:

* the optimizer **learns** from train only (failures, judge feedback, demos);
* candidates are **selected** on validation, and only when the gain is
  unlikely to be noise (paired bootstrap confidence);
* a **held-out** slice neither step sees can only *veto* a candidate — for
  regressing, for widening the validation/held-out gap, or for a validation
  gain that does not transfer (the signature of memorised validation answers);
* the ``test`` split is never used by optimization at all.

The simulated agent below answers correctly only for the queries its prompt
"knows": ``GENERAL`` knows everything, ``MEMO: <queries>`` only those listed —
the textbook overfit prompt.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentomatic.optimize import (
    CustomMetric,
    DataPoint,
    Dataset,
    PromptFitter,
)
from agentomatic.optimize.config import PromptCandidate, PromptRuntimeConfig
from agentomatic.optimize.events import OptimizationEvent
from agentomatic.optimize.learning import check_generalization, paired_improvement_confidence
from agentomatic.optimize.runner import AgentRunner

TRAIN = [f"t{i}" for i in range(4)]
VAL = [f"v{i}" for i in range(4)]
HOLD = [f"h{i}" for i in range(3)]


def _ds(queries: list[str]) -> Dataset:
    return Dataset(points=[DataPoint(query=q, expected_answer=f"ans-{q}") for q in queries])


async def _agent(query: str, *, prompt_override=None, context=None, invoke=None) -> str:
    prompt = prompt_override or ""
    if "GENERAL" in prompt:
        return f"ans-{query}"
    if prompt.startswith("MEMO:") and query in prompt.split():
        return f"ans-{query}"
    return "no idea"


def _metric() -> CustomMetric:
    return CustomMetric(lambda q, r, e, c: 1.0 if r == e else 0.0, name="exact")


class _Proposer:
    """Duck-typed optimizer that proposes fixed prompts and records what it saw."""

    name = "scripted"

    def __init__(self, rounds: list[list[str]]) -> None:
        self._rounds = rounds
        self.seen_eval_queries: list[str] = []
        self.seen_sample_queries: list[str] = []

    async def propose(
        self,
        current_config,
        eval_results,
        dataset_sample,
        search_space,
        iteration=0,
        context=None,
    ) -> list[PromptCandidate]:
        self.seen_eval_queries += [str(r.get("query")) for r in eval_results]
        self.seen_sample_queries += [str(p.get("query")) for p in dataset_sample]
        prompts = self._rounds[iteration] if iteration < len(self._rounds) else []
        # Like the real optimizers: change the prompt, keep everything else.
        return [
            PromptCandidate(
                name=f"c{iteration}_{i}",
                config=dataclasses.replace(current_config, system_prompt=text),
                source="scripted",
            )
            for i, text in enumerate(prompts)
        ]


class _Events:
    def __init__(self) -> None:
        self.events: list[OptimizationEvent] = []

    async def on_event(self, event, data) -> None:
        self.events.append(event)


def _fitter(tmp_path, proposer: Any, **kwargs: Any) -> PromptFitter:
    fitter = PromptFitter(
        agent="sim",
        optimizer=proposer,
        baseline_system_prompt="You are a baseline assistant.",
        experiment_dir=str(tmp_path),
        auto_report=False,
        drain_seconds=0,
        **{"max_trials": 8, "patience": 3, **kwargs},
    )
    fitter._runner = AgentRunner(agent="sim", agent_callable=_agent)  # noqa: SLF001
    fitter._failure_clusterer.cluster = AsyncMock(return_value=[])  # noqa: SLF001
    return fitter


def _decisions(result) -> dict[str, tuple[str, str]]:
    return {
        t["system_prompt"]: (t["decision"], t.get("reason", ""))
        for t in result.trials
        if t.get("phase") in ("full_val", "skipped")
    }


class TestAcceptance:
    async def test_memorised_validation_answers_are_rejected(self, tmp_path) -> None:
        memo = "MEMO: " + " ".join(VAL)
        fitter = _fitter(tmp_path, _Proposer([[memo]]))
        result = await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))

        decision, reason = _decisions(result)[memo]
        assert decision == "rejected"
        # Validation 0 → 1 while held-out stays 0: vetoed by the safety net.
        assert "Generalization safety net" in reason
        assert not result.improved
        assert result.best_prompt == "You are a baseline assistant."

    async def test_a_generalizing_prompt_is_accepted(self, tmp_path) -> None:
        fitter = _fitter(tmp_path, _Proposer([["GENERAL assistant"]]))
        result = await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))

        assert _decisions(result)["GENERAL assistant"][0] == "accepted"
        assert result.improved
        assert result.best_score == pytest.approx(1.0)
        assert result.holdout_score == pytest.approx(1.0)

    async def test_the_generalizing_prompt_wins_over_the_memorising_one(self, tmp_path) -> None:
        memo = "MEMO: " + " ".join(VAL)
        fitter = _fitter(tmp_path, _Proposer([[memo, "GENERAL assistant"]]))
        result = await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))
        assert result.best_prompt == "GENERAL assistant"

    async def test_duplicates_of_the_incumbent_are_not_rescored(self, tmp_path) -> None:
        fitter = _fitter(tmp_path, _Proposer([["You are a baseline assistant."]]))
        result = await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))
        assert _decisions(result)["You are a baseline assistant."][0] == "duplicate"
        assert not any(t.get("phase") == "minibatch" for t in result.trials)


class TestProposerIsolation:
    async def test_the_optimizer_learns_from_train_only(self, tmp_path) -> None:
        proposer = _Proposer([["GENERAL assistant"], ["GENERAL assistant v2"]])
        fitter = _fitter(tmp_path, proposer)
        await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))

        seen = set(proposer.seen_eval_queries) | set(proposer.seen_sample_queries)
        assert seen, "the proposer must receive reflection data"
        assert seen <= set(TRAIN), f"validation/holdout leaked to the proposer: {seen}"

    async def test_legacy_reflection_on_validation_is_opt_in(self, tmp_path) -> None:
        proposer = _Proposer([["GENERAL assistant"]])
        fitter = _fitter(tmp_path, proposer, reflect_on="validation")
        await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))
        assert set(proposer.seen_eval_queries) <= set(VAL)


class TestBudgetAndEvents:
    async def test_max_trials_caps_distinct_candidates(self, tmp_path) -> None:
        many = [[f"cand {i}" for i in range(10)] for _ in range(5)]
        fitter = _fitter(tmp_path, _Proposer(many), max_trials=3)
        result = await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))
        assert sum(1 for t in result.trials if t.get("phase") == "minibatch") == 3

    async def test_round_end_is_emitted_for_rounds_without_candidates(self, tmp_path) -> None:
        events = _Events()
        fitter = _fitter(tmp_path, _Proposer([[], []]), max_trials=8, callbacks=[events])
        await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))
        starts = events.events.count(OptimizationEvent.ROUND_START)
        assert starts >= 1
        assert events.events.count(OptimizationEvent.ROUND_END) == starts

    async def test_every_scored_candidate_records_its_prompt_and_decision(self, tmp_path) -> None:
        fitter = _fitter(tmp_path, _Proposer([["no help", "GENERAL assistant"]]))
        result = await fitter.fit(_ds(TRAIN), _ds(VAL), _metric(), testset=_ds(HOLD))
        mini = [t for t in result.trials if t["phase"] == "minibatch"]
        assert {t["system_prompt"] for t in mini} == {"no help", "GENERAL assistant"}
        assert all(t["decision"] in {"promoted", "not_promoted"} and t["reason"] for t in mini)


class TestGeneralizationRules:
    def test_memorisation_does_not_transfer(self) -> None:
        check = check_generalization(
            fit_score=0.19,
            holdout_score=0.085,
            baseline_fit=0.085,
            baseline_holdout=0.085,
            min_transfer_ratio=0.25,
        )
        assert not check.ok and "Does not transfer" in check.reason

    def test_a_large_real_gain_is_accepted(self) -> None:
        check = check_generalization(
            fit_score=0.8,
            holdout_score=0.7,
            baseline_fit=0.1,
            baseline_holdout=0.2,
            min_transfer_ratio=0.25,
        )
        assert check.ok

    def test_held_out_regression_is_rejected(self) -> None:
        check = check_generalization(
            fit_score=0.9,
            holdout_score=0.4,
            baseline_fit=0.6,
            baseline_holdout=0.5,
            min_transfer_ratio=0.0,
        )
        assert not check.ok

    def test_legacy_absolute_gap(self) -> None:
        assert not check_generalization(fit_score=0.9, holdout_score=0.5, max_gap=0.15).ok
        assert check_generalization(fit_score=0.6, holdout_score=0.5, max_gap=0.15).ok

    def test_paired_confidence(self) -> None:
        assert paired_improvement_confidence([1, 1, 1], [0, 0, 0]) == 1.0
        assert paired_improvement_confidence([0.5, 0.6], [0.5, 0.6]) == 0.0
        noisy = paired_improvement_confidence([0.9, 0.1, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5])
        assert 0.0 < noisy < 0.8


class TestBridgeDataRoles:
    @staticmethod
    def _points(**splits: int) -> Dataset:
        points = []
        for split, n in splits.items():
            points += [
                DataPoint(query=f"{split}{i}", expected_answer="x", metadata={"split": split})
                for i in range(n)
            ]
        return Dataset(points=points)

    def test_test_split_never_reaches_the_fitter(self) -> None:
        from agentomatic.agents import PromptFitterBridge

        train, val, gate = PromptFitterBridge._split_three(  # noqa: SLF001
            self._points(train=4, validation=3, test=3)
        )
        used = {p.query for p in [*train, *val, *(gate or [])]}
        assert not any(q.startswith("test") for q in used)
        assert gate is None  # the fitter reserves its own slice of validation

    def test_holdout_split_becomes_the_gate(self) -> None:
        from agentomatic.agents import PromptFitterBridge

        _train, val, gate = PromptFitterBridge._split_three(  # noqa: SLF001
            self._points(train=4, validation=3, holdout=2, test=3)
        )
        assert gate is not None and {p.query for p in gate} == {"holdout0", "holdout1"}
        assert {p.query for p in val} == {"validation0", "validation1", "validation2"}

    def test_train_only_is_split_without_overlap(self) -> None:
        from agentomatic.agents import PromptFitterBridge

        train, val, _ = PromptFitterBridge._split_three(self._points(train=10))  # noqa: SLF001
        assert not ({p.query for p in train} & {p.query for p in val})
        assert len(val) >= 1

    def test_a_non_improving_result_changes_nothing(self) -> None:
        from agentomatic.agents import PromptFitterBridge
        from agentomatic.optimize import PromptFitResult

        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="from prompts.json"),
            baseline_config=PromptRuntimeConfig(system_prompt="from prompts.json"),
            best_score=0.5,
            baseline_score=0.5,
        )
        assert PromptFitterBridge._extract_config(None, result) == {}  # noqa: SLF001

    def test_few_shot_demos_are_served_as_they_were_scored(self) -> None:
        from agentomatic.agents import PromptFitterBridge
        from agentomatic.optimize import PromptFitResult

        best = PromptRuntimeConfig(
            system_prompt="Be precise.",
            few_shot_examples=[{"query": "q", "response": "a"}],
        )
        result = PromptFitResult(
            best_config=best,
            baseline_config=PromptRuntimeConfig(system_prompt="base"),
            best_score=0.9,
            baseline_score=0.1,
        )
        config = PromptFitterBridge._extract_config(None, result)  # noqa: SLF001
        assert config["system_prompt"].startswith("Be precise.")
        assert "## Few-shot examples" in config["system_prompt"]
