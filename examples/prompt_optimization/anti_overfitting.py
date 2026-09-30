#!/usr/bin/env python3
"""Anti-overfitting in prompt optimization — watch a memorizing prompt get rejected.

An optimizer that is rewarded only for its validation score will happily
"improve" a prompt by pasting the validation answers into it. That prompt
scores perfectly on validation and no better anywhere else. Agentomatic's
fitter guards against it with separate data roles and acceptance rules:

* TRAIN      — what the optimizer learns from (failures, judge feedback);
* VALIDATION — what candidates are selected on; a win must be *significant*
  (paired bootstrap over the validation examples, ``min_confidence``);
* HOLDOUT    — a veto-only gate: the candidate must not regress there
  (``holdout_tolerance``), must not open a validation/holdout gap
  (``max_generalization_gap``), and must carry a share of its validation
  gain over (``min_transfer_ratio``) — the rule memorisation fails;
* TEST       — never used by the fitter; measure it yourself afterwards.

Part 1 applies the rules to hand-made numbers (no model needed).
Part 2 runs ``PromptFitter`` with two scripted candidates — one that memorizes
the validation answers, one that states a general instruction — and shows
which one survives, and what each would have scored on the test split.

Run::

    python examples/prompt_optimization/anti_overfitting.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import SupportAgent

from common import (
    DATASET,
    build_parser,
    make_llm,
    must_include_score,
    settings_from,
    setup_logging,
)

GENERAL_PROMPT = (
    "You are a precise Acme Cloud support assistant. Quote the relevant policy snippet "
    "verbatim, then answer in one short sentence. Never invent policy."
)


def part1_rules() -> None:
    """The acceptance rules on hand-made numbers."""
    from agentomatic.optimize import check_generalization, paired_improvement_confidence

    print("── Part 1: the acceptance rules ──")
    print(
        "incumbent: validation 0.50, holdout 0.50; rules: gap 0.15, tolerance 0.02, transfer 0.25"
    )
    scenarios = {
        # name: (candidate validation, candidate holdout)
        "general improvement": (0.80, 0.75),
        "memorised validation": (1.00, 0.50),
        "gain does not transfer": (0.62, 0.52),
        "holdout regression": (0.52, 0.45),
        "gap blows up": (0.95, 0.60),
    }
    for name, (val, hold) in scenarios.items():
        check = check_generalization(
            fit_score=val,
            holdout_score=hold,
            baseline_fit=0.50,
            baseline_holdout=0.50,
            max_gap=0.15,
            holdout_tolerance=0.02,
            min_transfer_ratio=0.25,
        )
        verdict = "ACCEPT" if check.ok else "REJECT"
        print(f"  {name:<22} val {val:.2f} holdout {hold:.2f} → {verdict}\n      {check.reason}")

    # Significance: per-example validation scores of incumbent vs candidate.
    incumbent = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    lucky = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]  # one example flipped — could be judge noise
    solid = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # every failure fixed
    for name, cand in (("one example flipped", lucky), ("every failure fixed", solid)):
        p = paired_improvement_confidence(cand, incumbent)
        verdict = "ACCEPT" if p >= 0.8 else "REJECT (not significant)"
        print(f"  {name:<22} P(candidate > incumbent) = {p:.2f} → {verdict}")


class ScriptedCandidates:
    """A custom optimizer: any object with ``name`` and async ``propose(...)``.

    Real optimizers ("rewrite", "gepa_like", …) generate candidates from the
    train failures; this one proposes a fixed prompt per round so the outcome
    is easy to follow.
    """

    name = "scripted"

    def __init__(self, rounds: list[tuple[str, str]]) -> None:
        self.rounds = rounds

    async def propose(
        self,
        current_config: Any,
        eval_results: list[dict[str, Any]],
        dataset_sample: list[dict[str, Any]],
        search_space: Any,
        iteration: int = 0,
        context: Any = None,
    ) -> list[Any]:
        from agentomatic.optimize import PromptCandidate

        if iteration >= len(self.rounds):
            return []
        name, prompt = self.rounds[iteration]
        # Change the prompt, keep every other setting of the incumbent.
        config = dataclasses.replace(current_config, system_prompt=prompt)
        return [PromptCandidate(name=name, config=config, source=self.name)]


async def part2_fit(settings: Any) -> int:
    """A real fit where the memorizing candidate is vetoed."""
    from agentomatic.agents import CallableMetric
    from agentomatic.optimize import Dataset, PromptFitter, load_data

    print("\n── Part 2: a fit with a memorizing and a general candidate ──")
    data = load_data(DATASET)

    def points(examples: list[Any]) -> Dataset:
        return Dataset(points=[e.to_datapoint() for e in examples])

    agent = SupportAgent(llm=make_llm(settings))
    # The overfit prompt: the VALIDATION questions with their answers.
    memorized = "\n".join(
        f"Q: {e.input['current_query']}\nA: {e.expected_output['response']}"
        for e in data.validation
    )
    memorizing_prompt = (
        f"{agent.system_prompt}\nAnswer these questions exactly as follows:\n{memorized}"
    )

    facts = CallableMetric("facts", must_include_score)  # deterministic → reproducible
    fitter = PromptFitter(
        agent=agent.agent_name,
        local_agent=agent,
        optimizer=ScriptedCandidates(
            [("memorize_validation", memorizing_prompt), ("general_instruction", GENERAL_PROMPT)]
        ),
        task_model=settings.task_spec,
        rewrite_model=settings.rewrite_spec,
        llm_base_url=settings.base_url,
        llm_api_key=settings.api_key,
        baseline_system_prompt=agent.system_prompt,
        max_trials=8,  # 2 rounds (one scripted candidate each)
        patience=3,
        min_absolute_improvement=0.01,
        min_confidence=0.8,
        max_generalization_gap=0.15,
        holdout_tolerance=0.02,
        min_transfer_ratio=0.25,
        auto_report=False,
        drain_seconds=0,
        experiment_dir=str(settings.out_dir / ".fit_anti_overfitting"),
    )
    result = await fitter.fit(
        points(data.train), points(data.validation), facts, testset=points(data.holdout)
    )

    print("  candidate              validation  holdout  decision   reason")
    for trial in result.trials:
        if trial.get("phase") != "full_val":
            continue
        print(
            f"  {trial['name']:<22} {trial['score']:.2f}        "
            f"{trial.get('holdout_score') or 0.0:.2f}     {trial['decision']:<10} "
            f"{str(trial.get('reason'))[:70]}"
        )

    # What each prompt would have scored on the untouched TEST split.
    async def test_score(prompt: str) -> float:
        agent.compiled_config["system_prompt"] = prompt
        report = await asyncio.to_thread(agent.evaluate, data.test, [facts])
        return float(report.scores["facts"])

    print("  test split (never seen by the fitter):")
    for name, prompt in (
        ("baseline", agent.system_prompt),
        ("memorize_validation", memorizing_prompt),
        ("general_instruction", GENERAL_PROMPT),
    ):
        print(f"    {name:<22} facts = {await test_score(prompt):.2f}")
    kept = result.best_prompt == GENERAL_PROMPT
    print(f"  kept prompt: {'general_instruction' if kept else 'something else'}")
    return 0 if kept else 1


def main(argv: list[str] | None = None) -> int:
    """Run both parts."""
    args = build_parser(__doc__.splitlines()[0]).parse_args(argv)
    settings = settings_from(args)
    setup_logging(settings.log_level)
    part1_rules()
    return asyncio.run(part2_fit(settings))


if __name__ == "__main__":
    raise SystemExit(main())
