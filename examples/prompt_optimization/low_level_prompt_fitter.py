#!/usr/bin/env python3
"""Low-level prompt optimization with ``PromptFitter`` — every knob explicit.

Use this level when you want full control: your own train/validation/test
datasets, a hand-built composite metric, a declared search space, inner-loop
callbacks, and direct access to every trial of the result. The high-level
``agent.compile()/fit()`` path (``train.py``) drives this same engine.

How one ``fit`` works (the mechanics every other example relies on)::

    baseline prompt ──► score on VALIDATION (and on the HOLDOUT gate)
          │
          │   ┌──────────────── round × N ────────────────────────────┐
          │   ▼                                                        │
          │  reflect on TRAIN failures (clusters, judge feedback)      │
          │   ▼                                                        │
          │  optimizer proposes candidates (rewrite / GEPA / MIPRO /   │
          │  few-shot / params / APO); exact duplicates are skipped    │
          │   ▼                                                        │
          │  minibatch screen vs the incumbent ─► full VALIDATION score│
          │   ▼                                                        │
          │  accept only if ALL hold:                                  │
          │   • Δ validation ≥ min_absolute_improvement                │
          │   • paired bootstrap P(candidate > incumbent) ≥ min_confidence
          │   • HOLDOUT does not regress more than holdout_tolerance   │
          │   • validation−holdout gap stays ≤ max_generalization_gap  │
          │     (or grows by at most half the validation lift)         │
          │   • holdout lift ≥ min_transfer_ratio × validation lift    │
          │   ▼                                                        │
          └─ incumbent = best so far ─► stop after `patience` flat rounds
                                     ▼
            PromptFitResult (best config, every trial + decision + reason)

Data roles: TRAIN is what the optimizer learns from, VALIDATION selects,
the HOLDOUT gate (``testset=`` here; auto-carved from validation with
``holdout_fraction`` when omitted) only vetoes, and a separate TEST split is
never passed to the fitter — evaluate it yourself before and after (step 7).

Run::

    python examples/prompt_optimization/low_level_prompt_fitter.py --trials 8
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import SupportAgent
from loguru import logger

from common import (
    DATASET,
    build_parser,
    make_llm,
    must_include_score,
    settings_from,
    setup_logging,
)


async def run(argv: list[str] | None = None) -> int:
    """Build every component by hand and run one ``PromptFitter.fit``."""
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=8, help="Total candidate budget.")
    parser.add_argument("--optimizer", default="rewrite", help="See step 5 for the options.")
    args = parser.parse_args(argv)
    settings = settings_from(args)
    setup_logging(settings.log_level)

    # ------------------------------------------------------------------
    # 1. Datasets — explicit train / validation / holdout / test
    # ------------------------------------------------------------------
    # PromptFitter works on ``agentomatic.optimize.Dataset`` (a list of
    # DataPoint(query, expected_answer, context, metadata, tags)). Build one
    # from dicts, JSONL, or — as here — from an AgentDataset's splits:
    #     Dataset.from_list([{"query": "...", "expected_answer": "...",
    #                         "context": ["doc", ...], "tags": ["billing"],
    #                         "metadata": {"invoke": {"customer_plan": "team"},
    #                                      "must_include": ["30 days"]}}])
    #     Dataset.from_jsonl("eval.jsonl")
    #     trainset, valset = dataset.split(ratio=0.8)
    # What each field is for:
    #   context            documents → judges (groundedness) + the rewrite model;
    #                      sent to the agent as context.documents
    #   metadata.invoke    other agent inputs, sent to the agent as written
    #   metadata (rest)    labels → metrics, judges and the rewrite model
    #   tags               → judges, the rewrite model and the report
    from agentomatic.optimize import Dataset, load_data

    agent_data = load_data(DATASET)

    def to_points(examples: list) -> Dataset:
        # to_datapoint() keeps all of it: the inputs (``metadata.invoke`` —
        # customer_plan here), documents (``context``), labels such as
        # must_include (``metadata``), ``tags``, and the rubric (folded into
        # the expected reference the judge reads).
        return Dataset(points=[e.to_datapoint() for e in examples])

    trainset = to_points(agent_data.train)  # reflection: failures, feedback, few-shot
    valset = to_points(agent_data.validation)  # candidates are SELECTED on this
    gateset = to_points(agent_data.holdout)  # only VETOES candidates that do not transfer
    logger.info(
        "train={} validation={} holdout gate={} test (untouched)={}",
        len(trainset),
        len(valset),
        len(gateset),
        len(agent_data.test),
    )

    # ------------------------------------------------------------------
    # 2. Metric — the fit objective (async ``evaluate`` protocol)
    # ------------------------------------------------------------------
    # Built-in optimize metrics:
    #   ExactMatchMetric()                      fuzzy exact match vs expected
    #   ContainsMetric()                        expected keywords present
    #   DeterministicMetric(checks=[{"type": "contains"|"regex"|"max_length"|..., "value": ...}])
    #   CustomMetric(fn(query, response, expected, context) -> float, name=...)
    #   LLMJudgeMetric(criteria=..., model=...)          single-score judge
    #   LocalJudgeMetric(criteria=..., dimensions=[...], model=...)
    #                                           judge with per-dimension scores + rationale
    #   MultiJudgePanel(judges=[...], aggregation="average"|"median"|"min"|"max",
    #                   weights=[...])           several judges, one score
    #   GEvalMetric / DeepEvalMetric             DeepEval (pip install deepeval)
    #   LatencyMetric / CostMetric               efficiency terms
    # Combine them with CompositeMetric([WeightedMetric(name, metric, weight), ...]);
    # weights are normalised, each component is reported as a dimension, and a
    # component that FAILS (judge unreachable, unparsable) scores 0 — the
    # composite is marked failed when half its weight failed, never faked.
    # Class-agent metrics (score(example, prediction)) work too — they are
    # adapted automatically and receive the example's metadata.
    from agentomatic.agents import CallableMetric
    from agentomatic.optimize import CompositeMetric, LocalJudgeMetric, WeightedMetric

    judge = LocalJudgeMetric(
        name="judge",
        model=settings.judge_spec,
        criteria="Correct, grounded in the stated policy, complete and concise.",
        dimensions=["correctness", "completeness", "relevance"],
    )
    facts = CallableMetric("facts", must_include_score)  # sync class-agent metric
    metric = CompositeMetric(
        [
            WeightedMetric(name="judge", metric=judge, weight=0.6),
            WeightedMetric(name="facts", metric=facts, weight=0.4),
        ]
    )

    # ------------------------------------------------------------------
    # 3. Search space — what the optimizer may change
    # ------------------------------------------------------------------
    from agentomatic.optimize import PromptSearchSpace

    search_space = PromptSearchSpace(
        optimize_system_prompt=True,  # rewrite the system prompt
        optimize_user_template=False,  # rewrite the user-message template
        optimize_few_shot=False,  # select demonstrations from trainset
        optimize_model_params=False,  # search model_param_space
        # model_param_space={"temperature": [0.0, 0.2, 0.5], "top_p": [0.9, 1.0]},
        # rag_param_space={"top_k": [3, 5, 8]}, tool_param_space={...},
        # max_few_shot_examples=5,
        # few_shot_selection_strategy="diversity_weighted",
        # search_method="random",   # "grid" | "random" | "tpe" (param search)
    )

    # ------------------------------------------------------------------
    # 4. Inner-loop callbacks (agentomatic.optimize, not agentomatic.agents)
    # ------------------------------------------------------------------
    #   EarlyStopping(monitor="score", patience=3, min_delta=0.005, mode="max")
    #   ModelCheckpoint(save_dir=..., save_best_only=True, max_checkpoints=5)
    #   ScoreThreshold(threshold=0.9)          stop once good enough
    #   PlateauStopping(patience=2, factor=0.5) cool the rewrite temperature
    #   TemperatureScheduler(initial_temperature=0.7, decay_rate=0.9)
    #   NaNStopping(max_consecutive_nan=2)     abort on broken outputs
    #   ProgressLogger(show_prompt_diff=True)  log each trial's prompt diff
    from agentomatic.optimize import (
        ModelCheckpoint,
        OptimizeEarlyStopping,
        ProgressLogger,
        ScoreThreshold,
    )

    callbacks = [
        OptimizeEarlyStopping(monitor="score", patience=2, min_delta=0.01),
        ScoreThreshold(threshold=0.95),
        ModelCheckpoint(save_dir=str(settings.out_dir / "checkpoints")),
        ProgressLogger(show_prompt_diff=True),
    ]

    # ------------------------------------------------------------------
    # 5. The fitter — every constructor option
    # ------------------------------------------------------------------
    from agentomatic.optimize import PromptFitter

    agent = SupportAgent(llm=make_llm(settings))
    fitter = PromptFitter(
        agent=agent.agent_name,  # name (reports, prompts.json lookup)
        local_agent=agent,  # call the agent in-process (no server)
        # api_base="http://127.0.0.1:8001",   # …or optimize a RUNNING platform
        task_model=settings.task_spec,  # model the agent runs on
        rewrite_model=settings.rewrite_spec,  # model that proposes prompts
        llm_base_url=settings.base_url,  # OpenAI-compatible endpoint for both
        llm_api_key=settings.api_key,
        optimizer=args.optimizer,  # "rewrite" | "gepa_like" | "mipro_like" |
        #                            "few_shot" | "param_search" | "apo"
        search_space=search_space,
        max_trials=args.trials,  # total candidate budget
        patience=2,  # flat rounds before stopping
        # -- acceptance rules (anti-overfitting) --------------------------
        min_absolute_improvement=0.01,  # minimum validation lift
        min_confidence=0.8,  # paired-bootstrap P(better) on validation
        max_generalization_gap=0.15,  # validation − holdout gap allowed
        holdout_tolerance=0.02,  # max holdout regression vs incumbent
        min_transfer_ratio=0.25,  # holdout lift ≥ this × validation lift
        holdout_fraction=0.2,  # auto-holdout size when testset=None
        reflect_on="train",  # learn from TRAIN ("validation" leaks)
        reflection_size=8,  # train examples analysed per round
        # -- execution / outputs ------------------------------------------
        concurrency=4,  # parallel agent/judge calls
        callbacks=callbacks,
        auto_report=True,  # write the fitter's HTML trial report
        experiment_dir=str(settings.out_dir / ".fit"),
        baseline_system_prompt=agent.system_prompt,  # start from the agent's prompt
        # multipass=True, rewrite_passes=None   draft → critique → refine rewrites
        # local_judges=["omlx/other-model"]     add extra judges to the objective
        # dashboard=True                         live terminal dashboard
    )

    # The untouched TEST split, scored before optimization (step 7 compares).
    # agent.evaluate accepts either metric family.
    report_metrics = [judge, facts]
    before = agent.evaluate(agent_data.test, report_metrics)

    # ------------------------------------------------------------------
    # 6. Fit — testset= is the HOLDOUT gate (veto only), not the test split
    # ------------------------------------------------------------------
    result = await fitter.fit(trainset, valset, metric, testset=gateset)

    # ------------------------------------------------------------------
    # 7. Inspect the result, then measure the TEST split with the new prompt
    # ------------------------------------------------------------------
    print(result.summary())
    for trial in result.trials:
        # phase: "skipped" (duplicate) → "minibatch" (screen) → "full_val";
        # decision/reason say why each candidate was kept or dropped.
        logger.info(
            "  {:<12} {:<9} score={:<6} {:<12} {}",
            str(trial.get("name")),
            str(trial.get("phase")),
            f"{float(trial.get('score') or 0.0):.3f}",
            str(trial.get("decision") or ""),
            str(trial.get("reason") or "")[:80],
        )

    agent.compiled_config["system_prompt"] = result.best_prompt  # what fit() would keep
    after = agent.evaluate(agent_data.test, report_metrics)
    logger.info(
        "validation {:.3f} → {:.3f} | holdout gate {} → {}",
        result.baseline_score,
        result.best_score,
        result.baseline_holdout_score,
        result.holdout_score,
    )
    for name in after.scores:
        logger.info("test {:<6} {:.3f} → {:.3f}", name, before.scores[name], after.scores[name])

    from agentomatic.optimize import generate_fit_report

    report = generate_fit_report(
        result,
        output_path=settings.out_dir / "low_level_report.html",
        baseline_eval=before,
        final_eval=after,
        eval_dataset=agent_data.test,
        model_name=settings.model,
    )
    out = settings.out_dir / "low_level_result.json"
    out.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    logger.info("Report → {} | full result → {}", report, out)

    # ------------------------------------------------------------------
    # 8. Apply (guarded)
    # ------------------------------------------------------------------
    # result.apply(version="v2_fit", agent_dir="agents/support_agent")
    # writes prompts.json only when the candidate improved AND passed the
    # generalization gate; force=True overrides both (don't, without review).
    return 0


def main(argv: list[str] | None = None) -> int:
    """Synchronous entry point."""
    return asyncio.run(run(argv))


if __name__ == "__main__":
    raise SystemExit(main())
