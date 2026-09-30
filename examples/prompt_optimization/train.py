#!/usr/bin/env python3
"""Prompt optimization, Keras-style: data → metrics → compile → fit → evaluate → report.

The high-level class-agent workflow, step by step. It optimizes the system
prompt of the example :class:`~agent.SupportAgent` with an LLM-as-judge plus
a deterministic metric, rejects candidates that do not carry over to a
held-out gate, measures the untouched test split before and after, and writes
an HTML + JSON report with the prompt before/after, every candidate and why
it was accepted or rejected, and per-example answers and scores.

Run it against any OpenAI-compatible server (oMLX by default)::

    omlx serve --model Qwen3.5-9B-MLX-4bit          # http://127.0.0.1:8000/v1
    python examples/prompt_optimization/train.py --epochs 2 --trials 6

    # augment the training split with synthetic examples first
    python examples/prompt_optimization/train.py --augment --n-examples 40

    # every flag
    python examples/prompt_optimization/train.py --help

Outputs (``--out-dir``, default ``examples/prompt_optimization/out``):
``train_report.html`` (+ ``train_report.json``), ``train_summary.json``, and
the fitter's own per-epoch trial reports under ``.fit/``.
"""

from __future__ import annotations

import difflib
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


def main(argv: list[str] | None = None) -> int:
    """Run the full optimize → evaluate → report loop."""
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=2, help="Outer fit() epochs.")
    parser.add_argument("--trials", type=int, default=6, help="Candidate budget per epoch.")
    parser.add_argument(
        "--optimizer",
        default="gepa_like",
        choices=["rewrite", "gepa_like", "mipro_like", "few_shot", "param_search", "apo"],
        help="Candidate-generation strategy (see the table in the step 6 comment).",
    )
    parser.add_argument("--augment", action="store_true", help="LLM-augment the train split.")
    parser.add_argument("--n-examples", type=int, default=40, help="Target size after augment.")
    parser.add_argument(
        "--apply", action="store_true", help="Keep the optimized prompt in prompts.json."
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Settings & logging
    # ------------------------------------------------------------------
    # One place for the model server URL / model names (flags or env vars,
    # see common.py). In a scaffolded project you would instead load the
    # project stack:
    #     stacks = StackManager(ROOT / "stacks"); stacks.load("local")
    #     apply_stack_defaults(stacks)
    #     llm = get_llm_for_agent("my_agent", role="default", stack_manager=stacks)
    settings = settings_from(args)
    setup_logging(settings.log_level)
    logger.info("Step 1/10 — settings: server={} model={}", settings.base_url, settings.model)

    # ------------------------------------------------------------------
    # 2. The agent under optimization
    # ------------------------------------------------------------------
    # Any BaseGraphAgent works as long as its nodes read the prompt through
    # ``self.resolve_system_prompt(...)`` — that is how candidates are injected.
    agent = SupportAgent(llm=make_llm(settings))
    logger.info(
        "Step 2/10 — agent '{}' baseline prompt: {!r}", agent.agent_name, agent.system_prompt
    )

    # ------------------------------------------------------------------
    # 3. Data: load splits, optionally augment the TRAIN split, persist
    # ------------------------------------------------------------------
    # JSONL rows: {"id", "split", "input": {...}, "expected_output": {...},
    #              "metadata": {...}}   (see datasets/support.jsonl)
    # The four data roles — the anti-overfitting contract:
    # * train      → what the optimizer LEARNS from (failure analysis, judge
    #                feedback, few-shot demos). Augmentation only grows this.
    # * validation → what candidate prompts are SCORED and SELECTED on; a
    #                candidate must beat the incumbent with bootstrap confidence.
    # * holdout    → VETO-only gate: a candidate that wins on validation but
    #                regresses here (or its gap grows) is rejected. Optional —
    #                without it, a slice of validation is reserved automatically.
    # * test       → NEVER used by fit(); evaluated before and after (step 10).
    from agentomatic.optimize import load_data, prepare_dataset

    dataset = load_data(DATASET)
    # prepare_dataset options:
    #   augment=True/False        LLM augmentation of the TRAIN split only;
    #                             validation/holdout/test stay untouched and
    #                             near-copies of their questions are dropped
    #   n_examples=int            target TOTAL size after augmentation
    #   strategies=[...]          label-preserving (the seed's answer is kept):
    #                               "paraphrase" | "perturbation" | "add_noise" |
    #                               "simplify" | "formality_shift"
    #                             new questions (the LLM writes the answer):
    #                               "expansion" | "complicate" | "adversarial" |
    #                               "edge_case"
    #   model="provider/model"    augmentation LLM spec
    #   llm_base_url / llm_api_key  OpenAI-compatible endpoint for that model
    #   per_call=4                variations requested per LLM call (small
    #                             batches survive small local models)
    #   max_tokens=4096           reply budget per call
    #   strict=False              True → raise if fewer rows than asked were added
    #   persist=True, persist_path=...  write the result to JSONL for review
    # dataset.metadata["augment_stats"] records what happened (calls, parsed,
    # duplicates, near-duplicates, added per strategy) — shown in the report.
    dataset, written = prepare_dataset(
        dataset,
        augment=args.augment,
        n_examples=args.n_examples,
        strategies=["paraphrase", "expansion"],
        model=settings.task_spec,
        llm_base_url=settings.base_url,
        llm_api_key=settings.api_key,
        persist=args.augment,
        persist_path=settings.out_dir / "support.augmented.jsonl",
    )
    logger.info(
        "Step 3/10 — data: train={} validation={} holdout={} test={}{}",
        len(dataset.train),
        len(dataset.validation),
        len(dataset.holdout),
        len(dataset.test),
        f" (augmented → {written})" if written else "",
    )

    # ------------------------------------------------------------------
    # 4. Metrics — what "good" means
    # ------------------------------------------------------------------
    # Two families, and every API accepts either (they are adapted for you):
    #   agentomatic.agents  — sync  ``score(example, prediction) -> float``
    #       ExactKeyMatchMetric(keys), ContainsTermsMetric(terms),
    #       ResponseSimilarityMetric(), CallableMetric(name, fn),
    #       agents.WeightedMetric([(name, metric, weight), ...])
    #   agentomatic.optimize — async ``evaluate(query, response, expected, context)``
    #       ExactMatchMetric, ContainsMetric, LLMJudgeMetric, LocalJudgeMetric,
    #       MultiJudgePanel, GEvalMetric, CompositeMetric, DeterministicMetric,
    #       LatencyMetric, CostMetric, CustomMetric(fn)
    from agentomatic.agents import CallableMetric, WeightedMetric
    from agentomatic.optimize import LocalJudgeMetric

    # Deterministic: did the answer state the example's key facts?
    facts = CallableMetric("facts", must_include_score)
    # LLM-as-judge (0..1 overall score + per-dimension scores + rationale).
    # LocalJudgeMetric options: name, model, criteria, dimensions, weight,
    # temperature (keep 0.0 — reproducible scores across epochs).
    judge = LocalJudgeMetric(
        name="judge",
        model=settings.judge_spec,
        criteria=(
            "Is the answer correct and grounded in the stated policy, complete, "
            "and concise? Score 0-1."
        ),
        dimensions=["correctness", "completeness", "relevance"],
        temperature=0.0,
    )
    # A panel of judges (mixture of experts) is a drop-in replacement:
    #     from agentomatic.optimize import MultiJudgePanel
    #     judge = MultiJudgePanel(judges=[LocalJudgeMetric(name="a", model=...),
    #                                     LocalJudgeMetric(name="b", model=...)],
    #                             aggregation="average")   # or "majority"
    quality = WeightedMetric([("judge", judge, 0.6), ("facts", facts, 0.4)], name="quality")
    logger.info("Step 4/10 — metrics: judge (LLM) + facts (deterministic) → quality")

    # ------------------------------------------------------------------
    # 5. Loss — what fit() minimises epoch over epoch
    # ------------------------------------------------------------------
    # MetricLoss(metric) = 1 - score. Alternatives: CallableLoss(fn, name=...)
    # or any object with ``compute(example, prediction) -> float``.
    from agentomatic.agents import MetricLoss

    loss = MetricLoss(quality)

    # ------------------------------------------------------------------
    # 6. Optimizer — how candidate prompts are proposed and accepted
    # ------------------------------------------------------------------
    # optimizer=               "rewrite"      rewrite from failure analysis
    #                          "gepa_like"    targeted edits from judge feedback
    #                          "mipro_like"   prompt × few-shot combinations
    #                          "few_shot"     pick demonstrations from train
    #                          "param_search" grid/random/TPE over model params
    #                          "apo"          trace-aware critique + edit
    # metric=                  the objective candidates are SELECTED on — any
    #                          metric of either family (here the ``quality`` blend)
    # search_space=            what may change (PromptSearchSpace, below)
    # max_trials=              candidate budget per epoch
    # patience=                rounds without improvement before stopping
    # concurrency=             parallel agent/judge calls (1 = sequential)
    # auto_report=, experiment_dir=  the fitter's own per-epoch trial report
    #
    # Acceptance rules — how overfitting is prevented (all on by default):
    # min_absolute_improvement minimum validation lift over the incumbent
    # min_confidence=0.8       paired-bootstrap P(candidate > incumbent) on the
    #                          validation examples; below it the "win" is noise
    # max_generalization_gap   validation − holdout gap a candidate may reach
    # holdout_tolerance=0.02   max holdout regression vs the incumbent
    # min_transfer_ratio=0.25  holdout lift must be ≥ this × validation lift
    # holdout_fraction=0.2     share of validation reserved as the gate when
    #                          the dataset has no ``holdout`` split
    # reflect_on="train"       where failures/feedback/few-shot come from
    #                          ("validation" = the old, leakier behaviour)
    # reflection_size=8        train examples analysed per round
    from agentomatic.agents import PromptFitterBridge
    from agentomatic.optimize import PromptSearchSpace

    optimizer = PromptFitterBridge(
        agent_name=agent.agent_name,
        task_model=settings.task_spec,
        rewrite_model=settings.rewrite_spec,
        llm_base_url=settings.base_url,
        llm_api_key=settings.api_key,
        optimizer=args.optimizer,
        metric=quality,
        max_trials=args.trials,
        # PromptSearchSpace options: optimize_system_prompt, optimize_user_template,
        # optimize_few_shot (+ max_few_shot_examples, few_shot_selection_strategy),
        # optimize_model_params (+ model_param_space={"temperature": [0.0, 0.2]}),
        # optimize_rag_params (+ rag_param_space), optimize_tool_params
        # (+ tool_param_space), optimize_model_choice (+ model_choices),
        # search_method="grid"|"random"|"tpe".
        search_space=PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_user_template=False,
            optimize_model_params=False,
            optimize_few_shot=False,
        ),
        patience=2,
        min_absolute_improvement=0.01,
        min_confidence=0.8,
        max_generalization_gap=0.15,
        holdout_tolerance=0.02,
        min_transfer_ratio=0.25,
        concurrency=4,
        auto_report=True,
        experiment_dir=str(settings.out_dir / ".fit"),
    )
    logger.info("Step 6/10 — optimizer: {} (trials={})", args.optimizer, args.trials)

    # ------------------------------------------------------------------
    # 7. Compile
    # ------------------------------------------------------------------
    # metrics= are REPORTED every epoch (train and val_*); loss= drives
    # EarlyStopping; the optimizer's metric= SELECTS candidates.
    agent.compile(dataset, metrics=[judge, facts, quality], optimizer=optimizer, loss=loss)

    # Baseline on the untouched test split, for the before/after comparison.
    baseline_report = agent.evaluate(dataset.test)
    baseline_prompt = agent.resolve_system_prompt(default=agent.system_prompt)

    # ------------------------------------------------------------------
    # 8. Callbacks
    # ------------------------------------------------------------------
    # agentomatic.agents callbacks (fit()):
    #   EarlyStopping(monitor="val_loss", mode="min", patience=1, min_delta=0.0)
    #   EpochDiffCallback(epochs=N) — prints loss + a unified prompt diff per epoch
    # (agentomatic.optimize has its own EarlyStopping/ModelCheckpoint/… for
    #  PromptFitter's inner trial loop — see low_level_prompt_fitter.py.)
    from agentomatic.agents import EarlyStopping, EpochDiffCallback

    epoch_diff = EpochDiffCallback(epochs=args.epochs)
    callbacks = [epoch_diff, EarlyStopping(monitor="val_loss", mode="min", patience=1)]

    # ------------------------------------------------------------------
    # 9. Fit
    # ------------------------------------------------------------------
    # fit() options: epochs, verbose (0/1/2), callbacks, validation_data
    # (defaults to the validation split), and per-call overrides:
    # search_space, optimize_mode, optimize_prompt/params/few_shot,
    # model_param_space, max_trials, plus any PromptFitter kwarg.
    logger.info("Step 9/10 — fit: {} epoch(s)", args.epochs)
    history = agent.fit(dataset, epochs=args.epochs, verbose=1, callbacks=callbacks)
    logger.info("fit status: {}", getattr(agent, "_last_optimize_status", "n/a"))
    logger.info("loss per epoch: {}", history.history.get("loss"))
    logger.info("val_loss per epoch: {}", history.history.get("val_loss"))

    # ------------------------------------------------------------------
    # 10. Evaluate on the untouched test split + reports
    # ------------------------------------------------------------------
    final_report = agent.evaluate(dataset.test)
    best_prompt = agent.resolve_system_prompt(default=agent.system_prompt)

    from agentomatic.optimize import generate_fit_report, merge_fit_results

    # history.fit_results holds one PromptFitResult per epoch; merged, they
    # span the ORIGINAL prompt → the FINAL one (what apply() and the report use).
    fit_result = merge_fit_results(history.fit_results) if history.fit_results else None
    if fit_result is None:
        logger.warning("No optimization result — see the fit status above.")
        return 1

    # generate_fit_report options:
    #   result               History from fit() (all epochs + loss curves),
    #                        a PromptFitResult, a list of them, or the agent
    #   baseline_eval /      EvaluationReports before/after fit on the SAME
    #   final_eval           examples → per-example answers & scores table
    #   eval_dataset         those examples (questions + expected answers)
    #   dataset_stats        augmentation stats (dataset.metadata["augment_stats"])
    #   keras_history        taken from History automatically
    #   run_config / stack_name / model_name / optimizer_name  header details
    # Writes <name>.html plus a <name>.json sidecar with the same data.
    report_path = generate_fit_report(
        history,
        output_path=settings.out_dir / "train_report.html",
        baseline_eval=baseline_report,
        final_eval=final_report,
        eval_dataset=dataset.test,
        dataset_stats=dataset.metadata.get("augment_stats"),
        model_name=settings.model,
        optimizer_name=args.optimizer,
    )
    summary = {
        "baseline_prompt": baseline_prompt,
        "best_prompt": best_prompt,
        "prompt_changed": best_prompt != baseline_prompt,
        "validation": {"before": fit_result.baseline_score, "after": fit_result.best_score},
        "test_scores_before": baseline_report.scores,
        "test_scores_after": final_report.scores,
        "history": history.history,
        "fit": fit_result.to_dict(),
        "report": str(report_path),
    }
    summary_path = settings.out_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logger.info("Step 10/10 — results")
    logger.info(
        "  validation (selection): {:.3f} → {:.3f}",
        fit_result.baseline_score,
        fit_result.best_score,
    )
    for name in final_report.scores:
        logger.info(
            "  test {:<8} {:.3f} → {:.3f}",
            name,
            baseline_report.scores.get(name, float("nan")),
            final_report.scores[name],
        )
    if best_prompt != baseline_prompt:
        diff = difflib.unified_diff(
            baseline_prompt.splitlines(),
            best_prompt.splitlines(),
            "initial",
            "final",
            lineterm="",
        )
        logger.info("  prompt diff:\n{}", "\n".join(diff))
    else:
        logger.info("  prompt unchanged — no candidate passed the acceptance rules")
    logger.info("  report: {}", report_path)
    logger.info("  summary: {}", summary_path)

    if args.apply:
        # Refuses a non-improving or overfitting result (see PromptFitResult.apply()).
        fit_result.apply(version="v2_fit", agent_dir=settings.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
