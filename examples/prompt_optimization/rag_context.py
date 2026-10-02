#!/usr/bin/env python3
"""RAG prompt optimization where every row brings its own documents.

A row of ``datasets/policies.jsonl`` carries the whole case, not just a
question and an answer::

    {"id": "policy_05", "split": "train",
     "input": {"current_query": "How long are audit logs kept?",
               "context": {"documents": [{"source": "audit.md", "content":
                   "Audit logs are kept for 400 days on Enterprise and 90 days on Team."}]},
               "customer_plan": "team"},
     "expected_output": {"response": "On the Team plan, audit logs are kept for 90 days ..."},
     "metadata": {"topic": "security", "difficulty": "medium", "must_include": ["90 days"]},
     "tags": ["security", "compliance", "plan-specific"],
     "rubric": {"groundedness": "...", "plan": "...", "refusal": "..."}}

Every stage of the optimization receives it:

==============  ==============================================================
Stage           Sees
==============  ==============================================================
agent           ``input`` exactly as written — question, documents, plan
metrics         the whole row: ``example.input`` / ``.metadata`` / ``.tags`` /
                ``.rubric`` (here ``facts`` reads ``metadata.must_include`` and
                ``grounded`` reads ``input.context.documents``)
LLM judge       question, answer, expected answer + rubric, the documents, and
                the other inputs, metadata and tags
rewrite model   every failure, success and sample with its documents, inputs,
                metadata and tags — told to write instructions that USE such
                context, never to copy one row's facts into the prompt
augmenter       the seed's documents and inputs; variations inherit them
report          a ``context`` column: tags · documents · inputs · metadata
==============  ==============================================================

A RAG agent that retrieves for itself works the same way: whatever it lists
under ``retrieval_context`` / ``citations`` / ``sources`` is what judges and
the rewrite model see when a row brings no documents.

This script prints what each stage sees for one row, fits the agent's prompt
on these rows, then reports the untouched test split before → after, overall
and per tag.

Run it against any OpenAI-compatible server (oMLX by default)::

    omlx serve --model Qwen3.5-9B-MLX-4bit          # http://127.0.0.1:8000/v1
    python examples/prompt_optimization/rag_context.py --epochs 1 --trials 4

    # grow the train split first — variations keep their seed's documents
    python examples/prompt_optimization/rag_context.py --augment --n-examples 30

Outputs (``--out-dir``): ``rag_report.html`` (+ ``rag_report.json``) and
``rag_summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import SupportAgent
from loguru import logger

from common import (
    POLICIES_DATASET,
    build_parser,
    grounded_score,
    make_llm,
    must_include_score,
    scores_by_tag,
    settings_from,
    setup_logging,
)


def show_what_each_stage_sees(example: Any) -> None:
    """Print one row the way the agent, the judge and the rewrite model get it."""
    from agentomatic.optimize.briefing import format_dataset_samples
    from agentomatic.optimize.dataset import example_context

    point = example.to_datapoint()  # what PromptFitter works on
    labels = example_context(point.metadata, None, point.tags)
    labels.get("inputs", {}).pop("context", None)  # the judge lists documents separately

    print(f"── What each stage sees — row {example.id}")
    print("agent (input_to_state receives exactly this):")
    print("  " + json.dumps(example.input, ensure_ascii=False))
    print("LLM judge — context documents:")
    for doc in point.context:
        print(f"  - {doc}")
    print("LLM judge — other inputs, metadata and tags:")
    print("  " + json.dumps(labels, ensure_ascii=False))
    print("rewrite model (dataset sample in its briefing):")
    print(format_dataset_samples([point]))
    print()


def main(argv: list[str] | None = None) -> int:
    """Show the context flow, fit on RAG rows, and report per tag."""
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=1, help="Outer fit() epochs.")
    parser.add_argument("--trials", type=int, default=4, help="Candidate budget per epoch.")
    parser.add_argument(
        "--optimizer",
        default="rewrite",
        choices=["rewrite", "gepa_like", "mipro_like", "few_shot", "apo"],
        help="Candidate-generation strategy (see train.py, step 6).",
    )
    parser.add_argument("--augment", action="store_true", help="LLM-augment the train split.")
    parser.add_argument("--n-examples", type=int, default=30, help="Target size after augment.")
    args = parser.parse_args(argv)

    settings = settings_from(args)
    setup_logging(settings.log_level)

    # ------------------------------------------------------------------
    # 1. Agent — answers from the documents each request brings
    # ------------------------------------------------------------------
    # SupportAgent.input_to_state reads ``context.documents`` and
    # ``customer_plan``; ``retrieve`` uses the supplied documents instead of
    # its own knowledge base whenever there are any, and the output lists
    # them under ``sources``.
    agent = SupportAgent(llm=make_llm(settings))

    # ------------------------------------------------------------------
    # 2. Data — rows with documents, plan, metadata, tags and a rubric
    # ------------------------------------------------------------------
    from agentomatic.optimize import load_data, prepare_dataset

    data = load_data(POLICIES_DATASET)
    show_what_each_stage_sees(data.train[4])  # a plan-specific row

    if args.augment:
        # The augmenter is shown each seed's documents and inputs ("Seed
        # context") and is told to ask only what they answer; every variation
        # inherits a copy of its seed's inputs and tags (+ "augmented").
        data, _ = prepare_dataset(
            data,
            augment=True,
            n_examples=args.n_examples,
            strategies=["paraphrase", "perturbation"],
            model=settings.task_spec,
            llm_base_url=settings.base_url,
            llm_api_key=settings.api_key,
            persist=True,
            persist_path=settings.out_dir / "policies.augmented.jsonl",
        )
    logger.info(
        "data: train={} validation={} holdout={} test={}",
        len(data.train),
        len(data.validation),
        len(data.holdout),
        len(data.test),
    )

    # ------------------------------------------------------------------
    # 3. Metrics that account for the context
    # ------------------------------------------------------------------
    # * judge    — LLM judge; besides the answer and the expected answer it
    #              receives the row's documents, rubric, plan, metadata, tags
    # * facts    — metadata.must_include present in the answer
    # * grounded — share of the answer's words found in the row's documents
    from agentomatic.agents import CallableMetric, WeightedMetric
    from agentomatic.optimize import LocalJudgeMetric

    judge = LocalJudgeMetric(
        name="judge",
        model=settings.judge_spec,
        criteria=(
            "Is the answer supported by the context documents, applied to the "
            "customer's plan when one is given, and does it say so when the "
            "documents do not answer? Score 0-1."
        ),
        dimensions=["groundedness", "relevance", "completeness"],
        temperature=0.0,
    )
    facts = CallableMetric("facts", must_include_score)
    grounded = CallableMetric("grounded", grounded_score)
    quality = WeightedMetric(
        [("judge", judge, 0.5), ("facts", facts, 0.3), ("grounded", grounded, 0.2)],
        name="quality",
    )

    # ------------------------------------------------------------------
    # 4. Optimizer + compile
    # ------------------------------------------------------------------
    from agentomatic.agents import MetricLoss, PromptFitterBridge
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
        search_space=PromptSearchSpace(optimize_system_prompt=True, optimize_few_shot=False),
        min_confidence=0.8,
        concurrency=4,
        experiment_dir=str(settings.out_dir / ".fit_rag"),
    )
    agent.compile(
        data,
        metrics=[judge, facts, grounded, quality],
        optimizer=optimizer,
        loss=MetricLoss(quality),
    )

    # ------------------------------------------------------------------
    # 5. Test before → fit → test after (test is never seen by fit)
    # ------------------------------------------------------------------
    before = agent.evaluate(data.test)
    baseline_prompt = agent.resolve_system_prompt(default=agent.system_prompt)
    history = agent.fit(data, epochs=args.epochs, verbose=1)
    after = agent.evaluate(data.test)
    best_prompt = agent.resolve_system_prompt(default=agent.system_prompt)

    # ------------------------------------------------------------------
    # 6. Account for the tags: where did the new prompt help or hurt?
    # ------------------------------------------------------------------
    tags_before = scores_by_tag(before, data.test)
    tags_after = scores_by_tag(after, data.test)
    print("── Test quality by tag (before → after)")
    for tag, scores in tags_after.items():
        print(
            f"  {tag:<15} {tags_before.get(tag, {}).get('quality', float('nan')):.2f} → "
            f"{scores.get('quality', float('nan')):.2f}"
        )

    from agentomatic.optimize import generate_fit_report

    # The per-example table of the report carries a ``context`` column.
    report_path = generate_fit_report(
        history,
        output_path=settings.out_dir / "rag_report.html",
        baseline_eval=before,
        final_eval=after,
        eval_dataset=data.test,
        dataset_stats=data.metadata.get("augment_stats"),
        model_name=settings.model,
        optimizer_name=args.optimizer,
    )
    summary = {
        "baseline_prompt": baseline_prompt,
        "best_prompt": best_prompt,
        "prompt_changed": best_prompt != baseline_prompt,
        "fit_status": getattr(agent, "_last_optimize_status", None),
        "test_scores_before": before.scores,
        "test_scores_after": after.scores,
        "test_by_tag_before": tags_before,
        "test_by_tag_after": tags_after,
        "dataset_sizes": {
            "train": len(data.train),
            "validation": len(data.validation),
            "holdout": len(data.holdout),
            "test": len(data.test),
        },
        "report": str(report_path),
    }
    summary_path = settings.out_dir / "rag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for name in after.scores:
        logger.info(
            "test {:<9} {:.3f} → {:.3f}", name, before.scores.get(name, 0.0), after.scores[name]
        )
    logger.info("report: {} · summary: {}", report_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
