#!/usr/bin/env python3
"""Data augmentation for prompt optimization — grow TRAIN without leaking eval data.

A handful of hand-written examples is rarely enough for the optimizer to see
every way users phrase a question. ``prepare_dataset(augment=True)`` asks an
LLM for variations of your **train** rows, one seed and one strategy per call,
and keeps only rows that are safe to learn from:

* only the train split grows — validation / holdout / test stay untouched;
* generated questions that duplicate, or nearly duplicate (≥ 90 % similar),
  any validation / holdout / test question are dropped (a paraphrase of a
  test question in train inflates every score measured on it);
* label-preserving strategies keep the seed's answer and metadata, so
  metadata-driven metrics (``must_include``) still score them correctly;
* ``dataset.metadata["augment_stats"]`` records what happened, and a warning
  (or ``strict=True`` → error) fires when fewer rows than asked were added.

Run::

    python examples/prompt_optimization/data_augmentation.py --n-examples 40
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

from common import DATASET, build_parser, settings_from, setup_logging


def main(argv: list[str] | None = None) -> int:
    """Augment the example dataset and show what was added."""
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--n-examples", type=int, default=40, help="Target TOTAL size.")
    parser.add_argument(
        "--strategies",
        default="paraphrase,formality_shift,expansion",
        help="Comma-separated strategies (see the table in step 2).",
    )
    args = parser.parse_args(argv)
    settings = settings_from(args)
    setup_logging(settings.log_level)

    # ------------------------------------------------------------------
    # 1. Load the seed data
    # ------------------------------------------------------------------
    from agentomatic.optimize import load_data

    dataset = load_data(DATASET)
    logger.info(
        "seed: train={} validation={} holdout={} test={}",
        len(dataset.train),
        len(dataset.validation),
        len(dataset.holdout),
        len(dataset.test),
    )

    # ------------------------------------------------------------------
    # 2. Augment
    # ------------------------------------------------------------------
    # Strategies — label-preserving (the seed's expected answer is kept):
    #   paraphrase       same question, different words
    #   perturbation     small wording changes, typos, word order
    #   add_noise        irrelevant detail / filler around the question
    #   simplify         shorter, plainer phrasing
    #   formality_shift  more / less formal register
    # New questions (the LLM writes the expected answer — review them!):
    #   expansion        related questions on the same topic
    #   complicate       multi-part or more demanding versions
    #   adversarial      tricky, misleading phrasings
    #   edge_case        boundary conditions and unusual cases
    #
    # prepare_dataset options:
    #   augment=True            enable LLM augmentation (False = load/persist only)
    #   n_examples=int          target TOTAL size (default 3 × seed size)
    #   strategies=[...]        the list above (default ["paraphrase"])
    #   model="provider/model"  augmentation LLM ("omlx/…", "openai/…", "ollama/…")
    #   llm_base_url / llm_api_key  its OpenAI-compatible endpoint
    #   per_call=4              variations requested per call (small = robust
    #                           on small local models; truncated replies are
    #                           salvaged, not dropped)
    #   max_tokens=4096         reply budget per call
    #   strict=False            True → raise if fewer rows than asked were added
    #   persist=True            write the result as JSONL for review
    #   persist_path=...        where (default: <seed>.augmented.jsonl)
    #   seed_path=...           the seed file (for the default persist path;
    #                           the seed file itself is never overwritten)
    from agentomatic.optimize import prepare_dataset

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    augmented, written = prepare_dataset(
        dataset,
        augment=True,
        n_examples=args.n_examples,
        strategies=strategies,
        model=settings.rewrite_spec,
        llm_base_url=settings.base_url,
        llm_api_key=settings.api_key,
        per_call=4,
        persist=True,
        persist_path=settings.out_dir / "support.augmented.jsonl",
        seed_path=DATASET,
    )

    # ------------------------------------------------------------------
    # 3. Inspect what happened
    # ------------------------------------------------------------------
    stats = augmented.metadata["augment_stats"]
    logger.info("augment_stats: {}", json.dumps(stats))
    logger.info(
        "after: train={} validation={} holdout={} test={} → {}",
        len(augmented.train),
        len(augmented.validation),
        len(augmented.holdout),
        len(augmented.test),
        written,
    )
    new_rows = [e for e in augmented.examples if e.metadata.get("source") == "augment"]
    for example in new_rows[:8]:
        logger.info(
            "  [{:<15}] from {:<11} {!r} → {!r}",
            example.metadata.get("strategy"),
            example.metadata.get("parent_id"),
            example.input.get("current_query"),
            str((example.expected_output or {}).get("response", ""))[:60],
        )

    # ------------------------------------------------------------------
    # 4. Use it
    # ------------------------------------------------------------------
    # Pass ``augmented`` to agent.compile()/fit() (see train.py --augment), or
    # load the persisted JSONL later: load_data(settings.out_dir /
    # "support.augmented.jsonl"). Review rows from "new question" strategies
    # before trusting a score computed on them.
    return 0 if new_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
