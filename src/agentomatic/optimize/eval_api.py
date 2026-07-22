"""High-level evaluate API for class-owned agents.

Packages stack loading, dataset split selection, structured + LLM-judge
metrics, ``agent.evaluate``, and a HolySheet HTML report so project
``eval.py`` scripts stay thin — mirroring :mod:`train_api`.

Example::

    from agentomatic.optimize import EvalConfig, evaluate_and_report

    result = evaluate_and_report(
        agent,
        config=EvalConfig(
            agent_name="assistant",
            agent_dir=Path("agents/assistant"),
            stacks_dir=Path("stacks"),
            env_path=Path("../.env"),
            stack="gemini",
            required_keys=["content", "next_action"],
            split="test",
        ),
    )
    print(result.scores, result.report_path)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from agentomatic.optimize.train_api import (
    build_default_metrics,
    load_data,
    prepare_dataset,
    resolve_reports_dir,
)


@dataclass(slots=True)
class EvalConfig:
    """Configuration for :func:`run_eval` / :func:`evaluate_and_report`.

    Attributes:
        agent_name: Agent name (artefacts / reports).
        agent_dir: Agent package directory (datasets, reports).
        stacks_dir: Directory containing stack YAML files.
        env_path: Optional ``.env`` path for secrets / stack vars.
        stack: Stack name (or ``AGENTOMATIC_STACK``).
        dataset_path: JSONL dataset (default ``datasets/all.jsonl``, or
            ``all.augmented.jsonl`` when ``prefer_augmented=True``).
        reports_dir: Writable reports dir (auto-resolved when omitted).
        report_path: Explicit HTML report path (timestamped default).
        split: Dataset split — ``test``, ``validation``, ``train``,
            ``eval`` (val+test), or ``all``.
        limit: Optional cap on the number of examples.
        required_keys: Structured output keys for schema metrics.
        judge_criteria: LLM-as-judge criteria text.
        judge_dimensions: Judge dimension names.
        judge_weight: Weight of judge in the composite (informational;
            all metrics are reported individually).
        use_judge: When False, skip the LLM judge (deterministic only).
        prefer_augmented: Prefer ``datasets/all.augmented.jsonl`` when
            present and ``dataset_path`` is omitted.
        augment: Optionally LLM-augment before evaluate (rare; usually
            reuse a persisted augmented file via ``prefer_augmented``).
        n_examples: Target size when ``augment=True``.
        persist: Persist (possibly augmented) dataset to disk.
        persist_path: Explicit persist destination.
        json_out: Optional path to write a JSON summary beside the HTML.
    """

    agent_name: str
    agent_dir: Path
    stacks_dir: Path | None = None
    env_path: Path | None = None
    stack: str = ""
    dataset_path: Path | None = None
    reports_dir: Path | None = None
    report_path: Path | None = None
    split: str = "test"
    limit: int | None = None
    required_keys: list[str] = field(default_factory=lambda: ["content", "next_action"])
    judge_criteria: str = (
        "Evaluate whether the response is relevant, grounded in the provided "
        "context, accurate, and proposes a clear actionable next step."
    )
    judge_dimensions: list[str] = field(
        default_factory=lambda: ["relevance", "groundedness", "actionability"]
    )
    judge_weight: float = 0.30
    use_judge: bool = True
    prefer_augmented: bool = False
    augment: bool = False
    n_examples: int | None = None
    persist: bool = False
    persist_path: Path | None = None
    augment_strategies: list[str] | None = None
    augment_model: str | None = None
    json_out: Path | None = None


@dataclass(slots=True)
class EvaluateResult:
    """Outcome of :func:`run_eval` / :func:`evaluate_and_report`."""

    report: Any
    scores: dict[str, float]
    report_path: Path
    dataset_sizes: dict[str, int]
    n_examples: int
    split: str
    stack: str
    model: str
    example_results: list[Any] = field(default_factory=list)
    augmented: bool = False
    persist_path: Path | None = None
    json_out: Path | None = None
    dataset_path: Path | None = None


def _model_spec(entry: Any) -> str:
    return f"{entry.provider}/{entry.model}"


def select_examples(
    dataset: Any,
    *,
    split: str = "test",
    limit: int | None = None,
) -> list[Any]:
    """Pick examples for a named split (with sensible fallbacks).

    ``test`` falls back to ``validation`` when the test split is empty.
    ``eval`` is validation + test (held-out pool).
    """
    split_key = (split or "test").strip().lower()
    if split_key == "test":
        examples = list(dataset.test or dataset.validation or [])
    elif split_key in {"validation", "val"}:
        examples = list(dataset.validation or [])
    elif split_key == "train":
        examples = list(dataset.train or [])
    elif split_key == "eval":
        examples = list(dataset.validation or []) + list(dataset.test or [])
    else:
        examples = list(dataset.examples or [])
    if limit is not None and limit >= 0:
        examples = examples[: int(limit)]
    return examples


def resolve_eval_dataset_path(agent_dir: Path, *, prefer_augmented: bool = False) -> Path:
    """Resolve default eval JSONL path under ``agent_dir/datasets``."""
    datasets = Path(agent_dir) / "datasets"
    augmented = datasets / "all.augmented.jsonl"
    seed = datasets / "all.jsonl"
    if prefer_augmented and augmented.exists():
        return augmented
    return seed


def run_eval(
    agent: Any,
    *,
    config: EvalConfig,
    dataset: Any | None = None,
    metrics: Sequence[Any] | None = None,
) -> EvaluateResult:
    """Load data → metrics → ``agent.evaluate`` → HolySheet report.

    Args:
        agent: Live ``BaseGraphAgent`` instance (already constructed with LLM).
        config: Eval configuration.
        dataset: Optional pre-loaded ``AgentDataset``.
        metrics: Optional metric sequence (defaults to structured + judge).

    Returns:
        :class:`EvaluateResult` with scores, per-example rows, and report path.
    """
    from agentomatic.agents import AgentDataset
    from agentomatic.config.settings import load_environment
    from agentomatic.optimize.llm_caller import LLMCaller
    from agentomatic.optimize.report import generate_eval_report
    from agentomatic.providers import apply_stack_defaults
    from agentomatic.stacks.manager import StackManager

    agent_dir = Path(config.agent_dir).resolve()
    stack_name = (config.stack or os.getenv("AGENTOMATIC_STACK") or "local").strip()

    if config.env_path is not None:
        load_environment(Path(config.env_path))
    else:
        for candidate in (agent_dir.parent.parent / ".env", Path(".env")):
            if candidate.exists():
                load_environment(candidate)
                break

    stacks_dir = Path(config.stacks_dir) if config.stacks_dir else agent_dir.parent.parent / "stacks"
    stacks = StackManager(stacks_dir)
    stacks.load(stack_name)
    apply_stack_defaults(stacks)
    entry = stacks.get_llm_config("default")
    model = _model_spec(entry)

    # Judge / augment LLMCaller must hit the stack endpoint (Gemini, oMLX, …).
    if entry.base_url or entry.api_key:
        LLMCaller.configure(
            base_url=entry.base_url,
            api_key=entry.api_key or "local",
        )

    reports = (
        Path(config.reports_dir)
        if config.reports_dir
        else resolve_reports_dir(agent_dir, config.agent_name)
    )
    reports.mkdir(parents=True, exist_ok=True)

    ds_path = (
        Path(config.dataset_path)
        if config.dataset_path
        else resolve_eval_dataset_path(agent_dir, prefer_augmented=config.prefer_augmented)
    )
    if dataset is None:
        dataset = AgentDataset.from_jsonl(ds_path, name=config.agent_name)

    sizes_before = {
        "train": len(dataset.train),
        "validation": len(dataset.validation),
        "test": len(dataset.test),
        "all": len(dataset.examples),
    }

    persist_written: Path | None = None
    if config.augment or config.persist:
        try:
            rewrite_entry = stacks.get_llm_config("rewrite")
        except Exception:  # noqa: BLE001
            rewrite_entry = entry
        rewrite_model = _model_spec(rewrite_entry)
        dataset, persist_written = prepare_dataset(
            dataset,
            augment=config.augment,
            n_examples=config.n_examples,
            persist=config.persist,
            persist_path=config.persist_path,
            seed_path=ds_path,
            model=config.augment_model or rewrite_model or model,
            strategies=config.augment_strategies,
            llm_base_url=entry.base_url,
            llm_api_key=entry.api_key or "local",
        )

    sizes = {
        "train": len(dataset.train),
        "validation": len(dataset.validation),
        "test": len(dataset.test),
        "all": len(dataset.examples),
        "before_all": sizes_before["all"],
    }

    examples = select_examples(dataset, split=config.split, limit=config.limit)
    if not examples:
        logger.warning(
            "run_eval: no examples for split={!r} (dataset sizes={})",
            config.split,
            sizes,
        )

    if metrics is None:
        built, _loss, _fit = build_default_metrics(
            model=model,
            required_keys=config.required_keys,
            judge_criteria=config.judge_criteria,
            judge_dimensions=config.judge_dimensions,
            judge_weight=config.judge_weight,
        )
        if not config.use_judge:
            metrics = [m for m in built if getattr(m, "name", "") != "judge"]
        else:
            metrics = list(built)

    logger.info(
        "run_eval agent={} stack={} model={} split={} n={} judge={} dataset={}",
        config.agent_name,
        stack_name,
        model,
        config.split,
        len(examples),
        config.use_judge,
        ds_path,
    )

    eval_report = agent.evaluate(examples, metrics)
    scores = dict(getattr(eval_report, "scores", {}) or {})
    example_results = list(getattr(eval_report, "example_results", []) or [])

    if config.report_path is not None:
        out = Path(config.report_path)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out = reports / f"eval_{config.agent_name}_{stamp}.html"

    generate_eval_report(
        eval_report,
        output_path=out,
        stack_name=stack_name,
        model_name=model,
        split=config.split,
        dataset_sizes=sizes,
    )

    json_written: Path | None = None
    if config.json_out is not None:
        body = {
            "agent": config.agent_name,
            "stack": stack_name,
            "model": model,
            "split": config.split,
            "n": len(examples),
            "scores": scores,
            "dataset_path": str(ds_path),
            "dataset_sizes": sizes,
            "examples": [
                {
                    "id": getattr(er, "example_id", ""),
                    "scores": getattr(er, "scores", {}),
                    "error": getattr(er, "error", None),
                    "duration_ms": getattr(er, "duration_ms", None),
                }
                for er in example_results
            ],
        }
        json_path = Path(config.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        json_written = json_path

    return EvaluateResult(
        report=eval_report,
        scores=scores,
        report_path=out,
        dataset_sizes=sizes,
        n_examples=len(examples),
        split=config.split,
        stack=stack_name,
        model=model,
        example_results=example_results,
        augmented=bool(config.augment),
        persist_path=persist_written,
        json_out=json_written,
        dataset_path=ds_path,
    )


# Public aliases — preferred names in docs / templates.
evaluate_and_report = run_eval
run_evaluate = run_eval

__all__ = [
    "EvalConfig",
    "EvaluateResult",
    "evaluate_and_report",
    "load_data",
    "resolve_eval_dataset_path",
    "run_eval",
    "run_evaluate",
    "select_examples",
]
