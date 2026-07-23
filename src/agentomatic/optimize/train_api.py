"""High-level Keras-style train API for class-owned agents.

Two public styles share the same primitives:

1. **Staged (Keras-like)** — full control::

       data, _ = prepare_dataset(load_data(...), augment=True, persist=True)
       metrics, loss, fit_metric = build_default_metrics(...)
       compiled = compile_agent(agent, metrics=metrics, loss=loss, ...)
       history = fit_agent(compiled, data, epochs=2, trials=12)
       scores = evaluate_agent(compiled, data.test)

2. **One-shot** — thin scripts via :func:`train_and_report` (calls the same
   ``compile_agent`` → ``fit_agent`` → ``evaluate_agent`` path).

Example (one-shot)::

    from agentomatic.optimize import TrainConfig, load_data, train_and_report

    result = train_and_report(
        agent,
        config=TrainConfig(
            agent_name="assistant",
            agent_dir=Path("agents/assistant"),
            stacks_dir=Path("stacks"),
            env_path=Path("../.env"),
            required_keys=["content", "next_action"],
            epochs=2,
            optimizer="rewrite",
            augment=True,
            n_examples=100,
            persist=True,
        ),
    )
    print(result.report_path, result.dataset_sizes)
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(slots=True)
class TrainConfig:
    """Configuration for :func:`run_train` / :func:`train_and_report`.

    Attributes:
        agent_name: Agent name (artefacts / reports).
        agent_dir: Agent package directory (prompts, datasets, reports).
        stacks_dir: Directory containing stack YAML files.
        env_path: Optional ``.env`` path for secrets / stack vars.
        stack: Stack name (or ``AGENTOMATIC_STACK``).
        dataset_path: JSONL dataset (default ``agent_dir/datasets/all.jsonl``).
        reports_dir: Writable reports dir (auto-resolved when omitted).
        epochs: Outer Keras-style epochs (each runs a PromptFitter pass).
        max_trials: Inner PromptFitter trial budget.
        patience: EarlyStopping patience on ``val_loss``.
        optimizer: Fitter optimizer name (``rewrite``, ``gepa_like``, …).
        required_keys: Structured output keys for schema metrics.
        judge_criteria: LLM-as-judge criteria text.
        judge_dimensions: Judge dimension names.
        judge_weight: Weight of judge in the agent loss composite.
        apply: Persist best prompt when improved.
        apply_as: Version key for ``prompts.json`` (default ``v2_fit``).
        min_absolute_improvement: Acceptance threshold for candidates.
        sequential: Force sequential candidate evaluation.
        concurrency: Parallelism when not sequential.
        optimize_model_params: Whether to search temperature / top_p.
        verbose: Fit verbosity (0/1).
        augment: When True, expand the seed dataset via LLM augmentation.
        n_examples: Target total example count after augmentation
            (alias: ``nr_examples``). Defaults to ``3 ×`` seed size when
            ``augment=True`` and this is omitted.
        persist: Persist the (possibly augmented) dataset back to disk.
        persist_path: Explicit path for persisted JSONL (default
            ``datasets/all.augmented.jsonl`` beside the seed file).
        augment_strategies: Strategies forwarded to
            :meth:`~agentomatic.optimize.DataSynthesizer.augment`.
        augment_model: Model for augmentation (defaults to rewrite model).
        persist_fit_store: When ``True``, bind a SQLAlchemy fit store from
            ``fit_store_url`` / ``AGENTOMATIC_FIT_STORE_URL`` /
            ``DATABASE_URL`` so retrain artefacts survive process restart
            (any SQLAlchemy async backend).
        fit_store_url: Explicit DB URL for auditable retrain persistence
            (overrides env). Same dialects as platform storage
            (Postgres, SQLite, …).
    """

    agent_name: str
    agent_dir: Path
    stacks_dir: Path | None = None
    env_path: Path | None = None
    stack: str = ""
    dataset_path: Path | None = None
    reports_dir: Path | None = None
    epochs: int = 2
    max_trials: int = 12
    patience: int = 2
    optimizer: str = "rewrite"
    required_keys: list[str] = field(default_factory=lambda: ["content", "next_action"])
    judge_criteria: str = (
        "Evaluate whether the response is relevant, grounded in the provided "
        "context, accurate, and proposes a clear actionable next step."
    )
    judge_dimensions: list[str] = field(
        default_factory=lambda: ["relevance", "groundedness", "actionability"]
    )
    judge_weight: float = 0.30
    apply: bool = False
    apply_as: str | None = None
    min_absolute_improvement: float = 0.001
    sequential: bool = True
    concurrency: int = 2
    optimize_model_params: bool = False
    verbose: int = 1
    drain_seconds: float = 1.0
    base_prompt_version: str = "v1"
    # --- data augmentation (first-class) ---------------------------------
    augment: bool = False
    n_examples: int | None = None
    nr_examples: int | None = None  # alias for n_examples
    persist: bool = False
    persist_path: Path | None = None
    augment_strategies: list[str] | None = None
    augment_model: str | None = None
    # --- auditable fit / retrain persistence --------------------------------
    persist_fit_store: bool = False
    fit_store_url: str | None = None

    def __post_init__(self) -> None:
        """Normalise ``nr_examples`` → ``n_examples`` alias."""
        if self.n_examples is None and self.nr_examples is not None:
            object.__setattr__(self, "n_examples", int(self.nr_examples))


@dataclass(slots=True)
class TrainResult:
    """Outcome of :func:`run_train`."""

    history: Any
    fit_result: Any | None
    eval_scores: dict[str, float]
    report_path: Path
    optimize_status: str
    applied_version: str | None
    dataset_sizes: dict[str, int]
    stack: str
    model: str
    rewrite_model: str
    optimizer: str
    augmented: bool = False
    persist_path: Path | None = None
    agent_name: str = ""

    def print_summary(self, console: Any | None = None) -> None:
        """Pretty-print a Rich summary of this train run.

        Args:
            console: Optional ``rich.console.Console``. When omitted a private
                console is created so callers keep full control of styling.
        """
        print_train_result(self, console=console)


def print_train_result(result: TrainResult, *, console: Any | None = None) -> None:
    """Print a rich, scannable summary of a :class:`TrainResult`.

    Extracted from project ``train_*.py`` scripts so they stay thin while
    callers retain control (pass your own Console, or call selectively).

    Args:
        result: Train outcome from :func:`train_and_report`.
        console: Optional Rich console; created on demand when omitted.
    """
    if console is None:
        from rich.console import Console

        console = Console()

    agent = result.agent_name or "agent"
    console.print(
        f"[bold]{agent}[/bold] stack={result.stack!r} model={result.model} "
        f"optimizer={result.optimizer}"
    )
    console.print(f"dataset={result.dataset_sizes} augmented={result.augmented}")
    if result.persist_path:
        console.print(f"persisted={result.persist_path}")
    console.print(f"optimize_status={result.optimize_status!r}")

    hist = getattr(result.history, "history", {}) or {}
    if "loss" in hist:
        console.print(f"loss={hist.get('loss')}")
    if "val_loss" in hist:
        console.print(f"val_loss={hist.get('val_loss')}")

    fit = result.fit_result
    if fit is not None:
        console.print(
            f"fit baseline={fit.baseline_score:.4f} best={fit.best_score:.4f} "
            f"Δ={fit.absolute_improvement:+.4f} improved={fit.improved}"
        )
        console.print(f"score_history={list(fit.history)}")
        console.print(f"early_stop={getattr(fit, 'early_stop_reason', None)!r}")
        ph = list(getattr(fit, "prompt_history", None) or [])
        console.print(f"prompt_history_len={len(ph)}")
        for i, entry in enumerate(ph[:6]):
            if not isinstance(entry, dict):
                continue
            snap = str(entry.get("prompt_snapshot") or "")
            console.print(
                f"  ph[{i}] accepted={entry.get('accepted')} "
                f"score={float(entry.get('score') or 0):.4f} "
                f"cand={entry.get('candidate_name')!r} "
                f"prompt[:120]={snap[:120]!r}"
            )
        base_p = getattr(getattr(fit, "baseline_config", None), "system_prompt", "") or ""
        best_p = getattr(getattr(fit, "best_config", None), "system_prompt", "") or ""
        console.print(
            f"prompt_changed={base_p != best_p} base_len={len(base_p)} best_len={len(best_p)}"
        )

    console.print(f"evaluate scores={result.eval_scores}")
    console.print(f"[green]Report:[/green] {result.report_path}")
    try:
        rsize = Path(result.report_path).stat().st_size if result.report_path else 0
        console.print(f"report_bytes={rsize}")
    except OSError:
        pass
    if result.applied_version:
        console.print(f"[green]Applied {result.applied_version!r}[/green]")
    else:
        console.print("[dim]Demo-safe: prompts unchanged (pass --apply to persist).[/dim]")


def _resolve_fit_store_url(config: TrainConfig) -> str | None:
    """Return a DB URL for offline fit persistence, if configured."""
    if config.fit_store_url:
        return str(config.fit_store_url).strip() or None
    env_url = (os.getenv("AGENTOMATIC_FIT_STORE_URL") or "").strip()
    if env_url:
        return env_url
    persist = config.persist_fit_store or os.getenv(
        "AGENTOMATIC_PERSIST_FIT", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not persist:
        return None
    return (os.getenv("AGENTOMATIC_DB_URL") or os.getenv("DATABASE_URL") or "").strip() or None


def _bind_fit_store(config: TrainConfig) -> Any | None:
    """Optionally bind a SQLAlchemy fit store for auditable retrain history.

    Returns:
        The bound store (caller should clear via :func:`set_fit_store`), or
        ``None`` when persistence is disabled / unavailable.
    """
    url = _resolve_fit_store_url(config)
    if not url:
        return None
    try:
        from agentomatic.async_utils import run_sync
        from agentomatic.optimize.fit_store import set_fit_store
        from agentomatic.storage.sqlalchemy import SQLAlchemyStore

        store = SQLAlchemyStore(url)
        run_sync(store.initialize())
        set_fit_store(store)
        logger.info(
            "Fit store bound for retrain audit: {}",
            url.split("@")[-1] if "@" in url else url,
        )
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fit store bind skipped: {}", exc)
        return None


def load_data(path: str | Path, *, name: str = "") -> Any:
    """Load an :class:`~agentomatic.agents.AgentDataset` from JSONL.

    Thin helper so train scripts can stay declarative::

        dataset = load_data("agents/assistant/datasets/all.jsonl")
    """
    from agentomatic.agents import AgentDataset

    path = Path(path)
    return AgentDataset.from_jsonl(path, name=name or path.stem)


def resolve_reports_dir(agent_dir: Path, agent_name: str) -> Path:
    """Resolve a writable reports directory.

    Preference: ``AI_TRAIN_REPORTS`` → ``agent_dir/reports`` →
    ``$AI_ARTIFACT_ROOT/reports/<agent>``.
    """
    explicit = os.getenv("AI_TRAIN_REPORTS", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    local = agent_dir / "reports"
    try:
        local.mkdir(parents=True, exist_ok=True)
        probe = local / ".write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return local
    except OSError:
        root = Path(
            os.getenv("AI_ARTIFACT_ROOT", str(agent_dir.parent.parent / ".local" / "artifacts"))
        )
        fallback = root / "reports" / agent_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _model_spec(entry: Any) -> str:
    return f"{entry.provider}/{entry.model}"


def _datapoint_to_example(dp: Any, *, idx: int, split: str = "train") -> Any:
    """Convert an optimize ``DataPoint`` into an ``AgentExample``."""
    from agentomatic.agents.types import AgentExample

    expected_raw = getattr(dp, "expected_answer", None)
    expected: dict[str, Any] | None
    if isinstance(expected_raw, dict):
        expected = dict(expected_raw)
    elif isinstance(expected_raw, str) and expected_raw.strip():
        try:
            parsed = json.loads(expected_raw)
            expected = parsed if isinstance(parsed, dict) else {"content": expected_raw}
        except (json.JSONDecodeError, TypeError):
            expected = {"content": expected_raw, "next_action": "Follow up with stakeholder."}
    else:
        expected = {"content": "", "next_action": ""}

    query = str(getattr(dp, "query", "") or "")
    meta = dict(getattr(dp, "metadata", None) or {})
    meta.setdefault("source", "augment")
    meta.setdefault("split", split)
    ctx = getattr(dp, "context", None) or []
    inp: dict[str, Any] = {
        "current_query": query,
        "query": query,
        "question": query,
    }
    if ctx:
        # Prefer structured context when the synthesizer echoed JSON.
        try:
            inp["context"] = json.loads(ctx[0]) if isinstance(ctx[0], str) else ctx[0]
        except (json.JSONDecodeError, TypeError, IndexError):
            inp["context"] = ctx

    return AgentExample(
        id=f"aug_{idx:04d}",
        input=inp,
        expected_output=expected,
        metadata=meta,
        tags=["augmented"],
        split=split,
    )


def prepare_dataset(
    dataset: Any,
    *,
    augment: bool = False,
    n_examples: int | None = None,
    persist: bool = False,
    persist_path: Path | str | None = None,
    seed_path: Path | str | None = None,
    model: str = "openai/gpt-4o-mini",
    strategies: list[str] | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
) -> tuple[Any, Path | None]:
    """Optionally augment (and persist) an :class:`AgentDataset`.

    Args:
        dataset: Seed ``AgentDataset``.
        augment: Run LLM augmentation when True.
        n_examples: Target total size (keeps originals + fills with synth).
        persist: Write the resulting dataset to disk.
        persist_path: Destination JSONL path.
        seed_path: Original seed path (used to derive default persist path).
        model: Augmentation LLM spec.
        strategies: Augmentation strategies.
        llm_base_url / llm_api_key: Optional OpenAI-compatible routing.

    Returns:
        ``(dataset, written_path_or_none)``.
    """
    from agentomatic.agents import AgentDataset
    from agentomatic.async_utils import run_sync
    from agentomatic.optimize.dataset import Dataset
    from agentomatic.optimize.llm_caller import LLMCaller
    from agentomatic.optimize.synthesizer import DataSynthesizer

    written: Path | None = None
    out_ds = dataset

    if augment:
        if llm_base_url or llm_api_key:
            LLMCaller.configure(base_url=llm_base_url, api_key=llm_api_key)

        seed_n = max(len(dataset.examples), 1)
        target = int(n_examples) if n_examples and n_examples > 0 else seed_n * 3
        need = max(0, target - seed_n)
        strategies = list(strategies or ["paraphrase", "perturbation", "expansion"])
        seed_used = min(10, seed_n)
        # Rough inverse of synthesizer yield: ~strategies × (M // strategies) × seeds
        multiplier = max(1, (need + max(seed_used, 1) - 1) // max(seed_used, 1))

        logger.info(
            "Augmenting dataset: seed={} target≈{} multiplier={} strategies={}",
            seed_n,
            target,
            multiplier,
            strategies,
        )

        train_examples = list(dataset.train) or list(dataset.examples)
        seed_opt = Dataset(points=[e.to_datapoint() for e in train_examples])
        synth = DataSynthesizer(model=model)

        async def _run() -> Dataset:
            return await synth.augment(
                seed_opt,
                strategies=strategies,
                multiplier=multiplier,
            )

        try:
            augmented_opt = run_sync(_run())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dataset augmentation failed ({}); using seed dataset", exc)
            augmented_opt = seed_opt

        seed_queries = {p.query for p in seed_opt.points}
        merged = list(dataset.examples)
        for dp in augmented_opt.points:
            if dp.query in seed_queries:
                continue
            merged.append(_datapoint_to_example(dp, idx=len(merged), split="train"))
            seed_queries.add(dp.query)
            if len(merged) >= target:
                break

        out_ds = AgentDataset(
            name=getattr(dataset, "name", "dataset"),
            examples=merged,
            metadata={
                **dict(getattr(dataset, "metadata", None) or {}),
                "augmented": True,
                "augment_target": target,
            },
        )
        logger.info("Augmented dataset: {} → {} examples", seed_n, len(out_ds.examples))

    if persist:
        if persist_path is not None:
            dest = Path(persist_path)
        elif seed_path is not None:
            seed = Path(seed_path)
            dest = seed.with_name(
                seed.stem + (".augmented.jsonl" if augment else seed.suffix or ".jsonl")
            )
            if not augment:
                dest = seed  # overwrite / rewrite seed when persist-only
        else:
            dest = Path("datasets") / "all.augmented.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_jsonl(dest)
        written = dest
        logger.info("Persisted dataset ({} examples) → {}", len(out_ds.examples), dest)

    return out_ds, written


def build_default_metrics(
    *,
    model: str,
    required_keys: Sequence[str],
    judge_criteria: str,
    judge_dimensions: Sequence[str],
    judge_weight: float = 0.30,
) -> tuple[list[Any], Any, Any]:
    """Build agent metrics, loss, and optimize-fit CustomMetric.

    Returns:
        ``(metrics, loss, fit_metric)`` ready for :func:`compile_agent`.
    """
    from agentomatic.agents import (
        CallableMetric,
        ExactKeyMatchMetric,
        MetricLoss,
        OptimizeMetricAdapter,
        WeightedMetric,
    )
    from agentomatic.optimize.judges import LocalJudgeMetric
    from agentomatic.optimize.structured_metrics import (
        agent_field_f1,
        agent_keyword_score,
        agent_schema_quality,
        make_structured_fit_metric,
    )

    keys = list(required_keys)
    judge = LocalJudgeMetric(
        name="judge",
        model=model,
        criteria=judge_criteria,
        dimensions=list(judge_dimensions),
    )
    judge_metric = OptimizeMetricAdapter(judge, name="judge")
    key_metric = ExactKeyMatchMetric(keys)
    json_metric = CallableMetric(
        "json_valid",
        lambda ex, pred: agent_schema_quality(ex, pred, required_keys=keys),
    )
    f1_metric = CallableMetric(
        "f1",
        lambda ex, pred: agent_field_f1(ex, pred, keys=keys),
    )
    keyword_metric = CallableMetric("keywords", agent_keyword_score)

    rem = max(0.0, 1.0 - judge_weight)
    metrics = [judge_metric, key_metric, json_metric, f1_metric, keyword_metric]
    loss = MetricLoss(
        WeightedMetric(
            [
                ("judge", judge_metric, judge_weight),
                ("f1", f1_metric, rem * 0.40),
                ("keywords", keyword_metric, rem * 0.30),
                ("json_valid", json_metric, rem * 0.15),
                ("key_match", key_metric, rem * 0.15),
            ],
            name="composite_loss",
        )
    )
    fit_metric = make_structured_fit_metric(keys, name="composite")
    return metrics, loss, fit_metric


@dataclass(slots=True)
class CompiledAgent:
    """Handle from :func:`compile_agent` — ready for :func:`fit_agent`.

    Wraps the live agent plus the metrics / fitter / search-space used at
    compile time so staged scripts do not re-thread those objects manually.
    ``train_and_report`` uses the same handle internally.
    """

    agent: Any
    metrics: list[Any]
    loss: Any
    fit_metric: Any | None = None
    fitter: Any | None = None
    search_space: Any | None = None
    optimizer_name: str = "rewrite"
    model: str = ""
    rewrite_model: str = ""
    stack: str = ""
    agent_name: str = ""
    max_trials: int = 12
    patience: int = 2
    optimize_model_params: bool = False
    verbose: int = 1

    @property
    def fit_result(self) -> Any | None:
        """Last :class:`~agentomatic.optimize.PromptFitResult` from fit."""
        return getattr(self.agent, "_last_fit_result", None)

    @property
    def optimize_status(self) -> str:
        """Last optimize status string from the fitter bridge."""
        return str(getattr(self.agent, "_last_optimize_status", "") or "")


def _as_compiled(compiled: CompiledAgent | Any) -> tuple[Any, CompiledAgent | None]:
    """Return ``(agent, compiled_or_none)`` for staged helpers."""
    if isinstance(compiled, CompiledAgent):
        return compiled.agent, compiled
    return compiled, None


def default_search_space(*, optimize_model_params: bool = False) -> Any:
    """Build the default :class:`~agentomatic.optimize.PromptSearchSpace`."""
    from agentomatic.optimize.search_space import PromptSearchSpace

    return PromptSearchSpace(
        optimize_system_prompt=True,
        optimize_few_shot=True,
        optimize_model_choice=False,
        optimize_model_params=optimize_model_params,
        model_param_space={
            "temperature": [0.0, 0.1, 0.2, 0.4, 0.7],
            "top_p": [0.7, 0.9, 1.0],
        },
        optimize_rag_params=False,
    )


def compile_agent(
    agent: Any,
    *,
    dataset: Any | None = None,
    metrics: Sequence[Any] | None = None,
    loss: Any | None = None,
    fit_metric: Any | None = None,
    optimizer: str | Any = "rewrite",
    task_model: str | None = None,
    rewrite_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    agent_name: str | None = None,
    max_trials: int = 12,
    patience: int = 2,
    search_space: Any | None = None,
    optimize_model_params: bool = False,
    min_absolute_improvement: float = 0.001,
    sequential: bool = True,
    concurrency: int = 2,
    experiment_dir: str | Path | None = None,
    base_prompt_version: str = "v1",
    drain_seconds: float = 1.0,
    auto_report: bool = True,
    stack: str = "",
    verbose: int = 1,
) -> CompiledAgent:
    """Compile an agent for prompt fitting (Keras-style ``model.compile``).

    Builds (or accepts) a :class:`~agentomatic.agents.PromptFitterBridge`,
    then calls ``agent.compile(...)``. Does **not** run trials — use
    :func:`fit_agent` next.

    Args:
        agent: Live ``BaseGraphAgent`` (or compatible) instance.
        dataset: Optional dataset stored on the agent at compile time.
        metrics: Agent-level metrics (from :func:`build_default_metrics` or
            custom). Required for a training signal.
        loss: Loss objective (typically the ``loss`` from
            :func:`build_default_metrics`).
        fit_metric: Metric used inside PromptFitter candidate ranking.
        optimizer: Optimizer name (``rewrite``, ``gepa_like``, …) or a
            pre-built ``PromptFitterBridge`` / optimizer instance.
        task_model / rewrite_model: LLM specs (``provider/model``).
        llm_base_url / llm_api_key: OpenAI-compatible routing for local stacks.
        agent_name: Artefact / report name (defaults to ``agent.agent_name``).
        max_trials: Inner PromptFitter trial budget.
        patience: Early-stop patience mirrored onto the bridge.
        search_space: Optional :class:`~agentomatic.optimize.PromptSearchSpace`.
        optimize_model_params: Whether the default search space tunes
            temperature / top_p.
        min_absolute_improvement: Candidate acceptance threshold.
        sequential / concurrency: Evaluation parallelism.
        experiment_dir: Directory for fitter artefacts.
        base_prompt_version: Starting prompt version key.
        drain_seconds: Post-fit async drain for HTTP clients.
        auto_report: Forwarded to ``PromptFitterBridge``.
        stack: Optional stack name kept for reporting metadata.
        verbose: Default verbosity for subsequent :func:`fit_agent` calls.

    Returns:
        :class:`CompiledAgent` handle for :func:`fit_agent` /
        :func:`evaluate_agent`.
    """
    from agentomatic.agents import PromptFitterBridge

    name = agent_name or getattr(agent, "agent_name", None) or type(agent).__name__
    metrics_list = list(metrics or [])
    space = (
        search_space
        if search_space is not None
        else default_search_space(optimize_model_params=optimize_model_params)
    )

    if isinstance(optimizer, str):
        opt_name = optimizer
        fitter = PromptFitterBridge(
            agent_name=name,
            task_model=task_model or "",
            rewrite_model=rewrite_model or task_model or "",
            local_agent=agent,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key or "local",
            max_trials=max_trials,
            metric=fit_metric,
            base_prompt_version=base_prompt_version,
            search_space=space,
            optimizer=opt_name,
            min_absolute_improvement=min_absolute_improvement,
            patience=patience,
            concurrency=concurrency,
            sequential=sequential,
            experiment_dir=str(experiment_dir) if experiment_dir else None,
            auto_report=auto_report,
            drain_seconds=drain_seconds,
        )
    else:
        fitter = optimizer
        raw_name = getattr(optimizer, "optimizer", None)
        opt_name = raw_name if isinstance(raw_name, str) and raw_name else type(optimizer).__name__

    agent.compile(dataset, metrics=metrics_list, optimizer=fitter, loss=loss)
    return CompiledAgent(
        agent=agent,
        metrics=metrics_list,
        loss=loss,
        fit_metric=fit_metric,
        fitter=fitter,
        search_space=space,
        optimizer_name=str(opt_name),
        model=task_model or "",
        rewrite_model=rewrite_model or task_model or "",
        stack=stack,
        agent_name=name,
        max_trials=max_trials,
        patience=patience,
        optimize_model_params=optimize_model_params,
        verbose=verbose,
    )


def fit_agent(
    compiled: CompiledAgent | Any,
    dataset: Any,
    *,
    epochs: int = 2,
    max_trials: int | None = None,
    trials: int | None = None,
    patience: int | None = None,
    validation_data: Any | None = None,
    callbacks: Sequence[Any] | None = None,
    optimize_mode: str | None = None,
    optimize_prompt: bool = True,
    optimize_params: bool | None = None,
    search_space: Any | None = None,
    verbose: int | None = None,
    **fit_kwargs: Any,
) -> Any:
    """Fit a compiled agent (Keras-style ``model.fit``).

    Args:
        compiled: :class:`CompiledAgent` from :func:`compile_agent`, or a
            raw agent already passed through ``agent.compile(...)``.
        dataset: Training ``AgentDataset`` (uses ``.train`` when present).
        epochs: Outer Keras-style epochs.
        max_trials / trials: Inner PromptFitter trial budget (``trials`` is
            an alias).
        patience: EarlyStopping patience on ``val_loss``.
        validation_data: Held-out split (defaults to ``dataset.validation``).
        callbacks: Optional callback list (default: EarlyStopping).
        optimize_mode: Fitter strategy override (defaults to compile-time
            optimizer name).
        optimize_prompt / optimize_params: Search-space toggles for this fit.
        search_space: Optional override search space.
        verbose: Fit verbosity (0/1).
        **fit_kwargs: Extra kwargs forwarded to ``agent.fit``.

    Returns:
        Keras-style :class:`~agentomatic.agents.History`.
    """
    from agentomatic.agents import EarlyStopping

    agent, handle = _as_compiled(compiled)
    if trials is not None:
        trial_budget = int(trials)
    elif max_trials is not None:
        trial_budget = int(max_trials)
    else:
        trial_budget = int(handle.max_trials) if handle is not None else 12
    if patience is not None:
        stop_patience = int(patience)
    else:
        stop_patience = int(handle.patience) if handle is not None else 2
    mode = optimize_mode or (handle.optimizer_name if handle else None) or "rewrite"
    space = search_space
    if space is None and handle is not None:
        space = handle.search_space
    if optimize_params is not None:
        opt_params = optimize_params
    else:
        opt_params = bool(handle.optimize_model_params) if handle is not None else False
    if verbose is not None:
        verb = int(verbose)
    else:
        verb = int(handle.verbose) if handle is not None else 1
    val = validation_data
    if val is None and hasattr(dataset, "validation"):
        val = dataset.validation
    if callbacks is not None:
        cbs = list(callbacks)
    else:
        cbs = [EarlyStopping(monitor="val_loss", patience=stop_patience, mode="min")]

    return agent.fit(
        dataset,
        epochs=epochs,
        validation_data=val,
        callbacks=cbs,
        max_trials=trial_budget,
        search_space=space,
        optimize_mode=mode,
        optimize_prompt=optimize_prompt,
        optimize_params=opt_params,
        verbose=verb,
        **fit_kwargs,
    )


def evaluate_agent(
    compiled: CompiledAgent | Any,
    dataset: Any,
    *,
    metrics: Sequence[Any] | None = None,
) -> Any:
    """Evaluate a compiled (or raw) agent on a split / dataset.

    Args:
        compiled: :class:`CompiledAgent` or agent with ``evaluate()``.
        dataset: Examples, ``AgentDataset``, or a split list
            (e.g. ``data.test``).
        metrics: Optional metric override; defaults to compile-time metrics.

    Returns:
        :class:`~agentomatic.agents.EvaluationReport` (use ``.scores`` for
        the flat score dict).
    """
    agent, handle = _as_compiled(compiled)
    resolved = (
        list(metrics)
        if metrics is not None
        else (list(handle.metrics) if handle is not None else None)
    )
    return agent.evaluate(dataset, resolved)


def run_train(
    agent: Any,
    *,
    config: TrainConfig,
    dataset: Any | None = None,
) -> TrainResult:
    """Compile → fit → evaluate → report for a class-owned agent.

    Thin convenience over the staged primitives::

        load_data / prepare_dataset → build_default_metrics → compile_agent
        → fit_agent → evaluate_agent → generate_fit_report → optional apply

    Args:
        agent: Live ``BaseGraphAgent`` instance (already constructed with LLM).
        config: Train configuration.
        dataset: Optional pre-loaded ``AgentDataset``; otherwise loaded from
            ``config.dataset_path`` / ``agent_dir/datasets/all.jsonl``.

    Returns:
        :class:`TrainResult` with history, fit artefacts, and report path.
    """
    from agentomatic.config.settings import load_environment
    from agentomatic.optimize.report import generate_fit_report
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

    stacks_dir = (
        Path(config.stacks_dir) if config.stacks_dir else agent_dir.parent.parent / "stacks"
    )
    stacks = StackManager(stacks_dir)
    stacks.load(stack_name)
    apply_stack_defaults(stacks)
    entry = stacks.get_llm_config("default")
    try:
        rewrite_entry = stacks.get_llm_config("rewrite")
    except Exception:  # noqa: BLE001
        rewrite_entry = entry
    model = _model_spec(entry)
    rewrite_model = _model_spec(rewrite_entry)

    reports = (
        Path(config.reports_dir)
        if config.reports_dir
        else resolve_reports_dir(agent_dir, config.agent_name)
    )
    reports.mkdir(parents=True, exist_ok=True)

    ds_path = (
        Path(config.dataset_path) if config.dataset_path else agent_dir / "datasets" / "all.jsonl"
    )
    if dataset is None:
        dataset = load_data(ds_path, name=config.agent_name)

    sizes_before = {
        "train": len(dataset.train),
        "validation": len(dataset.validation),
        "test": len(dataset.test),
        "all": len(dataset.examples),
    }

    persist_written: Path | None = None
    if config.augment or config.persist:
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
    logger.info(
        "run_train agent={} stack={} model={} optimizer={} "
        "train={} val={} test={} epochs={} trials={} augment={}",
        config.agent_name,
        stack_name,
        model,
        config.optimizer,
        sizes["train"],
        sizes["validation"],
        sizes["test"],
        config.epochs,
        config.max_trials,
        config.augment,
    )

    metrics, loss, fit_metric = build_default_metrics(
        model=model,
        required_keys=config.required_keys,
        judge_criteria=config.judge_criteria,
        judge_dimensions=config.judge_dimensions,
        judge_weight=config.judge_weight,
    )

    bound_fit_store = _bind_fit_store(config)
    try:
        compiled = compile_agent(
            agent,
            dataset=dataset,
            metrics=metrics,
            loss=loss,
            fit_metric=fit_metric,
            optimizer=config.optimizer,
            task_model=model,
            rewrite_model=rewrite_model,
            llm_base_url=entry.base_url,
            llm_api_key=entry.api_key or "local",
            agent_name=config.agent_name,
            max_trials=config.max_trials,
            patience=config.patience,
            search_space=default_search_space(optimize_model_params=config.optimize_model_params),
            optimize_model_params=config.optimize_model_params,
            min_absolute_improvement=config.min_absolute_improvement,
            sequential=config.sequential,
            concurrency=config.concurrency,
            experiment_dir=reports / ".fit",
            base_prompt_version=config.base_prompt_version,
            drain_seconds=config.drain_seconds,
            stack=stack_name,
            verbose=config.verbose,
        )

        history = fit_agent(
            compiled,
            dataset,
            epochs=config.epochs,
            max_trials=config.max_trials,
            patience=config.patience,
            validation_data=dataset.validation,
            optimize_mode=config.optimizer,
            optimize_prompt=True,
            optimize_params=config.optimize_model_params,
            verbose=config.verbose,
        )

        status = compiled.optimize_status
        fit_result = compiled.fit_result

        held_out = dataset.test or dataset.validation or dataset.train
        eval_report = evaluate_agent(compiled, held_out)
        eval_scores = dict(getattr(eval_report, "scores", {}) or {})

        out = reports / f"train_{config.agent_name}.html"
        if fit_result is not None:
            generate_fit_report(
                fit_result,
                output_path=out,
                keras_history=getattr(history, "history", None),
                eval_scores=eval_scores,
                dataset_sizes=sizes,
                optimizer_name=config.optimizer,
                stack_name=stack_name,
                model_name=model,
                run_config={
                    "stack": stack_name,
                    "model": model,
                    "optimizer": config.optimizer,
                    "epochs": config.epochs,
                    "max_trials": config.max_trials,
                    "patience": config.patience,
                    "augment": bool(config.augment),
                    "required_keys": list(config.required_keys or []),
                    "judge_dimensions": list(config.judge_dimensions or []),
                    "judge_criteria": config.judge_criteria,
                    "judge_weight": config.judge_weight,
                    "min_absolute_improvement": config.min_absolute_improvement,
                },
            )
        else:
            out.write_text(
                f"<html><body><h1>{config.agent_name} fit</h1>"
                f"<pre>{getattr(history, 'history', history)}</pre></body></html>",
                encoding="utf-8",
            )

        applied: str | None = None
        apply_as = config.apply_as or ("v2_fit" if config.apply else None)
        if apply_as and fit_result is not None and hasattr(fit_result, "apply"):
            if getattr(fit_result, "improved", False):
                written = fit_result.apply(version=apply_as, agent_dir=str(agent_dir))
                applied = written
                if written:
                    logger.info("Applied prompt version {!r} → {}", written, agent_dir)
            else:
                logger.info(
                    "Skip apply: absolute_improvement={:+.4f}",
                    getattr(fit_result, "absolute_improvement", 0.0),
                )

        return TrainResult(
            history=history,
            fit_result=fit_result,
            eval_scores=eval_scores,
            report_path=out,
            optimize_status=status,
            applied_version=applied,
            dataset_sizes=sizes,
            stack=stack_name,
            model=model,
            rewrite_model=rewrite_model,
            optimizer=config.optimizer,
            augmented=bool(config.augment),
            persist_path=persist_written,
            agent_name=config.agent_name,
        )
    finally:
        if bound_fit_store is not None:
            try:
                from agentomatic.async_utils import run_sync
                from agentomatic.optimize.fit_store import set_fit_store

                set_fit_store(None)
                run_sync(bound_fit_store.close())
            except Exception as exc:  # noqa: BLE001
                logger.debug("Fit store cleanup skipped: {}", exc)


# Public aliases — preferred names in docs / templates.
train_and_report = run_train
run_training = run_train
