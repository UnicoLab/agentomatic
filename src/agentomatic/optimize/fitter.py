"""PromptFitter — scikit-learn-like orchestrator for prompt/config optimisation.

Runs an end-to-end optimisation pipeline that evaluates a baseline
configuration, clusters failures, proposes candidate configs via a
pluggable :class:`BaseFitterOptimizer`, scores them on mini-batches
and full validation sets, applies dimensional acceptance logic, and
packages results into a :class:`PromptFitResult`.

Example::

    from agentomatic.optimize import (
        PromptFitter,
        Dataset,
        CompositeMetric,
        WeightedMetric,
        LLMJudgeMetric,
        ExactMatchMetric,
    )

    trainset = Dataset.from_jsonl("train.jsonl")
    valset   = Dataset.from_jsonl("val.jsonl")
    testset  = Dataset.from_jsonl("test.jsonl")

    metric = CompositeMetric(metrics=[
        WeightedMetric("relevance", LLMJudgeMetric(criteria="..."), weight=0.6),
        WeightedMetric("format",    ExactMatchMetric(),             weight=0.4),
    ])

    fitter = PromptFitter(
        agent="scope_agent",
        task_model="ollama/qwen2.5:7b",
        rewrite_model="openai/gpt-4.1",
        optimizer="gepa_like",
    )
    result = await fitter.fit(trainset, valset, metric, testset=testset)
    print(result.summary())
    result.apply(version="v2_fit")
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from agentomatic.optimize.config import (
    ParamDelta,
    PromptCandidate,
    PromptFitResult,
    PromptRuntimeConfig,
)
from agentomatic.optimize.context import (
    DatasetSummary,
    OptimizationContext,
    RoundStats,
)
from agentomatic.optimize.dataset import Dataset
from agentomatic.optimize.events import (
    CallbackManager,
    EventData,
    OptimizationCallback,
    OptimizationEvent,
)
from agentomatic.optimize.failure_analysis import (
    DimensionAnalyzer,
    FailureClusterer,
)
from agentomatic.optimize.metrics import (
    BaseMetric,
    CompositeMetric,
    EvalResult,
)
from agentomatic.optimize.runner import AgentRunner, RunResult
from agentomatic.optimize.search_space import PromptSearchSpace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAILURE_THRESHOLD: float = 0.5
"""Data points scoring below this are considered failures."""

_MINIBATCH_FRACTION: float = 0.30
"""Fraction of valset used for cheap minibatch pre-screening."""

_MINIBATCH_MIN: int = 5
"""Absolute minimum number of points in a minibatch."""

_TOP_K_CANDIDATES: int = 3
"""How many top minibatch candidates get promoted to full evaluation."""

_EARLY_STOP_PATIENCE: int = 3
"""Stop if no improvement for this many consecutive rounds."""

_CANDIDATES_PER_ROUND: int = 4
"""Number of candidates proposed per optimisation round."""


# =====================================================================
# PromptFitter
# =====================================================================


class PromptFitter:
    """Scikit-learn-like orchestrator for prompt and config optimisation.

    ``PromptFitter`` treats the *entire* runtime configuration surface
    — system prompt, few-shot examples, model hyper-parameters, RAG
    knobs, tool policy — as an optimisable object.  It evaluates a
    baseline, clusters failures, proposes candidates via a pluggable
    :class:`BaseFitterOptimizer`, screens them on a mini-batch, then
    validates the best on the full set (and optional test set).

    Parameters
    ----------
    agent : str
        Agent name (used to locate ``prompts.json`` and for API calls).
    base_prompt_version : str
        Key inside ``prompts.json`` to load as the baseline prompt.
    task_model : str
        LLM used for evaluation tasks and candidate generation.
    rewrite_model : str | None
        Separate LLM for prompt rewriting (defaults to *task_model*).
    local_judges : list[str] | None
        Optional list of model names for a local judge panel.
    search_space : PromptSearchSpace | None
        Defines which knobs may be tuned.
    optimizer : str | BaseFitterOptimizer
        Optimisation strategy — either a name (``"gepa_like"``,
        ``"rewrite"``, ``"param_search"``, …) or an instance.
    max_trials : int
        Total budget of candidate evaluations.
    min_absolute_improvement : float
        Minimum composite-score lift for a candidate to be accepted.
    api_base : str
        Base URL of the agent API server.
    api_prefix : str
        URL prefix for the agent API endpoints.
    concurrency : int
        Maximum concurrent agent requests.
    experiment_dir : str
        Directory for experiment artefacts and reports.
    auto_report : bool
        Whether to generate an HTML report at the end.

    Examples
    --------
    >>> fitter = PromptFitter(
    ...     agent="scope_agent",
    ...     task_model="ollama/qwen2.5:7b",
    ...     rewrite_model="openai/gpt-4.1",
    ...     optimizer="gepa_like",
    ... )
    >>> result = await fitter.fit(trainset, valset, metric)
    >>> print(result.summary())
    >>> result.apply(version="v2_fit")
    """

    def __init__(
        self,
        agent: str,
        base_prompt_version: str = "v1",
        task_model: str = "ollama/qwen2.5:7b",
        rewrite_model: str | None = None,
        local_judges: list[str] | None = None,
        search_space: PromptSearchSpace | None = None,
        optimizer: str | Any = "gepa_like",
        max_trials: int = 30,
        min_absolute_improvement: float = 0.05,
        api_base: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        concurrency: int = 5,
        experiment_dir: str = ".optimize",
        auto_report: bool = True,
        callbacks: list[OptimizationCallback] | None = None,
        dashboard: bool = False,
    ) -> None:
        # Public configuration
        self.agent = agent
        self.base_prompt_version = base_prompt_version
        self.task_model = task_model
        self.rewrite_model = rewrite_model or task_model
        self.local_judges = local_judges or []
        self.max_trials = max_trials
        self.min_absolute_improvement = min_absolute_improvement
        self.concurrency = concurrency
        self.experiment_dir = experiment_dir
        self.auto_report = auto_report

        # Runner for agent invocations
        self._runner = AgentRunner(
            agent=agent,
            api_base=api_base,
            api_prefix=api_prefix,
        )

        # Search space
        self._search_space = search_space or PromptSearchSpace()

        # Fitter optimizer — lazy-imported to avoid circular deps
        from agentomatic.optimize.fitter_optimizers import (
            BaseFitterOptimizer,
            resolve_fitter_optimizer,
        )

        self._optimizer: BaseFitterOptimizer = resolve_fitter_optimizer(
            optimizer,
            model=task_model,
            rewrite_model=self.rewrite_model,
        )

        # Failure analysis
        self._failure_clusterer = FailureClusterer(model=task_model)
        self._dimension_analyzer = DimensionAnalyzer()

        # Callback / event system
        self._callbacks = CallbackManager(callbacks)
        if dashboard:
            try:
                from agentomatic.optimize.dashboard import (
                    DashboardCallback,
                    launch_dashboard,
                )

                dcb = DashboardCallback()
                self._callbacks.add(dcb)
                launch_dashboard(dcb)
            except Exception as exc:
                logger.warning("Dashboard launch failed: {}", exc)
        if not self._callbacks:
            from agentomatic.optimize.progress import (
                auto_progress_callback,
            )

            self._callbacks.add(auto_progress_callback())

    # ================================================================
    # Public API
    # ================================================================

    async def fit(
        self,
        trainset: Dataset,
        valset: Dataset,
        metric: CompositeMetric | BaseMetric,
        testset: Dataset | None = None,
    ) -> PromptFitResult:
        """Run the full prompt-fitting optimisation loop.

        Executes a 10-step pipeline:

        1. Load baseline config from ``prompts.json``.
        2. Evaluate baseline on the validation set.
        3. Cluster failures from baseline evaluation.
        4. Generate candidates via the optimizer.
        5. Score candidates on a minibatch (30 % of valset).
        6. Promote top candidates to full validation.
        7. Compare via dimensional analysis and acceptance criteria.
        8. Produce parameter-change suggestions.
        9. Validate on testset (if provided).
        10. Build and return :class:`PromptFitResult`.

        Steps 4–7 loop for up to ``max_trials // candidates_per_round``
        rounds, with early stopping after 3 rounds without improvement.

        Parameters
        ----------
        trainset : Dataset
            Training data — used for few-shot bootstrap and proposal
            context.
        valset : Dataset
            Validation data — used for scoring candidates.
        metric : CompositeMetric | BaseMetric
            Evaluation metric.  ``CompositeMetric`` enables per-dimension
            tracking.
        testset : Dataset | None
            Optional hold-out test set for final validation.

        Returns
        -------
        PromptFitResult
            The full optimisation result with best config, baseline,
            metric deltas, failure clusters, suggestions, and trial
            history.

        Examples
        --------
        >>> fitter = PromptFitter(agent="scope_agent")
        >>> result = await fitter.fit(trainset, valset, metric)
        >>> print(f"Improvement: {result.absolute_improvement:+.4f}")
        """
        experiment_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        trials: list[dict[str, Any]] = []
        score_history: list[RoundStats] = []

        logger.info(
            "🚀 PromptFitter.fit — experiment={} agent={} max_trials={}",
            experiment_id,
            self.agent,
            self.max_trials,
        )
        await self._callbacks.emit(
            OptimizationEvent.FIT_START,
            EventData(
                agent=self.agent,
                experiment_id=experiment_id,
                optimizer_name=self._optimizer.name,
                total_rounds=max(1, self.max_trials // _CANDIDATES_PER_ROUND),
            ),
        )

        # ── Step 1: Load baseline config ─────────────────────────────
        logger.info("📦 Step 1/10 — Loading baseline config")
        baseline_config = self._load_baseline_config()
        logger.debug(
            "Baseline system_prompt length: {} chars",
            len(baseline_config.system_prompt),
        )

        # ── Step 2: Evaluate baseline on validation set ──────────────
        logger.info(
            "📊 Step 2/10 — Evaluating baseline on valset ({} pts)",
            len(valset),
        )
        baseline_score, baseline_dims, baseline_details = await self._evaluate_config(
            baseline_config,
            valset,
            metric,
        )
        logger.info("📊 Baseline score: {:.4f}", baseline_score)
        if baseline_dims:
            for dim, val in baseline_dims.items():
                logger.debug("   {}: {:.4f}", dim, val)
        await self._callbacks.emit(
            OptimizationEvent.BASELINE_EVALUATED,
            EventData(
                agent=self.agent,
                experiment_id=experiment_id,
                score=baseline_score,
                baseline_score=baseline_score,
                dimensions=dict(baseline_dims),
            ),
        )

        # ── Step 3: Cluster failures ─────────────────────────────────
        logger.info("🔍 Step 3/10 — Clustering baseline failures")
        failures = [d for d in baseline_details if d.get("avg_score", 1.0) < _FAILURE_THRESHOLD]
        failure_clusters_raw: list[dict[str, Any]] = []
        if failures:
            logger.info("   Found {} failures (score < {:.1f})", len(failures), _FAILURE_THRESHOLD)
            try:
                clusters = await self._failure_clusterer.cluster(failures)
                failure_clusters_raw = [
                    {
                        "label": c.label,
                        "description": c.description,
                        "count": c.count,
                        "suggested_fix": c.suggested_fix,
                        "severity": c.severity,
                    }
                    for c in clusters
                ]
                for c in clusters:
                    logger.info(
                        "   🏷️  {} ({} failures, severity {:.2f}): {}",
                        c.label,
                        c.count,
                        c.severity,
                        c.suggested_fix[:80],
                    )
            except Exception as exc:
                logger.warning("   Failure clustering failed: {}", exc)
        else:
            logger.info("   No failures detected — baseline is strong.")

        # ── Iterative optimisation loop (Steps 4-7) ──────────────────
        best_config = baseline_config
        best_score = baseline_score
        best_dims = dict(baseline_dims)
        no_improvement_rounds = 0

        max_rounds = max(1, self.max_trials // _CANDIDATES_PER_ROUND)
        logger.info(
            "🔄 Starting optimisation: {} rounds × {} candidates = {} max evals",
            max_rounds,
            _CANDIDATES_PER_ROUND,
            max_rounds * _CANDIDATES_PER_ROUND,
        )

        # Prepare dataset points for the runner/optimizer
        val_points = [dp.to_dict() for dp in valset]
        train_points = [dp.to_dict() for dp in trainset]

        # Prepare minibatch
        minibatch_size = max(
            _MINIBATCH_MIN,
            int(len(val_points) * _MINIBATCH_FRACTION),
        )
        minibatch_points = val_points[:minibatch_size]
        minibatch_dataset = Dataset.from_list(minibatch_points)

        # Build dataset summary for context
        dataset_summary = DatasetSummary(
            n_samples=len(val_points),
            avg_query_length=(
                sum(len(str(p.get("query", ""))) for p in val_points) // max(len(val_points), 1)
            ),
            avg_expected_length=(
                sum(len(str(p.get("expected", ""))) for p in val_points) // max(len(val_points), 1)
            ),
        )

        logger.info(
            "   Minibatch: {} / {} points ({:.0%})",
            minibatch_size,
            len(val_points),
            minibatch_size / max(len(val_points), 1),
        )

        eval_results = self._build_eval_results(baseline_details, metric)

        for round_idx in range(max_rounds):
            round_t0 = time.perf_counter()
            round_num = round_idx + 1
            logger.info("── Round {}/{} ──", round_num, max_rounds)
            await self._callbacks.emit(
                OptimizationEvent.ROUND_START,
                EventData(
                    agent=self.agent,
                    experiment_id=experiment_id,
                    round_idx=round_idx,
                    total_rounds=max_rounds,
                    best_score=best_score,
                    baseline_score=baseline_score,
                    score_history=[rs.score for rs in score_history],
                ),
            )

            # ── Build OptimizationContext for the optimizer ───────
            opt_context = OptimizationContext(
                baseline_score=baseline_score,
                baseline_dims=dict(baseline_dims),
                current_score=best_score,
                current_dims=dict(best_dims),
                score_history=list(score_history),
                failure_clusters=failure_clusters_raw,
                eval_details=eval_results,
                dataset_summary=dataset_summary,
                metric_names=(
                    [wm.name for wm in metric.metrics]
                    if isinstance(metric, CompositeMetric)
                    else [getattr(metric, "name", "metric")]
                ),
                round_idx=round_idx,
                total_rounds=max_rounds,
            )

            # ── Step 4: Generate candidates ──────────────────────
            logger.info(
                "   💡 Step 4 — Proposing {} candidates",
                _CANDIDATES_PER_ROUND,
            )
            try:
                candidates: list[PromptCandidate] = await self._optimizer.propose(
                    current_config=best_config,
                    eval_results=eval_results,
                    dataset_sample=train_points[:20],
                    search_space=self._search_space,
                    iteration=round_idx,
                    context=opt_context,
                )
            except Exception as exc:
                logger.error("   Candidate proposal failed: {}", exc)
                no_improvement_rounds += 1
                if no_improvement_rounds >= _EARLY_STOP_PATIENCE:
                    logger.warning(
                        "   ⏹️  Early stop: {} rounds without improvement", _EARLY_STOP_PATIENCE
                    )
                    break
                continue

            if not candidates:
                logger.warning("   No candidates produced — skipping round")
                no_improvement_rounds += 1
                if no_improvement_rounds >= _EARLY_STOP_PATIENCE:
                    break
                continue

            logger.info("   Received {} candidate(s)", len(candidates))

            # ── Step 5: Score candidates on minibatch ────────────────
            logger.info("   ⚡ Step 5 — Minibatch scoring ({} pts)", len(minibatch_dataset))
            candidate_scores: list[tuple[PromptCandidate, float, dict[str, float]]] = []

            for cand in candidates:
                try:
                    cand_score, cand_dims, cand_details = await self._evaluate_config(
                        cand.config,
                        minibatch_dataset,
                        metric,
                    )
                    cand.composite_score = cand_score
                    cand.scores = dict(cand_dims)
                    candidate_scores.append((cand, cand_score, cand_dims))

                    trials.append(
                        {
                            "round": round_num,
                            "name": cand.name,
                            "source": cand.source,
                            "phase": "minibatch",
                            "score": cand_score,
                            "dimensions": dict(cand_dims),
                            "mutation_notes": cand.mutation_notes,
                        }
                    )
                    await self._callbacks.emit(
                        OptimizationEvent.CANDIDATE_EVALUATED,
                        EventData(
                            agent=self.agent,
                            experiment_id=experiment_id,
                            round_idx=round_idx,
                            candidate_name=cand.name,
                            candidate_source=cand.source,
                            score=cand_score,
                            best_score=best_score,
                            dimensions=dict(cand_dims),
                        ),
                    )
                    logger.info(
                        "      {} ({}): {:.4f}",
                        cand.name,
                        cand.source,
                        cand_score,
                    )
                except Exception as exc:
                    logger.warning(
                        "      {} evaluation failed: {}",
                        cand.name,
                        exc,
                    )

            if not candidate_scores:
                logger.warning("   All candidates failed — skipping round")
                no_improvement_rounds += 1
                if no_improvement_rounds >= _EARLY_STOP_PATIENCE:
                    break
                continue

            # ── Step 6: Promote top candidates to full valset ────────
            candidate_scores.sort(key=lambda t: t[1], reverse=True)
            promotable = [
                (cand, sc, dims) for cand, sc, dims in candidate_scores if sc > best_score
            ][:_TOP_K_CANDIDATES]

            if not promotable:
                logger.info("   No candidates beat current best ({:.4f})", best_score)
                no_improvement_rounds += 1
                if no_improvement_rounds >= _EARLY_STOP_PATIENCE:
                    logger.warning(
                        "   ⏹️  Early stop: {} rounds without improvement",
                        _EARLY_STOP_PATIENCE,
                    )
                    break
                continue

            logger.info(
                "   🏆 Step 6 — Full evaluation for {} promoted candidate(s)", len(promotable)
            )

            round_improved = False
            for cand, mini_score, _ in promotable:
                try:
                    full_score, full_dims, full_details = await self._evaluate_config(
                        cand.config,
                        valset,
                        metric,
                    )
                    trials.append(
                        {
                            "round": round_num,
                            "name": cand.name,
                            "source": cand.source,
                            "phase": "full_val",
                            "score": full_score,
                            "dimensions": dict(full_dims),
                        }
                    )
                    logger.info(
                        "      {} full-val: {:.4f} (minibatch was {:.4f})",
                        cand.name,
                        full_score,
                        mini_score,
                    )

                    # ── Step 7: Dimensional acceptance ───────────────
                    comparisons = self._dimension_analyzer.compare(
                        best_dims,
                        full_dims,
                    )
                    accept, reason = self._dimension_analyzer.should_accept(
                        comparisons,
                        min_composite_delta=self.min_absolute_improvement,
                        composite_baseline=best_score,
                        composite_candidate=full_score,
                    )

                    if accept:
                        logger.info(
                            "      ✅ Accepted: {} — {}",
                            cand.name,
                            reason,
                        )
                        await self._callbacks.emit(
                            OptimizationEvent.CANDIDATE_ACCEPTED,
                            EventData(
                                agent=self.agent,
                                experiment_id=experiment_id,
                                round_idx=round_idx,
                                candidate_name=cand.name,
                                score=full_score,
                                best_score=best_score,
                                accept_reason=reason,
                                improvement=(full_score - best_score),
                            ),
                        )
                        best_config = cand.config
                        best_score = full_score
                        best_dims = dict(full_dims)
                        eval_results = self._build_eval_results(full_details, metric)
                        round_improved = True
                    else:
                        logger.info(
                            "      ❌ Rejected: {} — {}",
                            cand.name,
                            reason,
                        )
                        await self._callbacks.emit(
                            OptimizationEvent.CANDIDATE_REJECTED,
                            EventData(
                                agent=self.agent,
                                experiment_id=experiment_id,
                                round_idx=round_idx,
                                candidate_name=cand.name,
                                score=full_score,
                                best_score=best_score,
                                accept_reason=reason,
                            ),
                        )
                        dim_table = self._dimension_analyzer.format_table(comparisons)
                        logger.debug(
                            "      Dimension table:\n{}",
                            dim_table,
                        )

                except Exception as exc:
                    logger.warning(
                        "      Full evaluation of {} failed: {}",
                        cand.name,
                        exc,
                    )

            if round_improved:
                no_improvement_rounds = 0
                logger.info(
                    "   📈 Round {} best: {:.4f} (Δ {:.4f} vs baseline)",
                    round_num,
                    best_score,
                    best_score - baseline_score,
                )
            else:
                no_improvement_rounds += 1
                if no_improvement_rounds >= _EARLY_STOP_PATIENCE:
                    logger.warning(
                        "   ⏹️  Early stop: {} rounds without improvement",
                        _EARLY_STOP_PATIENCE,
                    )
                    await self._callbacks.emit(
                        OptimizationEvent.EARLY_STOP,
                        EventData(
                            agent=self.agent,
                            experiment_id=experiment_id,
                            round_idx=round_idx,
                            best_score=best_score,
                        ),
                    )
                    break

            round_elapsed = time.perf_counter() - round_t0
            logger.info(
                "   ⏱️  Round {} completed in {:.1f}s",
                round_num,
                round_elapsed,
            )
            score_history.append(
                RoundStats(
                    round_idx=round_idx,
                    score=best_score,
                    dims=dict(best_dims),
                    accepted=round_improved,
                    n_candidates=len(candidates) if candidates else 0,
                    elapsed_seconds=round_elapsed,
                )
            )
            await self._callbacks.emit(
                OptimizationEvent.ROUND_END,
                EventData(
                    agent=self.agent,
                    experiment_id=experiment_id,
                    round_idx=round_idx,
                    total_rounds=max_rounds,
                    score=best_score,
                    best_score=best_score,
                    baseline_score=baseline_score,
                    elapsed_seconds=round_elapsed,
                    score_history=[rs.score for rs in score_history],
                ),
            )

        # ── Step 8: Produce param suggestions ────────────────────────
        logger.info("📝 Step 8/10 — Producing param suggestions")
        param_suggestions, suggestions = self._build_param_suggestions(
            baseline_config,
            best_config,
        )

        # Compute metric deltas
        metric_deltas: dict[str, float] = {}
        for dim in sorted(set(baseline_dims) | set(best_dims)):
            delta = best_dims.get(dim, 0.0) - baseline_dims.get(dim, 0.0)
            if abs(delta) > 1e-6:
                metric_deltas[dim] = round(delta, 4)
        metric_deltas["composite"] = round(best_score - baseline_score, 4)

        # ── Step 9: Validate on testset ──────────────────────────────
        test_score: float | None = None
        if testset is not None:
            logger.info("🧪 Step 9/10 — Test-set validation ({} pts)", len(testset))
            try:
                test_score_val, test_dims, _ = await self._evaluate_config(
                    best_config,
                    testset,
                    metric,
                )
                test_score = test_score_val
                logger.info("🧪 Test score: {:.4f}", test_score)
                if test_dims:
                    for dim, val in test_dims.items():
                        logger.debug("   {}: {:.4f}", dim, val)
            except Exception as exc:
                logger.warning("Test-set evaluation failed: {}", exc)
        else:
            logger.info("⏭️  Step 9/10 — No testset provided, skipping")

        # ── Step 10: Build and return PromptFitResult ────────────────
        duration = time.perf_counter() - t0
        logger.info("📦 Step 10/10 — Building PromptFitResult")

        result = PromptFitResult(
            best_config=best_config,
            baseline_config=baseline_config,
            best_score=best_score,
            baseline_score=baseline_score,
            metric_deltas=metric_deltas,
            param_suggestions=param_suggestions,
            failure_clusters=failure_clusters_raw,
            trials=trials,
            suggestions=suggestions,
            duration_seconds=round(duration, 2),
            experiment_id=experiment_id,
            agent=self.agent,
        )

        # Build deployment recommendation
        try:
            from agentomatic.optimize.deployment import build_deployment_recommendation

            result.deployment_recommendation = build_deployment_recommendation(
                result,
                version=f"v2_fit_{experiment_id}",
            )
            logger.info(
                "🚀 Deployment: {} (confidence: {}, rollout: {} @ {:.0%})",
                result.deployment_recommendation.prompt_version,
                result.deployment_recommendation.confidence,
                result.deployment_recommendation.rollout.strategy,
                result.deployment_recommendation.rollout.initial_weight,
            )
        except Exception as exc:
            logger.warning("Deployment recommendation failed: {}", exc)

        # Auto-generate HTML report
        if self.auto_report:
            try:
                from agentomatic.optimize.report import generate_html_report

                report_dir = Path(self.experiment_dir) / self.agent
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = generate_html_report(
                    result,
                    output_path=report_dir / f"fitter_report_{experiment_id}.html",
                )
                logger.info("📄 Report: {}", report_path)
            except Exception as exc:
                logger.warning("Report generation failed: {}", exc)

        # Save experiment artefacts
        try:
            artefact_dir = Path(self.experiment_dir) / self.agent
            artefact_dir.mkdir(parents=True, exist_ok=True)
            artefact_path = artefact_dir / f"fit_result_{experiment_id}.json"
            artefact_path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info("💾 Artefacts: {}", artefact_path)
        except Exception as exc:
            logger.warning("Failed to save artefacts: {}", exc)

        # Final summary
        logger.info("\n{}", result.summary())

        if test_score is not None:
            logger.info("🧪 Test score: {:.4f}", test_score)

        await self._callbacks.emit(
            OptimizationEvent.FIT_COMPLETE,
            EventData(
                agent=self.agent,
                experiment_id=experiment_id,
                score=best_score,
                baseline_score=baseline_score,
                best_score=best_score,
                improvement=best_score - baseline_score,
                elapsed_seconds=time.perf_counter() - t0,
            ),
        )

        return result

    # ================================================================
    # Private helpers
    # ================================================================

    async def _evaluate_config(
        self,
        config: PromptRuntimeConfig,
        dataset: Dataset,
        metric: CompositeMetric | BaseMetric,
    ) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
        """Evaluate a runtime config against a dataset.

        Runs every data point through the agent with the config's
        system prompt as an override, then scores each response.

        Parameters
        ----------
        config : PromptRuntimeConfig
            Configuration to evaluate.
        dataset : Dataset
            Data points to run.
        metric : CompositeMetric | BaseMetric
            Evaluation metric.

        Returns
        -------
        tuple[float, dict[str, float], list[dict[str, Any]]]
            ``(avg_score, per_dimension_scores, eval_details)`` where
            *eval_details* is a list of dicts with per-point results.
        """
        points = [dp.to_dict() for dp in dataset]

        run_results: list[RunResult] = await self._runner.run_dataset(
            points,
            prompt_override=config.system_prompt,
            concurrency=self.concurrency,
        )

        scores: list[float] = []
        dim_accumulators: dict[str, list[float]] = {}
        eval_details: list[dict[str, Any]] = []

        for rr in run_results:
            if rr.error:
                logger.debug("Skipping errored result for '{}'", rr.query[:50])
                eval_details.append(
                    {
                        "query": rr.query,
                        "response": rr.response,
                        "expected": rr.expected,
                        "avg_score": 0.0,
                        "error": rr.error,
                        "details": [],
                    }
                )
                scores.append(0.0)
                continue

            try:
                eval_result: EvalResult = await metric.evaluate(
                    query=rr.query,
                    response=rr.response,
                    expected=rr.expected,
                    context=rr.context or rr.retrieval_context,
                )
                point_score = eval_result.score
                scores.append(point_score)

                # Extract per-dimension scores from composite metrics
                point_dims: dict[str, float] = {}
                if isinstance(metric, CompositeMetric):
                    point_dims = eval_result.metadata.get("dimensions", {})
                    for dim, val in point_dims.items():
                        dim_accumulators.setdefault(dim, []).append(val)

                eval_details.append(
                    {
                        "query": rr.query,
                        "response": rr.response,
                        "expected": rr.expected,
                        "avg_score": point_score,
                        "dimensions": point_dims,
                        "feedback": eval_result.reason,
                        "details": [
                            {
                                "metric": eval_result.metric_name,
                                "score": eval_result.score,
                                "reason": eval_result.reason,
                            },
                        ],
                    }
                )
            except Exception as exc:
                logger.warning("Metric evaluation failed for '{}': {}", rr.query[:50], exc)
                scores.append(0.0)
                eval_details.append(
                    {
                        "query": rr.query,
                        "response": rr.response,
                        "expected": rr.expected,
                        "avg_score": 0.0,
                        "error": str(exc),
                        "details": [],
                    }
                )

        avg_score = sum(scores) / len(scores) if scores else 0.0
        per_dim: dict[str, float] = {
            dim: sum(vals) / len(vals) for dim, vals in dim_accumulators.items() if vals
        }

        return avg_score, per_dim, eval_details

    def _load_baseline_config(self) -> PromptRuntimeConfig:
        """Load baseline config from the agent's ``prompts.json``.

        Searches for ``prompts.json`` in standard locations
        (``agents/<agent>/``, ``<agent>/``, cwd).  Falls back to a
        generic system prompt if no file is found.

        Returns
        -------
        PromptRuntimeConfig
            Baseline configuration with system prompt and default
            model parameters from the search space.
        """
        system_prompt = self._load_prompt_text()

        # Build default model_params from search space (first value of each)
        model_params: dict[str, Any] = {}
        if self._search_space.optimize_model_params:
            for param, values in self._search_space.model_param_space.items():
                if values:
                    model_params[param] = values[0]

        return PromptRuntimeConfig(
            system_prompt=system_prompt,
            model_params=model_params,
        )

    def _load_prompt_text(self) -> str:
        """Locate and read the system prompt from ``prompts.json``.

        Mirrors the logic from :meth:`PromptOptimizer._load_prompt`:
        searches common directories for the agent's prompt file.
        """
        search_dirs = [
            Path(f"agents/{self.agent}"),
            Path(self.agent),
            Path("."),
        ]

        for search_dir in search_dirs:
            prompts_file = search_dir / "prompts.json"
            if not prompts_file.exists():
                continue

            try:
                data = json.loads(prompts_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not parse {}: {}",
                    prompts_file,
                    exc,
                )
                continue

            # Try the specified base version first, then common fallbacks
            for version in [self.base_prompt_version, "v1", "default"]:
                entry = data.get(version)
                if isinstance(entry, dict):
                    # Support both "system" and "system_prompt" keys
                    prompt = entry.get("system") or entry.get("system_prompt")
                    if prompt:
                        logger.info(
                            "Loaded prompt version '{}' from {}",
                            version,
                            prompts_file,
                        )
                        return str(prompt)

            # Fallback: first entry with a system prompt
            for key, val in data.items():
                if isinstance(val, dict):
                    prompt = val.get("system") or val.get("system_prompt")
                    if prompt:
                        logger.info(
                            "Loaded prompt from first entry '{}' in {}",
                            key,
                            prompts_file,
                        )
                        return str(prompt)

        logger.warning(
            "No prompts.json found for agent '{}' — using default prompt",
            self.agent,
        )
        return "You are a helpful AI assistant."

    def _build_eval_results(
        self,
        eval_details: list[dict[str, Any]],
        metric: CompositeMetric | BaseMetric,
    ) -> list[dict[str, Any]]:
        """Convert eval details into the format expected by optimizers.

        Parameters
        ----------
        eval_details : list[dict[str, Any]]
            Per-point evaluation details from :meth:`_evaluate_config`.
        metric : CompositeMetric | BaseMetric
            The metric used for evaluation (for name extraction).

        Returns
        -------
        list[dict[str, Any]]
            Optimizer-compatible evaluation results.
        """
        results: list[dict[str, Any]] = []
        for detail in eval_details:
            results.append(
                {
                    "query": detail.get("query", ""),
                    "response": detail.get("response", ""),
                    "expected": detail.get("expected"),
                    "score": detail.get("avg_score", 0.0),
                    "dimensions": detail.get("dimensions", {}),
                    "feedback": detail.get("feedback", ""),
                    "is_failure": detail.get("avg_score", 0.0) < _FAILURE_THRESHOLD,
                }
            )
        return results

    def _build_param_suggestions(
        self,
        baseline: PromptRuntimeConfig,
        best: PromptRuntimeConfig,
    ) -> tuple[dict[str, ParamDelta], list[str]]:
        """Compare baseline and best configs to produce change suggestions.

        Parameters
        ----------
        baseline : PromptRuntimeConfig
            Original configuration.
        best : PromptRuntimeConfig
            Optimised configuration.

        Returns
        -------
        tuple[dict[str, ParamDelta], list[str]]
            ``(param_suggestions, text_suggestions)`` — structured
            deltas and human-readable descriptions.
        """
        param_suggestions: dict[str, ParamDelta] = {}
        text_suggestions: list[str] = []
        diff = best.diff(baseline)

        if "system_prompt" in diff:
            old_len = len(str(diff["system_prompt"]["old"]))
            new_len = len(str(diff["system_prompt"]["new"]))
            reason = f"System prompt revised ({old_len} → {new_len} chars)"
            param_suggestions["system_prompt"] = ParamDelta(
                param_name="system_prompt",
                old_value=f"[{old_len} chars]",
                new_value=f"[{new_len} chars]",
                reason=reason,
            )
            text_suggestions.append(reason)

        if "few_shot_examples" in diff:
            old_count = len(diff["few_shot_examples"]["old"] or [])
            new_count = len(diff["few_shot_examples"]["new"] or [])
            reason = f"Few-shot examples changed ({old_count} → {new_count})"
            param_suggestions["few_shot_examples"] = ParamDelta(
                param_name="few_shot_examples",
                old_value=old_count,
                new_value=new_count,
                reason=reason,
            )
            text_suggestions.append(reason)

        if "model_params" in diff:
            old_params = diff["model_params"]["old"] or {}
            new_params = diff["model_params"]["new"] or {}
            all_keys = sorted(set(old_params) | set(new_params))
            for key in all_keys:
                old_val = old_params.get(key)
                new_val = new_params.get(key)
                if old_val != new_val:
                    reason = f"Model param '{key}': {old_val} → {new_val}"
                    param_suggestions[f"model_params.{key}"] = ParamDelta(
                        param_name=f"model_params.{key}",
                        old_value=old_val,
                        new_value=new_val,
                        reason=reason,
                    )
                    text_suggestions.append(reason)

        if "user_template" in diff:
            reason = "User template was modified"
            param_suggestions["user_template"] = ParamDelta(
                param_name="user_template",
                old_value=diff["user_template"]["old"],
                new_value=diff["user_template"]["new"],
                reason=reason,
            )
            text_suggestions.append(reason)

        if "output_contract" in diff:
            reason = "Output contract was modified"
            param_suggestions["output_contract"] = ParamDelta(
                param_name="output_contract",
                old_value=diff["output_contract"]["old"],
                new_value=diff["output_contract"]["new"],
                reason=reason,
            )
            text_suggestions.append(reason)

        if "rag_params" in diff:
            text_suggestions.append("RAG parameters were adjusted")

        if "tool_params" in diff:
            text_suggestions.append("Tool parameters were adjusted")

        if not text_suggestions:
            text_suggestions.append("No configuration changes improved over the baseline.")

        return param_suggestions, text_suggestions
