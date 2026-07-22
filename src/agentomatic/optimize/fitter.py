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
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from agentomatic.optimize.llm_types import LLMSpec

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

_SATURATION_SCORE: float = 0.995
"""Baseline at/above this is treated as a saturated (non-informative) metric."""

_MIN_RELIABLE_TRAIN: int = 4
"""Below this train size, emit an honest tiny-data warning."""

_MIN_RELIABLE_VAL: int = 2
"""Below this fit-val size, emit an honest tiny-data warning."""

_CANDIDATES_PER_ROUND: int = 4
"""Number of candidates proposed per optimisation round."""


# =====================================================================
# Local agent wrapper
# =====================================================================


def _wrap_local_agent(agent: Any) -> Any:
    """Return an async callable compatible with :class:`AgentRunner`.

    Adapts a :class:`~agentomatic.agents.base.BaseGraphAgent` instance
    (or any object with ``transform()`` / ``atransform()`` / ``invoke()``)
    to the signature::

        async fn(query, *, prompt_override, context, invoke) -> str

    The *prompt_override* is injected via two mechanisms so that agents
    which read from state metadata honour it:

    1. Temporarily set ``agent.system_prompt = prompt_override`` (if the
       attribute exists) and restore the original value after the call.
    2. Include ``metadata.system_prompt_override`` in the input dict so
       graph nodes that inspect metadata also receive the override.

    The output dict is serialised to JSON; plain string outputs are
    returned as-is.
    """
    import asyncio
    import inspect
    import json as _json

    async def _callable(
        query: str,
        *,
        prompt_override: str | None = None,
        context: list | None = None,
        invoke: dict | None = None,
    ) -> str:
        input_data: dict[str, Any] = {"current_query": query}
        if invoke:
            input_data.update({k: v for k, v in invoke.items() if k != "current_query"})
        if context:
            input_data.setdefault("context", {})
            if isinstance(input_data["context"], dict):
                input_data["context"]["documents"] = list(context)
        if prompt_override:
            # Top-level + metadata so resolve_system_prompt / transform stash both see it.
            input_data["system_prompt_override"] = prompt_override
            input_data.setdefault("metadata", {})
            if isinstance(input_data["metadata"], dict):
                input_data["metadata"]["system_prompt_override"] = prompt_override

        # Surface model_params / temperature for agents that honour them.
        model_params = None
        if isinstance(invoke, dict):
            model_params = invoke.get("model_params")
            if model_params is None and "temperature" in invoke:
                model_params = {"temperature": invoke["temperature"]}
        if isinstance(model_params, dict) and model_params:
            input_data["model_params"] = model_params
            input_data.setdefault("metadata", {})
            if isinstance(input_data["metadata"], dict):
                input_data["metadata"]["model_params"] = model_params
            if "temperature" in model_params:
                input_data.setdefault("temperature", model_params["temperature"])

        # Temporarily override the agent's system_prompt attribute so that
        # nodes which read self.system_prompt pick up the candidate prompt.
        original_prompt: str | None = None
        has_prompt_attr = hasattr(agent, "system_prompt") and prompt_override is not None
        if has_prompt_attr:
            original_prompt = agent.system_prompt  # type: ignore[union-attr]
            try:
                agent.system_prompt = prompt_override  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                has_prompt_attr = False

        original_compiled: dict[str, Any] | None = None
        if isinstance(model_params, dict) and model_params and hasattr(agent, "compiled_config"):
            try:
                original_compiled = dict(getattr(agent, "compiled_config") or {})
                merged = dict(original_compiled)
                merged.update(model_params)
                agent.compiled_config = merged  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                original_compiled = None

        try:
            if hasattr(agent, "atransform") and inspect.iscoroutinefunction(agent.atransform):
                output = await agent.atransform(input_data)
            elif hasattr(agent, "transform"):
                output = await asyncio.to_thread(agent.transform, input_data)
            elif hasattr(agent, "invoke"):
                output = await asyncio.to_thread(agent.invoke, input_data)
            else:
                raise TypeError(f"Agent {type(agent)} has no transform/invoke method")
        finally:
            if has_prompt_attr and original_prompt is not None:
                try:
                    agent.system_prompt = original_prompt  # type: ignore[union-attr]
                except (AttributeError, TypeError):
                    pass
            if original_compiled is not None:
                try:
                    agent.compiled_config = original_compiled  # type: ignore[union-attr]
                except (AttributeError, TypeError):
                    pass

        if isinstance(output, dict):
            # Prefer structured output, else response field, else full dict
            inner = output.get("output")
            if isinstance(inner, dict) and inner:
                return _json.dumps(inner, ensure_ascii=False)
            response = output.get("response")
            if response is not None:
                return str(response)
            return _json.dumps(output, ensure_ascii=False)
        return str(output) if output is not None else ""

    return _callable


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
        Minimum composite-score lift for a candidate to be accepted
        (default ``0.001`` — small enough for LLM-judge noise, large enough
        to ignore pure float jitter).
    patience : int | None
        Rounds without improvement before early-stop (default module constant).
    api_base : str
        Base URL of the agent API server (ignored when *local_agent* is set).
    api_prefix : str
        URL prefix for the agent API endpoints (ignored when *local_agent* is set).
    concurrency : int
        Maximum concurrent agent requests.
    experiment_dir : str
        Directory for experiment artefacts and reports.
    auto_report : bool
        Whether to generate an HTML report at the end.
    local_agent : Any | None
        A live agent instance (e.g. a :class:`~agentomatic.agents.base.BaseGraphAgent`
        subclass).  When provided, **no HTTP server is required** — every
        evaluation invokes the agent's ``transform()`` / ``atransform()``
        method directly.  The runner builds a lightweight async wrapper
        via :func:`_wrap_local_agent`.
    llm_base_url : str | None
        Base URL for the OpenAI-compatible server used by the **optimizer**
        LLM (proposal rewriting, failure clustering, etc.).  Sets
        :meth:`~agentomatic.optimize.llm_caller.LLMCaller.configure` so
        all subsequent ``openai/`` model calls are routed correctly.
        Example: ``"http://127.0.0.1:8000/v1"`` for a local omlx/vLLM
        server.
    llm_api_key : str | None
        API key for the optimizer LLM server (may be arbitrary for local
        servers, e.g. ``"any-key"``).
    rewrite_passes : int | None
        Multi-pass refine count for :class:`RewriteOptimizer`. ``None`` =
        auto (3 for SLMs / local providers, 2 for frontier LLMs).
    multipass : bool
        Master switch for auto multi-pass refine (default True).
    slm_multipass : bool
        When True (default), auto multi-pass for small/local rewrite models
        (``omlx/``, ``ollama/``, ``7b``, …).
    llm_multipass : bool
        When True (default), auto multi-pass for frontier / cloud rewrite
        LLMs (``openai/``, ``anthropic/``, …).
    slm_default_passes : int
        Pass count when SLM auto-detect fires (default 3).
    llm_default_passes : int
        Pass count when LLM auto-detect fires (default 2).

    Examples
    --------
    >>> fitter = PromptFitter(
    ...     agent="scope_agent",
    ...     task_model="ollama/qwen2.5:7b",
    ...     rewrite_model="openai/gpt-4.1",
    ...     optimizer="rewrite",
    ...     # frontier rewrite LLM → draft + self-check (2 passes)
    ... )
    >>> result = await fitter.fit(trainset, valset, metric)
    >>> print(result.summary())
    >>> result.apply(version="v2_fit")

    Local-only (no HTTP server needed)::

    >>> fitter = PromptFitter(
    ...     agent="assistant",
    ...     task_model="openai/LFM2.5-8B-A1B-MLX-4bit",
    ...     local_agent=my_agent_instance,
    ...     llm_base_url="http://127.0.0.1:8000/v1",
    ...     llm_api_key="any-key",
    ...     optimizer="rewrite",
    ...     # SLMs get draft→critique→revise automatically
    ... )
    """

    def __init__(
        self,
        agent: str,
        base_prompt_version: str = "v1",
        task_model: LLMSpec = "ollama/qwen2.5:7b",
        rewrite_model: LLMSpec | None = None,
        local_judges: list[str] | None = None,
        search_space: PromptSearchSpace | None = None,
        optimizer: str | Any = "gepa_like",
        max_trials: int = 30,
        min_absolute_improvement: float = 0.001,
        patience: int | None = None,
        api_base: str = "http://localhost:8000",
        api_prefix: str = "/api/v1",
        concurrency: int = 1,
        experiment_dir: str = ".optimize",
        auto_report: bool = True,
        callbacks: list[OptimizationCallback] | None = None,
        dashboard: bool = False,
        local_agent: Any | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        rewrite_passes: int | None = None,
        multipass: bool = True,
        slm_multipass: bool = True,
        llm_multipass: bool = True,
        slm_default_passes: int = 3,
        llm_default_passes: int = 2,
        baseline_system_prompt: str | None = None,
        max_generalization_gap: float = 0.15,
        holdout_fraction: float = 0.2,
        drain_seconds: float = 1.5,
        sequential: bool | None = None,
    ) -> None:
        # Public configuration
        self.agent = agent
        self.base_prompt_version = base_prompt_version
        self.task_model = task_model
        self.rewrite_model = rewrite_model or task_model
        self.local_judges = local_judges or []
        self.max_trials = max_trials
        self.min_absolute_improvement = min_absolute_improvement
        self.patience = (
            int(patience) if patience is not None else _EARLY_STOP_PATIENCE
        )
        # Default concurrency=1 / sequential=True: local SLMs often spawn many
        # idle worker threads when hammered concurrently. Explicit concurrency>1
        # auto-disables sequential unless sequential=True is forced.
        if sequential is None:
            self.sequential = concurrency <= 1
        else:
            self.sequential = sequential
        self.concurrency = 1 if self.sequential else max(1, concurrency)
        self.experiment_dir = experiment_dir
        self.auto_report = auto_report
        self.rewrite_passes = rewrite_passes
        self.multipass = multipass
        self.slm_multipass = slm_multipass
        self.llm_multipass = llm_multipass
        self.slm_default_passes = slm_default_passes
        self.llm_default_passes = llm_default_passes
        self.max_generalization_gap = max_generalization_gap
        self.holdout_fraction = holdout_fraction
        self.drain_seconds = drain_seconds
        # Optional pre-set baseline: overrides prompts.json on first step.
        # Used by PromptFitterBridge to compound improvements across epochs.
        self._baseline_system_prompt: str | None = baseline_system_prompt

        # Configure LLMCaller for OpenAI-compatible local servers if requested.
        # This propagates base_url/api_key to all optimizer LLM calls automatically.
        if llm_base_url or llm_api_key:
            from agentomatic.optimize.llm_caller import LLMCaller

            LLMCaller.configure(base_url=llm_base_url, api_key=llm_api_key)

        # Runner for agent invocations
        agent_callable = _wrap_local_agent(local_agent) if local_agent is not None else None
        self._runner = AgentRunner(
            agent=agent,
            api_base=api_base,
            api_prefix=api_prefix,
            agent_callable=agent_callable,
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
            rewrite_passes=rewrite_passes,
            multipass=multipass,
            slm_multipass=slm_multipass,
            llm_multipass=llm_multipass,
            slm_default_passes=slm_default_passes,
            llm_default_passes=llm_default_passes,
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
        learnings_history: list[Any] = []
        early_stop_reason: str | None = None

        logger.info(
            "🚀 PromptFitter.fit — experiment={} agent={} max_trials={} "
            "concurrency={} sequential={}",
            experiment_id,
            self.agent,
            self.max_trials,
            self.concurrency,
            self.sequential,
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

        metric = self._augment_metric_with_local_judges(metric)

        # ── Generalization safety net: always reserve a holdout ──────
        from agentomatic.optimize.learning import (
            check_generalization,
            split_holdout,
            synthesize_epoch_learning,
        )

        fit_valset = valset
        holdout_set: Dataset | None = testset
        if holdout_set is None:

            def _as_dicts(pts: list[Any]) -> list[dict[str, Any]]:
                return [p.to_dict() if hasattr(p, "to_dict") else p for p in pts]

            fit_pts, hold_pts = split_holdout(
                list(valset),
                fraction=self.holdout_fraction,
                min_size=1,
            )
            holdout_source = "valset"
            # Tiny valsets: borrow holdout from train so generalization
            # safety stays always-on (never silently skip).
            if not hold_pts and trainset is not None and len(trainset) >= 2:
                _fit_train, hold_pts = split_holdout(
                    list(trainset),
                    fraction=self.holdout_fraction,
                    min_size=1,
                )
                fit_pts = list(valset)
                holdout_source = "trainset"
            if hold_pts:
                # Rebuild fit valset only when we carved holdout out of it.
                if holdout_source == "valset":
                    fit_valset = Dataset.from_list(_as_dicts(fit_pts))
                holdout_set = Dataset.from_list(_as_dicts(hold_pts))
                logger.info(
                    "🛡️  Generalization holdout: {} fit / {} holdout (from {})",
                    len(fit_valset),
                    len(holdout_set),
                    holdout_source,
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
            len(fit_valset),
        )
        baseline_score, baseline_dims, baseline_details = await self._evaluate_config(
            baseline_config,
            fit_valset,
            metric,
        )
        logger.info("📊 Baseline score: {:.4f}", baseline_score)
        if baseline_dims:
            for dim, val in baseline_dims.items():
                logger.debug("   {}: {:.4f}", dim, val)

        saturation_warning = ""
        if baseline_score >= _SATURATION_SCORE:
            saturation_warning = (
                f"Baseline already saturated at {baseline_score:.4f}. "
                "The fit metric is too easy (or the dataset is trivial) — "
                "prompt candidates cannot show improvement. Harden the metric "
                "(content overlap / must_include / judge rubric) and expand demos."
            )
            logger.warning("⚠️  {}", saturation_warning)

        tiny_data_warning = ""
        if len(trainset) < _MIN_RELIABLE_TRAIN or len(fit_valset) < _MIN_RELIABLE_VAL:
            tiny_data_warning = (
                f"Dataset too small for reliable prompt learning "
                f"(train={len(trainset)}, fit_val={len(fit_valset)}; "
                f"recommend ≥{_MIN_RELIABLE_TRAIN} train / ≥{_MIN_RELIABLE_VAL} val). "
                "Scores may look flat even when the rewrite path works."
            )
            logger.warning("⚠️  {}", tiny_data_warning)

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

        # Baseline holdout score (generalization reference)
        baseline_holdout_score: float | None = None
        if holdout_set is not None and len(holdout_set) > 0:
            try:
                baseline_holdout_score, _, _ = await self._evaluate_config(
                    baseline_config,
                    holdout_set,
                    metric,
                )
                logger.info(
                    "🛡️  Baseline holdout score: {:.4f} (gap vs fit: {:+.4f})",
                    baseline_holdout_score,
                    baseline_score - baseline_holdout_score,
                )
            except Exception as exc:
                logger.warning("Baseline holdout evaluation failed: {}", exc)

        # ── Iterative optimisation loop (Steps 4-7) ──────────────────
        best_config = baseline_config
        best_score = baseline_score
        best_dims = dict(baseline_dims)
        best_holdout_score = baseline_holdout_score
        no_improvement_rounds = 0

        max_rounds = max(1, self.max_trials // _CANDIDATES_PER_ROUND)
        logger.info(
            "🔄 Starting optimisation: {} rounds × {} candidates = {} max evals",
            max_rounds,
            _CANDIDATES_PER_ROUND,
            max_rounds * _CANDIDATES_PER_ROUND,
        )

        # Prepare dataset points for the runner/optimizer (fit split only)
        val_points = [dp.to_dict() for dp in fit_valset]
        train_points = [dp.to_dict() for dp in trainset]

        # Prepare minibatch — clamp to available data so we never claim
        # more points than exist (avoids misleading "500%" log messages).
        minibatch_size = min(
            len(val_points),
            max(
                _MINIBATCH_MIN,
                int(len(val_points) * _MINIBATCH_FRACTION),
            ),
        )
        minibatch_size = max(1, minibatch_size)  # Always at least 1 if data exists
        minibatch_points = val_points[:minibatch_size]
        minibatch_dataset = Dataset.from_list(minibatch_points)

        # Build dataset summary for context
        dataset_summary = DatasetSummary(
            n_samples=len(val_points),
            avg_query_length=(
                sum(len(str(p.get("query", ""))) for p in val_points) // max(len(val_points), 1)
            ),
            avg_expected_length=(
                sum(
                    len(str(p.get("expected_answer") or p.get("expected") or ""))
                    for p in val_points
                )
                // max(len(val_points), 1)
            ),
        )

        logger.info(
            "   Minibatch: {} / {} points ({:.0%})",
            minibatch_size,
            len(val_points),
            minibatch_size / max(len(val_points), 1),
        )

        eval_results = self._build_eval_results(baseline_details, metric)

        # Effective patience: never exceed max_rounds so early stopping can
        # still fire, but honour the configured ``self.patience`` (wired from
        # TrainConfig) instead of always using the module default.
        effective_patience = min(max(1, self.patience), max(1, max_rounds))

        # Seed curve with baseline so reports always show a trajectory start.
        score_history.append(
            RoundStats(
                round_idx=-1,
                score=baseline_score,
                dims=dict(baseline_dims),
                accepted=False,
                n_candidates=0,
                elapsed_seconds=0.0,
            )
        )

        def _record_round(
            *,
            round_idx: int,
            round_t0: float,
            round_improved: bool,
            accepted_name: str,
            n_candidates: int,
        ) -> None:
            """Always append score + prompt history for Keras-style curves."""
            nonlocal eval_results
            round_elapsed = time.perf_counter() - round_t0
            logger.info(
                "   ⏱️  Round {} completed in {:.1f}s",
                round_idx + 1,
                round_elapsed,
            )
            score_history.append(
                RoundStats(
                    round_idx=round_idx,
                    score=best_score,
                    dims=dict(best_dims),
                    accepted=round_improved,
                    n_candidates=n_candidates,
                    elapsed_seconds=round_elapsed,
                )
            )
            epoch_learning = synthesize_epoch_learning(
                round_idx=round_idx,
                prompt_snapshot=best_config.system_prompt,
                score=best_score,
                dims=dict(best_dims),
                eval_details=eval_results,
                accepted=round_improved,
                candidate_name=accepted_name,
                train_score=best_score,
                holdout_score=best_holdout_score,
            )
            learnings_history.append(epoch_learning)
            logger.info(
                "   🧠 Epoch learnings: {} worked / {} failed / focus={}",
                len(epoch_learning.what_worked),
                len(epoch_learning.what_failed),
                "; ".join(epoch_learning.next_focus[:2]) or "n/a",
            )

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
                learnings_history=list(learnings_history),
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
                agent_name=self.agent,
            )

            # ── Step 4: Generate candidates ──────────────────────
            logger.info(
                "   💡 Step 4 — Proposing {} candidates",
                _CANDIDATES_PER_ROUND,
            )
            optimizer_name = getattr(self._optimizer, "name", "") or ""
            if any(token in optimizer_name for token in ("rewrite", "gepa", "mipro")):
                await self._callbacks.emit(
                    OptimizationEvent.REWRITE_START,
                    EventData(
                        agent=self.agent,
                        experiment_id=experiment_id,
                        round_idx=round_idx,
                        optimizer_name=optimizer_name,
                    ),
                )
            candidates: list[PromptCandidate] = []
            try:
                candidates = await self._optimizer.propose(
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
                _record_round(
                    round_idx=round_idx,
                    round_t0=round_t0,
                    round_improved=False,
                    accepted_name="",
                    n_candidates=0,
                )
                if no_improvement_rounds >= effective_patience:
                    logger.warning(
                        "   ⏹️  Early stop: {} rounds without improvement", effective_patience
                    )
                    break
                continue

            if not candidates:
                logger.warning("   No candidates produced — skipping round")
                no_improvement_rounds += 1
                _record_round(
                    round_idx=round_idx,
                    round_t0=round_t0,
                    round_improved=False,
                    accepted_name="",
                    n_candidates=0,
                )
                if no_improvement_rounds >= effective_patience:
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
                _record_round(
                    round_idx=round_idx,
                    round_t0=round_t0,
                    round_improved=False,
                    accepted_name="",
                    n_candidates=len(candidates),
                )
                if no_improvement_rounds >= effective_patience:
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
                _record_round(
                    round_idx=round_idx,
                    round_t0=round_t0,
                    round_improved=False,
                    accepted_name="",
                    n_candidates=len(candidates),
                )
                if no_improvement_rounds >= effective_patience:
                    logger.warning(
                        "   ⏹️  Early stop: {} rounds without improvement",
                        effective_patience,
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
                continue

            logger.info(
                "   🏆 Step 6 — Full evaluation for {} promoted candidate(s)", len(promotable)
            )

            round_improved = False
            accepted_name = ""
            for cand, mini_score, _ in promotable:
                try:
                    full_score, full_dims, full_details = await self._evaluate_config(
                        cand.config,
                        fit_valset,
                        metric,
                    )
                    # Always-on generalization check on holdout
                    cand_holdout: float | None = None
                    if holdout_set is not None and len(holdout_set) > 0:
                        try:
                            cand_holdout, _, _ = await self._evaluate_config(
                                cand.config,
                                holdout_set,
                                metric,
                            )
                        except Exception as exc:
                            logger.warning(
                                "      Holdout eval failed for {}: {}",
                                cand.name,
                                exc,
                            )
                    gen_check = check_generalization(
                        fit_score=full_score,
                        holdout_score=cand_holdout,
                        max_gap=self.max_generalization_gap,
                        min_holdout_improvement=0.0,
                        baseline_holdout=baseline_holdout_score,
                    )
                    trials.append(
                        {
                            "round": round_num,
                            "name": cand.name,
                            "source": cand.source,
                            "phase": "full_val",
                            "score": full_score,
                            "holdout_score": cand_holdout,
                            "generalization": gen_check.to_dict(),
                            "dimensions": dict(full_dims),
                            "system_prompt": getattr(cand.config, "system_prompt", "") or "",
                            "prompt_preview": (
                                (getattr(cand.config, "system_prompt", "") or "")[:240]
                            ),
                        }
                    )
                    logger.info(
                        "      {} full-val: {:.4f} (minibatch was {:.4f})"
                        + (f", holdout={cand_holdout:.4f}" if cand_holdout is not None else ""),
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
                    if accept and not gen_check.ok:
                        accept = False
                        reason = f"Generalization safety net: {gen_check.reason}"
                        logger.warning("      🛡️  {}", reason)

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
                        if "rewrite" in (cand.source or ""):
                            await self._callbacks.emit(
                                OptimizationEvent.REWRITE_ACCEPTED,
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
                        best_holdout_score = cand_holdout
                        eval_results = self._build_eval_results(full_details, metric)
                        round_improved = True
                        accepted_name = cand.name
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
                        if "rewrite" in (cand.source or ""):
                            await self._callbacks.emit(
                                OptimizationEvent.REWRITE_REJECTED,
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

            _record_round(
                round_idx=round_idx,
                round_t0=round_t0,
                round_improved=round_improved,
                accepted_name=accepted_name,
                n_candidates=len(candidates) if candidates else 0,
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
                    elapsed_seconds=time.perf_counter() - round_t0,
                    score_history=[rs.score for rs in score_history],
                ),
            )
            if not round_improved and no_improvement_rounds >= effective_patience:
                early_stop_reason = (
                    f"no improvement for {effective_patience} round(s) "
                    f"(patience={effective_patience}, monitor=best_score)"
                )
                logger.warning(
                    "   ⏹️  Early stop: {} rounds without improvement",
                    effective_patience,
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

        # ── Step 8: Produce param suggestions ────────────────────────
        logger.info("📝 Step 8/10 — Producing param suggestions")
        param_suggestions, suggestions = self._build_param_suggestions(
            baseline_config,
            best_config,
        )
        if saturation_warning:
            suggestions.insert(0, saturation_warning)
        if tiny_data_warning:
            suggestions.insert(0 if not saturation_warning else 1, tiny_data_warning)

        # Compute metric deltas
        metric_deltas: dict[str, float] = {}
        for dim in sorted(set(baseline_dims) | set(best_dims)):
            delta = best_dims.get(dim, 0.0) - baseline_dims.get(dim, 0.0)
            if abs(delta) > 1e-6:
                metric_deltas[dim] = round(delta, 4)
        metric_deltas["composite"] = round(best_score - baseline_score, 4)

        # ── Step 9: Final holdout / test-set validation ──────────────
        test_score: float | None = best_holdout_score
        if testset is not None and holdout_set is not testset:
            # Explicit testset was provided separately from auto-holdout
            logger.info("🧪 Step 9/10 — Test-set validation ({} pts)", len(testset))
            try:
                test_score_val, test_dims, _ = await self._evaluate_config(
                    best_config,
                    testset,
                    metric,
                )
                test_score = test_score_val
                best_holdout_score = test_score_val
                logger.info("🧪 Test score: {:.4f}", test_score)
                if test_dims:
                    for dim, val in test_dims.items():
                        logger.debug("   {}: {:.4f}", dim, val)
            except Exception as exc:
                logger.warning("Test-set evaluation failed: {}", exc)
        elif best_holdout_score is not None:
            logger.info(
                "🧪 Step 9/10 — Holdout generalization score: {:.4f}",
                best_holdout_score,
            )
        else:
            logger.info("⏭️  Step 9/10 — No holdout/testset available")

        # ── Step 10: Build and return PromptFitResult ────────────────
        duration = time.perf_counter() - t0
        logger.info("📦 Step 10/10 — Building PromptFitResult")

        gen_gap = None
        if best_holdout_score is not None:
            gen_gap = best_score - best_holdout_score

        if early_stop_reason is None:
            early_stop_reason = (
                f"completed all {max_rounds} optimize round(s) "
                f"(max_trials={self.max_trials})"
            )

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
            score_history=[rs.score for rs in score_history],
            prompt_history=[e.to_dict() for e in learnings_history],
            holdout_score=best_holdout_score,
            baseline_holdout_score=baseline_holdout_score,
            generalization_gap=gen_gap,
            optimizer_name=str(getattr(self._optimizer, "name", "") or ""),
            early_stop_reason=early_stop_reason,
            dataset_sizes={
                "train": len(trainset),
                "fit_val": len(fit_valset),
                "holdout": len(holdout_set) if holdout_set is not None else 0,
                "test": len(testset) if testset is not None else 0,
            },
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
                from agentomatic.optimize.report import generate_fit_report

                report_dir = Path(self.experiment_dir) / self.agent
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = generate_fit_report(
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

        # Persist auditable retrain artefacts into the platform store
        try:
            from agentomatic.optimize.fit_store import persist_fit_result

            await persist_fit_result(result, experiment_dir=self.experiment_dir)
        except Exception as exc:
            logger.debug("Optional fit store persist skipped: {}", exc)

        # Always write local retrain_history.jsonl for offline auditability
        try:
            artefact_dir = Path(self.experiment_dir) / self.agent
            artefact_dir.mkdir(parents=True, exist_ok=True)
            index_path = artefact_dir / "retrain_history.jsonl"
            index_row = {
                "experiment_id": experiment_id,
                "agent": self.agent,
                "baseline_score": baseline_score,
                "best_score": best_score,
                "absolute_improvement": best_score - baseline_score,
                "holdout_score": best_holdout_score,
                "generalization_gap": gen_gap,
                "n_epochs": len(learnings_history),
                "duration_seconds": round(duration, 2),
            }
            with index_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(index_row, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to append retrain_history.jsonl: {}", exc)

        # Final summary (Keras-style)
        logger.info("\n{}", result.summary())
        if result.score_history:
            from agentomatic.optimize.config import _score_sparkline

            logger.info("📈 Loss/score curve: {}", _score_sparkline(result.score_history))

        if test_score is not None:
            logger.info("🧪 Holdout/test score: {:.4f}", test_score)

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

        # Drain: give local LLM servers / HTTP pools time to release
        # connections after the async optimisation loop (avoids "connection
        # reset" / half-closed socket errors on the next call).
        if self.drain_seconds > 0:
            import asyncio

            logger.info("⏳ Draining for {:.1f}s after fit loop…", self.drain_seconds)
            await asyncio.sleep(self.drain_seconds)

        return result

    # ================================================================
    # Private helpers
    # ================================================================

    def _augment_metric_with_local_judges(
        self,
        metric: CompositeMetric | BaseMetric,
    ) -> CompositeMetric | BaseMetric:
        """Fold configured ``local_judges`` into the evaluation metric.

        When ``PromptFitter(..., local_judges=[...])`` is set, each model is
        wrapped as a :class:`~agentomatic.optimize.judges.LocalJudgeMetric`
        and mixed into a :class:`CompositeMetric` so LLM-as-judge feedback
        participates in every reevaluation round.
        """
        if not self.local_judges:
            return metric

        from agentomatic.optimize.judges import LocalJudgeMetric, MultiJudgePanel
        from agentomatic.optimize.metrics import WeightedMetric

        judges = [
            LocalJudgeMetric(
                name=f"local_judge_{idx}",
                model=model_name,
                criteria="Evaluate correctness, completeness, and usefulness.",
            )
            for idx, model_name in enumerate(self.local_judges)
        ]
        panel: BaseMetric = judges[0] if len(judges) == 1 else MultiJudgePanel(judges=judges)

        if isinstance(metric, CompositeMetric):
            metric._metrics.append(  # noqa: SLF001 - intentional augmentation
                WeightedMetric(name="local_judges", metric=panel, weight=0.35)
            )
            metric._total_weight = sum(wm.weight for wm in metric._metrics)  # noqa: SLF001
            logger.info(
                "Augmented composite metric with {} local judge(s)",
                len(judges),
            )
            return metric

        augmented = CompositeMetric(
            metrics=[
                WeightedMetric(name="base", metric=metric, weight=0.65),
                WeightedMetric(name="local_judges", metric=panel, weight=0.35),
            ]
        )
        logger.info("Wrapped metric with {} local judge(s)", len(judges))
        return augmented

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

        # Inject model_params for deterministic eval. Prefer candidate
        # overrides; default temperature=0.0 so scoring does not oscillate
        # from sampling noise (common 0.33↔0.67 judge/agent jitter).
        eval_params = dict(config.model_params or {})
        eval_params.setdefault("temperature", 0.0)
        augmented_points: list[dict[str, Any]] = []
        for point in points:
            p = dict(point)
            meta = dict(p.get("metadata") or {})
            invoke = dict(meta.get("invoke") or {})
            invoke.setdefault("model_params", eval_params)
            invoke.setdefault("temperature", eval_params.get("temperature", 0.0))
            meta["invoke"] = invoke
            meta["model_params"] = eval_params
            p["metadata"] = meta
            augmented_points.append(p)

        # Bake few-shot examples into the system prompt override so local
        # agents (and HTTP runners) that only honour prompt_override still
        # see them during candidate evaluation.
        prompt_override = self._prompt_with_few_shot(config)

        run_results: list[RunResult] = await self._runner.run_dataset(
            augmented_points,
            prompt_override=prompt_override,
            concurrency=self.concurrency,
        )

        scores: list[float] = []
        dim_accumulators: dict[str, list[float]] = {}
        eval_details: list[dict[str, Any]] = []

        scored_count = 0
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
                continue

            try:
                eval_result: EvalResult = await metric.evaluate(
                    query=rr.query,
                    response=rr.response,
                    expected=rr.expected,
                    context=rr.context or rr.retrieval_context,
                )
                # Honest eval: skip fabricated / failed judge scores from the
                # average so a broken LLM-as-judge cannot invent mid-scale loss.
                if (eval_result.metadata or {}).get("evaluation_failed"):
                    logger.warning(
                        "Skipping failed metric eval for '{}': {}",
                        rr.query[:50],
                        eval_result.reason,
                    )
                    eval_details.append(
                        {
                            "query": rr.query,
                            "response": rr.response,
                            "expected": rr.expected,
                            "avg_score": 0.0,
                            "error": eval_result.reason,
                            "details": [],
                        }
                    )
                    continue

                point_score = eval_result.score
                scores.append(point_score)
                scored_count += 1

                # Extract per-dimension scores from composite metrics
                point_dims: dict[str, float] = {}
                if isinstance(metric, CompositeMetric):
                    point_dims = eval_result.metadata.get("dimensions", {})
                    for dim, val in point_dims.items():
                        dim_accumulators.setdefault(dim, []).append(val)

                meta = eval_result.metadata or {}
                eval_details.append(
                    {
                        "query": rr.query,
                        "response": rr.response,
                        "expected": rr.expected,
                        "avg_score": point_score,
                        "score": point_score,
                        "dimensions": point_dims or meta.get("dimensions", {}),
                        "feedback": eval_result.reason,
                        "motivation": meta.get("motivation", ""),
                        "what_worked": meta.get("what_worked", []),
                        "what_failed": meta.get("what_failed", []),
                        "improvement_hints": meta.get("improvement_hints", []),
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

        # If every point failed evaluation, report 0.0 (not a fabricated mid score).
        avg_score = sum(scores) / scored_count if scored_count else 0.0
        per_dim: dict[str, float] = {
            dim: sum(vals) / len(vals) for dim, vals in dim_accumulators.items() if vals
        }

        return avg_score, per_dim, eval_details

    @staticmethod
    def _prompt_with_few_shot(config: PromptRuntimeConfig) -> str:
        """Merge ``system_prompt`` + few-shot examples into one override string."""
        prompt = (config.system_prompt or "").strip()
        examples = list(config.few_shot_examples or [])
        if not examples:
            return prompt
        blocks = ["## Few-shot examples (follow this style and grounding)"]
        for idx, ex in enumerate(examples[:6], 1):
            if not isinstance(ex, dict):
                continue
            q = str(ex.get("query") or ex.get("input") or "").strip()
            r = str(ex.get("response") or ex.get("output") or "").strip()
            if not q and not r:
                continue
            blocks.append(f"Example {idx}\nQ: {q[:400]}\nA: {r[:600]}")
        if len(blocks) == 1:
            return prompt
        return (prompt + "\n\n" + "\n\n".join(blocks)).strip()

    def _load_baseline_config(self) -> PromptRuntimeConfig:
        """Load baseline config from the agent's ``prompts.json``.

        When ``baseline_system_prompt`` was provided at construction (e.g. by
        :class:`PromptFitterBridge` carrying the best prompt from a previous
        epoch), that value takes precedence over anything in ``prompts.json``
        so that successive ``fit()`` calls compound improvement rather than
        restarting from the file each time.

        Searches for ``prompts.json`` in standard locations
        (``agents/<agent>/``, ``<agent>/``, cwd).  Falls back to a
        generic system prompt if no file is found.
        """
        if self._baseline_system_prompt:
            logger.info(
                "📌 Using caller-supplied baseline prompt ({} chars) — compounding from previous epoch",
                len(self._baseline_system_prompt),
            )
            system_prompt = self._baseline_system_prompt
        else:
            system_prompt = self._load_prompt_text()

        # Build default model_params from search space (first value of each).
        # Always default temperature to 0.0 for reproducible fit evaluation
        # unless the search space (or an explicit candidate) overrides it.
        model_params: dict[str, Any] = {"temperature": 0.0}
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
