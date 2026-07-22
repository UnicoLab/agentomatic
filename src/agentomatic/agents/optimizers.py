"""MVP optimizers for class-owned graph agents.

Example::

    from agentomatic.agents.optimizers import NoOpOptimizer

    agent.compile(dataset, metrics, optimizer=NoOpOptimizer())
    agent.fit(dataset)
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from .types import AgentDataset, Metric

if TYPE_CHECKING:
    from agentomatic.optimize.llm_types import LLMSpec

# ---------------------------------------------------------------------------
# NoOpOptimizer
# ---------------------------------------------------------------------------


class NoOpOptimizer:
    """Optimizer that does nothing — useful as default.

    Returns an empty config dict.
    """

    def optimize(
        self,
        agent: Any,
        dataset: AgentDataset,
        metrics: Sequence[Metric],
    ) -> dict[str, Any]:
        """Return empty config (no changes)."""
        return {}


# ---------------------------------------------------------------------------
# GridSearchOptimizer
# ---------------------------------------------------------------------------


class GridSearchOptimizer:
    """Test combinations of parameter values.

    Evaluates the agent with each combination and returns
    the best-performing configuration.

    Args:
        param_grid: Mapping of parameter names to lists of values.
        max_examples: Max examples to evaluate per combination.

    Example::

        optimizer = GridSearchOptimizer({
            "temperature": [0.0, 0.2, 0.5],
            "retrieval_top_k": [3, 5, 8],
            "prompt_version": ["v1", "v2"],
        })
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]],
        max_examples: int = 10,
    ) -> None:
        self.param_grid = param_grid
        self.max_examples = max_examples

    def optimize(
        self,
        agent: Any,
        dataset: AgentDataset,
        metrics: Sequence[Metric],
    ) -> dict[str, Any]:
        """Search parameter grid for best config.

        Args:
            agent: The agent to optimize.
            dataset: Training dataset.
            metrics: Metrics to evaluate.

        Returns:
            Best parameter combination.
        """
        if not self.param_grid:
            return {}

        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]
        combinations = list(itertools.product(*value_lists))

        examples = dataset.train[: self.max_examples]
        if not examples:
            logger.warning("No training examples for grid search")
            return {}

        best_config: dict[str, Any] = {}
        best_score = -1.0

        # Save original attribute + compiled_config values so we can restore.
        originals: dict[str, Any] = {}
        compiled_originals: dict[str, Any] = {}
        compiled = getattr(agent, "compiled_config", None)
        for key in keys:
            if hasattr(agent, key):
                originals[key] = getattr(agent, key)
            if isinstance(compiled, dict) and key in compiled:
                compiled_originals[key] = compiled[key]

        for combo in combinations:
            config = dict(zip(keys, combo))
            self._apply_config(agent, config)

            # Evaluate
            total_score = 0.0
            count = 0
            for example in examples:
                try:
                    prediction = agent.transform(example.input)
                    for metric in metrics:
                        total_score += metric.score(
                            example,
                            prediction,
                        )
                        count += 1
                except Exception as exc:
                    logger.debug(f"Grid search eval error: {exc}")

            avg_score = total_score / count if count else 0.0
            logger.debug(f"Grid search: {config} -> {avg_score:.3f}")

            if avg_score > best_score:
                best_score = avg_score
                best_config = config

        # Restore original values (fit() will apply best_config)
        for key, value in originals.items():
            setattr(agent, key, value)
        if isinstance(compiled, dict):
            for key in keys:
                if key in compiled_originals:
                    compiled[key] = compiled_originals[key]
                else:
                    compiled.pop(key, None)

        logger.info(f"Grid search best: {best_config} (score: {best_score:.3f})")
        return best_config

    @staticmethod
    def _apply_config(agent: Any, config: dict[str, Any]) -> None:
        """Apply a candidate config so evaluation actually sees the change.

        Attributes are set when present; otherwise values land in
        ``compiled_config`` so :meth:`BaseGraphAgent.resolve_system_prompt`
        (and similar) pick them up during ``transform``.
        """
        compiled = getattr(agent, "compiled_config", None)
        for key, value in config.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
            if isinstance(compiled, dict):
                compiled[key] = value


# ---------------------------------------------------------------------------
# PromptFitterBridge
# ---------------------------------------------------------------------------


class PromptFitterBridge:
    """Bridge that runs the ``optimize.PromptFitter`` engine from ``fit()``.

    Wraps the powerful async PromptFitter so it can be used as a synchronous
    :class:`~agentomatic.agents.types.Optimizer` for ``BaseGraphAgent``. When
    ``fit()`` calls :meth:`optimize`, this bridge builds (or reuses) a
    ``PromptFitter``, converts the ``AgentDataset`` to the optimize format,
    runs the async optimization to completion, stashes the full
    :class:`~agentomatic.optimize.PromptFitResult` on ``agent._last_fit_result``,
    and returns the best prompt config as applicable agent attributes.

    Args:
        agent_name: Name for the fitter to use.
        task_model: Model for running tasks.
        rewrite_model: Model for prompt rewriting.
        metric: An ``optimize.BaseMetric`` used as the fit objective. Defaults
            to ``ExactMatchMetric`` when omitted.
        max_trials: Maximum optimization trials.
        fitter: Pre-built fitter instance (mainly for testing / advanced use);
            when given, construction kwargs are ignored.
        local_agent: A live agent instance (e.g. a
            :class:`~agentomatic.agents.base.BaseGraphAgent` subclass).  When
            provided, **no HTTP server is required** — evaluations call the
            agent's ``transform()`` / ``atransform()`` method directly.  When
            ``None`` (default) the bridge automatically uses the agent passed
            to :meth:`optimize` so local-mode works transparently.
        llm_base_url: Base URL for the OpenAI-compatible server used by the
            optimizer LLM (prompt rewriting, failure clustering, etc.).
            Example: ``"http://127.0.0.1:8000/v1"`` for omlx / Ollama / vLLM.
        llm_api_key: API key for the optimizer LLM server.
        kwargs: Extra keyword arguments forwarded to ``PromptFitter``.

    Example::

        optimizer = PromptFitterBridge(
            agent_name="my_agent",
            task_model="ollama/qwen2.5:7b",
            rewrite_model="openai/gpt-4.1",
        )
        agent.compile(dataset, metrics, optimizer=optimizer)
        history = agent.fit(dataset)
        result = agent._last_fit_result  # full PromptFitResult

    Local-only (no HTTP server needed)::

        optimizer = PromptFitterBridge(
            agent_name="assistant",
            task_model="openai/LFM2.5-8B-A1B-MLX-4bit",
            llm_base_url="http://127.0.0.1:8000/v1",
            llm_api_key="any-key",
        )
        agent.compile(dataset, metrics, optimizer=optimizer)
        agent.fit(dataset)
    """

    # Best-config keys we know how to apply back onto an agent.
    _APPLICABLE_KEYS = ("system_prompt", "user_template", "model_choice")

    def __init__(
        self,
        agent_name: str = "",
        task_model: LLMSpec = "ollama/qwen2.5:7b",
        rewrite_model: LLMSpec = "openai/gpt-4.1",
        optimizer: str = "rewrite",
        metric: Any | None = None,
        max_trials: int = 8,
        fitter: Any | None = None,
        local_agent: Any | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.agent_name = agent_name
        self.task_model = task_model
        self.rewrite_model = rewrite_model
        self.optimizer = optimizer
        self.metric = metric
        self.max_trials = max_trials
        self.fitter = fitter
        self.local_agent = local_agent
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.kwargs = kwargs

    def optimize(
        self,
        agent: Any,
        dataset: AgentDataset,
        metrics: Sequence[Metric],
    ) -> dict[str, Any]:
        """Run PromptFitter optimization and return applicable config.

        Returns:
            A dict of best-config values that map to agent attributes. The full
            ``PromptFitResult`` is stored on ``agent._last_fit_result``. On any
            failure (missing extra, no data, running event loop) an empty dict
            is returned so ``fit()`` degrades to a baseline pass. A structured
            reason is always recorded on ``agent._last_optimize_status`` (e.g.
            ``"skipped: empty dataset"`` / ``"ok"``) so callers can tell whether
            optimization actually ran instead of silently no-op'ing.
        """
        name = self.agent_name or getattr(agent, "agent_name", "agent")

        def _skip(reason: str) -> dict[str, Any]:
            """Record a structured skip reason and return the no-op config."""
            logger.warning(f"PromptFitterBridge: {reason} — skipping")
            agent._last_optimize_status = f"skipped: {reason}"  # noqa: SLF001
            return {}

        try:
            fitter = self._build_fitter(agent, name)
        except Exception as exc:  # noqa: BLE001
            return _skip(f"fitter unavailable ({exc})")

        opt_dataset = dataset.to_optimize_dataset()
        if not len(opt_dataset):
            return _skip("empty dataset")

        trainset, valset = self._split(opt_dataset)
        metric = self._resolve_metric()
        if metric is None:
            return _skip("no usable metric")

        try:
            result = self._run_async(fitter.fit(trainset, valset, metric))
        except RuntimeError as exc:
            return _skip(f"cannot run fitter here ({exc})")
        except Exception as exc:  # noqa: BLE001
            return _skip(f"fit failed ({exc})")

        agent._last_fit_result = result  # noqa: SLF001 - intentional handoff
        agent._last_optimize_status = "ok"  # noqa: SLF001
        logger.info(f"PromptFitterBridge: fit complete for '{name}'")
        return self._extract_config(agent, result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_fitter(self, agent: Any, name: str) -> Any:
        """Return the injected fitter or construct a new ``PromptFitter``.

        Honours per-call overrides set by ``BaseGraphAgent.fit(...)`` via
        ``agent._fit_optimize_options`` (search_space, optimizer mode,
        max_trials, and other PromptFitter kwargs).

        When *self.local_agent* is provided it is forwarded as the local
        callable so no HTTP server is required.  If *self.local_agent* is
        ``None`` the live *agent* argument is used as a fallback so that
        ``agent.compile(…, optimizer=fitter)`` / ``agent.fit(…)`` always
        invokes the agent locally when no ``api_base`` is configured.
        """
        if self.fitter is not None:
            return self.fitter
        from agentomatic.optimize.fitter import PromptFitter

        overrides = dict(getattr(agent, "_fit_optimize_options", None) or {})
        # Bridge-level defaults, then per-fit() overrides win.
        kwargs = dict(self.kwargs)
        # Seed kwargs with the explicit optimizer attribute so it is always
        # present even when the user didn't pass it through **kwargs.
        kwargs.setdefault("optimizer", self.optimizer)
        kwargs.update(overrides)
        max_trials = kwargs.pop("max_trials", self.max_trials)
        # Drop agent-facing / bridge-only keys that are not PromptFitter args
        for key in (
            "optimize_prompt",
            "optimize_params",
            "optimize_few_shot",
            "metric",
            "fitter",
            "agent_name",
        ):
            kwargs.pop(key, None)

        # Resolve local agent: explicit arg wins, otherwise use the live agent.
        local_agent = self.local_agent if self.local_agent is not None else agent

        # Carry the best prompt found so far (from a previous epoch) as the
        # baseline so each epoch compounds improvement instead of restarting
        # from the original prompts.json every time.
        baseline_prompt: str | None = None
        compiled_cfg = getattr(agent, "compiled_config", None)
        if isinstance(compiled_cfg, dict):
            baseline_prompt = compiled_cfg.get("system_prompt") or None
        if baseline_prompt:
            logger.info(
                "PromptFitterBridge: using compiled system_prompt as baseline ({} chars)",
                len(baseline_prompt),
            )

        return PromptFitter(
            agent=name,
            task_model=kwargs.pop("task_model", self.task_model),
            rewrite_model=kwargs.pop("rewrite_model", self.rewrite_model),
            max_trials=max_trials,
            local_agent=local_agent,
            llm_base_url=kwargs.pop("llm_base_url", self.llm_base_url),
            llm_api_key=kwargs.pop("llm_api_key", self.llm_api_key),
            baseline_system_prompt=baseline_prompt,
            **kwargs,
        )

    def _resolve_metric(self) -> Any | None:
        """Return the fit objective metric (default: ExactMatch)."""
        if self.metric is not None:
            return self.metric
        try:
            from agentomatic.optimize.metrics import ExactMatchMetric

            return ExactMatchMetric()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _split(opt_dataset: Any) -> tuple[Any, Any]:
        """Prefer ``metadata.split`` labels; else fall back to an 80/20 cut."""
        from agentomatic.optimize.dataset import Dataset

        points = list(getattr(opt_dataset, "points", []) or [])
        train = [
            p
            for p in points
            if (getattr(p, "metadata", None) or {}).get("split", "train") == "train"
        ]
        # Include "test" in the fit val pool so tiny datasets still have
        # enough points for always-on holdout + minibatch screening.
        val = [
            p
            for p in points
            if (getattr(p, "metadata", None) or {}).get("split") in ("validation", "val", "test")
        ]
        if train and val:
            return Dataset(points=train), Dataset(points=val)
        try:
            train_ds, val_ds = opt_dataset.split(0.8)
            if len(train_ds) and len(val_ds):
                return train_ds, val_ds
        except Exception:  # noqa: BLE001
            pass
        return opt_dataset, opt_dataset

    @classmethod
    def _extract_config(cls, agent: Any, result: Any) -> dict[str, Any]:
        """Map the fit result's best config onto ``compiled_config``.

        Always returns applicable keys present on the best config so
        ``fit()`` can store them in ``compiled_config`` (and
        :meth:`~agentomatic.agents.base.BaseGraphAgent.resolve_system_prompt`
        can pick them up). Attribute assignment is still gated by
        ``hasattr`` inside ``BaseGraphAgent.fit``.
        """
        best = getattr(result, "best_config", None)
        if best is None:
            return {}
        raw = best.to_dict() if hasattr(best, "to_dict") else dict(getattr(best, "__dict__", {}))
        config: dict[str, Any] = {}
        for key in cls._APPLICABLE_KEYS:
            if key not in raw:
                continue
            value = raw[key]
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            config[key] = value
        return config

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """Run ``coro`` to completion (works inside a live event loop too).

        Uses a persistent thread-local loop (see
        :func:`agentomatic.async_utils.run_sync`) instead of
        :func:`asyncio.run`, which closes the loop and breaks LangChain /
        OpenAI async HTTP clients for subsequent ``ainvoke`` calls.
        """
        from agentomatic.async_utils import run_sync

        return run_sync(coro)
