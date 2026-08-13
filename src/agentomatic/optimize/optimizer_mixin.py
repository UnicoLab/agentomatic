"""OptimiserMixin — adds ``agent.fit()`` prompt optimisation to any agent.

A lightweight mixin that bridges the full :class:`PromptFitter` pipeline
behind the familiar Keras-style ``fit()`` name.

Example::

    from agentomatic.optimize.optimizer_mixin import OptimizerMixin
    from agentomatic.agents import BaseGraphAgent

    class MyAgent(OptimizerMixin, BaseGraphAgent):
        agent_name = "my_agent"
        ...

    agent = MyAgent()
    result = agent.fit(test_cases, max_iterations=5)
    print(result.summary())
    # FitResult(my_agent): 0.52 → 0.84 (↑ +0.32) after 4 rounds

When mixed with :class:`~agentomatic.agents.BaseGraphAgent`, ``fit()``
smart-dispatches:

* **Prompt path** (default) — list of test cases / DeepEval cases →
  returns :class:`FitResult` (runs :class:`PromptFitter` via
  :func:`~agentomatic.async_utils.run_sync`).
* **Keras path** — ``AgentDataset`` / ``epochs=`` / ``validation_data=`` /
  ``optimize_mode=`` → delegates to :meth:`BaseGraphAgent.fit` and
  returns :class:`~agentomatic.agents.history.History`.

``optimize_prompts`` remains as an explicit async alias for the prompt path.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentomatic.optimize.config import PromptFitResult


# =====================================================================
# Result wrapper
# =====================================================================


@dataclass
class FitResult:
    """Lightweight result returned by ``agent.fit()`` (prompt path).

    Mirrors the richer :class:`~agentomatic.optimize.config.PromptFitResult`
    but exposes only the key fields for quick inspection.

    Example::

        result = agent.fit(test_cases)
        print(result.summary())
        # "Initial: 0.52 → Final: 0.84 (↑ +0.32) after 4 rounds"
    """

    agent_name: str = ""
    initial_score: float = 0.0
    final_score: float = 0.0
    improvement: float = 0.0
    iterations: int = 0
    improved: bool = False
    best_prompt: str = ""
    strategy: str = ""
    model: str = ""
    raw_result: Any = None

    def summary(self) -> str:
        arrow = "↑" if self.improvement > 0 else "↓" if self.improvement < 0 else "→"
        return (
            f"FitResult({self.agent_name}): "
            f"{self.initial_score:.3f} → {self.final_score:.3f} "
            f"({arrow} {abs(self.improvement):+.3f}) "
            f"after {self.iterations} rounds"
        )

    @classmethod
    def from_prompt_fit_result(cls, result: PromptFitResult) -> FitResult:
        """Create a :class:`FitResult` from a full :class:`PromptFitResult`."""
        history = list(result.history)  # property → list[float]
        initial = (
            float(result.baseline_score)
            if getattr(result, "baseline_score", None) is not None
            else (float(history[0]) if history else 0.0)
        )
        final = (
            float(result.best_score)
            if getattr(result, "best_score", None) is not None
            else (float(history[-1]) if history else 0.0)
        )
        # Prefer prompt_history (actual optimize rounds). ``score_history`` from
        # PromptFitter seeds the baseline as the first point, which would
        # over-count "rounds" by one if used directly.
        prompt_hist = getattr(result, "prompt_history", None) or []
        if prompt_hist:
            iterations = len(prompt_hist)
        else:
            iterations = len(history)
        return cls(
            agent_name=str(getattr(result, "agent", "") or ""),
            initial_score=initial,
            final_score=final,
            improvement=final - initial,
            iterations=iterations,
            improved=bool(result.improved),
            best_prompt=str(result.best_prompt or ""),
            strategy=str(getattr(result, "optimizer_name", "") or ""),
            model="",
            raw_result=result,
        )


# Keras-style kwargs that mean "delegate to BaseGraphAgent.fit".
# NOTE: ``optimizer`` / ``max_trials`` are shared with the prompt path and
# must NOT trigger Keras dispatch on their own.
_KERAS_FIT_KEYS = frozenset(
    {
        "epochs",
        "validation_data",
        "search_space",
        "optimize_mode",
        "optimize_prompt",
        "optimize_params",
        "optimize_few_shot",
        "model_param_space",
    }
)


# =====================================================================
# OptimizerMixin
# =====================================================================


class OptimizerMixin:
    """Mixin that adds a discoverable ``fit()`` for prompt optimisation.

    Prefer putting this mixin **before** :class:`BaseGraphAgent` when you
    want ``agent.fit(test_cases)`` as the one-liner. Keras-style calls
    (``epochs=``, ``AgentDataset``, …) still dispatch to
    :meth:`BaseGraphAgent.fit` automatically.

    Example::

        class MyAgent(OptimizerMixin, BaseGraphAgent):
            agent_name = "my_agent"
            ...

        result = agent.fit(test_cases)                 # → FitResult
        history = agent.compile(...).fit(ds, epochs=2) # → History
    """

    _optimized_prompt: str | None = None
    _last_fit_result: FitResult | None = None
    # Declared (not initialized) at class level so instances create their
    # own list on first use; avoids sharing mutable state across instances.
    _optimization_history: list[dict[str, Any]]  # pyright: ignore[reportUninitializedInstanceVariable]

    def fit(
        self,
        test_cases: Any | None = None,
        /,
        *args: Any,
        trainer_model: str = "ollama/mistral:7b",
        max_iterations: int = 5,
        target_score: float = 0.85,
        strategy: str = "iterative_refinement",
        verbose: bool | int = True,
        metric_name: str = "answer_relevancy",
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fit the agent — prompt optimisation by default.

        Args:
            test_cases: Prompt-path test cases (objects with ``input``/
                ``query``), or a Keras-path ``AgentDataset`` /
                ``list[AgentExample]``.
            trainer_model: LiteLLM model for rewriting (prompt path).
            max_iterations: Max optimisation rounds (prompt path).
            target_score: Early-stop threshold (prompt path).
            strategy: ``iterative_refinement`` / ``bootstrap_few_shot`` /
                ``mipro`` / ``combined`` (prompt path).
            verbose: Progress flag (prompt path) or Keras verbosity int.
            metric_name: Metric name for the LLM judge (prompt path).
            callbacks: Optimisation callbacks (prompt path) or Keras
                training callbacks (Keras path).
            kwargs: Extra PromptFitter kwargs, or Keras ``fit`` kwargs
                (``epochs``, ``validation_data``, ``optimize_mode``, …).
                Prompt path also accepts ``metric=`` to inject a custom
                metric (e.g. ``ExactMatchMetric`` for offline runs).

        Returns:
            :class:`FitResult` for the prompt path, or
            :class:`~agentomatic.agents.history.History` for the Keras path.
        """
        # After compile(), bare agent.fit() / agent.fit(epochs=…) is Keras.
        compiled = getattr(self, "_compile_dataset", None) is not None
        if _is_keras_fit_call(test_cases, args, kwargs, compiled=compiled):
            return _delegate_keras_fit(
                self,
                test_cases,
                *args,
                verbose=verbose,
                callbacks=callbacks,
                **kwargs,
            )

        from agentomatic.async_utils import run_sync

        return run_sync(
            self._fit_prompts_async(
                test_cases,
                trainer_model=trainer_model,
                max_iterations=max_iterations,
                target_score=target_score,
                strategy=strategy,
                verbose=bool(verbose),
                metric_name=metric_name,
                callbacks=callbacks,
                **kwargs,
            )
        )

    async def optimize_prompts(
        self,
        test_cases: list[Any] | None = None,
        *,
        trainer_model: str = "ollama/mistral:7b",
        max_iterations: int = 5,
        target_score: float = 0.85,
        strategy: str = "iterative_refinement",
        verbose: bool = True,
        metric_name: str = "answer_relevancy",
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> FitResult:
        """Async alias of the prompt-optimisation path of :meth:`fit`.

        Prefer ``agent.fit(test_cases)`` in application code. Use this when
        you are already inside an async context and want to ``await``
        without ``run_sync``.
        """
        return await self._fit_prompts_async(
            test_cases,
            trainer_model=trainer_model,
            max_iterations=max_iterations,
            target_score=target_score,
            strategy=strategy,
            verbose=verbose,
            metric_name=metric_name,
            callbacks=callbacks,
            **kwargs,
        )

    async def _fit_prompts_async(
        self,
        test_cases: list[Any] | None,
        *,
        trainer_model: str,
        max_iterations: int,
        target_score: float,
        strategy: str,
        verbose: bool,
        metric_name: str,
        callbacks: list[Any] | None,
        **kwargs: Any,
    ) -> FitResult:
        del verbose  # reserved; PromptFitter uses loguru
        agent_name: str = getattr(self, "agent_name", "unknown")

        if test_cases is None or len(test_cases) == 0:
            from loguru import logger

            logger.warning(
                "agent.fit() called with no test cases for '{}' — returning dummy result",
                agent_name,
            )
            return FitResult(
                agent_name=agent_name,
                improved=False,
                strategy=strategy,
                model=trainer_model,
            )

        dataset = _test_cases_to_dataset(test_cases)

        from agentomatic.optimize.callbacks import ScoreThreshold, default_callbacks
        from agentomatic.optimize.fitter import PromptFitter
        from agentomatic.optimize.metrics import LLMJudgeMetric
        from agentomatic.optimize.presets import Preset, to_fitter_kwargs

        metric = kwargs.pop("metric", None)
        if metric is None:
            metric = LLMJudgeMetric(
                name=metric_name,
                criteria=(
                    "Evaluate if the response is accurate, helpful, and addresses the query."
                ),
                model=trainer_model,
            )

        # Allow injecting a noop/custom optimizer for offline / E2E runs.
        optimizer = kwargs.pop("optimizer", None)

        preset = Preset(
            name="mixin",
            description="OptimizerMixin ad-hoc preset",
            model=trainer_model,
            max_iterations=max_iterations,
            strategy=strategy,
            target_score=target_score,
            parallel_evals=1,
            temperature=0.7,
            verbose=True,
        )
        fitter_kwargs = to_fitter_kwargs(preset, **kwargs)
        if optimizer is not None:
            fitter_kwargs["optimizer"] = optimizer
        cb_list = (
            list(callbacks)
            if callbacks is not None
            else default_callbacks(
                patience=max(1, max_iterations // 2),
                target_score=target_score,
            )
        )
        if not any(isinstance(c, ScoreThreshold) for c in cb_list):
            cb_list.append(ScoreThreshold(threshold=target_score))
        fitter_kwargs["callbacks"] = cb_list

        fitter = PromptFitter(
            agent=agent_name,
            local_agent=self,
            **fitter_kwargs,
        )

        result = await fitter.fit(dataset, dataset, metric)

        fit_result = FitResult.from_prompt_fit_result(result)
        fit_result.strategy = strategy
        fit_result.model = trainer_model

        if not hasattr(self, "_optimization_history"):
            self._optimization_history = []

        self._optimized_prompt = fit_result.best_prompt
        self._optimization_history.append(
            {
                "agent_name": agent_name,
                "initial_score": fit_result.initial_score,
                "final_score": fit_result.final_score,
                "improvement": fit_result.improvement,
                "iterations": fit_result.iterations,
                "strategy": strategy,
                "model": trainer_model,
            }
        )
        self._last_fit_result = fit_result
        return fit_result

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_optimized_prompt(self) -> str | None:
        """Return the best prompt from the most recent ``fit()`` call."""
        return self._optimized_prompt

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Return the history of all prompt ``fit()`` calls on this instance."""
        return list(getattr(self, "_optimization_history", []))

    @property
    def last_fit_result(self) -> FitResult | None:
        """The most recent :class:`FitResult`, if any."""
        return self._last_fit_result

    def reset_optimization(self) -> None:
        """Clear cached optimisation state."""
        self._optimized_prompt = None
        self._last_fit_result = None
        history = getattr(self, "_optimization_history", None)
        if history is not None:
            history.clear()
        else:
            self._optimization_history = []


# =====================================================================
# Helpers
# =====================================================================


def _is_keras_fit_call(
    first: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    compiled: bool = False,
) -> bool:
    """Return True when the call looks like BaseGraphAgent.fit(...)."""
    if any(k in kwargs for k in _KERAS_FIT_KEYS):
        return True
    if first is None:
        # Bare fit() after compile() → Keras History path.
        return compiled
    type_name = type(first).__name__
    if type_name in {"AgentDataset", "AgentExample"}:
        return True
    if isinstance(first, list) and first:
        sample = first[0]
        if type(sample).__name__ == "AgentExample":
            return True
        # DeepEval / prompt test cases have input/query — NOT keras.
        if hasattr(sample, "input") or hasattr(sample, "query"):
            return False
        if hasattr(sample, "expected_output") or hasattr(sample, "expected_answer"):
            return False
    del args  # reserved for future positional Keras overloads
    return False


def _delegate_keras_fit(self: Any, dataset: Any, *args: Any, **kwargs: Any) -> Any:
    """Call the next non-mixin ``fit`` in the MRO (usually BaseGraphAgent)."""
    for cls in type(self).mro():
        if cls is OptimizerMixin or cls is object:
            continue
        method = cls.__dict__.get("fit")
        if method is None or inspect.iscoroutinefunction(method):
            continue
        return method(self, dataset, *args, **kwargs)
    raise TypeError(
        "OptimizerMixin.fit() received Keras-style arguments but no "
        "BaseGraphAgent.fit was found in the MRO. Inherit from BaseGraphAgent "
        "or call with test_cases for the prompt-optimisation path."
    )


def _test_cases_to_dataset(cases: list[Any]) -> Any:
    """Convert a list of test cases to a :class:`Dataset`.

    Handles DeepEval ``LLMTestCase`` objects, :class:`DataPoint`, and
    plain objects with ``input`` / ``query`` attributes.
    """
    from agentomatic.optimize.dataset import DataPoint, Dataset

    points: list[DataPoint] = []
    for case in cases:
        query = getattr(case, "input", None) or getattr(case, "query", None) or ""
        expected = (
            getattr(case, "expected_output", None) or getattr(case, "expected_answer", None) or ""
        )
        context = list(getattr(case, "context", []) or [])
        points.append(DataPoint(query=str(query), expected_answer=str(expected), context=context))

    return Dataset(points=points)
