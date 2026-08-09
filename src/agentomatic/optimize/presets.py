"""Optimisation presets for common scenarios.

Provides factory methods that return ready-to-use :class:`Preset`
objects convertible into :class:`PromptFitter` kwargs or a
:class:`~agentomatic.optimize.config.PromptRuntimeConfig` seed.

Example::

    from agentomatic.optimize.presets import Presets, to_fitter_kwargs
    from agentomatic.optimize import PromptFitter

    preset = Presets.for_local()       # Free Ollama, 5 rounds
    fitter = PromptFitter(agent="bot", **to_fitter_kwargs(preset))
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# Map user-facing strategy names → PromptFitter optimizer keys.
_STRATEGY_TO_OPTIMIZER: dict[str, str] = {
    "iterative_refinement": "gepa_like",
    "bootstrap_few_shot": "few_shot",
    "few_shot": "few_shot",
    "mipro": "mipro_like",
    "mipro_like": "mipro_like",
    "combined": "gepa_like",
    "gepa": "gepa_like",
    "gepa_like": "gepa_like",
    "rewrite": "rewrite",
}


@dataclass
class Preset:
    """A named, documented optimisation preset."""

    name: str
    description: str
    model: str
    max_iterations: int
    strategy: str  # "iterative_refinement", "bootstrap_few_shot", "combined", ...
    target_score: float
    parallel_evals: int
    temperature: float
    verbose: bool

    def to_config(self, *, system_prompt: str = "") -> Any:
        """Convert this preset into a :class:`PromptRuntimeConfig`.

        Args:
            system_prompt: Optional baseline system prompt to seed the config.

        Returns:
            A :class:`~agentomatic.optimize.config.PromptRuntimeConfig`.
        """
        from agentomatic.optimize.config import PromptRuntimeConfig

        return PromptRuntimeConfig(
            system_prompt=system_prompt,
            model_params={"temperature": self.temperature},
            model_choice=self.model,
        )

    def to_fitter_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Convert this preset into keyword arguments for :class:`PromptFitter`."""
        return to_fitter_kwargs(self, **overrides)


# =====================================================================
# Built-in presets
# =====================================================================

_LOCAL_PRESET = Preset(
    name="local",
    description="Free local Ollama model, 5 iterations — great for development",
    model="ollama/mistral:7b",
    max_iterations=5,
    strategy="iterative_refinement",
    target_score=0.85,
    parallel_evals=1,
    temperature=0.7,
    verbose=True,
)

_QUALITY_PRESET = Preset(
    name="quality",
    description="GPT-4o with combined strategies and 10 iterations — best results",
    model="openai/gpt-4o",
    max_iterations=10,
    strategy="combined",
    target_score=0.90,
    parallel_evals=4,
    temperature=0.5,
    verbose=True,
)

_QUICK_PRESET = Preset(
    name="quick",
    description="2 bootstrap rounds with Ollama — fast smoke-test",
    model="ollama/mistral:7b",
    max_iterations=2,
    strategy="bootstrap_few_shot",
    target_score=0.80,
    parallel_evals=1,
    temperature=0.7,
    verbose=False,
)


# =====================================================================
# Presets API
# =====================================================================


class Presets:
    """Factory for optimisation presets.

    Each classmethod returns a :class:`Preset` that can be converted
    into :class:`PromptFitter` kwargs via :meth:`Preset.to_fitter_kwargs`
    or :func:`to_fitter_kwargs`.

    Example::

        preset = Presets.for_local()
        config = preset.to_config(system_prompt="You are helpful.")
        fitter_kwargs = preset.to_fitter_kwargs()
    """

    @classmethod
    def for_local(cls) -> Preset:
        """Free, fast preset using local Ollama (mistral:7b)."""
        return replace(_LOCAL_PRESET)

    @classmethod
    def for_quality(cls) -> Preset:
        """Best-quality preset using GPT-4o and combined strategies."""
        return replace(_QUALITY_PRESET)

    @classmethod
    def for_quick(cls) -> Preset:
        """Ultra-fast preset — 2 bootstrap rounds."""
        return replace(_QUICK_PRESET)

    @classmethod
    def for_model(cls, model: str, iterations: int = 5) -> Preset:
        """Create an ad-hoc preset for any model.

        Args:
            model: LiteLLM-style model string (e.g. ``"ollama/llama3:70b"``).
            iterations: Number of optimisation rounds.
        """
        return Preset(
            name=f"custom_{model.replace('/', '_')}",
            description=f"Custom preset for {model}",
            model=model,
            max_iterations=iterations,
            strategy="iterative_refinement",
            target_score=0.85,
            parallel_evals=1,
            temperature=0.7,
            verbose=True,
        )

    @classmethod
    def all(cls) -> list[Preset]:
        """Return all built-in presets."""
        return [cls.for_local(), cls.for_quality(), cls.for_quick()]

    @classmethod
    def display(cls) -> None:
        """Print a table of all available presets."""
        presets = cls.all()
        header = f"{'Name':<10} {'Iters':>5} {'Model':<25} {'Strategy':<25} {'Target':>6}"
        sep = "─" * len(header)
        print(sep)
        print(header)
        print(sep)
        for p in presets:
            print(
                f"{p.name:<10} {p.max_iterations:>5} {p.model:<25} "
                f"{p.strategy:<25} {p.target_score:>6.2f}"
            )
        print(sep)


# PromptFitter constructor keys that presets may populate.
_FITTER_KEYS = frozenset(
    {
        "task_model",
        "rewrite_model",
        "max_trials",
        "patience",
        "optimizer",
        "concurrency",
        "n_runners",
        "experiment_dir",
        "auto_report",
        "callbacks",
        "dashboard",
        "local_agent",
        "llm_base_url",
        "llm_api_key",
        "rewrite_passes",
        "multipass",
        "baseline_system_prompt",
        "holdout_fraction",
        "sequential",
        "search_space",
        "local_judges",
        "api_base",
        "api_prefix",
        "base_prompt_version",
        "min_absolute_improvement",
        "max_generalization_gap",
        "drain_seconds",
        "trace_store_path",
    }
)


def to_fitter_kwargs(preset: Preset, **overrides: Any) -> dict[str, Any]:
    """Convert a :class:`Preset` into keyword arguments for :class:`PromptFitter`.

    Only keys accepted by :class:`~agentomatic.optimize.fitter.PromptFitter`
    are returned (safe for ``PromptFitter(agent=..., **kwargs)``).

    Args:
        preset: The preset to convert.
        overrides: Optional overrides. Legacy aliases ``rewrite_llm``,
            ``eval_llm``, ``max_iterations``, and ``strategy`` are mapped
            automatically. Extra non-fitter keys are ignored.

    Returns:
        Dictionary suitable for ``PromptFitter(agent=..., **kwargs)``.
    """
    strategy = overrides.pop("strategy", preset.strategy)
    optimizer = overrides.pop(
        "optimizer",
        _STRATEGY_TO_OPTIMIZER.get(strategy, strategy),
    )
    max_iterations = int(
        overrides.pop(
            "max_iterations",
            overrides.pop("max_trials", preset.max_iterations),
        )
    )
    parallel = int(
        overrides.pop(
            "parallel_evals",
            overrides.pop("concurrency", preset.parallel_evals),
        )
    )
    model = (
        overrides.pop("model", None)
        or overrides.pop("rewrite_llm", None)
        or overrides.pop("eval_llm", None)
        or preset.model
    )
    # Drop non-fitter convenience fields if present.
    overrides.pop("target_score", None)
    overrides.pop("temperature", None)
    overrides.pop("verbose", None)

    kwargs: dict[str, Any] = {
        "task_model": overrides.pop("task_model", model),
        "rewrite_model": overrides.pop("rewrite_model", model),
        "max_trials": max_iterations,
        "patience": overrides.pop("patience", max(1, max_iterations // 2)),
        "optimizer": optimizer,
        "concurrency": max(1, parallel),
    }
    for key, value in overrides.items():
        if key in _FITTER_KEYS:
            kwargs[key] = value
    return kwargs
