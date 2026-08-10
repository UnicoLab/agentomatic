"""ML-style training callbacks for prompt optimisation (PromptFitter).

.. note::

    These callbacks are for **prompt optimisation** via
    :class:~agentomatic.optimize.fitter.PromptFitter.

    For **agent training** (agent.compile() / agent.fit()), use the
    callbacks in :mod:agentomatic.agents.history instead::

        from agentomatic.agents import EarlyStopping  # agent training
        from agentomatic.optimize import EarlyStopping  # prompt optimisation

Inspired by Keras callbacks, this module provides pluggable hooks that
can be registered with :class:`PromptFitter` to control the optimisation
loop — early stopping, checkpointing, NaN detection, temperature
scheduling, and more.

Example::

    from agentomatic.optimize.callbacks import (
        EarlyStopping, ModelCheckpoint, NaNStopping,
        TemperatureScheduler, PlateauStopping, ScoreThreshold,
        CallbackContext, default_callbacks, Callback,
    )
    from agentomatic.optimize import PromptFitter

    cbs = [
        EarlyStopping(patience=3, min_delta=0.01),
        ModelCheckpoint(save_dir="checkpoints/"),
        ScoreThreshold(threshold=0.85),
    ]
    fitter = PromptFitter(..., callbacks=cbs)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.optimize.events import EventData, OptimizationEvent


# =====================================================================
# Callback Context
# =====================================================================


@dataclass
class CallbackContext:
    """Mutable context passed through the callback chain.

    Stores the running state that callbacks can inspect and modify
    (e.g., requesting early stop via ``stop_requested``).
    """

    agent_name: str = ""
    experiment_id: str = ""
    best_score: float = 0.0
    best_iteration: int = 0
    current_iteration: int = 0
    current_score: float = 0.0
    total_iterations: int = 0
    stop_requested: bool = False
    scores_history: list[float] = field(default_factory=list)
    no_improvement_count: int = 0
    current_temperature: float | None = None
    """Set only by TemperatureScheduler / PlateauStopping; None = untouched."""
    checkpoint_dir: str = "optimization_results/checkpoints"
    current_prompt: str = ""
    best_prompt: str = ""
    prompt_override: bool = False
    """True when a callback restored/overrode ``current_prompt``."""
    start_time: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Base Callback
# =====================================================================


class Callback:
    """Base class for optimisation callbacks.

    Implements :class:`~agentomatic.optimize.events.OptimizationCallback`
    so callbacks work directly with :class:`PromptFitter` — **no adapter**::

        from agentomatic.optimize.callbacks import EarlyStopping
        from agentomatic.optimize import PromptFitter

        fitter = PromptFitter(..., callbacks=[EarlyStopping(patience=3)])

    Subclass and override ``on_*`` hooks.  Each receives a
    :class:`CallbackContext`.
    """

    def __init__(self) -> None:
        self._ctx = CallbackContext()

    @property
    def context(self) -> CallbackContext:
        return self._ctx

    def on_train_begin(self, ctx: CallbackContext) -> None: ...
    def on_train_end(self, ctx: CallbackContext) -> None: ...
    def on_iteration_begin(self, ctx: CallbackContext) -> None: ...
    def on_iteration_end(self, ctx: CallbackContext) -> None: ...
    def on_evaluation_begin(self, ctx: CallbackContext) -> None: ...
    def on_evaluation_end(self, ctx: CallbackContext) -> None: ...
    def on_optimization_begin(self, ctx: CallbackContext) -> None: ...
    def on_optimization_end(self, ctx: CallbackContext) -> None: ...

    # -- OptimizationCallback protocol ------------------------------------

    async def on_event(self, event: OptimizationEvent, data: EventData) -> None:
        """Dispatch PromptFitter events to ML-style hooks automatically."""
        ctx = self._ctx
        name = getattr(event, "name", str(event))

        prompt = str(getattr(data, "prompt", "") or "")
        # Do not clobber a callback-restored prompt (NaN / early-stop rollback).
        if prompt and not ctx.prompt_override:
            ctx.current_prompt = prompt

        best_score = getattr(data, "best_score", None)

        if name == "FIT_START":
            ctx.agent_name = str(getattr(data, "agent", "") or "")
            ctx.experiment_id = str(getattr(data, "experiment_id", "") or "")
            ctx.total_iterations = int(getattr(data, "total_rounds", 0) or 0)
            if prompt:
                ctx.best_prompt = prompt
                ctx.current_prompt = prompt
            self.on_train_begin(ctx)
            self.on_optimization_begin(ctx)
        elif name == "BASELINE_EVALUATED":
            score = float(getattr(data, "score", 0.0) or 0.0)
            ctx.current_score = score
            ctx.best_score = score
            if prompt:
                ctx.best_prompt = prompt
                ctx.current_prompt = prompt
        elif name == "ROUND_START":
            ctx.current_iteration = (getattr(data, "round_idx", 0) or 0) + 1
            if best_score is not None:
                ctx.best_score = float(best_score)
            self.on_iteration_begin(ctx)
        elif name == "CANDIDATE_EVALUATED":
            ctx.current_score = float(getattr(data, "score", 0.0) or 0.0)
            ctx.scores_history.append(ctx.current_score)
            self.on_evaluation_end(ctx)
        elif name == "CANDIDATE_PROMOTED":
            score = float(getattr(data, "score", None) or getattr(data, "best_score", None) or 0.0)
            ctx.current_score = score
            if score >= ctx.best_score:
                ctx.best_score = score
                if prompt:
                    ctx.best_prompt = prompt
        elif name == "ROUND_END":
            round_score = getattr(data, "score", None)
            if round_score is not None:
                ctx.current_score = float(round_score)
            if best_score is not None:
                ctx.best_score = float(best_score)
            if prompt and best_score is not None:
                if abs(float(best_score) - ctx.current_score) < 1e-9:
                    ctx.best_prompt = prompt
            self.on_iteration_end(ctx)
        elif name == "FIT_COMPLETE":
            self.on_optimization_end(ctx)
            self.on_train_end(ctx)
        elif name == "EARLY_STOP":
            ctx.stop_requested = True


# =====================================================================
# EarlyStopping
# =====================================================================


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Mirrors Keras' ``EarlyStopping`` callback.

    Args:
        monitor: Metric name to monitor (unused currently; we track the
            composite score).
        patience: Number of rounds with no improvement after which
            training is stopped.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"max"`` (higher is better) or ``"min"``.
        restore_best_weights: If ``True``, the best prompt is restored
            to ``ctx.current_prompt`` when training ends.
        verbose: Print messages on early stop.

    Example::

        cb = EarlyStopping(patience=3, min_delta=0.01)
    """

    def __init__(
        self,
        monitor: str = "score",
        patience: int = 3,
        min_delta: float = 0.005,
        mode: str = "max",
        restore_best_weights: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self._wait: int = 0
        self._best: float | None = None

    def on_train_begin(self, ctx: CallbackContext) -> None:
        self._wait = 0
        self._best = None

    def on_iteration_end(self, ctx: CallbackContext) -> None:
        current = ctx.current_score
        if self._best is None:
            self._best = current
            ctx.best_score = current
            ctx.no_improvement_count = 0
            return

        improved = (current - self._best) if self.mode == "max" else (self._best - current)

        if improved > self.min_delta:
            self._best = current
            self._wait = 0
            ctx.best_score = current
            ctx.no_improvement_count = 0
        else:
            self._wait += 1
            ctx.no_improvement_count = self._wait
            if self._wait >= self.patience:
                ctx.stop_requested = True
                logger.info(
                    f"⏹ EarlyStopping: no improvement for {self._wait} rounds "
                    f"(best={self._best:.4f}, current={current:.4f})"
                )
                if self.restore_best_weights and ctx.best_prompt:
                    ctx.current_prompt = ctx.best_prompt
                    ctx.prompt_override = True


# =====================================================================
# ModelCheckpoint
# =====================================================================


class ModelCheckpoint(Callback):
    """Save the best prompt to disk during training.

    Mirrors Keras' ``ModelCheckpoint`` callback.

    Args:
        save_dir: Directory for checkpoints.
        save_best_only: Only save when score improves.
        save_freq: Save every N rounds (1 = every round).
        max_checkpoints: Keep at most this many checkpoints.
        verbose: Print save messages.

    Example::

        cb = ModelCheckpoint(save_dir="checkpoints/my_agent/")
    """

    def __init__(
        self,
        save_dir: str = "optimization_results/checkpoints",
        save_best_only: bool = True,
        save_freq: int = 1,
        max_checkpoints: int = 5,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.verbose = verbose
        self._best_score: float | None = None

    def on_train_begin(self, ctx: CallbackContext) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self._best_score = None

    def on_iteration_end(self, ctx: CallbackContext) -> None:
        if ctx.current_iteration % self.save_freq != 0:
            return

        score = ctx.current_score
        is_best = self._best_score is None or score > self._best_score
        if is_best:
            self._best_score = score

        if self.save_best_only and not is_best:
            return

        self._save_checkpoint(ctx, is_best=is_best)

    def _save_checkpoint(self, ctx: CallbackContext, *, is_best: bool) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iter_str = f"iter_{ctx.current_iteration:03d}"
        fname = f"ckpt_{iter_str}_{timestamp}.json"
        fpath = self.save_dir / fname

        data: dict[str, Any] = {
            "agent_name": ctx.agent_name,
            "iteration": ctx.current_iteration,
            "score": ctx.current_score,
            "best_score": ctx.best_score,
            "prompt": ctx.current_prompt,
            "is_best": is_best,
            "timestamp": timestamp,
        }
        fpath.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.debug(f"💾 Checkpoint saved: {fpath.name} score={ctx.current_score:.4f}")

        # Prune old checkpoints
        self._prune()

    def _prune(self) -> None:
        ckpts = sorted(self.save_dir.glob("ckpt_*.json"), key=os.path.getmtime, reverse=True)
        for old in ckpts[self.max_checkpoints :]:
            old.unlink(missing_ok=True)


# =====================================================================
# NaNStopping
# =====================================================================


class NaNStopping(Callback):
    """Halt training when scores become NaN or invalid.

    Detects NaN, inf, and structurally empty outputs to prevent
    corrupted prompts from propagating through further rounds.

    Args:
        max_consecutive_nan: Number of consecutive NaN rounds before
            forcing a stop.
        validate_output: If ``True``, also checks output text for
            emptiness / whitespace-only responses.
        nan_rollback: If ``True``, restores the best prompt when NaN
            is detected.
    """

    def __init__(
        self,
        max_consecutive_nan: int = 2,
        validate_output: bool = True,
        nan_rollback: bool = True,
    ) -> None:
        super().__init__()
        self.max_consecutive_nan = max_consecutive_nan
        self.check_output_validity = validate_output
        self.nan_rollback = nan_rollback
        self._nan_count: int = 0

    @property
    def validate_output(self) -> bool:
        """Whether empty/zero scores are treated as invalid."""
        return self.check_output_validity

    @validate_output.setter
    def validate_output(self, value: bool) -> None:
        self.check_output_validity = bool(value)

    def on_train_begin(self, ctx: CallbackContext) -> None:
        self._nan_count = 0

    def on_evaluation_end(self, ctx: CallbackContext) -> None:
        import math

        score = ctx.current_score

        is_nan = math.isnan(score) or math.isinf(score)
        # Optionally reject empty response text (not zero scores — those can
        # be legitimate).
        if self.check_output_validity:
            output_text = str(ctx.extra.get("last_output", "") or "")
            if output_text and not self.is_valid_output(output_text):
                is_nan = True

        if is_nan:
            self._nan_count += 1
            score_repr = (
                "nan" if math.isnan(score) else ("inf" if math.isinf(score) else f"{score:.4f}")
            )
            logger.warning(
                f"⚠ NaNStopping: invalid score {score_repr} "
                f"({self._nan_count}/{self.max_consecutive_nan})"
            )

            if self.nan_rollback and ctx.best_prompt:
                ctx.current_prompt = ctx.best_prompt
                ctx.prompt_override = True

            if self._nan_count >= self.max_consecutive_nan:
                ctx.stop_requested = True
                logger.error("⛔ NaNStopping: too many invalid scores, stopping")
        else:
            self._nan_count = 0

    @staticmethod
    def is_valid_output(text: str) -> bool:
        """Check if output text looks valid (non-empty, not just whitespace)."""
        return bool(text and text.strip())


# =====================================================================
# TemperatureScheduler
# =====================================================================


class TemperatureScheduler(Callback):
    """Gradually adjust temperature during training.

    Reduces temperature from an initial value toward a minimum,
    encouraging exploration early and exploitation later.

    Args:
        initial_temperature: Starting temperature.
        min_temperature: Floor temperature.
        decay_rate: Multiplicative decay per round.
        decay_type: ``"exponential"``, ``"linear"``, or ``"step"``.
        step_size: Rounds between temperature changes (for ``"step"``).
    """

    def __init__(
        self,
        initial_temperature: float = 0.7,
        min_temperature: float = 0.1,
        decay_rate: float = 0.9,
        decay_type: str = "exponential",
        step_size: int = 3,
    ) -> None:
        super().__init__()
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.step_size = step_size

    def on_iteration_begin(self, ctx: CallbackContext) -> None:
        # Round 1 keeps the initial temperature (exponent 0).
        step = max(0, int(ctx.current_iteration) - 1)
        if self.decay_type == "exponential":
            temp = max(
                self.min_temperature,
                self.initial_temperature * (self.decay_rate**step),
            )
        elif self.decay_type == "linear":
            total = max(ctx.total_iterations, 1)
            ratio = step / total
            temp = self.initial_temperature * (1 - ratio) + self.min_temperature * ratio
        elif self.decay_type == "step":
            steps = step // self.step_size
            temp = max(
                self.min_temperature,
                self.initial_temperature * (self.decay_rate**steps),
            )
        else:
            temp = self.initial_temperature

        ctx.current_temperature = round(temp, 4)


# =====================================================================
# ProgressLogger
# =====================================================================


class ProgressLogger(Callback):
    """Log training progress at every iteration.

    Prints a compact summary line with score, delta, and timing info.

    Args:
        show_prompt_diff: If ``True``, shows a compact diff of prompt
            changes when the prompt improves.
        show_delta_chars: Number of sparkline characters to show.
    """

    def __init__(self, show_prompt_diff: bool = False, show_delta_chars: int = 0) -> None:
        super().__init__()
        self.show_prompt_diff = show_prompt_diff
        self.show_delta_chars = show_delta_chars
        self._prev_scores: list[float] = []

    def on_train_begin(self, ctx: CallbackContext) -> None:
        ctx.start_time = datetime.now().timestamp()
        self._prev_scores = []
        logger.info(f"🚀 Training started: agent={ctx.agent_name}, rounds={ctx.total_iterations}")

    def on_iteration_end(self, ctx: CallbackContext) -> None:
        self._prev_scores.append(ctx.current_score)
        # Use ``is not None`` — a legitimate best_score of 0.0 is falsy.
        delta = ctx.current_score - ctx.best_score if ctx.best_score is not None else 0.0
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
        spark = (
            _spark(self._prev_scores, width=self.show_delta_chars) if self.show_delta_chars else ""
        )
        logger.info(
            f"  Round {ctx.current_iteration:>3d}: "
            f"score={ctx.current_score:.4f} {arrow}{abs(delta):.4f} "
            f" best={ctx.best_score:.4f}"
            f" {spark}".rstrip()
        )

    def on_train_end(self, ctx: CallbackContext) -> None:
        elapsed = datetime.now().timestamp() - ctx.start_time
        logger.info(
            f"✅ Training complete: rounds={ctx.current_iteration}, "
            f"best={ctx.best_score:.4f}, elapsed={elapsed:.1f}s"
        )


# =====================================================================
# PlateauStopping
# =====================================================================


class PlateauStopping(Callback):
    """Reduce temperature when the score plateaus.

    When the score hasn't improved for *patience* rounds, the
    temperature is reduced by *factor* to encourage different
    exploration.

    Args:
        patience: Rounds without improvement before triggering.
        factor: Temperature multiplier when reducing.
        min_temperature: Floor temperature.
        verbose: Log temperature changes.
    """

    def __init__(
        self,
        patience: int = 2,
        factor: float = 0.5,
        min_temperature: float = 0.1,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_temperature = min_temperature
        self.verbose = verbose
        self._wait: int = 0
        self._best: float | None = None

    def on_train_begin(self, ctx: CallbackContext) -> None:
        self._wait = 0
        self._best = None

    def on_iteration_end(self, ctx: CallbackContext) -> None:
        if self._best is None or ctx.current_score > self._best:
            self._best = ctx.current_score
            self._wait = 0
            return

        self._wait += 1
        if self._wait >= self.patience:
            old_temp = (
                float(ctx.current_temperature) if ctx.current_temperature is not None else 0.7
            )
            ctx.current_temperature = max(self.min_temperature, old_temp * self.factor)
            self._wait = 0
            if self.verbose:
                logger.info(
                    f"📉 Plateau: reducing temperature {old_temp:.3f} → {ctx.current_temperature:.3f}"
                )


# =====================================================================
# ScoreThreshold
# =====================================================================


class ScoreThreshold(Callback):
    """Stop training when a target score is reached.

    Args:
        threshold: Target composite score (0.0–1.0).
        mode: ``"max"`` or ``"min"``.
    """

    def __init__(self, threshold: float = 0.85, mode: str = "max") -> None:
        super().__init__()
        self.threshold = threshold
        self.mode = mode

    def on_iteration_end(self, ctx: CallbackContext) -> None:
        reached = (
            ctx.current_score >= self.threshold
            if self.mode == "max"
            else ctx.current_score <= self.threshold
        )
        if reached:
            ctx.stop_requested = True
            logger.info(
                f"🎯 Target score {self.threshold} reached ({ctx.current_score:.4f}), stopping"
            )


# =====================================================================
# Helpers
# =====================================================================

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _spark(scores: list[float], width: int = 8) -> str:
    """Return a compact sparkline string for the given scores."""
    if not scores or width <= 0:
        return ""
    recent = scores[-width:] if len(scores) > width else scores
    lo, hi = min(recent), max(recent)
    if hi - lo < 1e-9:
        return _SPARK_CHARS[4] * len(recent)
    chars = []
    for s in recent:
        idx = int((s - lo) / (hi - lo) * (len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[min(idx, len(_SPARK_CHARS) - 1)])
    return "".join(chars)


def default_callbacks(patience: int = 3, target_score: float = 0.85) -> list[Callback]:
    """Return a sensible default callback stack.

    Includes:
    - :class:`EarlyStopping` with *patience* rounds.
    - :class:`ModelCheckpoint` (best-only).
    - :class:`ScoreThreshold` at *target_score*.
    - :class:`NaNStopping`.
    - :class:`ProgressLogger`.
    """
    return [
        EarlyStopping(patience=patience),
        ModelCheckpoint(save_best_only=True),
        ScoreThreshold(threshold=target_score),
        NaNStopping(),
        ProgressLogger(),
    ]
