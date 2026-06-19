"""Event system and callback protocol for prompt optimisation observability.

Provides a structured event/callback architecture that both
:class:`PromptFitter` and :class:`PromptOptimizationLoop` use to
emit real-time progress updates.  Consumers implement the
:class:`OptimizationCallback` protocol (or register plain callables)
to receive :class:`EventData` payloads without subclassing.

Quick-start::

    from agentomatic.optimize.events import (
        CallbackManager,
        OptimizationCallback,
        OptimizationEvent,
    )

    class MyReporter(OptimizationCallback):
        async def on_event(self, event, data):
            print(f"{event.value}: score={data.score}")

    mgr = CallbackManager()
    mgr.add(MyReporter())
    await mgr.emit(OptimizationEvent.FIT_START, EventData(agent="bot"))
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    pass


# =====================================================================
# Event types
# =====================================================================


class OptimizationEvent(enum.Enum):
    """Lifecycle events emitted during prompt optimisation."""

    # ── Fit / Run lifecycle ──────────────────────────────────────────
    FIT_START = "fit_start"
    BASELINE_EVALUATED = "baseline_evaluated"
    ROUND_START = "round_start"
    CANDIDATES_PROPOSED = "candidates_proposed"
    CANDIDATE_EVALUATED = "candidate_evaluated"
    CANDIDATE_PROMOTED = "candidate_promoted"
    CANDIDATE_ACCEPTED = "candidate_accepted"
    CANDIDATE_REJECTED = "candidate_rejected"
    ROUND_END = "round_end"
    EARLY_STOP = "early_stop"
    FIT_COMPLETE = "fit_complete"

    # ── Per-sample events ────────────────────────────────────────────
    EVAL_POINT_SCORED = "eval_point_scored"

    # ── PromptOptimizationLoop-specific ──────────────────────────────
    RUN_START = "run_start"
    STEP_START = "step_start"
    SAMPLE_RESULT = "sample_result"
    STEP_COMPLETE = "step_complete"
    RUN_COMPLETE = "run_complete"
    REWRITE_START = "rewrite_start"
    REWRITE_COMPLETE = "rewrite_complete"
    REWRITE_ACCEPTED = "rewrite_accepted"
    REWRITE_REJECTED = "rewrite_rejected"


# =====================================================================
# Event data payload
# =====================================================================


@dataclass
class EventData:
    """Payload carried by each :class:`OptimizationEvent`.

    All fields are optional — different events populate different
    subsets.  Consumers should use ``getattr`` or check for ``None``.
    """

    # ── Identity ─────────────────────────────────────────────────────
    agent: str = ""
    experiment_id: str = ""
    optimizer_name: str = ""

    # ── Progress ─────────────────────────────────────────────────────
    round_idx: int | None = None
    total_rounds: int | None = None
    step_idx: int | None = None
    total_steps: int | None = None
    candidate_idx: int | None = None
    total_candidates: int | None = None

    # ── Scores ───────────────────────────────────────────────────────
    score: float | None = None
    baseline_score: float | None = None
    best_score: float | None = None
    dimensions: dict[str, float] = field(default_factory=dict)
    score_history: list[float] = field(default_factory=list)

    # ── Candidate info ───────────────────────────────────────────────
    candidate_name: str = ""
    candidate_source: str = ""
    mutation_notes: str = ""
    accept_reason: str = ""

    # ── Sample-level ─────────────────────────────────────────────────
    query: str = ""
    response: str = ""
    expected: str = ""
    sample_score: float | None = None

    # ── Timing ───────────────────────────────────────────────────────
    elapsed_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # ── Prompt ───────────────────────────────────────────────────────
    prompt: str = ""
    prompt_length: int = 0

    # ── Misc ─────────────────────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)
    n_failures: int = 0
    n_successes: int = 0
    accuracy: float = 0.0

    # ── Rewrite acceptance ───────────────────────────────────────────
    improvement: float = 0.0
    was_accepted: bool = True


# =====================================================================
# Callback protocol
# =====================================================================


@runtime_checkable
class OptimizationCallback(Protocol):
    """Protocol for receiving optimisation events.

    Implement this to get live updates without subclassing the
    optimiser.  All methods are *optional* — the dispatcher falls
    back to :meth:`on_event` for any event that does not have a
    dedicated method.

    Example::

        class ScoreTracker(OptimizationCallback):
            scores: list[float] = []

            async def on_event(self, event, data):
                if data.score is not None:
                    self.scores.append(data.score)
    """

    async def on_event(
        self,
        event: OptimizationEvent,
        data: EventData,
    ) -> None:
        """Called for every optimisation event."""
        ...


# =====================================================================
# Callback manager
# =====================================================================


class CallbackManager:
    """Dispatches :class:`OptimizationEvent` to registered callbacks.

    Thread-safe, exception-tolerant: a failing callback never breaks
    the optimisation loop.

    Args:
        callbacks: Initial list of callbacks to register.

    Example::

        mgr = CallbackManager()
        mgr.add(my_callback)
        await mgr.emit(OptimizationEvent.FIT_START, EventData(agent="bot"))
    """

    def __init__(
        self,
        callbacks: list[OptimizationCallback] | None = None,
    ) -> None:
        self._callbacks: list[OptimizationCallback] = list(callbacks or [])

    # ── Registration ─────────────────────────────────────────────────

    def add(self, callback: OptimizationCallback) -> None:
        """Register a new callback."""
        self._callbacks.append(callback)

    def remove(self, callback: OptimizationCallback) -> None:
        """Unregister a callback (no-op if not found)."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    @property
    def callbacks(self) -> list[OptimizationCallback]:
        """Registered callbacks (read-only snapshot)."""
        return list(self._callbacks)

    # ── Dispatch ─────────────────────────────────────────────────────

    async def emit(
        self,
        event: OptimizationEvent,
        data: EventData | None = None,
    ) -> None:
        """Dispatch *event* with *data* to all registered callbacks.

        Exceptions in individual callbacks are logged but never
        propagated — the optimisation loop is never interrupted.
        """
        if data is None:
            data = EventData()
        for cb in self._callbacks:
            try:
                await cb.on_event(event, data)
            except Exception as exc:
                logger.debug(
                    "Callback {} raised on {}: {}",
                    type(cb).__name__,
                    event.value,
                    exc,
                )

    # ── Convenience ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._callbacks)

    def __bool__(self) -> bool:
        return bool(self._callbacks)
