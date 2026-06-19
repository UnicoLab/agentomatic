"""Rich-powered progress reporting callbacks for prompt optimisation.

Provides two :class:`~agentomatic.optimize.events.OptimizationCallback`
implementations for real-time progress feedback during optimisation runs:

* :class:`RichProgressCallback` — interactive terminal UI with progress
  bars, sparklines, and colour-coded score deltas (requires ``rich``).
* :class:`LogProgressCallback` — minimal fallback using :mod:`loguru`
  for non-interactive or headless environments.

The :func:`auto_progress_callback` factory picks the right one at
runtime based on ``rich`` availability and TTY detection.

Example::

    from agentomatic.optimize.progress import auto_progress_callback

    cb = auto_progress_callback()
    fitter = PromptFitter(..., callbacks=[cb])
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.optimize.events import EventData, OptimizationEvent

if TYPE_CHECKING:
    from agentomatic.optimize.events import OptimizationCallback

# Block characters for sparkline rendering (lowest → highest).
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


# =====================================================================
# Helpers
# =====================================================================


def _make_sparkline(scores: list[float]) -> str:
    """Convert a list of scores to a block-character sparkline.

    Each score is mapped to one of eight block characters (``▁``–``█``)
    based on its position within the ``[min, max]`` range of *scores*.

    Args:
        scores: Numeric score values.  An empty list returns ``""``.

    Returns:
        A compact sparkline string, e.g. ``"▁▂▃▅▇"``.
    """
    if not scores:
        return ""
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span == 0:
        # All identical — render middle-height blocks.
        return _SPARK_BLOCKS[4] * len(scores)
    chars: list[str] = []
    max_idx = len(_SPARK_BLOCKS) - 1
    for s in scores:
        idx = int((s - lo) / span * max_idx)
        idx = min(idx, max_idx)
        chars.append(_SPARK_BLOCKS[idx])
    return "".join(chars)


def _fmt_time(seconds: float) -> str:
    """Format elapsed seconds as a human-readable duration.

    Args:
        seconds: Non-negative number of seconds.

    Returns:
        A string like ``"1m 23s"`` or ``"45s"``.
    """
    if seconds < 0:
        seconds = 0.0
    mins, secs = divmod(int(seconds), 60)
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


# =====================================================================
# RichProgressCallback
# =====================================================================


try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False
    Console = None  # type: ignore[assignment,misc]
    Progress = None  # type: ignore[assignment,misc]
    TaskID = None  # type: ignore[assignment,misc]


class RichProgressCallback:
    """Interactive Rich-powered progress callback.

    Provides:

    * **Overall progress bar** showing round advancement, best score,
      score delta, and estimated time remaining.
    * **Per-round sub-progress** for candidate evaluation.
    * **Score-trend sparkline** using Unicode block characters.
    * **Live stats panel** with rounds completed, best score,
      improvement, candidates tried, and failure count.
    * **Colour coding** — green for improvement, yellow for
      stagnation, red for regression.

    If ``rich`` is not installed the callback degrades to a silent
    no-op with a one-time warning.

    Args:
        show_samples: Whether to print per-sample results.

    Example::

        from agentomatic.optimize.progress import RichProgressCallback

        cb = RichProgressCallback(show_samples=True)
        fitter = PromptFitter(..., callbacks=[cb])
    """

    def __init__(self, *, show_samples: bool = False) -> None:
        self.show_samples: bool = show_samples

        self._scores: list[float] = []
        self._start_time: float = 0.0
        self._candidates_tried: int = 0
        self._n_failures: int = 0
        self._rounds_completed: int = 0
        self._best_score: float = 0.0
        self._baseline_score: float = 0.0

        # Rich objects — initialised lazily on FIT_START / RUN_START.
        self._console: Console | None = None  # type: ignore[assignment]
        self._progress: Progress | None = None  # type: ignore[assignment]
        self._overall_task: TaskID | None = None  # type: ignore[assignment]
        self._round_task: TaskID | None = None  # type: ignore[assignment]

        if not _RICH_AVAILABLE:
            logger.warning(
                "rich is not installed — RichProgressCallback will "
                "be a no-op.  Install with: pip install rich"
            )

    # ── Protocol method ──────────────────────────────────────────────

    async def on_event(
        self,
        event: OptimizationEvent,
        data: EventData,
    ) -> None:
        """Dispatch *event* to the appropriate Rich rendering method.

        Args:
            event: The optimisation lifecycle event.
            data: Payload with contextual metrics.
        """
        if not _RICH_AVAILABLE:
            return

        handler = _RICH_HANDLERS.get(event)
        if handler is not None:
            handler(self, data)

    # ── Internal handlers ────────────────────────────────────────────

    def _on_fit_start(self, data: EventData) -> None:
        """Initialise Rich console and progress bars."""
        self._console = Console()  # type: ignore[assignment]
        self._start_time = time.monotonic()
        self._scores.clear()
        self._candidates_tried = 0
        self._n_failures = 0
        self._rounds_completed = 0
        self._best_score = 0.0
        self._baseline_score = 0.0

        total_rounds = data.total_rounds or 1
        self._progress = Progress(  # type: ignore[assignment]
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self._console,
        )
        self._progress.start()  # type: ignore[union-attr]
        self._overall_task = self._progress.add_task(  # type: ignore[union-attr]
            "Optimising",
            total=total_rounds,
        )
        agent_label = data.agent or "unknown"
        self._console.print(  # type: ignore[union-attr]
            f"\n🚀 [bold green]Optimisation started[/] "
            f"for agent [cyan]{agent_label}[/]  "
            f"({total_rounds} round(s))\n"
        )

    def _on_baseline_evaluated(self, data: EventData) -> None:
        """Record and display baseline score."""
        score = data.score if data.score is not None else 0.0
        self._baseline_score = score
        self._best_score = score
        self._scores.append(score)
        if self._console is not None:
            self._console.print(  # type: ignore[union-attr]
                f"  📊 Baseline score: [bold]{score:.4f}[/]"
            )

    def _on_round_start(self, data: EventData) -> None:
        """Start a per-round sub-progress bar."""
        total_cands = data.total_candidates or 1
        if self._progress is not None:
            self._round_task = self._progress.add_task(  # type: ignore[union-attr]
                f"Round {(data.round_idx or 0) + 1}",
                total=total_cands,
            )

    def _on_candidate_evaluated(self, data: EventData) -> None:
        """Advance per-round progress and colour-code the score."""
        self._candidates_tried += 1
        score = data.score if data.score is not None else 0.0
        name = data.candidate_name or "?"
        colour = self._score_colour(score)

        if self._progress is not None and self._round_task is not None:
            self._progress.update(  # type: ignore[union-attr]
                self._round_task,
                advance=1,
                description=(f"  {name}: [{colour}]{score:.4f}[/]"),
            )

    def _on_candidate_accepted(self, data: EventData) -> None:
        """Show an accepted candidate in green."""
        score = data.score if data.score is not None else 0.0
        name = data.candidate_name or "?"
        if self._console is not None:
            self._console.print(  # type: ignore[union-attr]
                f"  ✅ [green]Accepted[/] {name}: {score:.4f}  ({data.accept_reason})"
            )

    def _on_candidate_rejected(self, data: EventData) -> None:
        """Show a rejected candidate in red."""
        score = data.score if data.score is not None else 0.0
        name = data.candidate_name or "?"
        if self._console is not None:
            self._console.print(  # type: ignore[union-attr]
                f"  ❌ [red]Rejected[/] {name}: {score:.4f}  ({data.accept_reason})"
            )

    def _on_round_end(self, data: EventData) -> None:
        """Finish the round — update overall bar and sparkline."""
        score = data.best_score if data.best_score is not None else 0.0
        self._best_score = max(self._best_score, score)
        self._scores.append(score)
        self._rounds_completed += 1
        self._n_failures += data.n_failures

        # Remove completed round task.
        if self._progress is not None and self._round_task is not None:
            self._progress.remove_task(  # type: ignore[union-attr]
                self._round_task,
            )
            self._round_task = None

        # Advance overall bar.
        if self._progress is not None and self._overall_task is not None:
            delta = self._best_score - self._baseline_score
            sign = "+" if delta >= 0 else ""
            colour = self._score_colour(self._best_score)
            self._progress.update(  # type: ignore[union-attr]
                self._overall_task,
                advance=1,
                description=(f"Best: [{colour}]{self._best_score:.3f}[/] | Δ {sign}{delta:.3f}"),
            )

        # Sparkline.
        if self._console is not None and len(self._scores) > 1:
            spark = _make_sparkline(self._scores)
            first = self._scores[0]
            last = self._scores[-1]
            self._console.print(  # type: ignore[union-attr]
                f"  📈 {spark} {first:.2f} → {last:.2f}"
            )

    def _on_early_stop(self, data: EventData) -> None:
        """Log early-stop notification."""
        if self._console is not None:
            self._console.print(  # type: ignore[union-attr]
                "\n  ⏹️  [yellow]Early stop triggered[/]"
            )

    def _on_fit_complete(self, data: EventData) -> None:
        """Print final summary table and stop progress bars."""
        if self._progress is not None:
            self._progress.stop()  # type: ignore[union-attr]

        if self._console is None:
            return

        elapsed = time.monotonic() - self._start_time
        delta = self._best_score - self._baseline_score
        self._print_summary_table(elapsed, delta, data)

    def _on_sample_result(self, data: EventData) -> None:
        """Optionally log per-sample results."""
        if not self.show_samples:
            return
        if self._console is None:
            return
        s_score = data.sample_score if data.sample_score is not None else 0.0
        query_preview = (data.query or "")[:60]
        self._console.print(  # type: ignore[union-attr]
            f"    🔹 {s_score:.3f}  {query_preview}"
        )

    def _on_rewrite_accepted(self, data: EventData) -> None:
        """Display rewrite acceptance info."""
        if self._console is not None:
            self._console.print(  # type: ignore[union-attr]
                f"  ✏️  [green]Rewrite accepted[/]  "
                f"len={data.prompt_length}  "
                f"Δ {data.improvement:+.4f}"
            )

    def _on_rewrite_rejected(self, data: EventData) -> None:
        """Display rewrite rejection info."""
        if self._console is not None:
            self._console.print(  # type: ignore[union-attr]
                f"  ✏️  [red]Rewrite rejected[/]  "
                f"len={data.prompt_length}  "
                f"Δ {data.improvement:+.4f}"
            )

    # ── Helpers ──────────────────────────────────────────────────────

    def _score_colour(self, score: float) -> str:
        """Return a Rich colour tag based on score vs. baseline.

        Args:
            score: The score to compare against the baseline.

        Returns:
            ``"green"``, ``"yellow"``, or ``"red"``.
        """
        if score > self._baseline_score:
            return "green"
        if score == self._baseline_score:
            return "yellow"
        return "red"

    def _print_summary_table(
        self,
        elapsed: float,
        delta: float,
        data: EventData,
    ) -> None:
        """Render a Rich table with final statistics.

        Args:
            elapsed: Total wall-clock seconds.
            delta: Best score minus baseline.
            data: Final event payload.
        """
        if self._console is None:  # pragma: no cover
            return

        table = Table(  # type: ignore[assignment]
            title="Optimisation Summary",
            show_header=False,
            border_style="dim",
        )
        table.add_column("Key", style="bold")
        table.add_column("Value")

        colour = self._score_colour(self._best_score)
        sign = "+" if delta >= 0 else ""

        table.add_row("Agent", data.agent or "–")
        table.add_row("Rounds completed", str(self._rounds_completed))
        table.add_row(
            "Best score",
            f"[{colour}]{self._best_score:.4f}[/]",
        )
        table.add_row("Baseline", f"{self._baseline_score:.4f}")
        table.add_row(
            "Improvement",
            f"[{colour}]{sign}{delta:.4f}[/]",
        )
        table.add_row(
            "Candidates tried",
            str(self._candidates_tried),
        )
        table.add_row("Failures", str(self._n_failures))
        table.add_row("Elapsed", _fmt_time(elapsed))

        if len(self._scores) > 1:
            spark = _make_sparkline(self._scores)
            table.add_row(
                "Trend",
                f"{spark}  {self._scores[0]:.2f} → {self._scores[-1]:.2f}",
            )

        self._console.print()  # type: ignore[union-attr]
        self._console.print(table)  # type: ignore[union-attr]
        self._console.print()  # type: ignore[union-attr]


# Handler dispatch table — avoids a long if/elif chain.
_RICH_HANDLERS: dict[
    OptimizationEvent,
    Any,
] = {}

if _RICH_AVAILABLE:
    _RICH_HANDLERS = {
        OptimizationEvent.FIT_START: RichProgressCallback._on_fit_start,
        OptimizationEvent.RUN_START: RichProgressCallback._on_fit_start,
        OptimizationEvent.BASELINE_EVALUATED: (RichProgressCallback._on_baseline_evaluated),
        OptimizationEvent.ROUND_START: (RichProgressCallback._on_round_start),
        OptimizationEvent.STEP_START: (RichProgressCallback._on_round_start),
        OptimizationEvent.CANDIDATE_EVALUATED: (RichProgressCallback._on_candidate_evaluated),
        OptimizationEvent.CANDIDATE_ACCEPTED: (RichProgressCallback._on_candidate_accepted),
        OptimizationEvent.CANDIDATE_REJECTED: (RichProgressCallback._on_candidate_rejected),
        OptimizationEvent.ROUND_END: (RichProgressCallback._on_round_end),
        OptimizationEvent.STEP_COMPLETE: (RichProgressCallback._on_round_end),
        OptimizationEvent.EARLY_STOP: (RichProgressCallback._on_early_stop),
        OptimizationEvent.FIT_COMPLETE: (RichProgressCallback._on_fit_complete),
        OptimizationEvent.RUN_COMPLETE: (RichProgressCallback._on_fit_complete),
        OptimizationEvent.SAMPLE_RESULT: (RichProgressCallback._on_sample_result),
        OptimizationEvent.REWRITE_ACCEPTED: (RichProgressCallback._on_rewrite_accepted),
        OptimizationEvent.REWRITE_REJECTED: (RichProgressCallback._on_rewrite_rejected),
    }


# =====================================================================
# LogProgressCallback
# =====================================================================


class LogProgressCallback:
    """Minimal loguru-based progress callback for non-interactive use.

    Logs key lifecycle events (start, baseline, round progress,
    candidate evaluation, acceptance/rejection, completion) via
    :mod:`loguru`.  Suitable for CI, Docker, or headless environments
    where Rich is unavailable or undesirable.

    Example::

        from agentomatic.optimize.progress import LogProgressCallback

        cb = LogProgressCallback()
        loop = PromptOptimizationLoop(..., callbacks=[cb])
    """

    async def on_event(
        self,
        event: OptimizationEvent,
        data: EventData,
    ) -> None:
        """Log a human-readable summary for each event.

        Args:
            event: The optimisation lifecycle event.
            data: Payload with contextual metrics.
        """
        handler = _LOG_HANDLERS.get(event)
        if handler is not None:
            handler(data)

    # ── Internal handlers ────────────────────────────────────────────

    @staticmethod
    def _on_fit_start(data: EventData) -> None:
        logger.info(
            "Optimisation started | agent={} experiment={}",
            data.agent or "–",
            data.experiment_id or "–",
        )

    @staticmethod
    def _on_baseline_evaluated(data: EventData) -> None:
        logger.info(
            "Baseline evaluated | score={:.4f}",
            data.score if data.score is not None else 0.0,
        )

    @staticmethod
    def _on_round_start(data: EventData) -> None:
        r_idx = (data.round_idx or 0) + 1
        total = data.total_rounds or "?"
        logger.info("Round {}/{} started", r_idx, total)

    @staticmethod
    def _on_step_start(data: EventData) -> None:
        s_idx = (data.step_idx or 0) + 1
        total = data.total_steps or "?"
        logger.info("Step {}/{} started", s_idx, total)

    @staticmethod
    def _on_candidate_evaluated(data: EventData) -> None:
        logger.info(
            "Candidate evaluated | name={} score={:.4f}",
            data.candidate_name or "?",
            data.score if data.score is not None else 0.0,
        )

    @staticmethod
    def _on_candidate_accepted(data: EventData) -> None:
        logger.info(
            "Candidate accepted | name={} reason={}",
            data.candidate_name or "?",
            data.accept_reason or "–",
        )

    @staticmethod
    def _on_candidate_rejected(data: EventData) -> None:
        logger.info(
            "Candidate rejected | name={} reason={}",
            data.candidate_name or "?",
            data.accept_reason or "–",
        )

    @staticmethod
    def _on_round_end(data: EventData) -> None:
        logger.info(
            "Round ended | best_score={:.4f} elapsed={:.1f}s",
            data.best_score if data.best_score is not None else 0.0,
            data.elapsed_seconds,
        )

    @staticmethod
    def _on_step_complete(data: EventData) -> None:
        logger.info(
            "Step complete | score={:.4f} elapsed={:.1f}s",
            data.score if data.score is not None else 0.0,
            data.elapsed_seconds,
        )

    @staticmethod
    def _on_fit_complete(data: EventData) -> None:
        best = data.best_score if data.best_score is not None else 0.0
        baseline = data.baseline_score if data.baseline_score is not None else 0.0
        logger.info(
            "Optimisation complete | best={:.4f} "
            "baseline={:.4f} improvement={:+.4f} "
            "elapsed={:.1f}s",
            best,
            baseline,
            best - baseline,
            data.elapsed_seconds,
        )

    @staticmethod
    def _on_early_stop(data: EventData) -> None:
        logger.info("Early stop triggered")

    @staticmethod
    def _on_rewrite_accepted(data: EventData) -> None:
        logger.info(
            "Rewrite accepted | prompt_length={} Δ={:+.4f}",
            data.prompt_length,
            data.improvement,
        )

    @staticmethod
    def _on_rewrite_rejected(data: EventData) -> None:
        logger.info(
            "Rewrite rejected | prompt_length={} Δ={:+.4f}",
            data.prompt_length,
            data.improvement,
        )


# Handler dispatch table for LogProgressCallback.
_LOG_HANDLERS: dict[OptimizationEvent, Any] = {
    OptimizationEvent.FIT_START: LogProgressCallback._on_fit_start,
    OptimizationEvent.RUN_START: LogProgressCallback._on_fit_start,
    OptimizationEvent.REWRITE_REJECTED: (LogProgressCallback._on_rewrite_rejected),
}


# =====================================================================
# Factory
# =====================================================================


def auto_progress_callback() -> OptimizationCallback:
    """Return the best-available progress callback for the environment.

    * If ``rich`` is installed **and** ``sys.stdout`` is a TTY,
      returns :class:`RichProgressCallback`.
    * Otherwise returns :class:`LogProgressCallback`.

    Returns:
        An :class:`~agentomatic.optimize.events.OptimizationCallback`
        implementation ready to be passed to an optimiser.

    Example::

        from agentomatic.optimize.progress import auto_progress_callback

        cb = auto_progress_callback()
    """
    if _RICH_AVAILABLE and _is_tty():
        return RichProgressCallback()  # type: ignore[return-value]
    return LogProgressCallback()  # type: ignore[return-value]


def _is_tty() -> bool:
    """Check whether ``sys.stdout`` is connected to a TTY.

    Returns:
        ``True`` if stdout is a terminal, ``False`` otherwise.
    """
    try:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False
