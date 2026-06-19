"""Textual-based TUI dashboard for live optimisation observability.

Provides a rich terminal dashboard that visualises prompt-optimisation
progress in real time — scores, candidates, per-dimension metrics, and
a scrolling event log — all driven by :class:`OptimizationCallback`.

.. note::

   ``textual`` is an **optional** dependency.  When it is not installed
   every public symbol still importable: :class:`DashboardCallback`
   degrades to a no-op logger and :func:`launch_dashboard` becomes a
   silent no-op.

Quick-start::

    from agentomatic.optimize.events import CallbackManager
    from agentomatic.optimize.dashboard import DashboardCallback, launch_dashboard

    cb = DashboardCallback()
    launch_dashboard(cb)           # opens TUI in background thread
    mgr = CallbackManager([cb])    # wire into the fitter / loop
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.optimize.events import EventData, OptimizationEvent

if TYPE_CHECKING:
    pass

# ------------------------------------------------------------------
# Optional textual import — everything degrades gracefully.
# ------------------------------------------------------------------
_HAS_TEXTUAL = False
try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import (
        DataTable,
        ProgressBar,
        RichLog,
        Static,
    )

    _HAS_TEXTUAL = True
except ImportError:  # pragma: no cover
    pass

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_MAX_LOG_LINES = 30
_SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"
_STATUS_ACCEPTED = "✓ accepted"
_STATUS_REJECTED = "✗ rejected"
_STATUS_EVALUATED = "• evaluated"


# =====================================================================
# Sparkline helper
# =====================================================================


def _sparkline(values: list[float], width: int = 40) -> str:
    """Return an ASCII sparkline string for *values*.

    Args:
        values: Numeric values to chart.
        width: Maximum character width of the output.

    Returns:
        A compact sparkline string using Unicode block chars.
    """
    if not values:
        return ""
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    span = hi - lo if hi != lo else 1.0
    chars = _SPARKLINE_CHARS
    n = len(chars) - 1
    return "".join(chars[int((v - lo) / span * n)] for v in tail)


# =====================================================================
# Textual TUI application (only defined when textual is available)
# =====================================================================

if _HAS_TEXTUAL:
    _DASHBOARD_CSS = """
    Screen {
        background: $surface;
    }
    #header-bar {
        dock: top;
        height: 3;
        background: $primary-background;
        color: $text;
        padding: 0 2;
        content-align: left middle;
    }
    #progress-box {
        height: 3;
        padding: 0 2;
    }
    #score-chart {
        height: 5;
        padding: 0 2;
        background: $surface;
        color: $success;
    }
    #body {
        height: 1fr;
    }
    #candidates-pane {
        width: 2fr;
        border: solid $primary;
    }
    #right-col {
        width: 1fr;
    }
    #metrics-pane {
        height: 1fr;
        border: solid $secondary;
        padding: 0 1;
    }
    #log-pane {
        height: 2fr;
        border: solid $accent;
    }
    #footer-stats {
        dock: bottom;
        height: 1;
        background: $primary-background;
        color: $text-muted;
        padding: 0 2;
    }
    """

    class OptimizationDashboard(App):  # type: ignore[type-arg]
        """Textual application for live optimisation observability.

        The dashboard renders:
        * **Header bar** — agent, experiment, optimizer, elapsed time.
        * **Progress bar** — overall completion with ETA.
        * **Score chart** — sparkline of scores across rounds.
        * **Candidates table** — round, name, source, score, status.
        * **Metrics panel** — per-dimension current vs baseline (Δ).
        * **Log panel** — last ``_MAX_LOG_LINES`` events.
        * **Footer** — totals for candidates, accepted, rejected,
          best score.
        """

        CSS = _DASHBOARD_CSS
        TITLE = "Agentomatic · Prompt Optimisation"

        # ── internal state ──────────────────────────────────
        _agent: str = ""
        _experiment_id: str = ""
        _optimizer_name: str = ""
        _elapsed: float = 0.0
        _round_idx: int = 0
        _total_rounds: int = 1
        _scores: list[float]
        _baseline_score: float | None = None
        _best_score: float | None = None
        _dimensions: dict[str, float]
        _baseline_dims: dict[str, float]
        _total_cands: int = 0
        _accepted: int = 0
        _rejected: int = 0

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._scores = []
            self._dimensions = {}
            self._baseline_dims = {}

        # ── compose ─────────────────────────────────────────

        def compose(self) -> ComposeResult:
            """Build the widget tree."""
            yield Static("", id="header-bar")
            with Vertical(id="progress-box"):
                yield ProgressBar(total=100, show_eta=True, id="main-progress")
            yield Static("", id="score-chart")
            with Horizontal(id="body"):
                yield DataTable(id="candidates-pane")
                with Vertical(id="right-col"):
                    yield Static("", id="metrics-pane")
                    yield RichLog(
                        id="log-pane",
                        highlight=True,
                        max_lines=_MAX_LOG_LINES,
                    )
            yield Static("", id="footer-stats")

        def on_mount(self) -> None:
            """Initialise the candidates DataTable columns."""
            table: DataTable = self.query_one(  # type: ignore[type-arg]
                "#candidates-pane", DataTable
            )
            table.add_columns("Round", "Name", "Source", "Score", "Status")

        # ── public update entry-point ──────────────────────

        def update_from_event(
            self,
            event: OptimizationEvent,
            data: EventData,
        ) -> None:
            """Ingest an optimisation event and refresh widgets.

            This is the single entry-point called by
            :class:`DashboardCallback`.  It dispatches to the
            appropriate private helpers based on the event type.

            Args:
                event: The optimisation lifecycle event.
                data: Associated payload.
            """
            self._ingest_identity(data)
            self._ingest_progress(data)
            self._ingest_score(event, data)
            self._ingest_candidate(event, data)
            self._ingest_dimensions(event, data)

            self._refresh_header()
            self._refresh_progress()
            self._refresh_score_chart()
            self._refresh_metrics()
            self._refresh_footer()
            self._append_log(event, data)

        # ── private ingestors ──────────────────────────────

        def _ingest_identity(self, data: EventData) -> None:
            if data.agent:
                self._agent = data.agent
            if data.experiment_id:
                self._experiment_id = data.experiment_id
            if data.optimizer_name:
                self._optimizer_name = data.optimizer_name
            if data.elapsed_seconds:
                self._elapsed = data.elapsed_seconds

        def _ingest_progress(self, data: EventData) -> None:
            if data.round_idx is not None:
                self._round_idx = data.round_idx
            if data.total_rounds is not None:
                self._total_rounds = max(data.total_rounds, 1)

        def _ingest_score(
            self,
            event: OptimizationEvent,
            data: EventData,
        ) -> None:
            if event == OptimizationEvent.BASELINE_EVALUATED:
                self._baseline_score = data.score
            if data.score is not None and event in {
                OptimizationEvent.CANDIDATE_EVALUATED,
                OptimizationEvent.ROUND_END,
                OptimizationEvent.STEP_COMPLETE,
            }:
                self._scores.append(data.score)
            if data.best_score is not None:
                self._best_score = data.best_score

        def _ingest_candidate(
            self,
            event: OptimizationEvent,
            data: EventData,
        ) -> None:
            if event == OptimizationEvent.CANDIDATE_EVALUATED:
                self._total_cands += 1
                status = _STATUS_EVALUATED
            elif event == OptimizationEvent.CANDIDATE_ACCEPTED:
                self._accepted += 1
                status = _STATUS_ACCEPTED
            elif event == OptimizationEvent.CANDIDATE_REJECTED:
                self._rejected += 1
                status = _STATUS_REJECTED
            else:
                return

            try:
                table: DataTable = self.query_one(  # type: ignore[type-arg]
                    "#candidates-pane", DataTable
                )
                score_str = f"{data.score:.4f}" if data.score is not None else "—"
                table.add_row(
                    str(data.round_idx or self._round_idx),
                    data.candidate_name or "—",
                    data.candidate_source or "—",
                    score_str,
                    status,
                )
            except Exception:  # noqa: BLE001
                pass

        def _ingest_dimensions(
            self,
            event: OptimizationEvent,
            data: EventData,
        ) -> None:
            if data.dimensions:
                self._dimensions = dict(data.dimensions)
            if event == OptimizationEvent.BASELINE_EVALUATED and data.dimensions:
                self._baseline_dims = dict(data.dimensions)

        # ── private refreshers ─────────────────────────────

        def _refresh_header(self) -> None:
            minutes, secs = divmod(int(self._elapsed), 60)
            text = (
                f" Agent: [bold]{self._agent}[/]"
                f"  │  Experiment: {self._experiment_id}"
                f"  │  Optimizer: {self._optimizer_name}"
                f"  │  Elapsed: {minutes:02d}:{secs:02d}"
            )
            try:
                self.query_one("#header-bar", Static).update(text)
            except Exception:  # noqa: BLE001
                pass

        def _refresh_progress(self) -> None:
            pct = int(self._round_idx / self._total_rounds * 100)
            try:
                bar: ProgressBar = self.query_one("#main-progress", ProgressBar)
                bar.update(progress=pct)
            except Exception:  # noqa: BLE001
                pass

        def _refresh_score_chart(self) -> None:
            spark = _sparkline(self._scores, width=60)
            best = f"{self._best_score:.4f}" if self._best_score is not None else "—"
            baseline = f"{self._baseline_score:.4f}" if self._baseline_score is not None else "—"
            label = f" Scores  baseline={baseline}  best={best}\n {spark}"
            try:
                self.query_one("#score-chart", Static).update(label)
            except Exception:  # noqa: BLE001
                pass

        def _refresh_metrics(self) -> None:
            if not self._dimensions:
                return
            lines: list[str] = [" [bold]Dimension Scores[/]"]
            for dim, val in sorted(self._dimensions.items()):
                base = self._baseline_dims.get(dim)
                if base is not None:
                    delta = val - base
                    sign = "+" if delta >= 0 else ""
                    colour = "green" if delta >= 0 else "red"
                    lines.append(
                        f"  {dim}: {val:.4f}  "
                        f"(baseline {base:.4f}  "
                        f"[{colour}]{sign}{delta:.4f}[/])"
                    )
                else:
                    lines.append(f"  {dim}: {val:.4f}")
            try:
                self.query_one("#metrics-pane", Static).update("\n".join(lines))
            except Exception:  # noqa: BLE001
                pass

        def _refresh_footer(self) -> None:
            best_str = f"{self._best_score:.4f}" if self._best_score is not None else "—"
            text = (
                f" Candidates: {self._total_cands}"
                f"  │  Accepted: {self._accepted}"
                f"  │  Rejected: {self._rejected}"
                f"  │  Best Score: {best_str}"
            )
            try:
                self.query_one("#footer-stats", Static).update(text)
            except Exception:  # noqa: BLE001
                pass

        def _append_log(
            self,
            event: OptimizationEvent,
            data: EventData,
        ) -> None:
            parts: list[str] = [f"[dim]{event.value}[/]"]
            if data.round_idx is not None:
                parts.append(f"round={data.round_idx}")
            if data.candidate_name:
                parts.append(f"name={data.candidate_name}")
            if data.score is not None:
                parts.append(f"score={data.score:.4f}")
            if data.mutation_notes:
                notes = data.mutation_notes[:60]
                parts.append(f"notes={notes}")
            line = "  ".join(parts)
            try:
                log: RichLog = self.query_one("#log-pane", RichLog)
                log.write(line)
            except Exception:  # noqa: BLE001
                pass


# =====================================================================
# DashboardCallback
# =====================================================================


class DashboardCallback:
    """Callback that drives the TUI optimisation dashboard.

    Accumulates event data and forwards it to a running
    :class:`OptimizationDashboard` (if one exists and ``textual`` is
    installed).  When ``textual`` is **not** available the callback
    degrades to a silent no-op that only logs a one-time warning.

    Attributes:
        _app: Reference to the running Textual application, or
            ``None`` when no dashboard is active.
        _events: Full event log as ``(event, data)`` tuples.
        _scores: Flat list of scores for charting.
        _candidates: List of candidate evaluation dicts.

    Example::

        cb = DashboardCallback()
        launch_dashboard(cb)
        mgr = CallbackManager([cb])
        await mgr.emit(OptimizationEvent.FIT_START, EventData())
    """

    def __init__(self) -> None:
        if _HAS_TEXTUAL:
            self._app: Any | None = None
        else:
            self._app = None
            logger.warning(
                "textual is not installed — "
                "DashboardCallback will act as a no-op.  "
                "Install with: pip install textual"
            )

        self._events: list[tuple[OptimizationEvent, EventData]] = []
        self._scores: list[float] = []
        self._candidates: list[dict[str, Any]] = []

    # ── OptimizationCallback protocol ──────────────────────────

    async def on_event(
        self,
        event: OptimizationEvent,
        data: EventData,
    ) -> None:
        """Receive an optimisation event.

        Records the event internally and — when a dashboard app is
        attached — schedules a UI refresh on the Textual event loop.

        Args:
            event: The lifecycle event.
            data: Associated payload.
        """
        self._events.append((event, data))

        if data.score is not None:
            self._scores.append(data.score)

        if event in {
            OptimizationEvent.CANDIDATE_EVALUATED,
            OptimizationEvent.CANDIDATE_ACCEPTED,
            OptimizationEvent.CANDIDATE_REJECTED,
        }:
            self._candidates.append(
                {
                    "round": data.round_idx,
                    "name": data.candidate_name,
                    "source": data.candidate_source,
                    "score": data.score,
                    "event": event.value,
                }
            )

        if self._app is not None and _HAS_TEXTUAL:
            try:
                self._app.call_from_thread(self._app.update_from_event, event, data)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Failed to post event {} to dashboard",
                    event.value,
                )

    # ── Convenience accessors ──────────────────────────────────

    @property
    def event_count(self) -> int:
        """Total number of events received so far."""
        return len(self._events)

    @property
    def scores(self) -> list[float]:
        """Score history (read-only copy)."""
        return list(self._scores)

    @property
    def candidates(self) -> list[dict[str, Any]]:
        """Candidate evaluation history (read-only copy)."""
        return list(self._candidates)


# =====================================================================
# Launch helper
# =====================================================================


def launch_dashboard(callback: DashboardCallback) -> None:
    """Start the TUI dashboard in a background thread.

    The dashboard runs in its own daemon thread so that it does not
    block the optimisation loop.  The *callback* is linked to the
    app so that subsequent events are forwarded to the UI.

    If ``textual`` is not installed, this function is a silent no-op.

    Args:
        callback: The :class:`DashboardCallback` to bind to the
            dashboard application.

    Example::

        cb = DashboardCallback()
        launch_dashboard(cb)
        # … run optimisation with cb registered …
    """
    if not _HAS_TEXTUAL:
        logger.warning("textual is not installed — dashboard launch skipped.")
        return

    app = OptimizationDashboard()
    callback._app = app

    def _run() -> None:
        """Entry-point for the dashboard thread."""
        try:
            app.run()
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).warning("Dashboard exited with an error.")
        finally:
            callback._app = None

    thread = threading.Thread(
        target=_run,
        name="agentomatic-dashboard",
        daemon=True,
    )
    thread.start()
    # Give Textual a moment to mount widgets before events
    # start flowing in.
    time.sleep(0.3)
    logger.info("Optimisation dashboard launched in background.")
