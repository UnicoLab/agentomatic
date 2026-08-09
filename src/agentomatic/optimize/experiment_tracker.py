"""SQLite-backed experiment tracker for optimisation runs.

Provides durable tracking of all optimisation experiments with
per-iteration history, query methods, and rich table display.
Inspired by ML experiment trackers (MLflow, Weights & Biases).

Example::

    from agentomatic.optimize.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker("experiments.db")

    # Start tracking
    exp_id = await tracker.start_experiment(
        agent_name="my_agent",
        strategy="iterative_refinement",
        model="ollama/mistral:7b",
    )

    # Log each iteration
    await tracker.log_iteration(
        experiment_id=exp_id,
        iteration=1,
        score=0.72,
        prompt="You are a helpful assistant...",
        metrics={"answer_relevancy": 0.85, "geval": 0.72},
    )

    # End and display
    await tracker.end_experiment(exp_id, final_score=0.91)
    tracker.display_experiments()  # alias: show_experiments()
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# =====================================================================
# Schema
# =====================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    strategy TEXT NOT NULL,
    model TEXT NOT NULL,
    initial_score REAL,
    final_score REAL,
    best_score REAL,
    total_iterations INTEGER,
    target_score REAL,
    status TEXT DEFAULT 'running',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    extra JSON
);

CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    score REAL NOT NULL,
    delta REAL DEFAULT 0.0,
    prompt TEXT,
    metrics JSON,
    duration_ms REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_iterations_exp ON iterations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiments_agent ON experiments(agent_name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
"""


# =====================================================================
# ExperimentTracker
# =====================================================================


class ExperimentTracker:
    """Tracks optimisation experiments in a local SQLite database.

    Args:
        db_path: Path to the SQLite database file.
            Created automatically if it doesn't exist.

    Example::

        tracker = ExperimentTracker()
        exp_id = await tracker.start_experiment("bot", "iterative", "ollama/mistral")
    """

    def __init__(self, db_path: str = "optimization_results/experiments.db") -> None:
        self._db_path = Path(db_path)
        self._memory_conn: sqlite3.Connection | None = None
        if str(db_path) == ":memory:":
            self._memory_conn = self._get_conn()
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        # Reuse a process-local connection for file-backed DBs.
        conn = getattr(self, "_file_conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._file_conn = conn
        return conn

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    async def start_experiment(
        self,
        agent_name: str,
        strategy: str = "iterative_refinement",
        model: str = "ollama/mistral:7b",
        target_score: float = 0.85,
        experiment_id: str | None = None,
        **extra: Any,
    ) -> str:
        """Create a new experiment record.

        Args:
            agent_name: Name of the agent being optimised.
            strategy: Optimisation strategy identifier.
            model: Model used for rewriting.
            target_score: Target composite score.
            experiment_id: Optional stable ID (defaults to a short UUID).
            extra: Additional metadata stored as JSON.

        Returns:
            The experiment UUID.
        """
        exp_id = (experiment_id or str(uuid.uuid4()))[:12]
        import json

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO experiments
                   (id, agent_name, strategy, model, target_score,
                    status, created_at, extra)
                   VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
                (
                    exp_id,
                    agent_name,
                    strategy,
                    model,
                    target_score,
                    datetime.now().isoformat(),
                    json.dumps(extra),
                ),
            )
            conn.commit()

        logger.info(f"🧪 Experiment started: {exp_id} agent={agent_name}")
        return exp_id

    async def set_initial_score(self, experiment_id: str, score: float) -> None:
        """Record the baseline score for an experiment."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE experiments SET initial_score=? WHERE id=?",
                (float(score), experiment_id),
            )
            conn.commit()

    async def log_iteration(
        self,
        experiment_id: str,
        iteration: int,
        score: float,
        prompt: str = "",
        metrics: dict[str, float] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a single optimisation round.

        Args:
            experiment_id: The experiment UUID.
            iteration: Round number (1-based).
            score: Composite score for this round.
            prompt: The prompt used (truncated to 64KB).
            metrics: Per-metric scores.
            duration_ms: Round duration in milliseconds.
        """
        import json

        # Get previous score for delta calculation
        prev_score = 0.0
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT score FROM iterations WHERE experiment_id=? ORDER BY iteration DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if row:
                prev_score = row["score"]

        delta = score - prev_score

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO iterations
                   (experiment_id, iteration, score, delta, prompt, metrics, duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    iteration,
                    score,
                    round(delta, 6),
                    prompt[:65535] if prompt else "",
                    json.dumps(metrics) if metrics else None,
                    duration_ms,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    async def end_experiment(
        self,
        experiment_id: str,
        final_score: float = 0.0,
        best_score: float | None = None,
        total_iterations: int = 0,
        status: str = "completed",
    ) -> None:
        """Mark an experiment as finished.

        Args:
            experiment_id: The experiment UUID.
            final_score: Final composite score.
            best_score: Best score achieved (auto-computed if None).
            total_iterations: Total rounds executed.
            status: ``"completed"`` or ``"stopped"``.
        """
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            # Auto-compute best score if not provided
            if best_score is None:
                row = conn.execute(
                    "SELECT MAX(score) as best FROM iterations WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()
                best_score = row["best"] if row and row["best"] is not None else final_score

            # Auto-compute iterations
            if total_iterations == 0:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM iterations WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchone()
                total_iterations = row["cnt"] if row else 0

            conn.execute(
                """UPDATE experiments
                   SET final_score=?, best_score=?, total_iterations=?,
                       status=?, completed_at=?
                   WHERE id=?""",
                (final_score, best_score, total_iterations, status, now, experiment_id),
            )
            conn.commit()

        logger.info(
            f"🏁 Experiment completed: {experiment_id} "
            f"best={best_score:.4f} iters={total_iterations}"
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get a single experiment by ID.

        Returns:
            Dictionary with experiment fields, or ``None``.
        """
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        return dict(row) if row else None

    def get_experiments(
        self,
        agent_name: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List experiments, optionally filtered.

        Args:
            agent_name: Filter by agent name.
            status: Filter by status. ``None`` (default) returns completed
                **and** stopped runs so early-stopped fits stay visible.
            limit: Maximum number of results.

        Returns:
            List of experiment dictionaries.
        """
        params: list[Any] = []
        if status is None:
            query = "SELECT * FROM experiments WHERE status IN ('completed', 'stopped') "
        else:
            query = "SELECT * FROM experiments WHERE status=? "
            params.append(status)
        if agent_name:
            query += "AND agent_name=? "
            params.append(agent_name)
        query += "ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_iterations(self, experiment_id: str) -> list[dict[str, Any]]:
        """Get all iterations for an experiment.

        Returns:
            List of iteration dictionaries sorted by round.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM iterations WHERE experiment_id=? ORDER BY iteration",
                (experiment_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_best_experiment(self, agent_name: str) -> dict[str, Any] | None:
        """Return the best finished experiment for an agent.

        Includes both ``completed`` and ``stopped`` runs so early-stopped fits
        (patience / threshold / plateau) remain eligible as the best score.

        Args:
            agent_name: The agent name to query.

        Returns:
            Best experiment dictionary or ``None``.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT * FROM experiments
                   WHERE agent_name=?
                     AND status IN ('completed', 'stopped')
                     AND best_score IS NOT NULL
                   ORDER BY best_score DESC LIMIT 1""",
                (agent_name,),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display_experiments(self, agent_name: str | None = None, limit: int = 10) -> None:
        """Print a rich table of experiments.

        Args:
            agent_name: Filter by agent (shows all if ``None``).
            limit: Maximum rows to display.
        """
        experiments = self.get_experiments(agent_name=agent_name, limit=limit)
        if not experiments:
            print("No experiments found.")
            return

        header = (
            f"{'ID':<14} {'Agent':<20} {'Strategy':<22} {'Best':>8} {'Δ':>8} "
            f"{'Iters':>5} {'Status':<10}"
        )
        sep = "─" * len(header)
        print(sep)
        print(header)
        print(sep)

        for exp in experiments:
            eid = exp.get("id", "")[:12]
            agent = exp.get("agent_name", "")[:18]
            strategy = exp.get("strategy", "")[:20]
            best = exp.get("best_score") or 0.0
            initial = exp.get("initial_score") or 0.0
            delta = best - initial
            iters = exp.get("total_iterations") or 0
            status = exp.get("status", "?")[:8]
            delta_s = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            print(
                f"{eid:<14} {agent:<20} {strategy:<22} {best:>8.4f} "
                f"{delta_s:>8} {iters:>5} {status:<10}"
            )

        print(sep)
        print(f"  {len(experiments)} experiment(s) shown")

    def show_experiments(self, agent_name: str | None = None, limit: int = 10) -> None:
        """Alias for :meth:`display_experiments`."""
        self.display_experiments(agent_name=agent_name, limit=limit)

    def close(self) -> None:
        """No-op for API compatibility (SQLite connections are short-lived)."""


# =====================================================================
# Convenience singleton
# =====================================================================

_DEFAULT_TRACKER: ExperimentTracker | None = None


def get_tracker(db_path: str = "optimization_results/experiments.db") -> ExperimentTracker:
    """Get (or create) the default experiment tracker singleton.

    Args:
        db_path: Database path (only used on first call).

    Returns:
        Shared :class:`ExperimentTracker` instance.
    """
    global _DEFAULT_TRACKER
    if _DEFAULT_TRACKER is None:
        _DEFAULT_TRACKER = ExperimentTracker(db_path)
    return _DEFAULT_TRACKER


def reset_tracker() -> None:
    """Reset the singleton so the next :func:`get_tracker` call creates a fresh instance."""
    global _DEFAULT_TRACKER
    _DEFAULT_TRACKER = None
