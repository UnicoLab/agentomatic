"""Rich context objects for informed prompt rewriting.

Provides :class:`OptimizationContext` — a structured accumulator of
evaluation state, score history, pipeline metadata, and dataset
characteristics that optimisers receive to make deeply informed
rewrite decisions.

Example::

    ctx = OptimizationContext(
        baseline_score=0.72,
        current_score=0.85,
        score_history=[
            RoundStats(round_idx=0, score=0.72, dims={"relevance": 0.68}),
            RoundStats(round_idx=1, score=0.79, dims={"relevance": 0.75}),
        ],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# =====================================================================
# Score history per round
# =====================================================================


@dataclass(slots=True)
class RoundStats:
    """Snapshot of scores and metadata from a single optimisation round.

    Stored in :attr:`OptimizationContext.score_history` so the rewrite
    model can see performance trends across iterations.
    """

    round_idx: int = 0
    score: float = 0.0
    dims: dict[str, float] = field(default_factory=dict)
    best_candidate: str = ""
    accepted: bool = False
    n_candidates: int = 0
    elapsed_seconds: float = 0.0


# =====================================================================
# Dataset summary
# =====================================================================


@dataclass(slots=True)
class DatasetSummary:
    """Lightweight summary of dataset characteristics.

    Included in :class:`OptimizationContext` so optimisers can adapt
    their strategies based on dataset size and composition.
    """

    n_samples: int = 0
    categories: list[str] = field(default_factory=list)
    avg_query_length: int = 0
    avg_expected_length: int = 0
    has_context: bool = False


# =====================================================================
# Full optimisation context
# =====================================================================


@dataclass(slots=True)
class OptimizationContext:
    """Rich context passed to optimisers for informed prompt rewriting.

    Accumulates state across iterations so the rewriting model can
    reason about *trends* and *root causes* rather than treating each
    round as an independent rewrite.

    Attributes:
        baseline_score: Score of the original, unoptimised config.
        baseline_dims: Per-dimension scores at baseline.
        current_score: Score of the current-best config.
        current_dims: Per-dimension scores for the current best.
        score_history: List of :class:`RoundStats` for previous rounds.
        failure_clusters: Failure-analysis clusters (label, count, fix).
        eval_details: Full per-point evaluation results with pipeline
            context (``retrieval_context``, ``tool_calls``, ``reasoning``,
            ``citations``, ``metadata``, ``duration_ms``).
        dataset_summary: Size and shape of the evaluation dataset.
        metric_names: Names of the metrics being evaluated.
        round_idx: Current round index.
        total_rounds: Total planned rounds.
        agent_name: Name of the agent being optimised (for briefing).
    """

    # ── Scores ───────────────────────────────────────────────────────
    baseline_score: float = 0.0
    baseline_dims: dict[str, float] = field(default_factory=dict)
    current_score: float = 0.0
    current_dims: dict[str, float] = field(default_factory=dict)

    # ── History ──────────────────────────────────────────────────────
    score_history: list[RoundStats] = field(default_factory=list)

    # ── Failure analysis ─────────────────────────────────────────────
    failure_clusters: list[dict[str, Any]] = field(default_factory=list)

    # ── Evaluation detail (with pipeline context) ────────────────────
    eval_details: list[dict[str, Any]] = field(default_factory=list)

    # ── Dataset ──────────────────────────────────────────────────────
    dataset_summary: DatasetSummary = field(default_factory=DatasetSummary)
    metric_names: list[str] = field(default_factory=list)

    # ── Progress ─────────────────────────────────────────────────────
    round_idx: int = 0
    total_rounds: int = 0
    agent_name: str = ""

    # ── Formatting helpers ───────────────────────────────────────────

    def format_score_history(self, max_rounds: int = 5) -> str:
        """Format recent score history as a human-readable string.

        Args:
            max_rounds: Maximum number of recent rounds to include.

        Returns:
            Multi-line string showing score evolution with trend
            indicators.
        """
        if not self.score_history:
            return "No previous rounds."

        recent = self.score_history[-max_rounds:]
        lines: list[str] = []
        for i, rs in enumerate(recent):
            trend = ""
            if i > 0:
                prev = recent[i - 1].score
                delta = rs.score - prev
                trend = f" (Δ {delta:+.4f})" if abs(delta) > 1e-6 else " (=)"
            accepted = "✅" if rs.accepted else "❌"
            lines.append(
                f"  Round {rs.round_idx + 1}: {rs.score:.4f}{trend} "
                f"{accepted} ({rs.n_candidates} candidates, "
                f"{rs.elapsed_seconds:.1f}s)"
            )

        return "\n".join(lines)

    def format_score_sparkline(self) -> str:
        """Format score evolution as an inline sparkline.

        Returns:
            String like ``📈 0.72 → 0.79 → 0.85``.
        """
        if not self.score_history:
            return f"📊 baseline={self.baseline_score:.3f}"

        scores = [self.baseline_score] + [rs.score for rs in self.score_history]
        parts = [f"{s:.3f}" for s in scores]
        trend = "📈" if scores[-1] > scores[0] else "📉"
        return f"{trend} {' → '.join(parts)}"

    def format_dimension_table(self) -> str:
        """Format per-dimension comparison (baseline vs current).

        Returns:
            Multi-line table of dimension scores with deltas.
        """
        all_dims = sorted(set(self.baseline_dims) | set(self.current_dims))
        if not all_dims:
            return "No per-dimension scores available."

        lines: list[str] = [
            f"  {'Dimension':<20} {'Baseline':>10} {'Current':>10} {'Delta':>10}",
            f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10}",
        ]
        for dim in all_dims:
            base = self.baseline_dims.get(dim, 0.0)
            curr = self.current_dims.get(dim, 0.0)
            delta = curr - base
            lines.append(f"  {dim:<20} {base:>10.4f} {curr:>10.4f} {delta:>+10.4f}")
        return "\n".join(lines)

    def format_failure_clusters(self) -> str:
        """Format failure clusters for the rewrite prompt.

        Returns:
            Multi-line summary of failure categories.
        """
        if not self.failure_clusters:
            return "No failure clusters identified."

        lines: list[str] = []
        for i, cluster in enumerate(self.failure_clusters, 1):
            label = cluster.get("label", "unknown")
            count = cluster.get("count", 0)
            fix = cluster.get("suggested_fix", "N/A")
            severity = cluster.get("severity", 0.0)
            lines.append(f"  {i}. [{severity:.2f}] {label} ({count} cases): {fix[:120]}")
        return "\n".join(lines)

    def format_eval_details_for_rewrite(
        self,
        max_failures: int = 5,
        max_successes: int = 3,
    ) -> str:
        """Format evaluation details for inclusion in a rewrite prompt.

        Includes full pipeline context (retrieval docs, tool calls,
        reasoning) for the lowest-scoring and highest-scoring results.

        Args:
            max_failures: Number of worst results to include.
            max_successes: Number of best results to include.

        Returns:
            Multi-section string with failures and successes.
        """
        if not self.eval_details:
            return "No evaluation details available."

        scored = sorted(
            self.eval_details,
            key=lambda r: r.get("score", r.get("avg_score", 0.0)),
        )
        failures = scored[:max_failures]
        successes = scored[-max_successes:]

        lines: list[str] = []

        # ── Failures ─────────────────────────────────────────────────
        if failures:
            lines.append("### Failures (lowest-scoring)")
            for idx, f in enumerate(failures, 1):
                score = f.get("score", f.get("avg_score", 0.0))
                lines.append(f"\n**Failure {idx}** (score: {score:.3f})")
                lines.append(f"- Query: {f.get('query', 'N/A')[:300]}")
                lines.append(f"- Expected: {str(f.get('expected', 'N/A'))[:300]}")
                lines.append(f"- Response: {f.get('response', 'N/A')[:300]}")

                feedback = f.get("feedback") or f.get("reason") or f.get("details", "")
                if feedback:
                    lines.append(f"- Feedback: {str(feedback)[:250]}")

                dims = f.get("dimensions", {})
                if dims:
                    dim_str = ", ".join(f"{k}={v:.3f}" for k, v in dims.items())
                    lines.append(f"- Dimensions: {dim_str}")

                # Pipeline context
                ret_ctx = f.get("retrieval_context", [])
                if ret_ctx:
                    docs = "; ".join(str(d)[:100] for d in ret_ctx[:3])
                    lines.append(f"- Retrieved docs: {docs}")

                tool_calls = f.get("tool_calls", [])
                if tool_calls:
                    tools = ", ".join(str(t.get("name", t)) for t in tool_calls[:3])
                    lines.append(f"- Tool calls: {tools}")

                reasoning = f.get("reasoning", "")
                if reasoning:
                    lines.append(f"- Reasoning: {str(reasoning)[:200]}")

        # ── Successes ────────────────────────────────────────────────
        if successes:
            lines.append("\n### Successes (highest-scoring)")
            for idx, s in enumerate(successes, 1):
                score = s.get("score", s.get("avg_score", 0.0))
                lines.append(f"\n**Success {idx}** (score: {score:.3f})")
                lines.append(f"- Query: {s.get('query', 'N/A')[:200]}")
                lines.append(f"- Response: {s.get('response', 'N/A')[:200]}")

        return "\n".join(lines)
