"""Epoch learnings, prompt evolution history, and generalization safety.

Provides the progressive context that makes prompt fitting actually improve:

* :class:`EpochLearning` — per-round snapshot of prompt, scores, and
  synthesised learnings (what worked / failed / next focus).
* :func:`synthesize_epoch_learning` — build learnings from eval details
  without an extra LLM call (deterministic, budget-safe).
* :func:`check_generalization` — always-on safety net that rejects
  candidates that overfit the optimisation set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EpochLearning:
    """Auditable learning snapshot for one optimisation epoch/round.

    Attributes:
        round_idx: Zero-based round index.
        prompt_snapshot: System prompt used (or accepted) this round.
        score: Best composite score after the round.
        dims: Per-dimension scores.
        accepted: Whether a new candidate was accepted.
        what_worked: Patterns from high-scoring examples.
        what_failed: Patterns from low-scoring examples.
        judge_insights: Condensed judge feedback / motivations.
        next_focus: Actionable rewrite guidance for the next round.
        candidate_name: Accepted candidate name (if any).
        train_score: Optional train-split score (overfitting signal).
        holdout_score: Optional holdout / generalization score.
        generalization_gap: ``train_or_val − holdout`` when available.
        metadata: Extra audit fields.
    """

    round_idx: int = 0
    prompt_snapshot: str = ""
    score: float = 0.0
    dims: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    judge_insights: list[str] = field(default_factory=list)
    next_focus: list[str] = field(default_factory=list)
    candidate_name: str = ""
    train_score: float | None = None
    holdout_score: float | None = None
    generalization_gap: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON artefacts / DB persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpochLearning:
        """Restore from a dictionary."""
        return cls(
            round_idx=int(data.get("round_idx", 0)),
            prompt_snapshot=str(data.get("prompt_snapshot", "")),
            score=float(data.get("score", 0.0)),
            dims=dict(data.get("dims") or {}),
            accepted=bool(data.get("accepted", False)),
            what_worked=list(data.get("what_worked") or []),
            what_failed=list(data.get("what_failed") or []),
            judge_insights=list(data.get("judge_insights") or []),
            next_focus=list(data.get("next_focus") or []),
            candidate_name=str(data.get("candidate_name", "")),
            train_score=data.get("train_score"),
            holdout_score=data.get("holdout_score"),
            generalization_gap=data.get("generalization_gap"),
            metadata=dict(data.get("metadata") or {}),
        )

    def format_for_briefing(self, *, max_items: int = 5) -> str:
        """Compact multi-line block for rewrite briefings."""
        lines = [
            f"Epoch {self.round_idx + 1}: score={self.score:.4f} "
            f"{'ACCEPTED' if self.accepted else 'no-accept'}"
            + (f" ({self.candidate_name})" if self.candidate_name else "")
        ]
        if self.holdout_score is not None:
            gap = self.generalization_gap
            gap_s = f", gap={gap:+.4f}" if gap is not None else ""
            lines.append(f"  holdout={self.holdout_score:.4f}{gap_s}")
        for label, items in (
            ("Worked", self.what_worked),
            ("Failed", self.what_failed),
            ("Judge", self.judge_insights),
            ("Next", self.next_focus),
        ):
            for item in items[:max_items]:
                lines.append(f"  [{label}] {item[:220]}")
        return "\n".join(lines)


@dataclass(slots=True)
class GeneralizationCheck:
    """Result of the always-on generalization safety net."""

    ok: bool
    reason: str
    fit_score: float
    holdout_score: float | None
    gap: float | None
    max_gap: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for trials / artefacts."""
        return asdict(self)


def check_generalization(
    *,
    fit_score: float,
    holdout_score: float | None,
    max_gap: float = 0.15,
    min_holdout_improvement: float = 0.0,
    baseline_holdout: float | None = None,
) -> GeneralizationCheck:
    """Reject candidates that look overfit to the optimisation set.

    Rules (always applied when a holdout score is available):

    1. ``fit_score − holdout_score`` must not exceed *max_gap*.
    2. If *baseline_holdout* is given, holdout must not regress below
       ``baseline_holdout − epsilon`` and should improve by at least
       *min_holdout_improvement* when that threshold is > 0.

    Args:
        fit_score: Score on the set used for candidate selection (val).
        holdout_score: Score on a held-out generalization slice / test.
        max_gap: Maximum allowed ``fit − holdout`` gap.
        min_holdout_improvement: Required holdout lift vs baseline.
        baseline_holdout: Baseline holdout score for the same slice.

    Returns:
        :class:`GeneralizationCheck` with accept/reject decision.
    """
    if holdout_score is None:
        return GeneralizationCheck(
            ok=True,
            reason="No holdout score — generalization check skipped (warn).",
            fit_score=fit_score,
            holdout_score=None,
            gap=None,
            max_gap=max_gap,
        )

    gap = fit_score - holdout_score
    if gap > max_gap:
        return GeneralizationCheck(
            ok=False,
            reason=(
                f"Overfit risk: fit={fit_score:.4f} vs holdout={holdout_score:.4f} "
                f"(gap={gap:+.4f} > max_gap={max_gap:.4f})"
            ),
            fit_score=fit_score,
            holdout_score=holdout_score,
            gap=gap,
            max_gap=max_gap,
        )

    if baseline_holdout is not None:
        holdout_delta = holdout_score - baseline_holdout
        if holdout_delta < -0.02:
            return GeneralizationCheck(
                ok=False,
                reason=(
                    f"Holdout regression: {holdout_score:.4f} < baseline "
                    f"{baseline_holdout:.4f} (Δ={holdout_delta:+.4f})"
                ),
                fit_score=fit_score,
                holdout_score=holdout_score,
                gap=gap,
                max_gap=max_gap,
            )
        if min_holdout_improvement > 0 and holdout_delta < min_holdout_improvement:
            return GeneralizationCheck(
                ok=False,
                reason=(
                    f"Holdout improvement {holdout_delta:+.4f} below "
                    f"min_holdout_improvement={min_holdout_improvement:.4f}"
                ),
                fit_score=fit_score,
                holdout_score=holdout_score,
                gap=gap,
                max_gap=max_gap,
            )

    return GeneralizationCheck(
        ok=True,
        reason=(
            f"Generalization OK: fit={fit_score:.4f}, holdout={holdout_score:.4f}, gap={gap:+.4f}"
        ),
        fit_score=fit_score,
        holdout_score=holdout_score,
        gap=gap,
        max_gap=max_gap,
    )


def synthesize_epoch_learning(
    *,
    round_idx: int,
    prompt_snapshot: str,
    score: float,
    dims: dict[str, float] | None,
    eval_details: list[dict[str, Any]],
    accepted: bool = False,
    candidate_name: str = "",
    train_score: float | None = None,
    holdout_score: float | None = None,
    failure_threshold: float = 0.5,
    success_threshold: float = 0.75,
    max_items: int = 5,
) -> EpochLearning:
    """Build progressive learnings from evaluation details (no LLM).

    Extracts concrete failure/success patterns and judge motivations so
    the next rewrite pass has grounded signal instead of bare scores.
    """
    scored = sorted(
        eval_details,
        key=lambda r: float(r.get("score", r.get("avg_score", 0.0))),
    )
    failures = [
        r for r in scored if float(r.get("score", r.get("avg_score", 0.0))) < failure_threshold
    ]
    successes = [
        r for r in scored if float(r.get("score", r.get("avg_score", 0.0))) >= success_threshold
    ]

    what_failed: list[str] = []
    for f in failures[:max_items]:
        q = str(f.get("query", ""))[:120]
        fb = str(f.get("feedback") or f.get("reason") or f.get("motivation") or "")[:160]
        exp = str(f.get("expected", ""))[:100]
        what_failed.append(
            f"q={q!r} expected≈{exp!r} score={float(f.get('score', f.get('avg_score', 0))):.2f}"
            + (f" | {fb}" if fb else "")
        )

    what_worked: list[str] = []
    for s in successes[-max_items:]:
        q = str(s.get("query", ""))[:120]
        fb = str(s.get("feedback") or s.get("reason") or "")[:120]
        what_worked.append(
            f"q={q!r} score={float(s.get('score', s.get('avg_score', 0))):.2f}"
            + (f" | {fb}" if fb else "")
        )

    judge_insights: list[str] = []
    for r in scored:
        for key in ("motivation", "improvement_hints", "what_failed", "feedback", "reason"):
            val = r.get(key)
            if isinstance(val, list):
                for item in val:
                    text = str(item).strip()
                    if text and text not in judge_insights:
                        judge_insights.append(text[:220])
            elif isinstance(val, str) and val.strip():
                text = val.strip()
                if text not in judge_insights and not text.startswith("Judge evaluation failed"):
                    judge_insights.append(text[:220])
        if len(judge_insights) >= max_items * 2:
            break

    next_focus: list[str] = []
    if what_failed:
        next_focus.append(
            "Address lowest-scoring failure modes without hardcoding those exact queries."
        )
    if dims:
        weak = sorted(dims.items(), key=lambda kv: kv[1])[:2]
        for name, val in weak:
            if val < 0.7:
                next_focus.append(f"Improve dimension '{name}' (currently {val:.3f}).")
    if holdout_score is not None and train_score is not None:
        gap = train_score - holdout_score
        if gap > 0.1:
            next_focus.append(
                f"Reduce overfitting (train/holdout gap={gap:+.3f}): prefer general rules "
                "over example-specific instructions."
            )
    if not next_focus:
        next_focus.append("Preserve strengths; tighten output contract and edge-case coverage.")

    gap = None
    if holdout_score is not None and train_score is not None:
        gap = train_score - holdout_score
    elif holdout_score is not None:
        gap = score - holdout_score

    return EpochLearning(
        round_idx=round_idx,
        prompt_snapshot=prompt_snapshot,
        score=score,
        dims=dict(dims or {}),
        accepted=accepted,
        what_worked=what_worked,
        what_failed=what_failed,
        judge_insights=judge_insights[: max_items * 2],
        next_focus=next_focus[:max_items],
        candidate_name=candidate_name,
        train_score=train_score,
        holdout_score=holdout_score,
        generalization_gap=gap,
    )


def format_learnings_history(
    learnings: list[EpochLearning],
    *,
    max_epochs: int = 8,
) -> str:
    """Format recent epoch learnings for rewrite prompts / summaries."""
    if not learnings:
        return "No epoch learnings yet."
    recent = learnings[-max_epochs:]
    return "\n\n".join(e.format_for_briefing() for e in recent)


def split_holdout(
    points: list[Any],
    *,
    fraction: float = 0.2,
    min_size: int = 1,
    max_size: int = 50,
    seed: int = 42,
) -> tuple[list[Any], list[Any]]:
    """Split a list into (fit_points, holdout_points) deterministically.

    Always reserves a holdout when at least 2 points exist so generalization
    checks can run even without an explicit testset. Default ``min_size=1``
    keeps the safety net alive for tiny datasets (1 fit / 1 holdout).
    """
    n = len(points)
    if n < 2:
        return list(points), []

    hold_n = max(min_size, min(max_size, int(round(n * fraction))))
    # Keep ≥1 fit point. For tiny sets (n<4) reserve a single holdout;
    # for larger sets cap at half so fit remains majority.
    if n < 4:
        hold_n = min(hold_n, 1)
    else:
        hold_n = min(hold_n, n // 2)
    hold_n = max(1, min(hold_n, n - 1))
    # Deterministic shuffle via index permutation
    idxs = list(range(n))
    # Simple LCG shuffle for reproducibility without importing random globally
    state = seed & 0xFFFFFFFF
    for i in range(n - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        j = state % (i + 1)
        idxs[i], idxs[j] = idxs[j], idxs[i]

    hold_idxs = set(idxs[:hold_n])
    fit_pts = [p for i, p in enumerate(points) if i not in hold_idxs]
    hold_pts = [p for i, p in enumerate(points) if i in hold_idxs]
    return fit_pts, hold_pts
