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

from collections.abc import Sequence
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
    #: Candidate − incumbent on the selection (validation) set.
    fit_delta: float | None = None
    #: Candidate − incumbent on the held-out set.
    holdout_delta: float | None = None
    #: How much the fit−holdout gap widened vs the incumbent.
    gap_growth: float | None = None

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
    baseline_fit: float | None = None,
    min_transfer_ratio: float = 0.0,
    holdout_tolerance: float = 0.02,
) -> GeneralizationCheck:
    """Reject candidates that look overfit to the optimisation set.

    The held-out slice is never used to *pick* candidates — only to veto
    them. Two modes:

    **Incumbent-relative** (``baseline_fit`` and ``baseline_holdout`` given —
    what :class:`~agentomatic.optimize.PromptFitter` uses). Deltas are
    measured against the current best config on the same two slices:

    1. *Gap growth* — ``(fit − holdout) − (baseline_fit − baseline_holdout)``,
       the part of the validation gain that does not replicate on held-out
       data, must not exceed ``max_gap`` (or half the validation gain, for
       large gains). A dataset whose baseline already scores differently on
       the two slices is not penalised for that.
    2. *No held-out regression* — ``holdout − baseline_holdout`` must be
       ``≥ −holdout_tolerance``.
    3. *Transfer* — when the candidate improves validation by ``Δfit > 0``,
       held-out must improve by at least ``min_transfer_ratio × Δfit``.
       A prompt that memorised validation answers improves validation only;
       this is the rule that rejects it. Skipped when the held-out slice is
       already saturated (``baseline_holdout ≥ 0.99``).

    **Absolute** (legacy — no ``baseline_fit``): ``fit − holdout ≤ max_gap``,
    plus the regression rule when ``baseline_holdout`` is given.

    Args:
        fit_score: Candidate score on the selection (validation) set.
        holdout_score: Candidate score on the held-out slice.
        max_gap: Maximum allowed gap (absolute mode) or gap growth.
        min_holdout_improvement: Required held-out lift vs baseline.
        baseline_holdout: Incumbent / baseline score on the held-out slice.
        baseline_fit: Incumbent score on the selection set (enables the
            incumbent-relative mode).
        min_transfer_ratio: Share of the validation gain that must show up
            on held-out data (``0`` disables the transfer rule).
        holdout_tolerance: Held-out regression tolerated as noise.

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
    fit_delta = fit_score - baseline_fit if baseline_fit is not None else None
    holdout_delta = holdout_score - baseline_holdout if baseline_holdout is not None else None
    gap_growth = (
        gap - (baseline_fit - baseline_holdout)
        if baseline_fit is not None and baseline_holdout is not None
        else None
    )

    def _result(ok: bool, reason: str) -> GeneralizationCheck:
        return GeneralizationCheck(
            ok=ok,
            reason=reason,
            fit_score=fit_score,
            holdout_score=holdout_score,
            gap=gap,
            max_gap=max_gap,
            fit_delta=fit_delta,
            holdout_delta=holdout_delta,
            gap_growth=gap_growth,
        )

    if gap_growth is not None:
        # Large real gains leave some unreplicated noise on a small held-out
        # slice; allow up to half the gain before calling it overfitting.
        allowed = max(max_gap, 0.5 * fit_delta) if fit_delta and fit_delta > 0 else max_gap
        if gap_growth > allowed:
            return _result(
                False,
                f"Overfit risk: the validation/held-out gap widened by {gap_growth:+.4f} "
                f"(> {allowed:.4f}); fit={fit_score:.4f}, holdout={holdout_score:.4f}",
            )
    elif gap > max_gap:
        return _result(
            False,
            f"Overfit risk: fit={fit_score:.4f} vs holdout={holdout_score:.4f} "
            f"(gap={gap:+.4f} > max_gap={max_gap:.4f})",
        )

    if holdout_delta is not None and baseline_holdout is not None:
        if holdout_delta < -abs(holdout_tolerance):
            return _result(
                False,
                f"Holdout regression: {holdout_score:.4f} < baseline "
                f"{baseline_holdout:.4f} (Δ={holdout_delta:+.4f}, "
                f"tolerance {holdout_tolerance:.4f})",
            )
        if min_holdout_improvement > 0 and holdout_delta < min_holdout_improvement:
            return _result(
                False,
                f"Holdout improvement {holdout_delta:+.4f} below "
                f"min_holdout_improvement={min_holdout_improvement:.4f}",
            )
        if (
            min_transfer_ratio > 0
            and fit_delta is not None
            and fit_delta > 0
            and baseline_holdout < 0.99
        ):
            required = min_transfer_ratio * fit_delta
            if holdout_delta < required - 1e-9:
                return _result(
                    False,
                    f"Does not transfer: validation improved {fit_delta:+.4f} but held-out "
                    f"only {holdout_delta:+.4f} (< {min_transfer_ratio:.0%} of the gain = "
                    f"{required:+.4f}) — likely fitted to the validation examples",
                )

    return _result(
        True,
        f"Generalization OK: fit={fit_score:.4f}, holdout={holdout_score:.4f}, gap={gap:+.4f}"
        + (f", held-out Δ={holdout_delta:+.4f}" if holdout_delta is not None else ""),
    )


def paired_improvement_confidence(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> float:
    """Bootstrap confidence that ``candidate`` beats ``incumbent`` on average.

    Both sequences are per-example scores on the *same* examples in the same
    order. Resampling the paired differences estimates ``P(mean Δ > 0)`` —
    a guard against accepting a candidate whose "improvement" is judge noise
    on a handful of validation examples.

    Args:
        candidate: Candidate per-example scores.
        incumbent: Incumbent per-example scores (same examples, same order).
        n_boot: Bootstrap resamples.
        seed: RNG seed (deterministic decisions).

    Returns:
        A probability in ``[0, 1]``; ``1.0`` when every paired difference is
        positive, ``0.0`` when none is, ``0.5`` when the inputs are unusable.
    """
    import random

    if not candidate or len(candidate) != len(incumbent):
        return 0.5
    diffs = [float(c) - float(i) for c, i in zip(candidate, incumbent, strict=True)]
    if all(d > 0 for d in diffs):
        return 1.0
    if all(d <= 0 for d in diffs):
        return 0.0
    rng = random.Random(seed)
    n = len(diffs)
    wins = 0
    for _ in range(max(1, n_boot)):
        total = sum(diffs[rng.randrange(n)] for _ in range(n))
        if total > 0:
            wins += 1
    return wins / max(1, n_boot)


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
    gap: float | None = None
    if holdout_score is not None and train_score is not None:
        gap = train_score - holdout_score
        if gap > 0.1:
            next_focus.append(
                f"Reduce overfitting (train/holdout gap={gap:+.3f}): prefer general rules "
                "over example-specific instructions."
            )
    elif holdout_score is not None:
        gap = score - holdout_score
    if not next_focus:
        next_focus.append("Preserve strengths; tighten output contract and edge-case coverage.")

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
