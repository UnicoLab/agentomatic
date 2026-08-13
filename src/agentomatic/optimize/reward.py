"""Reward protocol — map metrics / judges / feedback onto scalar rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentomatic.optimize.metrics import EvalResult
from agentomatic.optimize.rollout import RewardSignal


@runtime_checkable
class RewardProtocol(Protocol):
    """Convert an evaluation result into a :class:`RewardSignal`."""

    def reward_from_eval(self, result: EvalResult) -> RewardSignal:
        """Map a metric :class:`EvalResult` to a reward signal."""
        ...


@dataclass(slots=True)
class MetricRewardAdapter:
    """Default adapter: composite metric score → reward."""

    source: str = "metric"

    def reward_from_eval(self, result: EvalResult) -> RewardSignal:
        """Build a reward from score / reason / dimensional metadata."""
        dims = {}
        raw_dims = result.metadata.get("dimensions") or {}
        if isinstance(raw_dims, dict):
            dims = {
                str(k): float(v) for k, v in raw_dims.items() if isinstance(v, (int, float))
            }
        return RewardSignal(
            value=float(result.score),
            dimensions=dims,
            source=self.source,
            reason=result.reason or "",
        )


@dataclass(slots=True)
class FeedbackRewardAdapter:
    """Map thumbs / rating feedback onto a ``[0, 1]`` reward."""

    thumbs_up: float = 1.0
    thumbs_down: float = 0.0

    def reward_from_eval(self, result: EvalResult) -> RewardSignal:
        """Treat metric score as feedback-shaped reward (pass-through)."""
        return RewardSignal(
            value=float(result.score),
            source="feedback",
            reason=result.reason or "",
        )

    def reward_from_rating(self, rating: int | None, *, comment: str = "") -> RewardSignal:
        """Convert a 1–5 style rating into a reward signal."""
        if rating is None:
            return RewardSignal(value=0.0, source="feedback", reason=comment)
        if rating <= 1:
            value = self.thumbs_down
        elif rating >= 5:
            value = self.thumbs_up
        else:
            # Linear map 1..5 → 0..1
            value = max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))
        return RewardSignal(
            value=value,
            dimensions={"rating": float(rating)},
            source="feedback",
            reason=comment,
        )

    def reward_from_record(self, record: dict[str, Any]) -> RewardSignal:
        """Convert a feedback dict (middleware export) into a reward."""
        rating = record.get("rating")
        comment = str(record.get("comment") or record.get("correction") or "")
        if isinstance(rating, (int, float)):
            return self.reward_from_rating(int(rating), comment=comment)
        # Correction present without rating → treat as positive supervision
        if record.get("correction"):
            return RewardSignal(value=1.0, source="feedback", reason=comment)
        return RewardSignal(value=0.0, source="feedback", reason=comment)


def resolve_reward_adapter(name: str | RewardProtocol = "metric") -> RewardProtocol:
    """Resolve a reward adapter by name or pass through an instance."""
    if isinstance(name, RewardProtocol) and not isinstance(name, str):
        return name
    key = str(name).strip().lower()
    if key in {"metric", "default", "composite"}:
        return MetricRewardAdapter()
    if key in {"feedback", "thumbs"}:
        return FeedbackRewardAdapter()
    raise ValueError(f"Unknown reward adapter '{name}'. Available: metric, feedback")
