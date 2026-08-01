"""Build optimisation datasets from production feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentomatic.optimize.dataset import DataPoint, Dataset
from agentomatic.optimize.reward import FeedbackRewardAdapter


def feedback_records_to_dataset(
    records: list[dict[str, Any]],
    *,
    min_rating: int | None = None,
    only_corrections: bool = False,
    include_failures: bool = True,
) -> Dataset:
    """Convert feedback dicts into a :class:`Dataset`.

    Prefer rows with ``correction`` as ``expected_answer``. Low ratings
    without corrections are kept when *include_failures* is True so APO /
    rewrite can learn from them (expected may be empty).

    Args:
        records: Feedback records (middleware export shape).
        min_rating: Drop rows with rating below this threshold (when set).
        only_corrections: Keep only rows that include a correction.
        include_failures: Keep thumbs-down rows even without correction.

    Returns:
        A :class:`Dataset` ready for ``fit()``.
    """
    points: list[DataPoint] = []
    reward_adapter = FeedbackRewardAdapter()
    for record in records:
        query = str(record.get("query") or "").strip()
        if not query:
            continue
        correction = record.get("correction")
        rating = record.get("rating")
        if only_corrections and not correction:
            continue
        if min_rating is not None and isinstance(rating, (int, float)):
            if int(rating) < min_rating and not correction:
                continue
        if (
            not include_failures
            and isinstance(rating, (int, float))
            and int(rating) <= 1
            and not correction
        ):
            continue

        expected = str(correction or "").strip() or None
        reward = reward_adapter.reward_from_record(record)
        points.append(
            DataPoint(
                query=query,
                expected_answer=expected,
                metadata={
                    "rating": rating,
                    "comment": record.get("comment"),
                    "feedback_type": record.get("feedback_type"),
                    "reward": reward.value,
                    "source": "feedback",
                    "agent_name": record.get("agent_name"),
                },
            )
        )
    return Dataset(points=points)


async def dataset_from_feedback_collector(
    collector: Any,
    agent_name: str | None = None,
    *,
    limit: int = 500,
    **kwargs: Any,
) -> Dataset:
    """Load feedback from a :class:`FeedbackCollector` into a dataset."""
    records = await collector.get_feedback(agent_name=agent_name, limit=limit)
    return feedback_records_to_dataset(list(records or []), **kwargs)


def dataset_from_feedback_jsonl(path: str | Path, **kwargs: Any) -> Dataset:
    """Load a feedback JSONL export into a dataset."""
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return feedback_records_to_dataset(records, **kwargs)
