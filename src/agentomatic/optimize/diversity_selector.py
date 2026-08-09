"""Diversity-aware test case selection for dataset curation.

Ensures evaluation datasets cover a wide range of scenarios
by selecting cases that differ in category, input length, and
semantic content.  Useful before optimisation runs to maximise
the signal per evaluation point.

Example::

    from agentomatic.optimize.diversity_selector import DiversitySelector

    selector = DiversitySelector()
    diverse = selector.select_diverse(all_cases, num_select=20)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class DiversityConfig:
    """Controls the diversity selection behaviour."""

    min_distance: float = 0.3
    """Minimum embedding distance between selected cases (future)."""

    ensure_category_coverage: bool = True
    """Ensure at least one case from each category."""

    ensure_length_diversity: bool = True
    """Distribute across short/medium/long buckets."""

    max_duplicate_categories: int = 3
    """Max cases allowed from the same category."""


class DiversitySelector:
    """Selects a diverse subset of test cases to maximise evaluation coverage.

    Uses lightweight heuristics (category, length buckets, keyword
    overlap) that work without embeddings.  Future versions may
    incorporate embedding-based semantic diversity.

    Example::

        selector = DiversitySelector()
        curated = selector.select_diverse(cases, num_select=15)
    """

    def __init__(self, config: DiversityConfig | None = None) -> None:
        self.config = config or DiversityConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_diverse(
        self,
        cases: list[Any],
        num_select: int,
        min_distance: float = 0.3,
    ) -> list[Any]:
        """Select a diverse subset from *cases*.

        Args:
            cases: Full list of test case objects.  Each case must have
                an ``input`` attribute (str) and optionally a
                ``category`` or ``category`` attribute.
            num_select: Target number of cases to return.
            min_distance: Unused (reserved for embedding-based selection).

        Returns:
            A subset of *cases* (up to *num_select* items).
        """
        if len(cases) <= num_select:
            return list(cases)

        # Extract metadata
        annotated = self._annotate(cases)

        selected: list[Any] = []
        categories_used: dict[str, int] = {}
        length_bucket_counts: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
        max_per_bucket = max(num_select // 3, 1)

        # First pass: ensure category coverage
        if self.config.ensure_category_coverage:
            seen_cats: set[str] = set()
            for case, meta in annotated:
                cat = meta["category"]
                if cat not in seen_cats and len(selected) < num_select:
                    selected.append(case)
                    seen_cats.add(cat)
                    categories_used[cat] = categories_used.get(cat, 0) + 1
                    bucket = meta["bucket"]
                    length_bucket_counts[bucket] += 1

        # Second pass: fill diversity-aware
        remaining = [(c, m) for c, m in annotated if c not in selected]
        random.shuffle(remaining)

        for case, meta in remaining:
            if len(selected) >= num_select:
                break

            cat = meta["category"]
            bucket = meta["bucket"]

            # Check category budget
            if categories_used.get(cat, 0) >= self.config.max_duplicate_categories:
                continue

            # Check length bucket budget
            if self.config.ensure_length_diversity:
                if length_bucket_counts.get(bucket, 0) >= max_per_bucket:
                    continue

            selected.append(case)
            categories_used[cat] = categories_used.get(cat, 0) + 1
            length_bucket_counts[bucket] += 1

        # Third pass: fill remaining slots
        if len(selected) < num_select:
            still_remaining = [c for c, _ in annotated if c not in selected]
            random.shuffle(still_remaining)
            selected.extend(still_remaining[: num_select - len(selected)])

        return selected[:num_select]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _annotate(self, cases: list[Any]) -> list[tuple[Any, dict[str, Any]]]:
        """Attach metadata (category, length bucket) to each case."""
        annotated: list[tuple[Any, dict[str, Any]]] = []
        for case in cases:
            inp = _case_input(case)
            cat = getattr(case, "category", None) or getattr(case, "label", None) or "general"
            bucket = _length_bucket(inp)
            annotated.append((case, {"category": str(cat), "bucket": bucket}))
        return annotated

    @staticmethod
    def estimate_diversity(cases: list[Any]) -> float:
        """Return a rough diversity score (0.0–1.0) for a case set.

        Higher = more diverse.  Based on category count and
        length-bucket distribution.

        Args:
            cases: List of test case objects.

        Returns:
            Diversity score between 0.0 and 1.0.
        """
        if not cases:
            return 0.0
        cats: set[str] = set()
        buckets: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
        for case in cases:
            inp = _case_input(case)
            cat = getattr(case, "category", "") or ""
            if cat:
                cats.add(str(cat))
            buckets[_length_bucket(inp)] += 1
        n = len(cases)
        cat_score = min(len(cats) / max(n, 1) * 2, 1.0)
        bucket_score = sum(1 for v in buckets.values() if v > 0) / 3
        return (cat_score + bucket_score) / 2


# =====================================================================
# Helpers
# =====================================================================


def _case_input(case: Any) -> str:
    """Extract input text from DeepEval cases or :class:`DataPoint`."""
    return str(getattr(case, "input", None) or getattr(case, "query", None) or "")


def _length_bucket(text: str) -> str:
    """Classify input text by length."""
    ln = len(str(text))
    if ln < 50:
        return "short"
    if ln > 200:
        return "long"
    return "medium"
