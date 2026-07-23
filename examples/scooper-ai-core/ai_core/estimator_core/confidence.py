"""Confidence scoring for combined effort estimates.

Ported from the legacy Scooper ``estimator_core.confidence`` module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _estimator_agreement(plugin_estimates: Sequence[float]) -> float:
    """Score inter-plugin agreement in ``[0, 1]`` from their P50 values.

    Fewer than two estimates yields a neutral ``0.5``; otherwise the score is
    ``max(0, 1 - coefficient_of_variation)``.
    """
    values = [float(v) for v in plugin_estimates if v is not None and v > 0]
    if len(values) < 2:
        return 0.5
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if mean <= 0:
        return 0.5
    cv = float(arr.std(ddof=1)) / mean
    return max(0.0, 1.0 - cv)


def _case_coverage_score(similar_cases: Sequence[Any]) -> float:
    """Score similarity coverage by the number of retrieved cases."""
    n = len(similar_cases)
    if n <= 0:
        return 0.0
    return min(1.0, math.log1p(n) / math.log1p(10))


def compute_confidence(
    plugin_estimates: Sequence[float],
    similar_cases: Sequence[Any] | None = None,
    *,
    similarity_enabled: bool = True,
) -> float:
    """Combine estimator agreement and case coverage into a ``[0, 1]`` score."""
    agreement_score = _estimator_agreement(plugin_estimates)
    if similarity_enabled and similar_cases is not None:
        case_score = _case_coverage_score(similar_cases)
        confidence = 0.6 * agreement_score + 0.4 * case_score
    else:
        confidence = agreement_score
    return round(float(np.clip(confidence, 0.0, 1.0)), 4)


def confidence_label(score: float) -> str:
    """Map a numeric confidence score to a ``low``/``medium``/``high`` label."""
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"
