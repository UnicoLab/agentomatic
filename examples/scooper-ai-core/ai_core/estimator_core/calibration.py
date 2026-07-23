"""Post-project calibration: compare estimates against actuals.

Cleanly re-derived from the legacy Scooper ``estimator_core.calibration``
module (the legacy suggested-multiplier heuristic was replaced with a simple,
correct mean-ratio estimator).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_estimation_error(estimated: float, actual: float) -> dict[str, Any]:
    """Compute absolute/percentage error and direction for one estimate.

    Args:
        estimated: The predicted person-days.
        actual: The observed person-days.

    Returns:
        A dict with ``absolute_error``, ``percentage_error`` and ``direction``.
    """
    est = float(estimated)
    act = float(actual)
    absolute_error = act - est
    percentage_error = (act - est) / max(est, 1.0)
    if abs(percentage_error) < 1e-9:
        direction = "accurate"
    elif percentage_error > 0:
        direction = "underestimated"
    else:
        direction = "overestimated"
    return {
        "estimated": est,
        "actual": act,
        "absolute_error": round(absolute_error, 2),
        "percentage_error": round(percentage_error, 4),
        "direction": direction,
    }


def _group_ratio(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Aggregate (estimated, actual) pairs into a suggested multiplier."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "mean_ratio": 1.0, "suggested_multiplier": 1.0}
    ratios = [act / max(est, 1.0) for est, act in pairs]
    mean_ratio = sum(ratios) / n
    suggested = mean_ratio if (n >= 3 and abs(mean_ratio - 1.0) > 0.1) else 1.0
    return {
        "n": n,
        "mean_ratio": round(mean_ratio, 4),
        "suggested_multiplier": round(suggested, 4),
    }


def compute_calibration_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate calibration statistics from a list of run records.

    Args:
        records: Each record must expose ``estimated``, ``actual`` and
            optionally ``archetype`` / ``complexity`` keys.

    Returns:
        A report with overall, per-archetype and per-complexity ratios.
    """
    overall: list[tuple[float, float]] = []
    by_archetype: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_complexity: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        est = float(rec.get("estimated", 0.0))
        act = float(rec.get("actual", 0.0))
        if est <= 0 and act <= 0:
            continue
        overall.append((est, act))
        if rec.get("archetype"):
            by_archetype[str(rec["archetype"])].append((est, act))
        if rec.get("complexity"):
            by_complexity[str(rec["complexity"])].append((est, act))
    return {
        "overall": _group_ratio(overall),
        "by_archetype": {k: _group_ratio(v) for k, v in by_archetype.items()},
        "by_complexity": {k: _group_ratio(v) for k, v in by_complexity.items()},
    }
