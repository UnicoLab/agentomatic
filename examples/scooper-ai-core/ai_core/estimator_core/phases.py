"""Phase-split distributions for the effort plan.

Ported from the legacy Scooper ``estimator_core.phases`` module.
"""

from __future__ import annotations

DEFAULT_DISTRIBUTION: dict[str, float] = {
    "discovery": 0.10,
    "design": 0.10,
    "build": 0.45,
    "integration": 0.15,
    "testing": 0.15,
    "stabilization": 0.05,
}

AI_HEAVY_DISTRIBUTION: dict[str, float] = {
    "discovery": 0.15,
    "design": 0.05,
    "build": 0.40,
    "integration": 0.10,
    "testing": 0.20,
    "stabilization": 0.10,
}

INTEGRATION_HEAVY_DISTRIBUTION: dict[str, float] = {
    "discovery": 0.10,
    "design": 0.05,
    "build": 0.35,
    "integration": 0.25,
    "testing": 0.15,
    "stabilization": 0.10,
}


def select_distribution(has_ml: bool, n_integrations: int) -> dict[str, float]:
    """Pick a phase distribution based on ML presence and integration count."""
    if has_ml:
        return AI_HEAVY_DISTRIBUTION
    if n_integrations >= 3:
        return INTEGRATION_HEAVY_DISTRIBUTION
    return DEFAULT_DISTRIBUTION


def compute_phases(total_p50: float, distribution: dict[str, float]) -> dict[str, float]:
    """Split a total P50 effort across phases per *distribution*."""
    return {phase: round(total_p50 * frac, 2) for phase, frac in distribution.items()}
