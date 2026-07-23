"""Deterministic estimation primitives shared across plugins and agents."""

from __future__ import annotations

from ai_core.estimator_core.calibration import (
    compute_calibration_report,
    compute_estimation_error,
)
from ai_core.estimator_core.confidence import (
    compute_confidence,
    confidence_label,
)
from ai_core.estimator_core.constants import (
    ARCHETYPE_MULTIPLIERS,
    BASE_DAYS_PER_MODULE,
    COMPLEXITY_MULTIPLIERS,
    COMPLEXITY_NUMERIC,
    COMPLEXITY_TO_BASE_KEY,
    DEVOPS_FRACTION,
    PM_FRACTION,
    TESTING_FRACTION,
    archetype_multiplier,
    base_days,
    complexity_multiplier,
)
from ai_core.estimator_core.distributions import (
    compute_percentiles,
    pert_sample,
    triangular_sample,
)
from ai_core.estimator_core.phases import (
    compute_phases,
    select_distribution,
)

__all__ = [
    "ARCHETYPE_MULTIPLIERS",
    "BASE_DAYS_PER_MODULE",
    "COMPLEXITY_MULTIPLIERS",
    "COMPLEXITY_NUMERIC",
    "COMPLEXITY_TO_BASE_KEY",
    "DEVOPS_FRACTION",
    "PM_FRACTION",
    "TESTING_FRACTION",
    "archetype_multiplier",
    "base_days",
    "complexity_multiplier",
    "compute_calibration_report",
    "compute_confidence",
    "compute_estimation_error",
    "compute_percentiles",
    "compute_phases",
    "confidence_label",
    "pert_sample",
    "select_distribution",
    "triangular_sample",
]
