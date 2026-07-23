"""Estimation constants: multipliers and base efforts.

Ported from the legacy Scooper ``estimator_core.constants`` module. These
values are the deterministic backbone shared by the rule-based and Monte Carlo
plugins. Effort *category names* (development, devops, …) are owned by each
plugin when building :class:`~ai_core.schemas.EffortCategory` rows.
"""

from __future__ import annotations

# Complexity level -> effort multiplier.
COMPLEXITY_MULTIPLIERS: dict[str, float] = {
    "low": 0.6,
    "medium": 1.0,
    "high": 1.5,
    "critical": 2.5,
}

# Base person-days per module by (mapped) complexity key.
BASE_DAYS_PER_MODULE: dict[str, int] = {
    "simple": 3,
    "medium": 8,
    "complex": 15,
    "critical": 30,
}

# Map a module complexity level to a base-days key.
COMPLEXITY_TO_BASE_KEY: dict[str, str] = {
    "low": "simple",
    "medium": "medium",
    "high": "complex",
    "critical": "critical",
}

# Archetype -> effort multiplier.
ARCHETYPE_MULTIPLIERS: dict[str, float] = {
    "greenfield": 1.0,
    "brownfield": 0.7,
    "bug_fix": 0.3,
    "integration": 1.2,
    "migration": 1.4,
    "refactor": 0.6,
    "data_pipeline": 1.1,
    "ai_ml": 1.8,
    "crud_admin": 0.5,
    "reporting_dashboard": 0.8,
    "security_compliance": 1.3,
    "infrastructure": 0.9,
}

# Cross-cutting effort as a fraction of the module subtotal.
DEVOPS_FRACTION: float = 0.15
TESTING_FRACTION: float = 0.20
PM_FRACTION: float = 0.10

# Default numeric complexity encoding.
COMPLEXITY_NUMERIC: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def base_days(complexity: str) -> int:
    """Return the base person-days for a module complexity level."""
    key = COMPLEXITY_TO_BASE_KEY.get(complexity, "medium")
    return BASE_DAYS_PER_MODULE[key]


def complexity_multiplier(complexity: str) -> float:
    """Return the effort multiplier for a module complexity level."""
    return COMPLEXITY_MULTIPLIERS.get(complexity, 1.0)


def archetype_multiplier(archetype: str) -> float:
    """Return the effort multiplier for a module archetype."""
    return ARCHETYPE_MULTIPLIERS.get(archetype, 1.0)
