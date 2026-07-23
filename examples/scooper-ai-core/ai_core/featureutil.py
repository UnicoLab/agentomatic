"""Helpers to derive plugin inputs and feature vectors from scope features.

Used by the similarity / Bayesian plugins and the historical-update pipeline
to turn :class:`~ai_core.schemas.FeaturesInput` (and case dicts) into stable
numeric rows and :class:`~ai_core.schemas.ProjectFeatures` objects.
"""

from __future__ import annotations

from typing import Any

from ai_core.estimator_core import COMPLEXITY_NUMERIC
from ai_core.schemas import (
    ComplexityFingerprint,
    FeaturesInput,
    ModuleFeature,
    ProjectFeatures,
)


def features_from_input(inputs: FeaturesInput) -> ProjectFeatures:
    """Rebuild a :class:`ProjectFeatures` from a flat plugin input bundle."""
    modules = []
    for module in inputs.modules:
        try:
            modules.append(
                ModuleFeature(
                    **{k: v for k, v in module.items() if k in ModuleFeature.model_fields}
                )
            )
        except Exception:  # noqa: BLE001 - tolerate loose module dicts
            modules.append(ModuleFeature(name=str(module.get("name", "Module"))))
    return ProjectFeatures(
        project_name=inputs.project_name,
        modules=modules,
        overall_complexity=inputs.overall_complexity,
        tech_stack=inputs.tech_stack,
        integrations=inputs.integrations,
        unknowns=inputs.unknowns,
        fingerprint=ComplexityFingerprint(),
    )


def bayesian_feature_vector(inputs: FeaturesInput) -> list[float]:
    """Compute the numeric feature vector used by the Bayesian model.

    Vector: ``[n_modules, mean_complexity, n_integrations, has_auth, has_ml]``.
    """
    modules = inputs.modules
    n_modules = float(len(modules))
    if modules:
        mean_complexity = sum(
            COMPLEXITY_NUMERIC.get(str(m.get("complexity", "medium")), 1) for m in modules
        ) / len(modules)
    else:
        mean_complexity = 1.0
    # Prefer explicit integrations; fall back to tech_stack length (seed cases).
    n_integrations = float(len(inputs.integrations) or len(inputs.tech_stack))
    has_auth = (
        1.0
        if any(m.get("has_auth") or "auth" in str(m.get("name", "")).lower() for m in modules)
        else 0.0
    )
    has_ml = (
        1.0
        if any(
            m.get("has_ml")
            or any(kw in str(m.get("archetype", "")).lower() for kw in ("ai_ml", "ml"))
            for m in modules
        )
        else 0.0
    )
    return [n_modules, mean_complexity, n_integrations, has_auth, has_ml]


BAYESIAN_FEATURE_NAMES: tuple[str, ...] = (
    "n_modules",
    "mean_complexity",
    "n_integrations",
    "has_auth",
    "has_ml",
)


def case_feature_row(case: dict[str, Any]) -> tuple[list[float], float]:
    """Build a ``(feature_vector, target_days)`` row from a case dict."""
    modules = case.get("modules", [])
    n_modules = float(len(modules))
    if modules:
        mean_complexity = sum(
            COMPLEXITY_NUMERIC.get(str(m.get("complexity", "medium")), 1) for m in modules
        ) / len(modules)
    else:
        mean_complexity = 1.0
    n_integrations = float(len(case.get("tech_stack", [])))
    has_auth = 1.0 if any("auth" in str(m.get("name", "")).lower() for m in modules) else 0.0
    has_ml = (
        1.0
        if any(
            kw in str(m.get("archetype", "")).lower() for m in modules for kw in ("ai_ml", "ml")
        )
        else 0.0
    )
    target = float(case.get("total_actual_days", 0.0))
    return [n_modules, mean_complexity, n_integrations, has_auth, has_ml], target
