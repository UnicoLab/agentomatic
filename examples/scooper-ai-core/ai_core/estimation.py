"""Cascade ensemble combiner and plugin runner for effort estimation.

Combines the outputs of the estimation plugins with the exact cascade used by
the legacy Scooper effort-estimation agent:

    rule-based baseline -> similarity calibration -> Bayesian adjustment ->
    overheads -> Monte Carlo spread.
"""

from __future__ import annotations

from typing import Any

from ai_core.estimator_core import (
    archetype_multiplier,
    base_days,
    complexity_multiplier,
    compute_confidence,
    compute_phases,
    select_distribution,
)
from ai_core.schemas import EffortOutput, FeaturesInput
from ai_core.settings import Settings, get_settings


def module_estimates(inputs: FeaturesInput) -> list[dict[str, Any]]:
    """Compute per-module P50/P10/P90 person-day estimates (rule formula).

    Used to populate the ``modules`` breakdown on the estimate output. Values
    are illustrative per-module splits derived from the same deterministic
    scorecard the rule-based plugin uses.
    """
    rows: list[dict[str, Any]] = []
    for mod in inputs.modules:
        complexity = str(mod.get("complexity", "medium"))
        archetype = str(mod.get("archetype", "greenfield"))
        days = (
            float(base_days(complexity))
            * complexity_multiplier(complexity)
            * archetype_multiplier(archetype)
        )
        if mod.get("has_legacy_code"):
            days *= 1.3
        if mod.get("has_data_migration"):
            days *= 1.2
        if mod.get("external_dependency"):
            days *= 1.15
        if mod.get("regulatory_risk"):
            days *= 1.25
        p50 = max(1, round(days))
        rows.append(
            {
                "name": str(mod.get("name", "Module")),
                "complexity": complexity,
                "archetype": archetype,
                "estimated_days": p50,
                "p10": max(1, round(days * 0.7)),
                "p90": max(1, round(days * 1.4)),
            }
        )
    return rows


async def run_plugins(
    inputs: FeaturesInput,
    settings: Settings | None = None,
) -> dict[str, EffortOutput]:
    """Instantiate and run the enabled estimation plugins.

    Fresh plugin instances load the active artifact bundle each call, so a
    historical-update promotion is picked up without an in-process reload.

    Args:
        inputs: Normalised feature bundle.
        settings: Optional settings override (defaults to :func:`get_settings`).

    Returns:
        Mapping of plugin name -> :class:`EffortOutput` (failures are skipped).
    """
    from plugins.monte_carlo_effort import MonteCarloEffortPlugin
    from plugins.pymc_bayesian_effort import PymcBayesianEffortPlugin
    from plugins.rule_based_effort import RuleBasedEffortPlugin
    from plugins.similarity_effort import SimilarityEffortPlugin

    cfg = settings or get_settings()
    enabled: list[Any] = []
    if cfg.use_rule_based:
        enabled.append(RuleBasedEffortPlugin())
    if cfg.use_monte_carlo:
        enabled.append(MonteCarloEffortPlugin())
    if cfg.use_similarity:
        enabled.append(SimilarityEffortPlugin())
    if cfg.use_pymc:
        enabled.append(PymcBayesianEffortPlugin())

    results: dict[str, EffortOutput] = {}
    for plugin in enabled:
        try:
            await plugin.load_model()
            results[plugin.plugin_name] = await plugin.predict(inputs)
        except Exception:  # noqa: BLE001 - one plugin failure must not block others
            continue
    return results


def _plugin_usable(output: EffortOutput | None) -> bool:
    """Return whether a plugin output may influence the ensemble."""
    return output is not None and float(output.confidence) > 0.0 and float(output.total_p50) > 0.0


def combine_cascade(
    plugin_results: dict[str, EffortOutput],
    inputs: FeaturesInput,
    similar_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Combine plugin outputs into a single estimate via the cascade ensemble.

    Zero-confidence / disabled plugins (e.g. uncalibrated similarity or PyMC
    without a trained trace) are ignored so they cannot poison the ensemble.

    Args:
        plugin_results: Mapping of plugin name -> output.
        inputs: The feature bundle (for overhead/phase computation).
        similar_cases: Retrieved similar cases (for confidence + drivers).

    Returns:
        A dict with combined percentiles, confidence, categories, overhead
        breakdown and phase plan.
    """
    similar_cases = similar_cases or []
    baseline = plugin_results.get("rule_based_effort")
    mc = plugin_results.get("monte_carlo_effort")
    similarity = plugin_results.get("similarity_effort")
    bayesian = plugin_results.get("pymc_bayesian_effort")

    if baseline is None or not _plugin_usable(baseline):
        return _weighted_average(plugin_results, inputs, similar_cases)

    baseline_p50 = float(baseline.total_p50)
    p50 = baseline_p50

    # 2. Similarity calibration (only when confident and non-zero).
    if _plugin_usable(similarity) and float(similarity.confidence) > 0.6:
        sim_weight = float(similarity.confidence)
        sim_p50 = float(similarity.total_p50 or p50)
        p50 = p50 * (1.0 - sim_weight * 0.3) + sim_p50 * (sim_weight * 0.3)

    # 3. Bayesian adjustment (ratio clamped to [0.5, 1.5]).
    if _plugin_usable(bayesian):
        bayes_ratio = bayesian.total_p50 / max(baseline_p50, 1.0)
        bayes_ratio = min(max(bayes_ratio, 0.5), 1.5)
        p50 *= bayes_ratio

    # 4. Overheads.
    n_modules = len(inputs.modules)
    n_integrations = len(inputs.integrations)
    n_unknowns = len(inputs.unknowns)
    integration_overhead = max(0.0, (n_modules - 1) * 0.5)
    coordination_overhead = float(n_integrations) * 1.0
    uncertainty_pct = min(0.30, n_unknowns * 0.05)
    uncertainty_buffer = p50 * uncertainty_pct
    p50_with_overhead = p50 + integration_overhead + coordination_overhead + uncertainty_buffer

    # 5. Monte Carlo spread (rescaled), else fallback multipliers.
    if _plugin_usable(mc):
        mc_ratio = p50_with_overhead / max(mc.total_p50, 1.0)
        total_p10 = round(mc.total_p10 * mc_ratio, 2)
        total_p80 = round(mc.total_p80 * mc_ratio, 2)
        total_p90 = round(mc.total_p90 * mc_ratio, 2)
        total_p95 = round(mc.total_p95 * mc_ratio, 2)
    else:
        total_p10 = round(p50_with_overhead * 0.70, 2)
        total_p80 = round(p50_with_overhead * 1.20, 2)
        total_p90 = round(p50_with_overhead * 1.50, 2)
        total_p95 = round(p50_with_overhead * 1.75, 2)

    total_p50 = round(p50_with_overhead, 2)

    categories = [c.model_dump() for c in baseline.categories]
    categories.extend(
        [
            {"category": "integration_overhead", "p50": round(integration_overhead, 2)},
            {"category": "coordination_overhead", "p50": round(coordination_overhead, 2)},
            {"category": "uncertainty_buffer", "p50": round(uncertainty_buffer, 2)},
        ]
    )

    plugin_p50s = [o.total_p50 for o in plugin_results.values() if _plugin_usable(o)]
    confidence = compute_confidence(
        plugin_p50s,
        similar_cases,
        similarity_enabled=_plugin_usable(similarity),
    )

    has_ml = any(m.get("has_ml") for m in inputs.modules)
    distribution = select_distribution(has_ml, n_integrations)
    phase_plan = [
        {"phase": phase, "p50Days": days}
        for phase, days in compute_phases(total_p50, distribution).items()
    ]

    similar_projects = (
        similarity.metadata.get("similar_projects", []) if _plugin_usable(similarity) else []
    )
    skipped = [
        {
            "plugin": name,
            "status": (out.metadata or {}).get("status", "skipped"),
            "reason": (out.notes[0] if out.notes else "zero confidence"),
        }
        for name, out in plugin_results.items()
        if not _plugin_usable(out)
    ]

    return {
        "method": "cascaded",
        "total_p10": total_p10,
        "total_p50": total_p50,
        "total_p80": total_p80,
        "total_p90": total_p90,
        "total_p95": total_p95,
        "confidence": confidence,
        "categories": categories,
        "overhead_breakdown": {
            "integrationOverhead": round(integration_overhead, 2),
            "coordinationOverhead": round(coordination_overhead, 2),
            "uncertaintyBuffer": round(uncertainty_buffer, 2),
        },
        "phase_plan": phase_plan,
        "similar_projects": similar_projects,
        "plugin_methods": [n for n, o in plugin_results.items() if _plugin_usable(o)],
        "skipped_plugins": skipped,
    }


def _weighted_average(
    plugin_results: dict[str, EffortOutput],
    inputs: FeaturesInput,
    similar_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Confidence-weighted fallback when the rule-based baseline is missing."""
    usable = [o for o in plugin_results.values() if _plugin_usable(o)]
    skipped = [
        {
            "plugin": name,
            "status": (out.metadata or {}).get("status", "skipped"),
            "reason": (out.notes[0] if out.notes else "zero confidence"),
        }
        for name, out in plugin_results.items()
        if not _plugin_usable(out)
    ]
    if not usable:
        return {
            "method": "empty",
            "total_p10": 0.0,
            "total_p50": 0.0,
            "total_p80": 0.0,
            "total_p90": 0.0,
            "total_p95": 0.0,
            "confidence": 0.0,
            "categories": [],
            "overhead_breakdown": {
                "integrationOverhead": 0.0,
                "coordinationOverhead": 0.0,
                "uncertaintyBuffer": 0.0,
            },
            "phase_plan": [],
            "similar_projects": [],
            "plugin_methods": [],
            "skipped_plugins": skipped,
        }
    total_w = sum(o.confidence for o in usable)
    p50 = sum(o.total_p50 * o.confidence for o in usable) / total_w
    return {
        "method": "weighted_average",
        "total_p10": round(p50 * 0.7, 2),
        "total_p50": round(p50, 2),
        "total_p80": round(p50 * 1.2, 2),
        "total_p90": round(p50 * 1.5, 2),
        "total_p95": round(p50 * 1.75, 2),
        "confidence": compute_confidence([o.total_p50 for o in usable], similar_cases),
        "categories": [{"category": "development", "p50": round(p50, 2)}],
        "overhead_breakdown": {
            "integrationOverhead": 0.0,
            "coordinationOverhead": 0.0,
            "uncertaintyBuffer": 0.0,
        },
        "phase_plan": [],
        "similar_projects": [],
        "plugin_methods": [n for n, o in plugin_results.items() if _plugin_usable(o)],
        "skipped_plugins": skipped,
    }
