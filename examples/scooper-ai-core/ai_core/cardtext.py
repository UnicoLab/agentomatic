"""Parallel card-text builders for cases and query features.

Retrieval text is kept short and natural-language oriented so embedding models
(especially EmbeddingGemma) produce meaningful cosine scores. Structured
multi-line cards historically collapsed similarity to near-zero.
"""

from __future__ import annotations

from typing import Any

from ai_core.schemas import ProjectCase, ProjectFeatures


def _fingerprint_line(fingerprint: dict[str, int] | None) -> str:
    """Render a fingerprint dict as a stable one-line summary."""
    if not fingerprint:
        return "Fingerprint: n/a"
    axes = ["ux", "frontend", "backend", "data", "ai", "integrations", "security", "ambiguity"]
    parts = [f"{a}={fingerprint.get(a, 3)}" for a in axes]
    return "Fingerprint: " + ", ".join(parts)


def _join(parts: list[str], *, sep: str = ", ") -> str:
    """Join non-empty trimmed parts."""
    return sep.join(p.strip() for p in parts if p and str(p).strip())


def case_to_card_text(case: ProjectCase) -> str:
    """Render a historical case as short retrieval text for embeddings."""
    modules = _join([m.name for m in case.modules])
    tech = _join(list(case.tech_stack))
    archetypes = _join(list(case.archetypes) or [m.archetype for m in case.modules])
    description = (case.description or "").strip()
    bits = [
        case.project_name,
        description[:280] if description else "",
        f"Modules: {modules}" if modules else "",
        f"Tech: {tech}" if tech else "",
        f"Archetypes: {archetypes}" if archetypes else "",
        f"Domain: {case.domain}" if case.domain else "",
    ]
    return ". ".join(b for b in bits if b)


def features_to_query_text(features: ProjectFeatures) -> str:
    """Render query features as short retrieval text (aligned with cases)."""
    modules = _join([m.name for m in features.modules])
    tech = _join(list(features.tech_stack))
    archetypes = _join(sorted({m.archetype for m in features.modules}))
    description = (features.description or "").strip()
    bits = [
        features.project_name,
        description[:280] if description else "",
        f"Modules: {modules}" if modules else "",
        f"Tech: {tech}" if tech else "",
        f"Archetypes: {archetypes}" if archetypes else "",
    ]
    return ". ".join(b for b in bits if b)


def loose_case_dict_to_card_text(case: dict[str, Any]) -> str:
    """Build retrieval text from a raw case dict (historical_update path)."""
    try:
        return case_to_card_text(ProjectCase.model_validate(case))
    except Exception:  # noqa: BLE001 - tolerate odd shapes during seeding
        modules_raw = case.get("modules") or []
        module_names = [
            str(m.get("name", "")) for m in modules_raw if isinstance(m, dict) and m.get("name")
        ]
        tech = [str(t) for t in (case.get("tech_stack") or [])]
        archetypes = [str(a) for a in (case.get("archetypes") or [])]
        description = str(case.get("description") or "").strip()
        bits = [
            str(case.get("project_name") or "Unnamed"),
            description[:280] if description else "",
            f"Modules: {_join(module_names)}" if module_names else "",
            f"Tech: {_join(tech)}" if tech else "",
            f"Archetypes: {_join(archetypes)}" if archetypes else "",
            f"Domain: {case.get('domain')}" if case.get("domain") else "",
        ]
        return ". ".join(b for b in bits if b)
