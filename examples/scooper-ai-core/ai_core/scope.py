"""Scope-analysis helpers: fingerprint, normalisation, heuristic fallback.

Ported from the legacy Scooper scope-analysis agent enrichment logic.
"""

from __future__ import annotations

import re
from typing import Any

from ai_core.schemas import (
    ComplexityFingerprint,
    ModuleFeature,
    ProjectFeatures,
)

_COMPLEXITY_MAP = {"low": 1, "medium": 2, "high": 3, "critical": 4}

_PLACEHOLDER_NAMES = frozenset(
    {
        "",
        "unnamed",
        "unnamed project",
        "projet sans nom",
        "n/a",
        "na",
        "unknown",
        "inconnu",
        "project",
        "projet",
        "titre",
        "title",
        "none",
        "null",
    }
)

_TITLE_LABEL_RE = re.compile(
    r"(?i)^(?:projet|project|titre|title|nom(?:\s+du\s+projet)?)\s*[:\-–]\s*(.+)$"
)
_H1_RE = re.compile(r"^#\s+(.+)$")


def default_project_name(*, language: str = "fr") -> str:
    """Return the language-aware placeholder when no title can be recovered."""
    code = (language or "fr").lower()[:2]
    return "Projet sans nom" if code == "fr" else "Unnamed Project"


def is_placeholder_name(name: str | None) -> bool:
    """Return whether *name* is empty or a known placeholder."""
    if name is None:
        return True
    cleaned = re.sub(r"\s+", " ", str(name)).strip().lower()
    return cleaned in _PLACEHOLDER_NAMES


def extract_project_title(text: str) -> str | None:
    """Extract a meaningful project title from markdown / labelled lines.

    Prefers the first Markdown H1, then labelled lines such as ``Projet: …``
    or ``Title: …`` near the top of the document.
    """
    if not text or not text.strip():
        return None

    def _clean(raw: str) -> str | None:
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
        title = re.sub(r"[*_`]+", "", title).strip(" \t.-–—:")
        title = re.sub(r"\s+", " ", title).strip()
        if 2 <= len(title) <= 120 and not is_placeholder_name(title):
            return title[:120]
        return None

    for line in text.splitlines():
        match = _H1_RE.match(line.strip())
        if match:
            cleaned = _clean(match.group(1))
            if cleaned:
                return cleaned

    for line in text.splitlines()[:40]:
        match = _TITLE_LABEL_RE.match(line.strip())
        if match:
            cleaned = _clean(match.group(1))
            if cleaned:
                return cleaned
    return None


def resolve_project_name(
    raw_name: Any,
    *,
    document_text: str = "",
    language: str = "fr",
) -> str:
    """Pick the best project name from LLM output, document title, or fallback."""
    candidates = [raw_name]
    if isinstance(raw_name, dict):
        candidates = [raw_name.get("project_name"), raw_name.get("name"), raw_name.get("title")]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not is_placeholder_name(text):
            return text[:120]
    titled = extract_project_title(document_text)
    if titled:
        return titled
    return default_project_name(language=language)


def _avg_complexity(modules: list[dict[str, Any]]) -> int:
    """Average module complexity mapped to 1..5 (legacy +1 offset, clamped)."""
    if not modules:
        return 3
    total = sum(_COMPLEXITY_MAP.get(str(m.get("complexity", "medium")), 2) for m in modules)
    avg = total / len(modules)
    return min(5, max(1, round(avg) + 1))


def _bool_score(trait: str, modules: list[dict[str, Any]]) -> int:
    """Score presence of a boolean trait across modules on a 1..5 scale."""
    count = sum(1 for m in modules if m.get(trait))
    if count == 0:
        return 1
    if count == 1:
        return 2
    if count == 2:
        return 3
    return min(5, count + 1)


def compute_fingerprint(result: dict[str, Any]) -> ComplexityFingerprint:
    """Compute the 8-axis complexity fingerprint from a scope result dict."""
    modules = result.get("modules", [])
    total_screens = sum(int(m.get("n_screens", 0) or 0) for m in modules) or len(modules)
    integrations = result.get("integrations", [])
    unknowns = result.get("unknowns", [])
    return ComplexityFingerprint(
        ux=min(5, max(1, 1 + total_screens // 3)),
        frontend=min(5, max(1, 1 + total_screens // 2)),
        backend=_avg_complexity(modules),
        data=_bool_score("has_db", modules),
        ai=_bool_score("has_ml", modules),
        integrations=min(5, max(1, len(integrations) + 1)),
        security=_bool_score("has_auth", modules),
        ambiguity=min(5, max(1, len(unknowns) + 1)),
    )


def module_questions(
    module: dict[str, Any],
    *,
    language: str = "fr",
    cap: int = 4,
) -> list[str]:
    """Generate clarification questions for an under-specified module."""
    questions: list[str] = []
    name = module.get("name", "ce module" if (language or "fr")[:2] == "fr" else "this module")
    fr = (language or "fr").lower()[:2] == "fr"
    if module.get("design_ready") is None:
        questions.append(
            f"Les maquettes UI/UX de « {name} » sont-elles finalisées ?"
            if fr
            else f"Are the UI/UX designs for '{name}' finalised?"
        )
    if module.get("api_ready") is None and module.get("has_api"):
        questions.append(
            f"Les contrats d'API de « {name} » sont-ils définis ?"
            if fr
            else f"Are the API contracts for '{name}' defined?"
        )
    if module.get("has_db") and not module.get("data_volume"):
        questions.append(
            f"Quel volume de données est attendu pour « {name} » ?"
            if fr
            else f"What data volume is expected for '{name}'?"
        )
    if module.get("external_dependency"):
        questions.append(
            f"Quels systèmes externes « {name} » utilise-t-il ?"
            if fr
            else f"Which external systems does '{name}' depend on?"
        )
    if str(module.get("complexity", "")) in {"high", "critical"}:
        questions.append(
            f"Quels sont les principaux risques techniques pour « {name} » ?"
            if fr
            else f"What are the main technical risks for '{name}'?"
        )
    return questions[:cap]


def normalize_scope(
    raw: dict[str, Any],
    *,
    document_text: str = "",
    language: str = "fr",
) -> ProjectFeatures:
    """Coerce a loose LLM scope dict into a validated :class:`ProjectFeatures`."""
    modules_raw = raw.get("modules", []) or []
    modules: list[ModuleFeature] = []
    for module in modules_raw:
        if not isinstance(module, dict):
            continue
        fields = {k: v for k, v in module.items() if k in ModuleFeature.model_fields}
        fields.setdefault("name", str(module.get("name", "Module")))
        try:
            modules.append(ModuleFeature(**fields))
        except Exception:  # noqa: BLE001 - tolerate bad enum values
            modules.append(ModuleFeature(name=str(fields.get("name", "Module"))))
    if not modules:
        core = "Fonctionnalité cœur" if (language or "fr")[:2] == "fr" else "Core functionality"
        modules = [ModuleFeature(name=core, confidence="low")]
    features = ProjectFeatures(
        project_name=resolve_project_name(
            raw.get("project_name") or raw.get("name") or raw.get("title"),
            document_text=document_text,
            language=language,
        ),
        description=str(raw.get("description", "")),
        modules=modules,
        overall_complexity=raw.get("overall_complexity", "medium")
        if raw.get("overall_complexity") in {"low", "medium", "high", "critical"}
        else "medium",
        tech_stack=[str(t) for t in raw.get("tech_stack", []) if t],
        integrations=[str(i) for i in raw.get("integrations", []) if i],
        unknowns=[str(u) for u in raw.get("unknowns", []) if u],
    )
    features.fingerprint = compute_fingerprint(
        {
            "modules": [m.model_dump() for m in features.modules],
            "integrations": features.integrations,
            "unknowns": features.unknowns,
        }
    )
    return features


_MODULE_KEYWORDS = {
    "auth": ("security_compliance", True, False, True, False),
    "login": ("security_compliance", False, False, True, False),
    "dashboard": ("reporting_dashboard", True, True, False, False),
    "report": ("reporting_dashboard", True, True, False, False),
    "api": ("integration", True, False, False, False),
    "integration": ("integration", True, False, False, False),
    "database": ("crud_admin", False, True, False, False),
    "data": ("data_pipeline", False, True, False, False),
    "ml": ("ai_ml", False, False, False, True),
    "model": ("ai_ml", False, False, False, True),
    "notification": ("greenfield", False, False, False, False),
    "admin": ("crud_admin", False, True, False, False),
}


def heuristic_scope(
    text: str,
    *,
    project_name: str | None = None,
    language: str = "fr",
) -> ProjectFeatures:
    """Derive a coarse scope from raw text when the LLM is unavailable.

    Splits on markdown headings/bullets and keyword-matches module archetypes so
    the estimation pipeline always has something to work with. The project name
    is taken from *project_name*, else the document H1 / labelled title.
    """
    resolved_name = resolve_project_name(project_name, document_text=text, language=language)
    title_lower = resolved_name.lower()
    candidates: list[str] = []
    for line in text.splitlines():
        stripped_line = line.strip()
        # Skip the H1 used as the project title so it is not also a module.
        if _H1_RE.match(stripped_line):
            continue
        stripped = stripped_line.lstrip("#").lstrip("-*").strip()
        if 3 <= len(stripped) <= 80 and stripped_line.startswith(("#", "-", "*")):
            if stripped.lower() == title_lower:
                continue
            candidates.append(stripped)
    seen: set[str] = set()
    modules: list[ModuleFeature] = []
    for cand in candidates:
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        archetype = "greenfield"
        has_api = has_db = has_auth = has_ml = False
        for kw, (arch, api, db, auth, ml) in _MODULE_KEYWORDS.items():
            if kw in key:
                archetype, has_api, has_db, has_auth, has_ml = arch, api, db, auth, ml
                break
        modules.append(
            ModuleFeature(
                name=cand[:80],
                archetype=archetype,  # type: ignore[arg-type]
                complexity="medium",
                confidence="low",
                has_api=has_api,
                has_db=has_db,
                has_auth=has_auth,
                has_ml=has_ml,
            )
        )
        if len(modules) >= 12:
            break
    if not modules:
        core = "Fonctionnalité cœur" if (language or "fr")[:2] == "fr" else "Core functionality"
        modules = [ModuleFeature(name=core, confidence="low")]
    integrations = sorted(
        {
            m.group(0)
            for m in re.finditer(
                r"\b(Stripe|Salesforce|SAP|Azure|AWS|GCP|Kafka|Twilio|SendGrid)\b",
                text,
                re.IGNORECASE,
            )
        }
    )
    features = ProjectFeatures(
        project_name=resolved_name,
        description=text[:400],
        modules=modules,
        integrations=integrations,
    )
    features.fingerprint = compute_fingerprint(
        {
            "modules": [m.model_dump() for m in features.modules],
            "integrations": features.integrations,
            "unknowns": features.unknowns,
        }
    )
    return features
