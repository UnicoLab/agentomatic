"""Shared Pydantic schemas for scope, features, historical cases and estimates.

Internal data contracts exchanged between agents and plugins. External callers
(e.g. a frontend) own any UI-specific wrapping of these shapes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Complexity = Literal["low", "medium", "high", "critical"]
ConfidenceLabel = Literal["low", "medium", "high"]

# Module archetypes recognised by the estimator (drive effort multipliers).
Archetype = Literal[
    "greenfield",
    "brownfield",
    "bug_fix",
    "integration",
    "migration",
    "refactor",
    "data_pipeline",
    "ai_ml",
    "crud_admin",
    "reporting_dashboard",
    "security_compliance",
    "infrastructure",
]

_ARCHETYPE_ALIASES: dict[str, Archetype] = {
    "reporting": "reporting_dashboard",
    "dashboard": "reporting_dashboard",
    "report": "reporting_dashboard",
    "reporting_dashboard": "reporting_dashboard",
    "ai": "ai_ml",
    "ai_integration": "ai_ml",
    "ai-ml": "ai_ml",
    "ml": "ai_ml",
    "llm": "ai_ml",
    "chatbot": "ai_ml",
    "data": "data_pipeline",
    "pipeline": "data_pipeline",
    "etl": "data_pipeline",
    "crud": "crud_admin",
    "admin": "crud_admin",
    "security": "security_compliance",
    "compliance": "security_compliance",
    "auth": "security_compliance",
    "infra": "infrastructure",
    "devops": "infrastructure",
    "bug": "bug_fix",
    "fix": "bug_fix",
    "legacy": "brownfield",
    "new": "greenfield",
}


def normalize_archetype(value: Any, *, default: Archetype = "greenfield") -> Archetype:
    """Coerce free-text / LLM archetypes onto the estimator Literal set."""
    if not isinstance(value, str) or not value.strip():
        return default
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    if key in _ARCHETYPE_ALIASES:
        return _ARCHETYPE_ALIASES[key]
    allowed = set(_ARCHETYPE_ALIASES.values()) | {
        "greenfield",
        "brownfield",
        "bug_fix",
        "integration",
        "migration",
        "refactor",
        "data_pipeline",
        "ai_ml",
        "crud_admin",
        "reporting_dashboard",
        "security_compliance",
        "infrastructure",
    }
    if key in allowed:
        return key  # type: ignore[return-value]
    return default


def normalize_complexity(value: Any, *, default: Complexity = "medium") -> Complexity:
    """Coerce free-text complexity onto the estimator Literal set."""
    if not isinstance(value, str) or not value.strip():
        return default
    key = value.strip().lower()
    if key in {"low", "medium", "high", "critical"}:
        return key  # type: ignore[return-value]
    if key in {"l", "simple", "facile"}:
        return "low"
    if key in {"m", "moyen", "moderee", "modérée"}:
        return "medium"
    if key in {"h", "complexe", "difficile"}:
        return "high"
    if key in {"c", "critique", "blocking"}:
        return "critical"
    return default


class ComplexityFingerprint(BaseModel):
    """8-axis complexity fingerprint (each axis an int in 1..5)."""

    ux: int = 3
    frontend: int = 3
    backend: int = 3
    data: int = 3
    ai: int = 1
    integrations: int = 1
    security: int = 2
    ambiguity: int = 3

    @property
    def total_score(self) -> int:
        """Sum of all eight axes (range 8..40)."""
        return (
            self.ux
            + self.frontend
            + self.backend
            + self.data
            + self.ai
            + self.integrations
            + self.security
            + self.ambiguity
        )

    def as_vector(self) -> list[float]:
        """Return the fingerprint as an ordered float vector for distance calc."""
        return [
            float(self.ux),
            float(self.frontend),
            float(self.backend),
            float(self.data),
            float(self.ai),
            float(self.integrations),
            float(self.security),
            float(self.ambiguity),
        ]


class ModuleFeature(BaseModel):
    """A single functional module with estimation-relevant attributes."""

    name: str
    complexity: Complexity = "medium"
    archetype: Archetype = "greenfield"
    confidence: ConfidenceLabel = "medium"
    n_screens: int = 0
    has_api: bool = False
    has_db: bool = False
    has_auth: bool = False
    has_ml: bool = False
    has_legacy_code: bool = False
    has_data_migration: bool = False
    external_dependency: bool = False
    regulatory_risk: bool = False
    design_ready: bool | None = None
    api_ready: bool | None = None
    missing_info: list[str] = Field(default_factory=list)


class ProjectFeatures(BaseModel):
    """Structured scope produced by the scope-analysis agent."""

    project_name: str = "Projet sans nom"
    description: str = ""
    modules: list[ModuleFeature] = Field(default_factory=list)
    overall_complexity: Complexity = "medium"
    tech_stack: list[str] = Field(default_factory=list)
    integrations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    fingerprint: ComplexityFingerprint = Field(default_factory=ComplexityFingerprint)


class CaseModule(BaseModel):
    """A module within a validated historical project case."""

    name: str
    actual_days: float = 0.0
    complexity: Complexity = "medium"
    archetype: Archetype = "greenfield"


class ProjectCase(BaseModel):
    """A validated historical project (source of truth for similarity/RAG)."""

    case_id: str
    project_name: str
    description: str = ""
    modules: list[CaseModule] = Field(default_factory=list)
    total_actual_days: float = 0.0
    team_size: int = 1
    tech_stack: list[str] = Field(default_factory=list)
    domain: str = "general"
    year: int = 2024
    archetypes: list[str] = Field(default_factory=list)
    fingerprint: dict[str, int] | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class EffortCategory(BaseModel):
    """Per-discipline effort breakdown row (person-days)."""

    category: str
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0


class EffortOutput(BaseModel):
    """Common output contract for all estimation plugins."""

    method: str = "unknown"
    total_p10: float = 0.0
    total_p50: float = 0.0
    total_p80: float = 0.0
    total_p90: float = 0.0
    total_p95: float = 0.0
    confidence: float = 0.0
    categories: list[EffortCategory] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeaturesInput(BaseModel):
    """Plugin input: normalised feature bundle for one project."""

    project_name: str = "Projet sans nom"
    modules: list[dict[str, Any]] = Field(default_factory=list)
    overall_complexity: Complexity = "medium"
    tech_stack: list[str] = Field(default_factory=list)
    integrations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)

    @classmethod
    def from_features(cls, features: ProjectFeatures) -> FeaturesInput:
        """Build a plugin input from a :class:`ProjectFeatures` object."""
        return cls(
            project_name=features.project_name,
            modules=[m.model_dump() for m in features.modules],
            overall_complexity=features.overall_complexity,
            tech_stack=features.tech_stack,
            integrations=features.integrations,
            unknowns=features.unknowns,
        )
