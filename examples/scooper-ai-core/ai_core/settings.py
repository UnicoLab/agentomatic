"""Runtime settings for domain logic (estimation, artifacts, embeddings).

LLM provider/model/base_url come from the active Agentomatic stack
(``stacks/*.yaml``) and are injected into agents — they are not duplicated
here. This module holds knobs that Agentomatic does not own: artifact paths,
plugin toggles, embedding endpoint overrides, and language defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean flag from the environment."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse an int flag from the environment, tolerating bad values."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float flag from the environment, tolerating bad values."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of domain settings (non-LLM)."""

    # Embeddings (OpenAI-compatible; stack ``embedding:`` selects model/dim).
    embed_enabled: bool = True
    embed_base_url: str = "http://127.0.0.1:8000/v1"
    embed_model: str = "embeddinggemma-300m-4bit"
    embed_api_key: str = "sk-local-mlx"
    embed_dimension: int = 768

    # Vector store selection + artifact locations.
    vector_provider: str = "local_npz"
    artifact_root: Path = field(default_factory=lambda: Path(".local/artifacts"))
    ingestion_root: Path = field(default_factory=lambda: Path(".local/ingestion"))
    runs_root: Path = field(default_factory=lambda: Path(".local/runs"))

    # Reliability knobs for chat helpers (0 = fail fast; no silent retries).
    llm_max_retries: int = 0
    # Soft ceiling for assistant LLM only. 0 = wait for the primary model
    # (preferred — never abort a working SLM into Secours). Positive values
    # yield source=error on expiry, not a labelled fallback blurb.
    op_timeout_seconds: float = 0.0

    # Estimation plugin toggles.
    use_rule_based: bool = True
    use_monte_carlo: bool = True
    use_similarity: bool = True
    use_pymc: bool = False

    # Retrieval knobs.
    top_k_semantic: int = 10
    top_k_final: int = 5
    monte_carlo_samples: int = 5000
    recency_half_life_years: float = 2.0

    # Ingestion knobs.
    chunk_size_tokens: int = 1200
    chunk_overlap_tokens: int = 150
    min_quality_score: float = 0.70
    context_max_words: int = 2500
    # Max chars of existing / new content fed into the consolidate LLM call.
    # 4k was truncating multi-doc merges and dropping material facts.
    context_input_chars: int = 24000
    # Max decoded base64 payload accepted by MarkItDown ingestion (bytes).
    max_ingest_bytes: int = 20 * 1024 * 1024

    # Language control.
    default_language: str = "fr"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache the :class:`Settings` snapshot from the environment."""
    return Settings(
        embed_enabled=_env_bool("AI_EMBED_ENABLED", True),
        embed_base_url=os.getenv("AI_EMBED_BASE_URL", "http://127.0.0.1:8000/v1"),
        embed_model=os.getenv("AI_EMBED_MODEL", "embeddinggemma-300m-4bit"),
        embed_api_key=os.getenv("AI_EMBED_API_KEY", "sk-local-mlx"),
        embed_dimension=_env_int("AI_EMBED_DIMENSION", 768),
        vector_provider=os.getenv("AI_VECTOR_PROVIDER", "local_npz"),
        artifact_root=Path(os.getenv("AI_ARTIFACT_ROOT", ".local/artifacts")),
        ingestion_root=Path(os.getenv("AI_INGESTION_ROOT", ".local/ingestion")),
        runs_root=Path(os.getenv("AI_RUNS_ROOT", ".local/runs")),
        llm_max_retries=_env_int("AI_LLM_MAX_RETRIES", 0),
        op_timeout_seconds=_env_float("AI_OP_TIMEOUT_SECONDS", 0.0),
        use_rule_based=_env_bool("ESTIMATION_USE_RULE_BASED", True),
        use_monte_carlo=_env_bool("ESTIMATION_USE_MONTE_CARLO", True),
        use_similarity=_env_bool("ESTIMATION_USE_SIMILARITY", True),
        use_pymc=_env_bool("ESTIMATION_USE_PYMC", False),
        top_k_semantic=_env_int("AI_TOP_K_SEMANTIC", 10),
        top_k_final=_env_int("AI_TOP_K_FINAL", 5),
        monte_carlo_samples=_env_int("AI_MONTE_CARLO_SAMPLES", 5000),
        recency_half_life_years=_env_float("AI_RECENCY_HALF_LIFE_YEARS", 2.0),
        chunk_size_tokens=_env_int("AI_CHUNK_SIZE_TOKENS", 1200),
        chunk_overlap_tokens=_env_int("AI_CHUNK_OVERLAP_TOKENS", 150),
        min_quality_score=_env_float("AI_MIN_QUALITY_SCORE", 0.70),
        context_max_words=_env_int("AI_CONTEXT_MAX_WORDS", 2500),
        context_input_chars=_env_int("AI_CONTEXT_INPUT_CHARS", 24000),
        max_ingest_bytes=_env_int("AI_MAX_INGEST_BYTES", 20 * 1024 * 1024),
        default_language=os.getenv("AI_DEFAULT_LANGUAGE", "fr"),
    )
