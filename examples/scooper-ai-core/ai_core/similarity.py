"""Similarity search and feature-aware reranking for case retrieval.

Ported from the legacy Scooper ``embeddings.similarity`` module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


def brute_force_cosine_search(
    query: np.ndarray,
    corpus: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Return the top-*k* ``(index, score)`` matches by cosine similarity.

    Both *query* and *corpus* are assumed L2-normalised, so the dot product is
    the cosine similarity.
    """
    if corpus is None or len(corpus) == 0:
        return []
    scores = corpus @ query
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    ranked = sorted(top_idx, key=lambda i: float(scores[i]), reverse=True)
    return [(int(i), float(scores[i])) for i in ranked]


def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two lists treated as sets (lowercased)."""
    sa = {x.lower() for x in a}
    sb = {x.lower() for x in b}
    if not sa and not sb:
        return 0.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _recency_score(year: int, half_life_years: float = 2.0) -> float:
    """Exponential recency decay based on project year."""
    current_year = datetime.now().year
    age = max(0, current_year - int(year))
    return float(2.0 ** (-age / max(half_life_years, 0.1)))


def _fingerprint_bonus(query_fp: dict[str, int] | None, case_fp: dict[str, int] | None) -> float:
    """Similarity bonus from the 8-axis fingerprint (1 - normalised L2)."""
    if not query_fp or not case_fp:
        return 0.0
    axes = ["ux", "frontend", "backend", "data", "ai", "integrations", "security", "ambiguity"]
    qv = np.array([float(query_fp.get(a, 3)) for a in axes])
    cv = np.array([float(case_fp.get(a, 3)) for a in axes])
    dist = float(np.linalg.norm(qv - cv))
    max_dist = float(np.linalg.norm(np.full(len(axes), 4.0)))  # 1..5 range
    return max(0.0, 1.0 - dist / max_dist) if max_dist else 0.0


def rerank_by_features(
    matches: list[tuple[int, float]],
    metadatas: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    half_life_years: float = 2.0,
) -> list[dict[str, Any]]:
    """Blend cosine score with feature-overlap signals to rerank matches.

    Weights (sum to 1.0): cosine 0.35, tech overlap 0.15, recency 0.13,
    fingerprint 0.13, domain 0.10, complexity 0.08, archetype 0.06.
    """
    query_tech = list(query.get("tech_stack", []))
    query_archetypes = list(query.get("archetypes", []))
    query_complexity = str(query.get("overall_complexity", ""))
    query_fp = query.get("fingerprint")
    query_domain = str(query.get("domain", "")).lower()

    reranked: list[dict[str, Any]] = []
    for idx, cosine_score in matches:
        if idx < 0 or idx >= len(metadatas):
            continue
        meta = metadatas[idx]
        tech_overlap = _jaccard(query_tech, list(meta.get("tech_stack", [])))
        recency = _recency_score(int(meta.get("year", 2024)), half_life_years)
        fp_bonus = _fingerprint_bonus(query_fp, meta.get("fingerprint"))
        domain_bonus = (
            1.0 if query_domain and query_domain in str(meta.get("domain", "")).lower() else 0.0
        )
        complexity_bonus = (
            1.0 if query_complexity and query_complexity == _dominant_complexity(meta) else 0.0
        )
        archetype_bonus = _jaccard(query_archetypes, list(meta.get("archetypes", [])))

        combined = (
            0.35 * cosine_score
            + 0.15 * tech_overlap
            + 0.13 * recency
            + 0.13 * fp_bonus
            + 0.10 * domain_bonus
            + 0.08 * complexity_bonus
            + 0.06 * archetype_bonus
        )
        reranked.append(
            {
                "index": idx,
                "cosine_score": round(cosine_score, 4),
                "combined_score": round(float(combined), 4),
                "metadata": meta,
            }
        )
    reranked.sort(key=lambda r: r["combined_score"], reverse=True)
    return reranked


def _dominant_complexity(meta: dict[str, Any]) -> str:
    """Infer a case's dominant module complexity for the complexity bonus."""
    modules = meta.get("modules", [])
    counts: dict[str, int] = {}
    for module in modules:
        level = str(module.get("complexity", "medium"))
        counts[level] = counts.get(level, 0) + 1
    if not counts:
        return "medium"
    return max(counts, key=lambda k: counts[k])
