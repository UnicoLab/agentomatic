"""Probability distribution samplers for Monte Carlo effort estimation.

Ported from the legacy Scooper ``estimator_core.distributions`` module.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PERCENTILES: tuple[int, ...] = (10, 50, 80, 90, 95)


def triangular_sample(low: float, mode: float, high: float, n: int = 1000) -> np.ndarray:
    """Draw *n* samples from a triangular distribution.

    Args:
        low: Optimistic bound.
        mode: Most-likely value.
        high: Pessimistic bound.
        n: Number of samples.

    Returns:
        A 1-D array of samples. Bounds are re-ordered and nudged apart when
        degenerate so ``numpy`` does not raise.
    """
    lo, md, hi = sorted((float(low), float(mode), float(high)))
    if hi <= lo:
        hi = lo + 1e-6
    md = min(max(md, lo), hi)
    rng = np.random.default_rng()
    return rng.triangular(left=lo, mode=md, right=hi, size=n)


def pert_sample(
    low: float,
    mode: float,
    high: float,
    n: int = 1000,
    lambd: float = 4.0,
) -> np.ndarray:
    """Draw *n* samples from a PERT (Beta-reparameterised) distribution."""
    lo, md, hi = sorted((float(low), float(mode), float(high)))
    if hi <= lo:
        return np.full(n, lo, dtype=float)
    md = min(max(md, lo), hi)
    spread = hi - lo
    mean = (lo + lambd * md + hi) / (lambd + 2)
    denom = (spread**2) / (lambd + 2 + 1)
    if denom == 0 or (mean - lo) == 0:
        return triangular_sample(lo, md, hi, n)
    alpha = ((mean - lo) / spread) * ((mean - lo) * (hi - mean) / denom - 1)
    beta_param = alpha * (hi - mean) / (mean - lo) if (mean - lo) != 0 else 1.0
    alpha = max(alpha, 0.1)
    beta_param = max(beta_param, 0.1)
    rng = np.random.default_rng()
    raw = rng.beta(a=alpha, b=beta_param, size=n)
    return lo + raw * spread


def compute_percentiles(
    samples: np.ndarray,
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Compute rounded percentiles from a sample array.

    Returns:
        A mapping like ``{"p10": ..., "p50": ..., "p80": ..., "p90": ...,
        "p95": ...}`` rounded to two decimals.
    """
    if samples is None or len(samples) == 0:
        return {f"p{p}": 0.0 for p in percentiles}
    values = np.percentile(samples, percentiles)
    return {f"p{p}": round(float(v), 2) for p, v in zip(percentiles, values, strict=True)}
