"""Shared helpers for HTTP middleware path matching."""

from __future__ import annotations

#: Liveness/readiness probe endpoints — called by orchestrators, not users.
#:
#: These must never require credentials and must never be rate limited. A
#: readiness probe that answers 401 keeps every pod out of service so the
#: Deployment never rolls out; one that answers 429 under load restarts pods
#: at exactly the wrong moment. The platform mounts ``/health``, ``/ready``
#: and ``/readiness``; the ``/healthz``, ``/livez`` and ``/readyz`` spellings
#: are included so an operator who aliases a conventional path is covered too.
PROBE_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/healthz",
        "/live",
        "/livez",
        "/ready",
        "/readiness",
        "/readyz",
    }
)

#: Everything infrastructure calls: the probes plus the Prometheus scrape.
#:
#: Middleware must not bill or throttle any of these. Behind a NAT, ingress,
#: or service mesh the kubelet and the scraper share one source IP with real
#: traffic, so counting them against a per-IP budget makes probes flap to 429
#: under exactly the load where the pod must stay up — and blanks the metrics
#: that would explain it.
OPERATIONAL_PATHS: frozenset[str] = PROBE_PATHS | {"/metrics"}


def path_is_skipped(path: str, skip_paths: set[str]) -> bool:
    """Return True when *path* matches an exact skip entry or a prefix entry.

    Prefix matching: a skip entry ``/studio`` matches ``/studio``,
    ``/studio/info``, ``/studio/ui/``, etc.
    """
    if path in skip_paths:
        return True
    for skip in skip_paths:
        if not skip or skip == "/":
            continue
        if path == skip or path.startswith(skip.rstrip("/") + "/"):
            return True
    return False
