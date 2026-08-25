"""Shared helpers for HTTP middleware path matching."""

from __future__ import annotations

#: Operational endpoints that infrastructure — not users — calls: liveness and
#: readiness probes and the Prometheus scrape.
#:
#: Middleware must never rate-limit or bill these. A kubelet probing every few
#: seconds and a scraper polling ``/metrics`` share one source IP with real
#: traffic behind a NAT, ingress, or service mesh, so counting them against a
#: per-IP budget makes probes flap to 429 under exactly the load where the
#: pod must stay up — and blanks the metrics that would explain it.
#:
#: The set is deliberately wider than the routes this platform mounts today
#: (it also covers the ``/healthz`` and ``/livez`` spellings) so operators who
#: alias a conventional probe path are covered too.
OPERATIONAL_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/healthz",
        "/live",
        "/livez",
        "/ready",
        "/readiness",
        "/readyz",
        "/metrics",
    }
)


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
