"""Shared helpers for HTTP middleware path matching."""

from __future__ import annotations


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
