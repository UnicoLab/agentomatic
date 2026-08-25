"""Filesystem confinement for ingestion jobs.

Ingestion requests carry caller-supplied paths (what to read, where to write).
Those arrive over HTTP, so treating them as trusted turns an ingestor into an
arbitrary file read/write primitive: ``source=/etc/passwd`` exfiltrates any
file the process can read, and ``output_dir``/``output_filename`` can escape
to any writable location (including overwriting code on an import path).

Every ingestor that touches caller-supplied paths should resolve them through
:func:`resolve_within_root`, which confines them to an ingestion root.

The root defaults to the current working directory (the project root for a
normal ``agentomatic run``), so ordinary relative paths keep working. Operators
who genuinely ingest from elsewhere set ``AGENTOMATIC_INGESTION_ROOT``. Setting
it to ``/`` restores the old unconfined behaviour and is deliberately explicit.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable naming the directory ingestion paths are confined to.
INGESTION_ROOT_ENV = "AGENTOMATIC_INGESTION_ROOT"


class IngestionPathError(ValueError):
    """Raised when a caller-supplied path escapes the ingestion root."""


def ingestion_root() -> Path:
    """Return the directory ingestion paths are confined to.

    Falls back to the current working directory when
    ``AGENTOMATIC_INGESTION_ROOT`` is unset.
    """
    configured = os.getenv(INGESTION_ROOT_ENV, "").strip()
    base = Path(configured).expanduser() if configured else Path.cwd()
    return base.resolve()


def resolve_within_root(
    candidate: str | Path,
    *,
    root: Path | None = None,
    description: str = "path",
) -> Path:
    """Resolve *candidate* and require the result to sit inside *root*.

    Args:
        candidate: Caller-supplied path (absolute or relative to the root).
        root: Confinement root; defaults to :func:`ingestion_root`.
        description: Field name used in the error message.

    Returns:
        The resolved, confined :class:`~pathlib.Path`.

    Raises:
        IngestionPathError: If the path escapes *root*. Symlinks are resolved
            before the check, so a symlink pointing outside is rejected too.
    """
    base = (root or ingestion_root()).resolve()
    raw = Path(candidate).expanduser()
    resolved = (raw if raw.is_absolute() else base / raw).resolve()

    if base == Path(resolved.anchor):
        # Root is the filesystem root — explicitly unconfined.
        return resolved

    if resolved != base and base not in resolved.parents:
        raise IngestionPathError(
            f"{description} {str(candidate)!r} resolves outside the ingestion root "
            f"({base}). Set {INGESTION_ROOT_ENV} to widen it."
        )
    return resolved


def safe_output_filename(name: str | None, *, default: str) -> str:
    """Return a bare filename, rejecting any path separators or traversal.

    ``output_filename`` is a *name*, not a path — allowing ``../../evil`` in it
    lets a caller escape an otherwise-confined output directory.

    Raises:
        IngestionPathError: If *name* contains a separator or is a traversal.
    """
    if not name:
        return default
    candidate = name.strip()
    if not candidate:
        return default
    if candidate in {".", ".."} or os.sep in candidate or "/" in candidate:
        raise IngestionPathError(f"output_filename {name!r} must be a bare filename, not a path")
    if os.altsep and os.altsep in candidate:
        raise IngestionPathError(f"output_filename {name!r} must be a bare filename, not a path")
    return candidate
