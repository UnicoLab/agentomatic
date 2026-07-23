"""Shared SQL database connection config (same Postgres as the backend).

Agents that need relational storage declare
:func:`shared_backend_database_connection` in their ``connections.py``.
The URL comes from ``DATABASE_URL`` (preferred) or ``SCOOPER_DATABASE_URL``
so Docker and native modes share one source of truth with the backend.
"""

from __future__ import annotations

import os

from agentomatic.connections import ConnectionPurpose, DatabaseConnectionConfig


def resolve_shared_database_url() -> str:
    """Return the SQLAlchemy async URL shared with the Scooper backend.

    Resolution order:
      1. ``DATABASE_URL`` (Agentomatic stack / ai_platform convention)
      2. ``SCOOPER_DATABASE_URL`` (backend convention)
      3. Local Postgres default (native ``make deps-infra``)
    """
    for key in ("DATABASE_URL", "SCOOPER_DATABASE_URL"):
        raw = os.getenv(key)
        if raw and raw.strip():
            return raw.strip()
    return "postgresql+asyncpg://scooper:scooper@localhost:5432/scooper"


def shared_backend_database_connection(
    *,
    name: str = "scooper_db",
    purpose: ConnectionPurpose = ConnectionPurpose.GENERAL,
) -> DatabaseConnectionConfig:
    """Build a :class:`DatabaseConnectionConfig` pointing at the backend DB.

    Args:
        name: Logical connection name for ``get_connections(...).database(name)``.
        purpose: Connection purpose tag (memory / analytics / general…).

    Returns:
        Config using ``${DATABASE_URL}`` interpolation when set, else the
        resolved shared URL baked in (still overridable at runtime via env
        before process start).
    """
    # Prefer ${ENV} interpolation so secrets stay out of code; fall back to
    # the resolved URL when neither var is present at config-build time.
    url = (
        "${DATABASE_URL}"
        if os.getenv("DATABASE_URL")
        else (
            "${SCOOPER_DATABASE_URL}"
            if os.getenv("SCOOPER_DATABASE_URL")
            else resolve_shared_database_url()
        )
    )
    return DatabaseConnectionConfig(
        name=name,
        purpose=purpose,
        url=url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
