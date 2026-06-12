"""Pluggable storage backends.

Available stores:
- ``MemoryStore``: In-memory store for development (always available).
- ``SQLAlchemyStore``: Production store with connection pooling (requires ``db`` extra).

Usage::

    from agentomatic.storage.memory import MemoryStore
    store = MemoryStore()

    # Production:
    from agentomatic.storage.sqlalchemy import SQLAlchemyStore
    store = SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/db")
    await store.initialize()
"""
from __future__ import annotations
