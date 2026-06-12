"""Pluggable storage backends.

Agentomatic ships with two storage backends:

- :class:`MemoryStore` — in-memory (dev, testing)
- :class:`SQLAlchemyStore` — async SQL with connection pooling (production)

You can implement your own backend by subclassing :class:`BaseStore`.

Usage::

    # Development
    from agentomatic.storage import MemoryStore
    store = MemoryStore()

    # Production
    from agentomatic.storage import SQLAlchemyStore
    store = SQLAlchemyStore("postgresql+asyncpg://...")
    await store.initialize()

    # Custom
    from agentomatic.storage import BaseStore
    class RedisStore(BaseStore): ...
"""

from __future__ import annotations

from .base import BaseStore
from .memory import MemoryStore

__all__ = ["BaseStore", "MemoryStore"]


# Lazy import for optional SQLAlchemy dependency
def __getattr__(name: str):
    if name == "SQLAlchemyStore":
        from .sqlalchemy import SQLAlchemyStore

        return SQLAlchemyStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
