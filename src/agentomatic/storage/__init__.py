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

from typing import TYPE_CHECKING

from .base import BaseStore
from .memory import MemoryStore

if TYPE_CHECKING:
    # Lazily imported at runtime via __getattr__ below; importing here
    # lets the type checker see the names exported in __all__.
    from .checkpointer import AgentomaticCheckpointer
    from .document import MinimalDocumentStore
    from .sqlalchemy import SQLAlchemyStore

__all__ = [
    "AgentomaticCheckpointer",
    "BaseStore",
    "MemoryStore",
    "MinimalDocumentStore",
    "SQLAlchemyStore",
]


# Lazy import for optional dependencies
def __getattr__(name: str):
    if name == "SQLAlchemyStore":
        from .sqlalchemy import SQLAlchemyStore

        return SQLAlchemyStore
    if name == "AgentomaticCheckpointer":
        from .checkpointer import AgentomaticCheckpointer

        return AgentomaticCheckpointer
    if name == "MinimalDocumentStore":
        from .document import MinimalDocumentStore

        return MinimalDocumentStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
