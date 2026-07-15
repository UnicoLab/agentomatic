"""Store factory: derive a :class:`BaseStore` from any connection.

Lets a platform (or an agent) reuse a declared connection as the backing
store for conversation memory — regardless of whether it is a SQL
database, a Cosmos DB for MongoDB (vCore) cluster, or something entirely
custom.  The registry below is keyed by connection *kind* (or a subclass
predicate) and returns a fully-initialised :class:`BaseStore`.

Register your own factory via :func:`register_store_provider`::

    from agentomatic.connections.stores import register_store_provider

    async def _my_store_builder(connection, **kwargs):
        return MyRedisStore(connection.client, **kwargs)

    register_store_provider(MyRedisConnection, _my_store_builder)

Then, given a connection tagged with ``ConnectionPurpose.MEMORY``,
:func:`create_store_from_connection` returns the matching store::

    conn = get_connections("agent").first_for_purpose(ConnectionPurpose.MEMORY)
    store = await create_store_from_connection(conn)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.storage.base import BaseStore

_STORE_PROVIDERS: dict[type, Any] = {}


def register_store_provider(connection_cls: type, builder: Any) -> None:
    """Register an async store builder for a connection class.

    Args:
        connection_cls: The connection class (not the config class) the
            builder applies to, e.g. :class:`DatabaseConnection`.
        builder: Async callable ``(connection, **kwargs) -> BaseStore``
            that returns a fully-initialised :class:`BaseStore`.
    """
    _STORE_PROVIDERS[connection_cls] = builder


def registered_store_providers() -> list[str]:
    """Return the names of connection classes with a registered store builder."""
    return sorted(cls.__name__ for cls in _STORE_PROVIDERS)


async def create_store_from_connection(
    connection: Any,
    **kwargs: Any,
) -> BaseStore:
    """Return a :class:`BaseStore` bound to ``connection``.

    Dispatches by exact type first, then falls back to :func:`isinstance`
    checks in insertion order so that subclasses inherit their parent's
    builder unless a more specific one is registered.

    Args:
        connection: A live connection (DatabaseConnection, CustomConnection…).
        **kwargs: Forwarded to the underlying store builder.

    Raises:
        TypeError: If no store builder is registered for the connection's
            type.
    """
    builder = _STORE_PROVIDERS.get(type(connection))
    if builder is None:
        for cls, candidate in _STORE_PROVIDERS.items():
            if isinstance(connection, cls):
                builder = candidate
                break
    if builder is None:
        raise TypeError(
            f"No store builder registered for connection type "
            f"'{type(connection).__name__}'. Register one with "
            "`register_store_provider(cls, builder)`."
        )
    logger.debug(
        f"Building store from connection '{getattr(connection, 'name', '?')}' "
        f"({type(connection).__name__})"
    )
    return await builder(connection, **kwargs)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------


async def _build_sql_store(connection: Any, **kwargs: Any) -> BaseStore:
    """Return a SQLAlchemyStore that reuses the connection's engine."""
    return await connection.create_store(**kwargs)


async def _build_custom_store(connection: Any, **kwargs: Any) -> BaseStore:
    """Wrap a :class:`CustomConnection` client in a document store.

    The connection's client is expected to expose the Mongo-compatible
    ``client[db][collection]`` surface consumed by
    :class:`~agentomatic.storage.document.MinimalDocumentStore`.
    """
    from agentomatic.storage.document import MinimalDocumentStore

    await connection.initialize()
    return MinimalDocumentStore(connection.client, **kwargs)


def _register_builtin_store_providers() -> None:
    """Register built-in store providers lazily to avoid import cycles."""
    from agentomatic.connections.custom import CustomConnection
    from agentomatic.connections.database import DatabaseConnection

    register_store_provider(DatabaseConnection, _build_sql_store)
    register_store_provider(CustomConnection, _build_custom_store)


_register_builtin_store_providers()
