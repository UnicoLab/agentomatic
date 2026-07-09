"""Connection manager and per-agent connection registry.

The :class:`ConnectionManager` owns a set of named connections for a single
scope (an agent, or the shared ``platform`` scope).  A module-level
registry maps scope names to managers so agent code can retrieve live,
initialised connections with a single call::

    from agentomatic.connections import get_connections

    db = get_connections("my_agent").database("main")
    async with db.session() as session:
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.connections.custom import CustomConnection
from agentomatic.connections.database import DatabaseConnection
from agentomatic.connections.http import HttpConnection
from agentomatic.connections.models import (
    ConnectionConfig,
    ConnectionPurpose,
    CustomConnectionConfig,
    DatabaseConnectionConfig,
    HttpConnectionConfig,
    VectorConnectionConfig,
)
from agentomatic.connections.vector import VectorConnection

if TYPE_CHECKING:
    from collections.abc import Callable

#: Scope name used for connections shared across the whole platform.
PLATFORM_SCOPE = "__platform__"

#: A live connection object (built-in or user-registered).
Connection = Any


# ---------------------------------------------------------------------------
# Extensible connection-type registry
# ---------------------------------------------------------------------------

#: Maps a config class to the callable that builds its live connection.
_CONNECTION_BUILDERS: dict[type, Callable[[Any], Connection]] = {
    DatabaseConnectionConfig: DatabaseConnection,
    HttpConnectionConfig: HttpConnection,
    VectorConnectionConfig: VectorConnection,
    CustomConnectionConfig: CustomConnection,
}


def register_connection_type(
    config_cls: type,
    builder: Callable[[Any], Connection],
) -> None:
    """Register a custom connection type so *any* backend is pluggable.

    This is the escape hatch that lets you connect to redis, an in-house
    feature store, a graph database, etc. with minimal code::

        from agentomatic.connections import register_connection_type

        register_connection_type(RedisConnectionConfig, RedisConnection)

    The ``builder`` receives the config instance and must return an object
    exposing ``name``, ``async initialize()``, ``async health_check()`` and
    ``async close()``.

    Args:
        config_cls: The Pydantic config class identifying the connection.
        builder: Callable building a live connection from the config.
    """
    _CONNECTION_BUILDERS[config_cls] = builder


def _build_connection(config: ConnectionConfig) -> Connection:
    """Build a live connection from its config via the type registry."""
    builder = _CONNECTION_BUILDERS.get(type(config))
    if builder is None:
        # Fall back to MRO so subclasses of a known config work too.
        for cls, candidate in _CONNECTION_BUILDERS.items():
            if isinstance(config, cls):
                builder = candidate
                break
    if builder is None:
        raise TypeError(
            f"Unsupported connection config: {type(config).__name__}. "
            "Register it with `register_connection_type(config_cls, builder)`."
        )
    return builder(config)


class ConnectionManager:
    """Manage a named set of connections for a single scope."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self._connections: dict[str, Connection] = {}

    @property
    def count(self) -> int:
        """Return the number of managed connections."""
        return len(self._connections)

    def add(self, config: ConnectionConfig) -> None:
        """Register a connection from its configuration."""
        self._connections[config.name] = _build_connection(config)

    def add_many(self, configs: list[ConnectionConfig]) -> None:
        """Register several connections at once."""
        for cfg in configs:
            self.add(cfg)

    def get(self, name: str) -> Connection | None:
        """Return a connection by name."""
        return self._connections.get(name)

    def database(self, name: str) -> DatabaseConnection:
        """Return a database connection by name.

        Raises:
            KeyError: If no database connection with ``name`` exists.
        """
        conn = self._connections.get(name)
        if not isinstance(conn, DatabaseConnection):
            raise KeyError(
                f"No database connection '{name}' in scope '{self.scope}'. "
                f"Available: {self.list_names()}"
            )
        return conn

    def http(self, name: str) -> HttpConnection:
        """Return an HTTP connection by name.

        Raises:
            KeyError: If no HTTP connection with ``name`` exists.
        """
        conn = self._connections.get(name)
        if not isinstance(conn, HttpConnection):
            raise KeyError(
                f"No HTTP connection '{name}' in scope '{self.scope}'. "
                f"Available: {self.list_names()}"
            )
        return conn

    def vector(self, name: str) -> VectorConnection:
        """Return a vector-store connection by name.

        Raises:
            KeyError: If no vector connection with ``name`` exists.
        """
        conn = self._connections.get(name)
        if not isinstance(conn, VectorConnection):
            raise KeyError(
                f"No vector connection '{name}' in scope '{self.scope}'. "
                f"Available: {self.list_names()}"
            )
        return conn

    def custom(self, name: str) -> CustomConnection:
        """Return a generic factory-based connection by name.

        Raises:
            KeyError: If no custom connection with ``name`` exists.
        """
        conn = self._connections.get(name)
        if not isinstance(conn, CustomConnection):
            raise KeyError(
                f"No custom connection '{name}' in scope '{self.scope}'. "
                f"Available: {self.list_names()}"
            )
        return conn

    async def client(self, name: str) -> Any:
        """Return the native client of any client-backed connection.

        Works uniformly across vector and custom connections, initialising the
        connection on demand so agent code needs a single line::

            redis = await get_connections("agent").client("cache")
        """
        conn = self._connections.get(name)
        if conn is None:
            raise KeyError(
                f"No connection '{name}' in scope '{self.scope}'. Available: {self.list_names()}"
            )
        if not hasattr(type(conn), "client"):
            raise TypeError(
                f"Connection '{name}' ({type(conn).__name__}) exposes no native client."
            )
        await conn.initialize()
        return conn.client

    def list_names(self) -> list[str]:
        """List the names of all managed connections."""
        return list(self._connections.keys())

    # -- purpose-based lookup -------------------------------------------

    def by_purpose(self, purpose: ConnectionPurpose | str) -> dict[str, Connection]:
        """Return ``{name: connection}`` for all connections of a purpose.

        Lets features find backends by intent regardless of kind, e.g. every
        connection tagged ``ConnectionPurpose.RAG`` (which may be a database
        *and* a vector store)::

            for name, conn in get_connections("agent").by_purpose("rag").items():
                ...
        """
        want = str(purpose)
        return {
            name: conn
            for name, conn in self._connections.items()
            if str(getattr(getattr(conn, "config", None), "purpose", ConnectionPurpose.GENERAL))
            == want
        }

    def for_purpose(self, purpose: ConnectionPurpose | str) -> list[Connection]:
        """Return the list of connections tagged with ``purpose``."""
        return list(self.by_purpose(purpose).values())

    def first_for_purpose(self, purpose: ConnectionPurpose | str) -> Connection | None:
        """Return the first connection tagged with ``purpose`` (or ``None``)."""
        conns = self.for_purpose(purpose)
        return conns[0] if conns else None

    async def initialize(self) -> None:
        """Initialise all connections in this scope."""
        for name, conn in self._connections.items():
            try:
                await conn.initialize()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to initialize connection '{name}': {exc}")

    async def health_check(self) -> dict[str, Any]:
        """Aggregate health across all connections."""
        results: dict[str, Any] = {}
        for name, conn in self._connections.items():
            try:
                results[name] = await conn.health_check()
            except Exception as exc:  # noqa: BLE001
                results[name] = {"connection": name, "status": "error", "error": str(exc)}
        return results

    async def close(self) -> None:
        """Close all connections in this scope."""
        for conn in self._connections.values():
            try:
                await conn.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Error closing connection '{conn.name}': {exc}")


# ---------------------------------------------------------------------------
# Module-level scope registry
# ---------------------------------------------------------------------------

_managers: dict[str, ConnectionManager] = {}


def register_connections(scope: str, configs: list[ConnectionConfig]) -> ConnectionManager:
    """Create (or extend) the connection manager for a scope.

    Args:
        scope: The scope name (agent name, or ``PLATFORM_SCOPE``).
        configs: Connection configurations to register.

    Returns:
        The :class:`ConnectionManager` for the scope.
    """
    manager = _managers.setdefault(scope, ConnectionManager(scope))
    manager.add_many(configs)
    return manager


def get_connections(scope: str = PLATFORM_SCOPE) -> ConnectionManager:
    """Return the connection manager for a scope, creating it if needed.

    Args:
        scope: The scope name (agent name, or ``PLATFORM_SCOPE`` for
            platform-wide connections).

    Returns:
        The scope's :class:`ConnectionManager` (possibly empty).
    """
    return _managers.setdefault(scope, ConnectionManager(scope))


def all_managers() -> dict[str, ConnectionManager]:
    """Return all registered connection managers keyed by scope."""
    return dict(_managers)


def reset_connections() -> None:
    """Clear all registered connection managers (for testing)."""
    _managers.clear()
