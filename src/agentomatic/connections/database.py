"""Async database connections for agents.

Wraps a SQLAlchemy async engine + session factory per named connection so
each agent can talk to its own database (with its own credentials) without
boilerplate.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus, urlsplit, urlunsplit

from loguru import logger

from agentomatic.connections.models import DatabaseConnectionConfig
from agentomatic.endpoints.auth import resolve_env

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


def _observe_connection(name: str, status: str) -> None:
    """Emit a best-effort connection-acquisition metric."""
    try:
        from agentomatic.observability.metrics import CONNECTION_CALL_COUNT

        CONNECTION_CALL_COUNT.labels(connection=name, status=status).inc()
    except Exception:  # noqa: BLE001 - metrics are optional
        pass


def _inject_credentials(url: str, username: str, password: str) -> str:
    """Splice ``username``/``password`` into a URL's netloc when provided."""
    if not username and not password:
        return url
    parts = urlsplit(url)
    host = parts.hostname or ""
    creds = quote_plus(username)
    if password:
        creds = f"{creds}:{quote_plus(password)}"
    netloc = f"{creds}@{host}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class DatabaseConnection:
    """A lazily-initialised async database connection.

    Example::

        db = DatabaseConnection(config)
        await db.initialize()
        async with db.session() as session:
            result = await session.execute(text("SELECT 1"))
    """

    def __init__(self, config: DatabaseConnectionConfig) -> None:
        self.config = config
        self._engine: AsyncEngine | None = None
        self._sessionmaker: Any = None
        self._resolved_url: str = ""

    @property
    def name(self) -> str:
        """Return the connection's logical name."""
        return self.config.name

    @property
    def engine(self) -> AsyncEngine:
        """Return the underlying engine (must call :meth:`initialize` first)."""
        if self._engine is None:
            raise RuntimeError(
                f"Database connection '{self.name}' is not initialized. "
                "Call `await connection.initialize()` first."
            )
        return self._engine

    async def initialize(self) -> None:
        """Create the async engine and session factory."""
        if self._engine is not None:
            return

        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SQLAlchemy is required for database connections. "
                "Install with: pip install 'agentomatic[db]'"
            ) from exc

        cfg = self.config
        url = _inject_credentials(
            resolve_env(cfg.url),
            resolve_env(cfg.username),
            resolve_env(cfg.password),
        )
        self._resolved_url = url

        engine_kwargs: dict[str, Any] = {
            "echo": cfg.echo,
            "pool_pre_ping": cfg.pool_pre_ping,
            "connect_args": cfg.connect_args,
        }
        # SQLite async engines do not support pool sizing kwargs.
        if not url.startswith("sqlite"):
            engine_kwargs.update(
                pool_size=cfg.pool_size,
                max_overflow=cfg.max_overflow,
                pool_timeout=cfg.pool_timeout,
            )

        self._engine = create_async_engine(url, **engine_kwargs)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        logger.info(f"🗄️ Database connection '{self.name}' initialized")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield an ``AsyncSession`` bound to this connection."""
        if self._sessionmaker is None:
            await self.initialize()
        try:
            session = self._sessionmaker()
        except Exception:
            _observe_connection(self.name, "error")
            raise
        _observe_connection(self.name, "ok")
        try:
            yield session
        finally:
            await session.close()

    async def create_store(self, **store_kwargs: Any) -> Any:
        """Build a :class:`SQLAlchemyStore` backed by this connection.

        Lets an agent reuse its own (authenticated) database as the backing
        store for conversation *memory* — declare the connection with
        ``purpose=ConnectionPurpose.MEMORY`` and wire the returned store into
        the platform::

            db = get_connections("agent").database("memory")
            store = await db.create_store()

        The store shares this connection's engine and pool, so closing the
        platform will not double-dispose it.

        Returns:
            An initialised :class:`~agentomatic.storage.SQLAlchemyStore`.
        """
        await self.initialize()
        from agentomatic.storage.sqlalchemy import SQLAlchemyStore

        store = SQLAlchemyStore(
            url=self._resolved_url or resolve_env(self.config.url),
            engine=self._engine,
            **store_kwargs,
        )
        await store.initialize()
        return store

    async def health_check(self) -> dict[str, Any]:
        """Run ``SELECT 1`` to verify the connection is healthy."""
        try:
            from sqlalchemy import text

            if self._engine is None:
                await self.initialize()
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"connection": self.name, "kind": "database", "status": "healthy"}
        except Exception as exc:  # noqa: BLE001
            return {
                "connection": self.name,
                "kind": "database",
                "status": "unhealthy",
                "error": str(exc),
            }

    async def close(self) -> None:
        """Dispose the underlying engine."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
            logger.debug(f"Database connection '{self.name}' closed")
