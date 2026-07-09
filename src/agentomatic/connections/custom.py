"""Generic factory-based connections for *any* backend.

The most flexible, lowest-code way to connect to a backend Agentomatic does
not ship first-class support for (redis, mongo, elasticsearch, neo4j, a
proprietary SDK, …).  Instead of writing a config *and* a connection class,
declare a single :class:`~agentomatic.connections.models.CustomConnectionConfig`
pointing at any factory callable — this module manages the rest (env
interpolation, lazy build, sync/async factories, and lifecycle)::

    CustomConnectionConfig(
        name="cache",
        factory="redis.asyncio.from_url",
        args=["${REDIS_URL}"],
        purpose=ConnectionPurpose.CACHE,
    )
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from loguru import logger

from agentomatic.connections.models import CustomConnectionConfig
from agentomatic.endpoints.auth import resolve_env


def resolve_env_deep(value: Any) -> Any:
    """Recursively resolve ``${ENV}`` placeholders in strings / lists / dicts."""
    if isinstance(value, str):
        return resolve_env(value)
    if isinstance(value, list):
        return [resolve_env_deep(v) for v in value]
    if isinstance(value, tuple):
        return tuple(resolve_env_deep(v) for v in value)
    if isinstance(value, dict):
        return {k: resolve_env_deep(v) for k, v in value.items()}
    return value


def import_from_path(path: str) -> Any:
    """Import a callable/object from a dotted path.

    Supports both ``package.module:attribute`` and ``package.module.attribute``.
    """
    module_path, _, attr = path.partition(":")
    if not attr:
        module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"Invalid factory path: '{path}' (expected 'pkg.mod:func').")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


class CustomConnection:
    """A generic, lazily-built connection wrapping an arbitrary client."""

    def __init__(self, config: CustomConnectionConfig) -> None:
        self.config = config
        self._client: Any = None

    @property
    def name(self) -> str:
        """Return the connection's logical name."""
        return self.config.name

    async def initialize(self) -> None:
        """Build the underlying client (idempotent)."""
        if self._client is not None:
            return

        factory = self.config.factory
        if isinstance(factory, str):
            factory = import_from_path(resolve_env(factory))
        if not callable(factory):
            raise TypeError(
                f"Custom connection '{self.name}' factory is not callable: {factory!r}"
            )

        args = resolve_env_deep(self.config.args)
        kwargs = resolve_env_deep(self.config.kwargs)
        result = factory(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        self._client = result
        logger.info(f"🔌 Custom connection '{self.name}' initialized")

    @property
    def client(self) -> Any:
        """Return the underlying client (must call :meth:`initialize` first)."""
        if self._client is None:
            raise RuntimeError(
                f"Custom connection '{self.name}' is not initialized. "
                "Call `await connection.initialize()` first."
            )
        return self._client

    async def _call(self, method: str) -> Any:
        """Call a (possibly async) zero-arg method on the client."""
        fn = getattr(self._client, method, None)
        if fn is None or not callable(fn):
            return None
        result = fn()
        if inspect.isawaitable(result):
            result = await result
        return result

    async def health_check(self) -> dict[str, Any]:
        """Report health via a configured/auto-detected method (best effort)."""
        base = {"connection": self.name, "kind": "custom"}
        if self._client is None:
            try:
                await self.initialize()
            except Exception as exc:  # noqa: BLE001
                return {**base, "status": "unhealthy", "error": str(exc)}

        method = self.config.health_method or next(
            (m for m in ("ping", "health_check", "is_connected") if hasattr(self._client, m)),
            "",
        )
        if not method:
            return {**base, "status": "configured"}
        try:
            await self._call(method)
            return {**base, "status": "healthy"}
        except Exception as exc:  # noqa: BLE001
            return {**base, "status": "unhealthy", "error": str(exc)}

    async def close(self) -> None:
        """Close the client via the configured/auto-detected shutdown method."""
        if self._client is None:
            return
        method = self.config.close_method or next(
            (m for m in ("aclose", "close", "disconnect") if hasattr(self._client, m)),
            "",
        )
        if method:
            try:
                await self._call(method)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Custom connection '{self.name}' close error: {exc}")
        self._client = None
