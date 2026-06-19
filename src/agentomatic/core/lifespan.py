"""Application lifespan management."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from fastapi import FastAPI


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru with a standard format.

    Removes all existing handlers and installs a single ``stdout``
    sink with coloured, structured output.

    Args:
        level: Minimum log level (e.g. ``"DEBUG"``, ``"INFO"``).
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
    )


@asynccontextmanager
async def create_lifespan(
    registry: Any,
    agents_dir: str | Path,
    package_prefix: str,
    settings: Any,
    on_startup: list[Callable[..., Any]] | None = None,
    on_shutdown: list[Callable[..., Any]] | None = None,
) -> AsyncIterator[Callable[[FastAPI], Any]]:
    """Create a FastAPI lifespan context manager.

    .. deprecated::
        This function is retained for backward compatibility but is no
        longer used by :meth:`AgentPlatform.build`, which creates its own
        inline lifespan.  Prefer :meth:`AgentPlatform.on_startup` /
        :meth:`AgentPlatform.on_shutdown` hooks instead.

    Startup sequence:
      1. Configure logging
      2. Discover agents
      3. Run custom startup hooks

    Shutdown sequence:
      1. Run custom shutdown hooks

    Args:
        registry: The :class:`AgentRegistry` instance.
        agents_dir: Path to the agents directory.
        package_prefix: Python package prefix for agent imports.
        settings: Application settings object.
        on_startup: Optional list of callables to run at startup.
        on_shutdown: Optional list of callables to run at shutdown.

    Yields:
        A lifespan callable suitable for :class:`~fastapi.FastAPI`.
    """

    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # --- Startup ---
        configure_logging(getattr(settings, "log_level", "INFO"))
        logger.info("🚀 Agentomatic platform starting...")

        # Discover agents
        agents_path = Path(agents_dir).resolve()
        if agents_path.parent not in [Path(p) for p in sys.path]:
            sys.path.insert(0, str(agents_path.parent))

        registry.discover(agents_path, package_prefix)
        logger.info(f"📦 {registry.count} agent(s) ready")

        # Custom startup hooks
        if on_startup:
            for hook in on_startup:
                if callable(hook):
                    result = hook()
                    if hasattr(result, "__await__"):
                        await result

        yield

        # --- Shutdown ---
        logger.info("🛑 Agentomatic platform shutting down...")
        if on_shutdown:
            for hook in on_shutdown:
                if callable(hook):
                    result = hook()
                    if hasattr(result, "__await__"):
                        await result

    yield _lifespan
