"""Safe polling auto-reload for versioned plugin artifact bundles."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from agentomatic.artifacts import ArtifactRegistry

from .registry import PluginRegistry


class PluginAutoReloader:
    """Watch an artifact registry's ``current`` version and reload plugins.

    The registry promotion write is atomic, so the active version id is the
    stable change signal.  A failed reload keeps each plugin's previous state;
    the watcher records the failed pointer and does not hot-loop on a broken
    promotion.  Promote a corrected version (or use the authenticated manual
    reload endpoint) to attempt another load.
    """

    def __init__(
        self,
        plugins: PluginRegistry,
        *,
        artifact_root: str | Path | None = None,
        interval: float = 5.0,
    ) -> None:
        if interval <= 0:
            raise ValueError("plugin auto-reload interval must be greater than zero")
        self._plugins = plugins
        self._artifacts = ArtifactRegistry(artifact_root)
        self._interval = interval
        self._last_observed_version: str | None = None
        self._started = False
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self.last_result: dict[str, Any] | None = None

    @property
    def current_version(self) -> str | None:
        """Return the version most recently observed by the watcher."""
        return self._last_observed_version

    async def start(self) -> None:
        """Begin watching, treating the already-loaded startup version as baseline."""
        if self._task is not None:
            return
        self._last_observed_version = self._artifacts.current_version()
        self._started = True
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._watch(), name="agentomatic-plugin-autoreload")
        logger.info(
            "🔄 Plugin artifact auto-reload enabled (root={}, interval={}s, current={})",
            self._artifacts.root,
            self._interval,
            self._last_observed_version or "none",
        )

    async def stop(self) -> None:
        """Stop the watcher and await its background task."""
        task, self._task = self._task, None
        if self._stop_event is not None:
            self._stop_event.set()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._stop_event = None
        self._started = False

    async def check_once(self) -> bool:
        """Reload once if the promoted artifact version changed.

        Returns:
            ``True`` when a changed non-empty pointer triggered a reload;
            ``False`` when no reload was needed or the pointer was removed.
        """
        current = self._artifacts.current_version()
        if not self._started or current == self._last_observed_version:
            return False

        previous = self._last_observed_version
        # Advance before invoking reload.  A broken promoted version is tried
        # exactly once, avoiding log storms and repeated pressure on a bad
        # model file.  A new promotion or explicit API reload is the recovery.
        self._last_observed_version = current
        if current is None:
            logger.warning(
                "Plugin artifact pointer was removed (was {}); keeping loaded models unchanged",
                previous,
            )
            return False

        logger.info("🔄 Artifact current changed: {} -> {}; reloading plugins", previous, current)
        result = await self._plugins.reload_all()
        self.last_result = result
        if result["failed"]:
            logger.error(
                "Plugin auto-reload for artifact {} completed with {} failure(s); "
                "previous plugin models remain active where reload failed",
                current,
                result["failed"],
            )
        else:
            logger.info("✅ Plugin auto-reload completed for artifact {}", current)
        return True

    async def _watch(self) -> None:
        """Poll until stopped, isolating transient registry/read failures."""
        assert self._stop_event is not None
        while True:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
                return
            except TimeoutError:
                try:
                    await self.check_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - retry the next poll safely
                    logger.exception("Plugin auto-reload watcher check failed: {}", exc)
