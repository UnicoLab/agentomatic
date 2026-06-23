"""Plugin registry — auto-discovery and management."""

from __future__ import annotations

import importlib
from pathlib import Path

from loguru import logger

from .ml import BaseMLPlugin


class PluginRegistry:
    """Central registry that auto-discovers and manages ML plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, BaseMLPlugin] = {}

    @property
    def count(self) -> int:
        """Return the number of registered plugins."""
        return len(self._plugins)

    def get_plugin(self, name: str) -> BaseMLPlugin | None:
        """Get a registered plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> dict[str, BaseMLPlugin]:
        """List all registered plugins."""
        return self._plugins

    def list_names(self) -> list[str]:
        """List names of all registered plugins."""
        return list(self._plugins.keys())

    def discover(self, plugins_dir: Path, package_prefix: str = "") -> None:
        """Auto-discover plugins from a directory.

        Scans for Python packages or modules containing BaseMLPlugin subclasses.
        """
        plugins_dir = Path(plugins_dir).resolve()
        if not plugins_dir.exists():
            logger.debug(f"Plugins directory not found: {plugins_dir}")
            return

        logger.info(f"🔍 Discovering plugins in {plugins_dir}")

        for entry in sorted(plugins_dir.iterdir()):
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue

            # Check if it's a directory with a Python package or a Python file
            if entry.is_dir():
                has_plugin_py = (entry / "plugin.py").exists()
                has_init_py = (entry / "__init__.py").exists()
                if not (has_plugin_py or has_init_py):
                    continue
                self._discover_from_dir(entry, package_prefix)
            elif entry.suffix == ".py":
                self._discover_from_file(entry, package_prefix)

        logger.info(f"📦 Discovery complete — {self.count} plugins registered")

    def _discover_from_dir(self, plugin_dir: Path, package_prefix: str) -> None:
        """Discover a plugin from a directory."""
        plugin_name = plugin_dir.name
        module_path = f"{package_prefix}.{plugin_name}" if package_prefix else plugin_name

        has_plugin_py = (plugin_dir / "plugin.py").exists()

        try:
            if has_plugin_py:
                mod = importlib.import_module(f"{module_path}.plugin")
            else:
                mod = importlib.import_module(module_path)

            self._register_subclasses(mod)
        except ImportError as exc:
            logger.warning(f"  ⚠️ Could not import {module_path}: {exc}")

    def _discover_from_file(self, plugin_file: Path, package_prefix: str) -> None:
        """Discover a plugin from a Python file."""
        module_name = plugin_file.stem
        module_path = f"{package_prefix}.{module_name}" if package_prefix else module_name

        try:
            mod = importlib.import_module(module_path)
            self._register_subclasses(mod)
        except ImportError as exc:
            logger.warning(f"  ⚠️ Could not import {module_path}: {exc}")

    def _register_subclasses(self, mod) -> None:
        """Find and register any BaseMLPlugin subclasses in the module."""
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseMLPlugin)
                and attr is not BaseMLPlugin
            ):
                plugin_instance = attr()
                name = plugin_instance.plugin_name
                if name not in self._plugins:
                    self._plugins[name] = plugin_instance
                    logger.info(
                        f"  ✅ Registered Plugin: {name} (v{plugin_instance.plugin_version})"
                    )
