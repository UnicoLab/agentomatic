"""Endpoint registry — auto-discovery and management.

Mirrors :class:`agentomatic.plugins.registry.PluginRegistry`: it scans a
directory for :class:`BaseEndpoint` subclasses and registers one instance
per endpoint.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from loguru import logger

from agentomatic.endpoints.base import BaseEndpoint


class EndpointRegistry:
    """Central registry that auto-discovers and manages custom endpoints."""

    def __init__(self) -> None:
        self._endpoints: dict[str, BaseEndpoint] = {}

    @property
    def count(self) -> int:
        """Return the number of registered endpoints."""
        return len(self._endpoints)

    def register(self, endpoint: BaseEndpoint) -> None:
        """Register an endpoint instance programmatically."""
        self._endpoints[endpoint.endpoint_name] = endpoint
        logger.info(
            f"  ✅ Registered Endpoint: {endpoint.endpoint_name} (v{endpoint.endpoint_version})"
        )

    def get(self, name: str) -> BaseEndpoint | None:
        """Get a registered endpoint by name."""
        return self._endpoints.get(name)

    def list_endpoints(self) -> dict[str, BaseEndpoint]:
        """List all registered endpoints keyed by name."""
        return self._endpoints

    def list_names(self) -> list[str]:
        """List the names of all registered endpoints."""
        return list(self._endpoints.keys())

    def discover(self, endpoints_dir: Path, package_prefix: str = "") -> None:
        """Auto-discover endpoints from a directory.

        Scans for Python packages or modules containing ``BaseEndpoint``
        subclasses, following the same directory conventions as plugins.

        Args:
            endpoints_dir: Path to the endpoints directory.
            package_prefix: Python import prefix for the endpoints package.
        """
        endpoints_dir = Path(endpoints_dir).resolve()
        if not endpoints_dir.exists():
            logger.debug(f"Endpoints directory not found: {endpoints_dir}")
            return

        logger.info(f"🔍 Discovering endpoints in {endpoints_dir}")

        for entry in sorted(endpoints_dir.iterdir()):
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue

            if entry.is_dir():
                has_endpoint_py = (entry / "endpoint.py").exists()
                has_init_py = (entry / "__init__.py").exists()
                if not (has_endpoint_py or has_init_py):
                    continue
                self._discover_from_dir(entry, package_prefix)
            elif entry.suffix == ".py":
                self._discover_from_file(entry, package_prefix)

        logger.info(f"📦 Discovery complete — {self.count} endpoint(s) registered")

    def _discover_from_dir(self, endpoint_dir: Path, package_prefix: str) -> None:
        """Discover an endpoint from a directory."""
        name = endpoint_dir.name
        module_path = f"{package_prefix}.{name}" if package_prefix else name
        has_endpoint_py = (endpoint_dir / "endpoint.py").exists()

        try:
            if has_endpoint_py:
                mod = importlib.import_module(f"{module_path}.endpoint")
            else:
                mod = importlib.import_module(module_path)
            self._register_subclasses(mod)
        except ImportError as exc:
            logger.warning(f"  ⚠️ Could not import {module_path}: {exc}")

    def _discover_from_file(self, endpoint_file: Path, package_prefix: str) -> None:
        """Discover an endpoint from a Python file."""
        module_name = endpoint_file.stem
        module_path = f"{package_prefix}.{module_name}" if package_prefix else module_name

        try:
            mod = importlib.import_module(module_path)
            self._register_subclasses(mod)
        except ImportError as exc:
            logger.warning(f"  ⚠️ Could not import {module_path}: {exc}")

    def _register_subclasses(self, mod: object) -> None:
        """Find and register any ``BaseEndpoint`` subclasses in ``mod``."""
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseEndpoint)
                and attr is not BaseEndpoint
            ):
                instance = attr()
                name = instance.endpoint_name
                if name not in self._endpoints:
                    self.register(instance)
