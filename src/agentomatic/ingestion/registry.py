"""Ingestor registry — auto-discovery and management."""

from __future__ import annotations

import importlib
from pathlib import Path

from loguru import logger

from .base import BaseIngestor


class IngestionRegistry:
    """Central registry that auto-discovers and manages ingestors.

    Discovery scans an ``ingestion/`` directory for Python packages or modules
    that define :class:`BaseIngestor` subclasses (in ``ingestor.py`` or the
    package ``__init__.py``), mirroring the plugin/endpoint discovery model.
    """

    def __init__(self) -> None:
        self._ingestors: dict[str, BaseIngestor] = {}

    @property
    def count(self) -> int:
        """Return the number of registered ingestors."""
        return len(self._ingestors)

    def register(self, ingestor: BaseIngestor) -> None:
        """Register an ingestor instance programmatically."""
        self._ingestors[ingestor.ingestor_name] = ingestor
        logger.info(
            f"  ✅ Registered Ingestor: {ingestor.ingestor_name} (v{ingestor.ingestor_version})"
        )

    def get(self, name: str) -> BaseIngestor | None:
        """Return a registered ingestor by name."""
        return self._ingestors.get(name)

    def list_ingestors(self) -> dict[str, BaseIngestor]:
        """Return all registered ingestors."""
        return self._ingestors

    def list_names(self) -> list[str]:
        """Return the names of all registered ingestors."""
        return list(self._ingestors.keys())

    def discover(self, ingestion_dir: Path, package_prefix: str = "") -> None:
        """Auto-discover ingestors from a directory."""
        ingestion_dir = Path(ingestion_dir).resolve()
        if not ingestion_dir.exists():
            logger.debug(f"Ingestion directory not found: {ingestion_dir}")
            return

        logger.info(f"🔍 Discovering ingestors in {ingestion_dir}")

        for entry in sorted(ingestion_dir.iterdir()):
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue
            if entry.is_dir():
                has_ingestor_py = (entry / "ingestor.py").exists()
                has_init_py = (entry / "__init__.py").exists()
                if not (has_ingestor_py or has_init_py):
                    continue
                self._discover_from_dir(entry, package_prefix)
            elif entry.suffix == ".py":
                self._discover_from_file(entry, package_prefix)

        logger.info(f"📦 Ingestion discovery complete — {self.count} ingestor(s) registered")

    def _discover_from_dir(self, ingestor_dir: Path, package_prefix: str) -> None:
        name = ingestor_dir.name
        module_path = f"{package_prefix}.{name}" if package_prefix else name
        try:
            if (ingestor_dir / "ingestor.py").exists():
                mod = importlib.import_module(f"{module_path}.ingestor")
            else:
                mod = importlib.import_module(module_path)
            self._register_subclasses(mod)
        except ImportError as exc:
            logger.warning(f"  ⚠️ Could not import {module_path}: {exc}")

    def _discover_from_file(self, ingestor_file: Path, package_prefix: str) -> None:
        module_name = ingestor_file.stem
        module_path = f"{package_prefix}.{module_name}" if package_prefix else module_name
        try:
            mod = importlib.import_module(module_path)
            self._register_subclasses(mod)
        except ImportError as exc:
            logger.warning(f"  ⚠️ Could not import {module_path}: {exc}")

    def _register_subclasses(self, mod: object) -> None:
        """Find and register any BaseIngestor subclasses in the module."""
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseIngestor)
                and attr is not BaseIngestor
            ):
                instance = attr()
                if instance.ingestor_name not in self._ingestors:
                    self.register(instance)
