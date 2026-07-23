"""Versioned artifact registry with blue/green promotion and rollback.

The historical-update pipeline writes a full artifact *bundle* to a candidate
version directory, validates it against an eval gate, then atomically flips the
``current`` pointer. Rollback re-points ``current`` at any prior version. Each
version records metadata, model cards and eval scores.

Bundle layout (per version dir)::

    <root>/<version>/
        cases.json          # validated historical cases
        embeddings.npz       # similarity embeddings
        metadata.jsonl       # aligned case metadata
        rule_stats.json      # refreshed rule-based stats
        monte_carlo.json     # recomputed Monte Carlo params
        pymc/trace.nc        # optional Bayesian trace
        manifest.json        # version metadata + model cards + eval scores
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ai_core.settings import get_settings


class ArtifactRegistry:
    """Manage versioned artifact bundles and the active ``current`` pointer."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else get_settings().artifact_root
        self._registry_path = self.root / "registry.json"

    # -- registry file -------------------------------------------------------

    def _load_registry(self) -> dict[str, Any]:
        """Load the registry file (or a fresh empty structure)."""
        if not self._registry_path.exists():
            return {"current": None, "versions": {}}
        try:
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"current": None, "versions": {}}

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Atomically write the registry file."""
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self._registry_path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    # -- versions ------------------------------------------------------------

    def new_version_id(self) -> str:
        """Return a fresh, monotonically increasing version id."""
        return f"v{time.strftime('%Y%m%dT%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"

    def candidate_dir(self, version: str) -> Path:
        """Return (creating) the directory for a candidate version bundle."""
        path = self.root / version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_candidate(
        self,
        version: str,
        *,
        model_cards: dict[str, Any] | None = None,
        eval_scores: dict[str, Any] | None = None,
        status: str = "candidate",
    ) -> None:
        """Record a candidate version's metadata without promoting it."""
        data = self._load_registry()
        data["versions"][version] = {
            "version": version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "model_cards": model_cards or {},
            "eval_scores": eval_scores or {},
        }
        self._save_registry(data)

    def promote(self, version: str) -> None:
        """Atomically flip ``current`` to *version* (blue/green go-live)."""
        data = self._load_registry()
        if version not in data["versions"]:
            raise KeyError(f"unknown artifact version: {version}")
        for meta in data["versions"].values():
            if meta.get("status") == "active":
                meta["status"] = "retired"
        data["versions"][version]["status"] = "active"
        data["current"] = version
        self._save_registry(data)

    def rollback(self, version: str) -> None:
        """Re-point ``current`` to a prior version (one-command rollback)."""
        self.promote(version)

    def current_version(self) -> str | None:
        """Return the active version id, or ``None`` when none is promoted."""
        return self._load_registry().get("current")

    def current_dir(self) -> Path | None:
        """Return the active version's bundle directory, or ``None``."""
        version = self.current_version()
        if not version:
            return None
        path = self.root / version
        return path if path.exists() else None

    def version_dir(self, version: str) -> Path:
        """Return the bundle directory for a specific version."""
        return self.root / version

    def list_versions(self) -> list[dict[str, Any]]:
        """Return all recorded versions (newest first)."""
        data = self._load_registry()
        versions = list(data["versions"].values())
        versions.sort(key=lambda v: v.get("created_at", ""), reverse=True)
        return versions

    def manifest(self, version: str | None = None) -> dict[str, Any] | None:
        """Return the registry metadata for a version (default: current)."""
        data = self._load_registry()
        version = version or data.get("current")
        if not version:
            return None
        return data["versions"].get(version)
