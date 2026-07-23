"""Versioned artifact bundles with blue/green promotion and rollback.

Use :class:`ArtifactRegistry` so ML plugins and pipelines can write a
candidate bundle, validate it, then atomically flip the ``current`` pointer.
After promotion, call ``POST /api/v1/plugins/reload`` so in-memory models
pick up the new weights.
"""

from __future__ import annotations

from agentomatic.artifacts.registry import ArtifactRegistry, default_artifact_root

__all__ = ["ArtifactRegistry", "default_artifact_root"]
