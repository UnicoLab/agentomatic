"""Versioned resource bundles for optimisation (NamedResources lite).

Wraps :class:`PromptRuntimeConfig` with a monotonic ``resource_id`` so
algorithms can attribute rollouts to a specific candidate configuration,
mirroring Agent Lightning's ``NamedResources`` / ``resources_id`` without
a distributed store.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agentomatic.optimize.config import PromptRuntimeConfig


@dataclass(slots=True)
class ResourceBundle:
    """A versioned snapshot of the optimisable runtime surface."""

    resource_id: str
    config: PromptRuntimeConfig
    label: str = ""
    parent_id: str | None = None
    notes: str = ""
    score: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "resource_id": self.resource_id,
            "config": self.config.to_dict(),
            "label": self.label,
            "parent_id": self.parent_id,
            "notes": self.notes,
            "score": self.score,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceBundle:
        """Deserialise from a plain dict."""
        return cls(
            resource_id=str(data.get("resource_id") or "v0"),
            config=PromptRuntimeConfig.from_dict(data.get("config") or {}),
            label=str(data.get("label") or ""),
            parent_id=data.get("parent_id"),
            notes=str(data.get("notes") or ""),
            score=data.get("score"),
            created_at=str(data.get("created_at") or datetime.now(UTC).isoformat()),
            metadata=dict(data.get("metadata") or {}),
        )


class ResourceRegistry:
    """In-process registry of versioned :class:`ResourceBundle` snapshots."""

    def __init__(self, *, prefix: str = "r") -> None:
        self._prefix = prefix
        self._counter = 0
        self._bundles: dict[str, ResourceBundle] = {}
        self._lock = threading.RLock()
        self._latest_id: str | None = None

    def publish(
        self,
        config: PromptRuntimeConfig,
        *,
        label: str = "",
        parent_id: str | None = None,
        notes: str = "",
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
        resource_id: str | None = None,
    ) -> ResourceBundle:
        """Register a new resource version and return the bundle."""
        with self._lock:
            if resource_id is None:
                resource_id = f"{self._prefix}{self._counter}"
                self._counter += 1
            bundle = ResourceBundle(
                resource_id=resource_id,
                config=config,
                label=label,
                parent_id=parent_id,
                notes=notes,
                score=score,
                metadata=dict(metadata or {}),
            )
            self._bundles[resource_id] = bundle
            self._latest_id = resource_id
            return bundle

    def get(self, resource_id: str) -> ResourceBundle | None:
        """Look up a bundle by id."""
        with self._lock:
            return self._bundles.get(resource_id)

    def latest(self) -> ResourceBundle | None:
        """Return the most recently published bundle."""
        with self._lock:
            if self._latest_id is None:
                return None
            return self._bundles.get(self._latest_id)

    def history(self) -> list[ResourceBundle]:
        """Return all bundles in publish order."""
        with self._lock:
            return list(self._bundles.values())

    def update_score(self, resource_id: str, score: float) -> None:
        """Attach an evaluation score to an existing bundle."""
        with self._lock:
            bundle = self._bundles.get(resource_id)
            if bundle is not None:
                bundle.score = score
