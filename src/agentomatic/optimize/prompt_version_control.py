"""Prompt version control with rollback support.

Tracks every version of a prompt during optimisation, enabling
rollback to any previous version and retrieval of the best-scoring
variant.

Example::

    from agentomatic.optimize.prompt_version_control import PromptVersionControl

    pvc = PromptVersionControl(agent_name="my_agent")

    pvc.add_version("You are a helpful assistant", score=0.72)
    pvc.add_version("You are an expert assistant", score=0.85)

    best = pvc.get_best()
    pvc.rollback(0)  # Go back to the first version
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PromptVersion:
    """A single versioned prompt with metadata."""

    prompt: str
    score: float = 0.0
    iteration: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


class PromptVersionControl:
    """Tracks prompt versions and supports rollback.

    Args:
        agent_name: Name of the agent (used for storage).
        save_dir: Directory for persisted versions.
        max_versions: Maximum versions to keep in memory.

    Example::

        vc = PromptVersionControl("my_agent")
        vc.add_version(prompt_text, score=0.85, iteration=3)
        best = vc.get_best()
        vc.rollback(1)  # Go back 1 version
    """

    def __init__(
        self,
        agent_name: str = "agent",
        save_dir: str = "optimization_results",
        max_versions: int = 50,
    ) -> None:
        self.agent_name = agent_name
        self.save_dir = Path(save_dir) / agent_name
        self.max_versions = max_versions
        self._versions: list[PromptVersion] = []
        self._current_idx: int = -1

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def add_version(
        self,
        prompt: str,
        score: float = 0.0,
        iteration: int = 0,
        metrics: dict[str, float] | None = None,
        notes: str = "",
    ) -> int:
        """Record a new prompt version.

        Args:
            prompt: The prompt text.
            score: Composite score for this version.
            iteration: Optimisation round.
            metrics: Per-metric scores.
            notes: Free-form notes.

        Returns:
            The version index (0-based).
        """
        version = PromptVersion(
            prompt=prompt,
            score=score,
            iteration=iteration,
            metrics=metrics or {},
            notes=notes,
        )
        self._versions.append(version)
        self._current_idx = len(self._versions) - 1

        # Prune old versions
        while len(self._versions) > self.max_versions:
            self._versions.pop(0)
            self._current_idx = max(0, self._current_idx - 1)

        return self._current_idx

    def get_best(self) -> PromptVersion | None:
        """Return the highest-scoring version.

        Returns:
            :class:`PromptVersion` or ``None`` if no versions exist.
        """
        if not self._versions:
            return None
        return max(self._versions, key=lambda v: v.score)

    def get_current(self) -> PromptVersion | None:
        """Return the most recently added version."""
        if self._current_idx < 0 or not self._versions:
            return None
        return self._versions[self._current_idx]

    def get_version(self, index: int) -> PromptVersion | None:
        """Get a version by index (0-based).

        Returns ``None`` if *index* is out of range.
        """
        if 0 <= index < len(self._versions):
            return self._versions[index]
        return None

    def rollback(self, steps_back: int = 1) -> PromptVersion | None:
        """Move the current pointer back by *steps_back*.

        Args:
            steps_back: Number of versions to go back.

        Returns:
            The version we rolled back to, or ``None`` if not possible.
        """
        target = self._current_idx - steps_back
        if target < 0:
            return None
        self._current_idx = target
        return self._versions[self._current_idx]

    def rollback_to(self, index: int) -> PromptVersion | None:
        """Roll back to a specific version index.

        Args:
            index: Target version index (0-based).

        Returns:
            The target version, or ``None`` if out of range.
        """
        if 0 <= index < len(self._versions):
            self._current_idx = index
            return self._versions[self._current_idx]
        return None

    @property
    def version_count(self) -> int:
        """Total number of stored versions."""
        return len(self._versions)

    @property
    def current_index(self) -> int:
        """Index of the current version."""
        return self._current_idx

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def score_history(self) -> list[float]:
        """Return the score of each version (chronological)."""
        return [v.score for v in self._versions]

    def improvement(self) -> float:
        """Best score minus first score."""
        if not self._versions:
            return 0.0
        first = self._versions[0].score
        best = self.get_best()
        return (best.score - first) if best else 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize all versions to a dictionary."""
        return {
            "agent_name": self.agent_name,
            "current_index": self._current_idx,
            "versions": [
                {
                    "prompt": v.prompt,
                    "score": v.score,
                    "iteration": v.iteration,
                    "timestamp": v.timestamp,
                    "metrics": v.metrics,
                    "notes": v.notes,
                }
                for v in self._versions
            ],
        }

    def save(self, filename: str = "prompt_versions.json") -> Path:
        """Persist all versions to a JSON file.

        Args:
            filename: Output filename (inside ``save_dir``).

        Returns:
            Path to the saved file.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fpath = self.save_dir / filename
        fpath.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return fpath

    @classmethod
    def load(cls, agent_name: str, save_dir: str = "optimization_results") -> PromptVersionControl:
        """Load versions from a JSON file.

        Args:
            agent_name: Agent name to load.
            save_dir: Base directory for version files.

        Returns:
            A :class:`PromptVersionControl` with restored versions.
        """
        fpath = Path(save_dir) / agent_name / "prompt_versions.json"
        vc = cls(agent_name=agent_name, save_dir=save_dir)

        if fpath.exists():
            data = json.loads(fpath.read_text())
            for vdata in data.get("versions", []):
                vc.add_version(
                    prompt=vdata.get("prompt", ""),
                    score=vdata.get("score", 0.0),
                    iteration=vdata.get("iteration", 0),
                    metrics=vdata.get("metrics", {}),
                    notes=vdata.get("notes", ""),
                )
            vc._current_idx = data.get("current_index", vc._current_idx)

        return vc
