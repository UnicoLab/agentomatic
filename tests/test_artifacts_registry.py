"""Tests for :class:`~agentomatic.artifacts.ArtifactRegistry`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentomatic.artifacts import ArtifactRegistry


def test_promote_and_current_dir(tmp_path: Path) -> None:
    """Candidate → promote flips current and returns the bundle dir."""
    registry = ArtifactRegistry(tmp_path)
    version = registry.new_version_id()
    cand = registry.candidate_dir(version)
    (cand / "model.bin").write_bytes(b"weights-v1")
    registry.register_candidate(
        version,
        model_cards={"demo": {"status": "ok"}},
        eval_scores={"accuracy": 0.9},
    )
    assert registry.current_version() is None

    registry.promote(version)
    assert registry.current_version() == version
    assert registry.current_dir() == cand
    assert (registry.current_dir() / "model.bin").read_bytes() == b"weights-v1"

    manifest = registry.manifest()
    assert manifest is not None
    assert manifest["status"] == "active"
    assert manifest["eval_scores"]["accuracy"] == 0.9


def test_rollback_retires_previous_active(tmp_path: Path) -> None:
    """Rollback re-points current and retires the previously active version."""
    registry = ArtifactRegistry(tmp_path)
    v1 = "v1"
    v2 = "v2"
    registry.candidate_dir(v1)
    registry.candidate_dir(v2)
    registry.register_candidate(v1)
    registry.register_candidate(v2)
    registry.promote(v1)
    registry.promote(v2)
    assert registry.current_version() == v2
    assert registry.manifest(v1)["status"] == "retired"

    registry.rollback(v1)
    assert registry.current_version() == v1
    assert registry.manifest(v1)["status"] == "active"
    assert registry.manifest(v2)["status"] == "retired"


def test_promote_unknown_raises(tmp_path: Path) -> None:
    """Promote of an unregistered version raises KeyError."""
    registry = ArtifactRegistry(tmp_path)
    with pytest.raises(KeyError, match="unknown artifact version"):
        registry.promote("missing")


def test_corrupt_registry_recovers(tmp_path: Path) -> None:
    """A corrupt registry.json is treated as empty."""
    registry = ArtifactRegistry(tmp_path)
    registry.root.mkdir(parents=True, exist_ok=True)
    registry._registry_path.write_text("{not-json", encoding="utf-8")
    assert registry.current_version() is None
    assert registry.list_versions() == []


def test_atomic_save_leaves_valid_json(tmp_path: Path) -> None:
    """Registry writes are valid JSON after promote."""
    registry = ArtifactRegistry(tmp_path)
    version = "v-atomic"
    registry.candidate_dir(version)
    registry.register_candidate(version)
    registry.promote(version)
    data = json.loads(registry._registry_path.read_text(encoding="utf-8"))
    assert data["current"] == version
    assert version in data["versions"]


def test_list_versions_newest_first(tmp_path: Path) -> None:
    """list_versions sorts by created_at descending."""
    registry = ArtifactRegistry(tmp_path)
    for name in ("a", "b"):
        registry.candidate_dir(name)
        registry.register_candidate(name)
    versions = registry.list_versions()
    assert len(versions) == 2
    assert versions[0]["created_at"] >= versions[1]["created_at"]
