"""Contracts for the repository's release-only tooling."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_semantic_release import CHANGELOG_MARKER, _validate_changelog_contract

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_semantic_release_wrapper_accepts_changelog_marker(tmp_path, monkeypatch) -> None:
    """A correctly prepared checkout passes the wrapper preflight."""
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n{CHANGELOG_MARKER}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    _validate_changelog_contract(["version", "--print"])


def test_semantic_release_wrapper_rejects_missing_changelog_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A release cannot silently bump/tag without updating release notes."""
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="Release aborted"):
        _validate_changelog_contract(["version", "--print"])


def test_semantic_release_wrapper_allows_explicit_no_changelog_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local diagnostic commands can deliberately opt out of changelog handling."""
    monkeypatch.chdir(tmp_path)

    _validate_changelog_contract(["version", "--no-changelog", "--print"])


def test_workflows_use_native_node24_action_releases() -> None:
    """Do not hide deprecated action runtimes behind GitHub's compatibility switch."""
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPOSITORY_ROOT / ".github/workflows").glob("*.yml"))
    )
    for retired in (
        "actions/checkout@v4",
        "actions/setup-python@v5",
        "actions/upload-artifact@v4",
        "actions/download-artifact@v4",
        "actions/create-github-app-token@v1",
        "astral-sh/setup-uv@v5",
        "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24",
    ):
        assert retired not in workflows

    for current in (
        "actions/checkout@v7",
        "actions/setup-python@v7",
        "actions/upload-artifact@v7",
        "actions/download-artifact@v8",
        "actions/create-github-app-token@v3",
        "astral-sh/setup-uv@v10.0.1",
    ):
        assert current in workflows
