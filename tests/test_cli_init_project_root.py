# pyright: reportMissingParameterType=none
"""Regression tests for ``agentomatic init`` target resolution.

``init`` used to resolve ``agents/`` against the *current* directory. Running
it from a subdirectory of a project — or from the parent it had just been
created in — silently built a second, parallel project and left the real
``agents/`` empty while reporting success.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from agentomatic.cli.commands import (
    _validate_agent_name,
    cli,
    find_child_project,
    find_child_projects,
    find_project_root,
)
from agentomatic.cli.project import scaffold_project


def _make_project(root: Path, name: str = "demo") -> Path:
    """Scaffold a project and return its path."""
    target = root / name
    scaffold_project(target, name)
    return target


class TestProjectRootDiscovery:
    def test_detects_scaffolded_project(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        assert find_project_root(project) == project.resolve()

    def test_walks_up_from_subdirectory(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        deep = project / "docs" / "nested"
        deep.mkdir(parents=True)
        assert find_project_root(deep) == project.resolve()

    def test_returns_none_outside_a_project(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare"
        bare.mkdir()
        assert find_project_root(bare) is None

    def test_finds_lone_child_project(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path)
        assert find_child_project(tmp_path) == project.resolve()

    def test_ignores_ambiguous_children(self, tmp_path: Path) -> None:
        _make_project(tmp_path, "one")
        _make_project(tmp_path, "two")
        assert find_child_project(tmp_path) is None


class TestInitTargetsTheRealProject:
    def test_init_from_subdirectory_writes_into_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _make_project(tmp_path)
        deep = project / "docs" / "nested"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)

        result = CliRunner().invoke(cli, ["init", "hello", "--template", "basic"])

        assert result.exit_code == 0, result.output
        assert (project / "agents" / "hello" / "agent.py").is_file()
        assert not (deep / "agents").exists(), "scaffolded a parallel project"

    def test_init_from_parent_writes_into_child_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``agentomatic new x`` then ``init`` without the ``cd`` in between."""
        project = _make_project(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["init", "hello", "--template", "basic"])

        assert result.exit_code == 0, result.output
        assert (project / "agents" / "hello" / "agent.py").is_file()
        assert not (tmp_path / "agents").exists(), "scaffolded a parallel project"

    def test_ambiguous_parent_does_not_redirect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two candidate projects must not silently pick one — and must say so."""
        one = _make_project(tmp_path, "one")
        two = _make_project(tmp_path, "two")
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["init", "hello", "--template", "basic"])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "agents" / "hello" / "agent.py").is_file()
        assert not (one / "agents" / "hello").exists()
        assert not (two / "agents" / "hello").exists()
        assert "Several projects here" in result.output

    def test_here_flag_overrides_detection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _make_project(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["init", "hello", "--template", "basic", "--here"])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "agents" / "hello" / "agent.py").is_file()
        assert not (project / "agents" / "hello").exists()

    def test_dir_flag_still_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project = _make_project(tmp_path)
        monkeypatch.chdir(project)
        explicit = tmp_path / "elsewhere"

        result = CliRunner().invoke(
            cli, ["init", "hello", "--template", "basic", "--dir", str(explicit)]
        )

        assert result.exit_code == 0, result.output
        assert (explicit / "hello" / "agent.py").is_file()

    def test_reports_an_absolute_location(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The path printed must be unambiguous about where code landed."""
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["init", "hello", "--template", "basic"])

        assert result.exit_code == 0, result.output
        assert str((tmp_path / "agents" / "hello").resolve()) in result.output


class TestInitBootstrapsRunnableProject:
    def test_bare_directory_gets_main_py_and_stacks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["init", "solo", "--template", "basic"])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "agents" / "solo" / "agent.py").is_file()
        # ``deploy`` renders ``uvicorn main:app``; without main.py the image
        # cannot boot.
        assert (tmp_path / "main.py").is_file()
        assert (tmp_path / "stacks" / "local.yaml").is_file()
        assert (tmp_path / ".agentomatic-stack").is_file()

    def test_existing_main_py_is_never_overwritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``uv init`` leaves its own main.py; the scaffold must not clobber it."""
        sentinel = "# hand-written, do not touch\n"
        (tmp_path / "main.py").write_text(sentinel)
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, ["init", "solo", "--template", "basic"])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "main.py").read_text() == sentinel

    def test_no_stray_component_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bootstrap must not restructure a directory it was not given."""
        monkeypatch.chdir(tmp_path)

        CliRunner().invoke(cli, ["init", "solo", "--template", "basic"])

        for junk in ("plugins", "endpoints", "ingestion", "pipelines"):
            assert not (tmp_path / junk).exists(), f"bootstrap created {junk}/"


class TestNameValidation:
    def test_accepts_ordinary_names(self) -> None:
        for name in ("hello", "my_agent", "my-agent", "_x9"):
            assert _validate_agent_name(name) is None

    def test_rejects_path_separators(self) -> None:
        assert _validate_agent_name("a/b") is not None

    def test_rejects_empty_and_dots(self) -> None:
        assert _validate_agent_name("") is not None
        assert _validate_agent_name(".") is not None

    def test_rejects_leading_digit(self) -> None:
        assert _validate_agent_name("9lives") is not None

    def test_cli_rejects_bad_name(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli, ["init", "bad/name", "--template", "basic", "--dir", str(tmp_path)]
        )
        assert result.exit_code == 1


class TestExplicitDirStaysContained:
    def test_dir_does_not_scatter_project_files_into_its_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dir`` places the package; it must not relocate the project.

        The bootstrap used to write ``stacks/``, ``main.py`` and
        ``.agentomatic-stack`` into ``--dir``'s parent — a home directory when
        the flag pointed at one. With a custom layout and no project in sight
        there is nothing to bootstrap, so nothing is written.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)

        result = CliRunner().invoke(
            cli,
            ["init", "hello", "--template", "basic", "--dir", str(workspace / "custom")],
        )

        assert result.exit_code == 0, result.output
        assert (workspace / "custom" / "hello" / "agent.py").is_file()
        for stray in (".agentomatic-stack", "stacks", "main.py"):
            assert not (tmp_path / stray).exists(), f"leaked {stray} above --dir"
            assert not (workspace / stray).exists(), f"scattered {stray} into cwd"

    def test_dir_inside_a_project_bootstraps_that_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom directory *within* a project still gets its stacks."""
        project = _make_project(tmp_path)
        monkeypatch.chdir(project)

        result = CliRunner().invoke(
            cli,
            ["init", "hello", "--template", "basic", "--dir", str(project / "custom")],
        )

        assert result.exit_code == 0, result.output
        assert (project / "custom" / "hello" / "agent.py").is_file()
        assert (project / "stacks" / "local.yaml").is_file()
        assert not (tmp_path / "stacks").exists()


class TestRootDetectionStaysBounded:
    def test_lone_stacks_dir_is_not_a_project(self, tmp_path: Path) -> None:
        """A ``stacks/`` directory alone is far too weak a signal."""
        (tmp_path / "stacks").mkdir()
        (tmp_path / "stacks" / "local.yaml").write_text("name: local\n")
        child = tmp_path / "child"
        child.mkdir()
        assert find_project_root(child) is None

    def test_lone_main_py_is_not_a_project(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("print('hi')\n")
        child = tmp_path / "child"
        child.mkdir()
        assert find_project_root(child) is None

    def test_walk_stops_at_a_git_boundary(self, tmp_path: Path) -> None:
        """A scaffold must not reach out of the repository it sits in."""
        outer = _make_project(tmp_path, "outer")
        repo = outer / "vendored"
        (repo / ".git").mkdir(parents=True)
        inner = repo / "src"
        inner.mkdir()
        assert find_project_root(inner) is None

    def test_walk_stops_at_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        (home / "projects").mkdir(parents=True)
        _make_project(tmp_path, "above_home")
        monkeypatch.setenv("HOME", str(home))
        assert find_project_root(home / "projects") is None


class TestAddResolvesTheProject:
    def test_add_connection_works_from_a_subdirectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _make_project(tmp_path)
        monkeypatch.chdir(project)
        assert CliRunner().invoke(cli, ["init", "hello", "--template", "basic"]).exit_code == 0

        sub = project / "notebooks"
        sub.mkdir()
        monkeypatch.chdir(sub)

        result = CliRunner().invoke(cli, ["add", "connection", "hello"])

        assert result.exit_code == 0, result.output
        assert (project / "agents" / "hello" / "connections.py").is_file()

    def test_add_ingestion_lands_in_the_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _make_project(tmp_path)
        sub = project / "notebooks"
        sub.mkdir()
        monkeypatch.chdir(sub)

        result = CliRunner().invoke(cli, ["add", "ingestion", "docs"])

        assert result.exit_code == 0, result.output
        assert (project / "ingestion" / "docs").is_dir()
        assert not (sub / "ingestion").exists()

    def test_add_connection_missing_agent_is_actionable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _make_project(tmp_path)
        monkeypatch.chdir(project)

        result = CliRunner().invoke(cli, ["add", "connection", "ghost"])

        assert result.exit_code == 1
        assert "agentomatic init ghost" in result.output


class TestChildProjectListing:
    def test_lists_every_child_project(self, tmp_path: Path) -> None:
        _make_project(tmp_path, "alpha")
        _make_project(tmp_path, "beta")
        (tmp_path / "unrelated").mkdir()
        found = {p.name for p in find_child_projects(tmp_path)}
        assert found == {"alpha", "beta"}

    def test_empty_without_projects(self, tmp_path: Path) -> None:
        (tmp_path / "unrelated").mkdir()
        assert find_child_projects(tmp_path) == []
