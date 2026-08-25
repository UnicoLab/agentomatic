# pyright: reportMissingParameterType=none
"""Static guards on the container build inputs shipped in this repo.

The Dockerfiles are not built in CI, so nothing caught that ``.dockerignore``
excluded a file ``pyproject.toml`` declares as required. These tests assert
the build context contract instead: whatever the project metadata says it
needs must survive ``.dockerignore`` and be copied by every Dockerfile that
installs the project.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
#: Dockerfiles that install *this* project (and so run the build backend).
PROJECT_DOCKERFILES = ("Dockerfile", "Dockerfile.distroless")


def _dockerignore_excludes(path: str) -> bool:
    """Return whether ``.dockerignore`` excludes *path*.

    Implements Docker's last-matching-pattern-wins rule for the simple
    (non-glob-directory) patterns this repo uses.

    Args:
        path: Repo-relative path to test, e.g. ``README.md``.

    Returns:
        ``True`` when the final matching pattern excludes the file.
    """
    import fnmatch

    excluded = False
    for raw in (REPO / ".dockerignore").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        pattern = line[1:] if negated else line
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.rstrip("/")):
            excluded = not negated
    return excluded


def _instructions(dockerfile: str) -> list[str]:
    """Return a Dockerfile's instruction lines, with comments stripped.

    Comments quote the very flags these tests look for, so matching raw text
    would pass (or fail) on prose rather than on the build itself.

    Args:
        dockerfile: Repo-relative Dockerfile name.

    Returns:
        Non-comment, non-blank lines.
    """
    return [
        line
        for line in (REPO / dockerfile).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _is_install_line(line: str) -> bool:
    """Return whether *line* installs dependencies into the image.

    Both install mechanisms in this repo count: ``uv sync`` for the venv-based
    image and ``uv pip install --target`` for the distroless one, which cannot
    use a venv built around a different interpreter.
    """
    stripped = line.strip()
    if stripped.startswith("#"):
        return False
    return "uv sync" in stripped or "uv pip install" in stripped


def _install_lines(dockerfile: str) -> list[str]:
    """Return the dependency-install instruction lines of *dockerfile*."""
    return [line for line in _instructions(dockerfile) if _is_install_line(line)]


def _declared_readme() -> str | None:
    """Return the readme filename declared in ``pyproject.toml``, if any."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    readme = data.get("project", {}).get("readme")
    if isinstance(readme, str):
        return readme
    if isinstance(readme, dict):
        return readme.get("file")
    return None


def test_declared_readme_exists() -> None:
    """The declared readme must actually be in the repo."""
    readme = _declared_readme()

    assert readme is not None
    assert (REPO / readme).is_file()


def test_declared_readme_survives_dockerignore() -> None:
    """Regression: ``README*`` was excluded, so ``uv sync`` failed in-image.

    The build backend raises ``OSError: Readme file does not exist`` when the
    file named by ``project.readme`` is missing from the build context.
    """
    readme = _declared_readme()
    assert readme is not None

    assert not _dockerignore_excludes(readme), (
        f".dockerignore excludes {readme}, which pyproject.toml declares as "
        "project.readme — the image build will fail installing the project"
    )


@pytest.mark.parametrize("dockerfile", PROJECT_DOCKERFILES)
def test_project_dockerfiles_copy_the_readme(dockerfile: str) -> None:
    """Each Dockerfile that installs the project must copy the readme in."""
    readme = _declared_readme()
    assert readme is not None
    content = (REPO / dockerfile).read_text()

    assert readme in content, f"{dockerfile} never COPYs {readme}"


@pytest.mark.parametrize("dockerfile", PROJECT_DOCKERFILES)
def test_project_dockerfiles_copy_metadata_before_installing(dockerfile: str) -> None:
    """The readme must be present *before* the project-install step runs."""
    content = (REPO / dockerfile).read_text()
    readme = _declared_readme()
    assert readme is not None
    lines = content.splitlines()

    copy_at = next(
        (i for i, line in enumerate(lines) if line.startswith("COPY") and readme in line),
        None,
    )
    # The step that installs the project itself (not just its dependencies).
    install_at = next(
        (
            i
            for i, line in enumerate(lines)
            if _is_install_line(line) and "--no-install-project" not in line
        ),
        None,
    )

    assert copy_at is not None, f"{dockerfile} never COPYs {readme}"
    assert install_at is not None, f"{dockerfile} has no project-install step"
    assert copy_at < install_at, f"{dockerfile} installs the project before copying {readme}"


def test_compose_bind_mounts_exist() -> None:
    """Every host path the root compose bind-mounts must be in the repo.

    A missing bind source makes ``docker compose up`` fail at the root of
    this repository — the first thing an evaluator tries.
    """
    import yaml

    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    missing: list[str] = []
    for name, service in (compose.get("services") or {}).items():
        for volume in service.get("volumes") or []:
            if not isinstance(volume, str) or not volume.startswith("."):
                continue
            source = volume.split(":")[0]
            if not (REPO / source).exists():
                missing.append(f"{name}: {source}")

    assert not missing, f"docker-compose.yml bind-mounts missing paths: {missing}"


#: Modules the platform needs for the features its own compose stack turns on.
RUNTIME_EXTRAS = {
    "sqlalchemy": "db",
    "prometheus-client": "metrics",
    "langgraph": "langgraph",
    "pyjwt": "security",
    "asyncpg": "db-postgres",
}


@pytest.mark.parametrize("dockerfile", PROJECT_DOCKERFILES)
def test_images_install_the_extras_their_features_need(dockerfile: str) -> None:
    """A bare install ships core deps only, silently disabling features.

    Regression: the image had no sqlalchemy, prometheus-client, langgraph or
    pyjwt, so ``/metrics`` served nothing, ``DATABASE_URL`` failed with
    "No module named 'sqlalchemy'", and JWT auth could not be turned on —
    all while the container reported healthy.
    """
    install_lines = _install_lines(dockerfile)

    assert install_lines, f"{dockerfile} has no dependency-install step"
    for line in install_lines:
        assert "all" in line, f"{dockerfile}: `{line.strip()}` omits the `all` extra"
        assert "db-postgres" in line, (
            f"{dockerfile}: `{line.strip()}` omits `db-postgres`, so the "
            "Postgres profile in docker-compose.yml cannot be reached"
        )


def test_all_extra_still_excludes_postgres_by_design() -> None:
    """The images name db-postgres explicitly because ``all`` omits it.

    If ``all`` ever absorbs the Postgres driver this test fails, prompting the
    Dockerfile comments (and the installation docs) to be revisited rather
    than left describing a split that no longer exists.
    """
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    all_extra = " ".join(data["project"]["optional-dependencies"]["all"])

    assert "db-postgres" not in all_extra


@pytest.mark.parametrize("dockerfile", PROJECT_DOCKERFILES)
def test_uv_is_pinned(dockerfile: str) -> None:
    """An unpinned build tool makes the image irreproducible."""
    content = "\n".join(_instructions(dockerfile))

    assert "astral-sh/uv:latest" not in content, (
        f"{dockerfile} pulls an unpinned uv tag — image contents change "
        "between builds with no diff to show for it"
    )
    assert "UV_VERSION=" in content, f"{dockerfile} does not pin a uv version"
