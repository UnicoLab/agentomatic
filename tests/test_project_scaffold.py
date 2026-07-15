"""Tests for project scaffold, init routing, and safe add."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from agentomatic.cli.commands import cli
from agentomatic.cli.project import get_project_files, scaffold_project
from agentomatic.cli.templates import get_template_files


class TestProjectScaffold:
    def test_project_files_contain_platform_entrypoint(self) -> None:
        files = get_project_files("demo_app")
        assert "main.py" in files
        assert "AgentPlatform.from_folder" in files["main.py"]
        assert "pipelines_dir=" not in files["main.py"]
        assert "_platform = create_platform()" in files["main.py"]
        assert "stacks/local.yaml" in files
        assert "stacks/remote.yaml" in files
        assert ".env.example" in files
        assert "agents/.gitkeep" in files
        assert "ingestion/.gitkeep" in files

    def test_project_emits_pinned_requirements(self) -> None:
        from agentomatic._version import __version__

        files = get_project_files("demo_app")
        assert "requirements.txt" in files
        assert f"agentomatic[all]=={__version__}" in files["requirements.txt"]

    def test_scaffolded_main_accepts_platform_kwargs(self) -> None:
        """Scaffolded main.py must construct AgentPlatform without TypeError."""
        import ast
        import inspect

        from agentomatic import AgentPlatform

        files = get_project_files("demo_app")
        tree = ast.parse(files["main.py"])
        kwargs: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_folder"
            ):
                kwargs = [k.arg for k in node.keywords if k.arg]
        sig = inspect.signature(AgentPlatform.__init__)
        for key in kwargs:
            assert key in sig.parameters, f"unknown Platform kwarg: {key}"

    def test_project_includes_package_inits(self) -> None:
        files = get_project_files("demo_app")
        assert "agents/__init__.py" in files
        assert "plugins/__init__.py" in files


class TestDeployRunParity:
    """The deployed ``uvicorn main:app`` must expose the same surface as run."""

    @staticmethod
    def _build_scaffold_app(
        tmp_path: Path,
        monkeypatch,
        env: dict[str, str] | None = None,
    ):
        """Scaffold a project, exec its ``main.py``, and return the built app."""
        scaffold_project(tmp_path / "proj", "demo_app", force=True)
        monkeypatch.chdir(tmp_path / "proj")
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        namespace: dict[str, object] = {}
        source = (tmp_path / "proj" / "main.py").read_text()
        exec(compile(source, "main.py", "exec"), namespace)  # noqa: S102
        return namespace["app"]

    def test_app_exposes_run_parity_routes(self, tmp_path: Path, monkeypatch) -> None:
        app = self._build_scaffold_app(tmp_path, monkeypatch)
        paths = {getattr(route, "path", None) for route in app.routes}
        # Platform surface `agentomatic run` always mounts.
        for expected in ("/health", "/readiness", "/docs", "/openapi.json", "/", "/studio"):
            assert expected in paths, f"missing route: {expected}"

    def test_env_flags_drive_features(self, tmp_path: Path, monkeypatch) -> None:
        app = self._build_scaffold_app(
            tmp_path,
            monkeypatch,
            env={"AGENTOMATIC_ENABLE_STUDIO": "0", "AGENTOMATIC_TITLE": "Env Title"},
        )
        paths = {getattr(route, "path", None) for route in app.routes}
        assert "/studio" not in paths  # studio redirect only added when enabled
        assert app.title == "Env Title"

    def test_minimal_profile_env_keeps_swagger_disables_studio(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Minimal deploy profile env: Studio off, but Swagger MUST stay on."""
        app = self._build_scaffold_app(
            tmp_path,
            monkeypatch,
            # Mirrors the env baked into a `deploy --profile minimal` image.
            env={"AGENTOMATIC_ENABLE_STUDIO": "0", "AGENTOMATIC_LOG_LEVEL": "WARNING"},
        )
        paths = {getattr(route, "path", None) for route in app.routes}
        # Studio debug UI is gone in minimal…
        assert "/studio" not in paths
        # …but Swagger/OpenAPI and health MUST remain (explicit requirement).
        for kept in ("/docs", "/redoc", "/openapi.json", "/health", "/api/v1/agents"):
            assert kept in paths, f"minimal profile dropped required route: {kept}"


class TestClassTemplateAlias:
    def test_class_alias_matches_basic(self) -> None:
        basic = get_template_files("basic", "hello")
        alias = get_template_files("class", "hello")
        assert "agent.py" in alias
        assert alias["agent.py"] == basic["agent.py"]

    def test_scaffold_project_writes_and_skips(self, tmp_path: Path) -> None:
        result = scaffold_project(tmp_path / "app", "app", force=False)
        assert (tmp_path / "app" / "main.py").exists()
        assert len(result["written"]) > 5

        # Second run without force should skip existing files
        result2 = scaffold_project(tmp_path / "app", "app", force=False)
        assert result2["skipped"]
        assert not result2["written"] or len(result2["written"]) == 0


class TestInitRouting:
    def test_ingestion_defaults_to_ingestion_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["init", "docs", "--template", "ingestion", "--dir", str(tmp_path / "ingestion")],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "ingestion" / "docs" / "ingestor.py").exists()

    def test_init_no_overwrite_without_force(self, tmp_path: Path) -> None:
        runner = CliRunner()
        agents = tmp_path / "agents"
        first = runner.invoke(
            cli,
            ["init", "bot", "--template", "basic", "--dir", str(agents)],
        )
        assert first.exit_code == 0, first.output
        init_py = agents / "bot" / "__init__.py"
        init_py.write_text("# CUSTOM MARKER\n")
        second = runner.invoke(
            cli,
            ["init", "bot", "--template", "basic", "--dir", str(agents)],
        )
        assert second.exit_code == 0, second.output
        assert "# CUSTOM MARKER" in init_py.read_text()


class TestAddConnection:
    def test_add_connection_to_existing_agent(self, tmp_path: Path) -> None:
        runner = CliRunner()
        agents = tmp_path / "agents"
        runner.invoke(cli, ["init", "bot", "--template", "basic", "--dir", str(agents)])
        result = runner.invoke(
            cli,
            ["add", "connection", "bot", "--dir", str(agents)],
        )
        assert result.exit_code == 0, result.output
        assert (agents / "bot" / "connections.py").exists()
        # Manifest preserved
        assert "AgentManifest" in (agents / "bot" / "__init__.py").read_text()


class TestTemplateManifests:
    def test_agent_templates_emit_manifest_and_llm(self) -> None:
        for tmpl in ("basic", "full", "rag", "chatbot", "coordinator", "extraction"):
            files = get_template_files(tmpl, "sample")
            assert "AgentManifest" in files["__init__.py"], tmpl
            assert "llm.py" in files, tmpl
