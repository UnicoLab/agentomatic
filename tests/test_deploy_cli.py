"""Tests for the ``agentomatic deploy`` CLI command and helpers.

Covers:
- ``agentomatic.cli.deploy.generate_deploy`` — file layout, contents,
  rootless USER declaration, distroless variant, and per-agent stubs.
- ``agentomatic.cli.deploy.render_env_example`` — stack-derived
  ``.env`` rendering with placeholder preservation.
- ``agentomatic stack export`` CLI subcommand end-to-end (via
  :class:`click.testing.CliRunner`).
- Wiring for ``require_auth_globally`` from :class:`AgentPlatform` to the
  underlying :class:`ZeroTrustEnforcer`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from agentomatic.cli import deploy as deploy_mod
from agentomatic.cli.commands import cli
from agentomatic.stacks.defaults import get_default_local_stack, get_default_remote_stack

# =========================================================================
# render_dockerfile / render_dockerfile_distroless — content assertions
# =========================================================================


class TestDockerfileRendering:
    def test_standard_dockerfile_is_rootless(self) -> None:
        content = deploy_mod.render_dockerfile()
        assert "FROM python:3.12-slim AS builder" in content
        assert "USER appuser" in content
        # Installs from PyPI pinned to the current version (project image, not
        # the framework repo image) and launches the user's main.py.
        assert 'pip install "agentomatic[all]==' in content
        assert 'CMD ["uvicorn", "main:app"' in content
        assert "COPY --chown=appuser:appuser main.py ./main.py" in content
        assert "HEALTHCHECK" in content
        assert "http://localhost:8000/health" in content
        assert "/api/v1/health" not in content

    def test_standard_dockerfile_pins_current_version(self) -> None:
        from agentomatic._version import __version__

        content = deploy_mod.render_dockerfile()
        assert f'pip install "agentomatic[all]=={__version__}"' in content

    def test_distroless_uses_nonroot_numeric_uid(self) -> None:
        content = deploy_mod.render_dockerfile_distroless()
        assert "gcr.io/distroless/python3-debian12:nonroot" in content
        assert "USER 65532:65532" in content
        assert '"/app/.venv/bin/python"' in content
        assert 'pip install "agentomatic[all]==' in content
        assert '"main:app"' in content

    def test_copy_lines_only_include_existing(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("app = None\n")
        (tmp_path / "agents").mkdir()
        content = deploy_mod.render_dockerfile(project_root=tmp_path)
        assert "COPY --chown=appuser:appuser main.py ./main.py" in content
        assert "COPY --chown=appuser:appuser agents/ ./agents/" in content
        # Non-existent dirs must not be emitted (would break docker build).
        assert "plugins/ ./plugins/" not in content


# =========================================================================
# render_docker_compose — service + agent stubs
# =========================================================================


class TestComposeRendering:
    def test_basic_compose_wires_stack(self) -> None:
        content = deploy_mod.render_docker_compose(stack_name="local")
        assert "services:" in content
        assert "platform:" in content
        assert "AGENTOMATIC_STACK=local" in content
        assert "8000:8000" in content
        assert "healthcheck:" in content
        assert "http://localhost:8000/health" in content
        assert "./ingestion:/app/ingestion:ro" in content
        assert "./pipelines:/app/pipelines:ro" in content
        assert "./plugins:/app/plugins:ro" in content
        assert "./endpoints:/app/endpoints:ro" in content

    def test_compose_agent_stubs(self) -> None:
        content = deploy_mod.render_docker_compose(
            stack_name="remote",
            agent_names=["weather_bot", "chat"],
        )
        assert "agent-weather-bot:" in content
        assert "agent-chat:" in content
        assert "AGENTOMATIC_AGENTS=weather_bot" in content

    def test_compose_references_distroless(self) -> None:
        content = deploy_mod.render_docker_compose(
            stack_name="local",
            dockerfile_name="Dockerfile.distroless",
        )
        assert "dockerfile: Dockerfile.distroless" in content


# =========================================================================
# render_env_example — StackConfig derivation
# =========================================================================


class TestEnvExampleRendering:
    def test_local_stack_env(self) -> None:
        stack = get_default_local_stack()
        env = deploy_mod.render_env_example(stack)
        assert "AGENTOMATIC_STACK=local" in env
        assert "LLM__PROVIDER=ollama" in env
        assert "LLM__MODEL=mistral:7b" in env
        assert "LLM__OLLAMA_BASE_URL=http://localhost:11434" in env
        assert "EMBEDDING__PROVIDER=ollama" in env
        assert "DB__URL=sqlite+aiosqlite:///data/platform.db" in env

    def test_remote_stack_preserves_env_placeholders(self) -> None:
        stack = get_default_remote_stack()
        env = deploy_mod.render_env_example(stack)
        assert "LLM__PROVIDER=openai" in env
        assert "${OPENAI_API_KEY}" in env
        assert "${DATABASE_URL}" in env
        assert "AUTH__JWKS_URL=${JWKS_URL}" in env

    def test_extra_llm_profiles_are_commented(self) -> None:
        stack = get_default_remote_stack()
        env = deploy_mod.render_env_example(stack)
        # 'fast' and 'judge' profiles must appear as commented lines so
        # they don't override the default LLM at container start.
        assert "# ~ profile: fast" in env
        assert "# ~ profile: judge" in env

    def test_default_env_when_no_stack(self) -> None:
        env = deploy_mod.render_env_example_default()
        assert "LLM__PROVIDER=ollama" in env
        assert "APP_NAME" in env


# =========================================================================
# generate_deploy — orchestrator writes all expected files
# =========================================================================


class TestGenerateDeploy:
    def test_generates_all_default_files(self, tmp_path: Path) -> None:
        out = tmp_path / "generated"
        plan = deploy_mod.generate_deploy(
            out_dir=out,
            stack_name="local",
            stacks_dir=tmp_path / "missing-stacks",  # forces built-in fallback
        )
        assert plan.out_dir == out
        assert "Dockerfile" in plan.files
        assert "docker-compose.yml" in plan.files
        assert ".env.example" in plan.files
        assert "nginx.conf" in plan.files
        assert "README.md" in plan.files
        for path in plan.files.values():
            assert path.exists()

    def test_dockerfile_content_matches_rootless_pattern(self, tmp_path: Path) -> None:
        plan = deploy_mod.generate_deploy(
            out_dir=tmp_path / "out",
            stack_name="local",
            stacks_dir=tmp_path / "no-stacks",
        )
        dockerfile = plan.files["Dockerfile"].read_text()
        assert "USER appuser" in dockerfile
        assert "agentomatic" in dockerfile
        assert "EXPOSE 8000" in dockerfile

    def test_distroless_variant(self, tmp_path: Path) -> None:
        plan = deploy_mod.generate_deploy(
            out_dir=tmp_path / "out",
            stack_name="local",
            stacks_dir=tmp_path / "no-stacks",
            distroless=True,
        )
        assert "Dockerfile.distroless" in plan.files
        assert "Dockerfile" not in plan.files  # only one dockerfile emitted
        dockerfile = plan.files["Dockerfile.distroless"].read_text()
        assert "USER 65532:65532" in dockerfile
        compose = plan.files["docker-compose.yml"].read_text()
        # Compose references the distroless Dockerfile (path is relative to the
        # build context, which points back at the project root).
        assert "Dockerfile.distroless" in compose
        assert "dockerfile:" in compose

    def test_agent_stubs_emitted_when_requested(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / "agents"
        (agents_dir / "weather_bot").mkdir(parents=True)
        (agents_dir / "weather_bot" / "agent.py").write_text("# stub\n")
        (agents_dir / "chatbot").mkdir(parents=True)
        (agents_dir / "chatbot" / "__init__.py").write_text("# stub\n")
        # A hidden directory that must be ignored by discovery.
        (agents_dir / "_private").mkdir()
        (agents_dir / "_private" / "__init__.py").write_text("# stub\n")

        plan = deploy_mod.generate_deploy(
            out_dir=tmp_path / "out",
            stack_name="local",
            stacks_dir=tmp_path / "no-stacks",
            agents_dir=agents_dir,
            include_agent_stubs=True,
        )
        compose = plan.files["docker-compose.yml"].read_text()
        assert "agent-weather-bot:" in compose
        assert "agent-chatbot:" in compose
        assert "agent-_private:" not in compose

    def test_compose_context_points_to_project_root(self, tmp_path: Path) -> None:
        """Artefacts under deploy/generated must build from the project root.

        Regression for P0-2: ``docker compose up --build`` from the generated
        directory has to find ``main.py`` / ``agents/`` at the project root.
        """
        project = tmp_path / "proj"
        (project / "agents").mkdir(parents=True)
        (project / "main.py").write_text("app = None\n")
        out = project / "deploy" / "generated"

        plan = deploy_mod.generate_deploy(
            out_dir=out,
            stack_name="local",
            stacks_dir=tmp_path / "no-stacks",
            agents_dir=project / "agents",
        )
        compose = plan.files["docker-compose.yml"].read_text()
        assert "context: ../.." in compose
        assert "dockerfile: deploy/generated/Dockerfile" in compose
        assert "../../agents:/app/agents:ro" in compose

        dockerfile = plan.files["Dockerfile"].read_text()
        assert "COPY --chown=appuser:appuser main.py ./main.py" in dockerfile
        assert "COPY --chown=appuser:appuser agents/ ./agents/" in dockerfile

    def test_nginx_can_be_disabled(self, tmp_path: Path) -> None:
        plan = deploy_mod.generate_deploy(
            out_dir=tmp_path / "out",
            stack_name="local",
            stacks_dir=tmp_path / "no-stacks",
            include_nginx=False,
        )
        assert "nginx.conf" not in plan.files


# =========================================================================
# CLI integration — `agentomatic deploy`
# =========================================================================


class TestDeployCliCommand:
    def test_deploy_command_writes_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "generated"
        result = runner.invoke(
            cli,
            [
                "deploy",
                "--stack",
                "local",
                "--out",
                str(out),
                "--stacks-dir",
                str(tmp_path / "no-stacks"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "Dockerfile").exists()
        assert (out / "docker-compose.yml").exists()
        assert (out / ".env.example").exists()

    def test_deploy_distroless_flag(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "generated"
        result = runner.invoke(
            cli,
            [
                "deploy",
                "--stack",
                "local",
                "--distroless",
                "--out",
                str(out),
                "--stacks-dir",
                str(tmp_path / "no-stacks"),
                "--no-nginx",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "Dockerfile.distroless").exists()
        content = (out / "Dockerfile.distroless").read_text()
        assert "USER 65532:65532" in content


# =========================================================================
# CLI integration — `agentomatic stack export`
# =========================================================================


class TestStackExportCli:
    def test_export_local_to_stdout(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "stack",
                "export",
                "--stack",
                "local",
                "--dir",
                str(tmp_path / "no-stacks"),  # forces built-in fallback
            ],
        )
        assert result.exit_code == 0, result.output
        assert "LLM__PROVIDER=ollama" in result.output
        assert "AGENTOMATIC_STACK=local" in result.output

    def test_export_to_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        env_file = tmp_path / ".env.exported"
        result = runner.invoke(
            cli,
            [
                "stack",
                "export",
                "--stack",
                "remote",
                "--env",
                str(env_file),
                "--dir",
                str(tmp_path / "no-stacks"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert env_file.exists()
        text = env_file.read_text()
        assert "LLM__PROVIDER=openai" in text
        assert "${OPENAI_API_KEY}" in text


# =========================================================================
# AgentPlatform wiring — require_auth_globally propagates to enforcer
# =========================================================================


class TestRequireAuthGloballyWiring:
    def test_flag_reaches_zero_trust_enforcer(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        from agentomatic.core.platform import AgentPlatform
        from agentomatic.security.jwt_auth import JWTConfig

        platform = AgentPlatform(
            agents_dir=agents_dir,
            plugins_dir=tmp_path / "plugins",
            endpoints_dir=tmp_path / "endpoints",
            ingestion_dir=tmp_path / "ingestion",
            enable_zero_trust=True,
            require_auth_globally=True,
            # Real verification must be configured under the global auth lock.
            enable_jwt_auth=True,
            jwt_config=JWTConfig(enabled=True, jwks_url="https://issuer.example/jwks"),
        )
        app = platform.build()
        enforcer = app.state.zero_trust_enforcer
        # Private attribute — reading it here is the most direct way to
        # prove the flag was forwarded through the platform to the
        # enforcer instance actually installed on the app.
        assert enforcer._require_auth_globally is True

    def test_require_auth_globally_without_verification_refuses_to_start(
        self, tmp_path: Path
    ) -> None:
        """P1-2: signature-disabled JWT under the auth lock must refuse to boot."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        from agentomatic.core.platform import AgentPlatform

        platform = AgentPlatform(
            agents_dir=agents_dir,
            plugins_dir=tmp_path / "plugins",
            endpoints_dir=tmp_path / "endpoints",
            ingestion_dir=tmp_path / "ingestion",
            enable_zero_trust=True,
            require_auth_globally=True,
            enable_jwt_auth=True,  # no jwks_url → unsafe under the auth lock
        )
        with pytest.raises(RuntimeError, match="forged/unsigned JWTs"):
            platform.build()

    def test_require_auth_globally_allows_api_key_auth(self, tmp_path: Path) -> None:
        """API-key auth is an acceptable guard instead of JWT verification."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        from agentomatic.core.platform import AgentPlatform

        platform = AgentPlatform(
            agents_dir=agents_dir,
            plugins_dir=tmp_path / "plugins",
            endpoints_dir=tmp_path / "endpoints",
            ingestion_dir=tmp_path / "ingestion",
            enable_zero_trust=True,
            require_auth_globally=True,
            enable_auth=True,
            auth_api_key="secret-key",
        )
        # Must not raise — API-key auth guards the platform.
        app = platform.build()
        assert app.state.zero_trust_enforcer._require_auth_globally is True

    def test_default_is_false(self, tmp_path: Path) -> None:
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        from agentomatic.core.platform import AgentPlatform

        platform = AgentPlatform(
            agents_dir=agents_dir,
            plugins_dir=tmp_path / "plugins",
            endpoints_dir=tmp_path / "endpoints",
            ingestion_dir=tmp_path / "ingestion",
            enable_zero_trust=True,
        )
        app = platform.build()
        enforcer = app.state.zero_trust_enforcer
        assert enforcer._require_auth_globally is False


# =========================================================================
# discover_agent_names helper
# =========================================================================


class TestDiscoverAgentNames:
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert deploy_mod.discover_agent_names(tmp_path / "missing") == []

    def test_returns_only_valid_packages(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "__init__.py").write_text("")
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "agent.py").write_text("")
        (tmp_path / "not_a_pkg").mkdir()  # no __init__.py / agent.py
        (tmp_path / "_hidden").mkdir()
        (tmp_path / "_hidden" / "__init__.py").write_text("")

        names = deploy_mod.discover_agent_names(tmp_path)
        assert names == ["a", "b"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
