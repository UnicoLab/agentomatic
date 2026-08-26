"""Guard the runnable production Docker fixture against configuration drift."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compose_mounts_every_discoverable_production_resource() -> None:
    """The default local stack must exercise every resource discovery path."""
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for resource in ("agents", "plugins", "endpoints", "ingestion", "pipelines"):
        assert f"e2e_demo/{resource}" in compose
        assert (ROOT / "e2e_demo" / resource).is_dir()


def test_production_fixture_includes_delegation_connection_and_live_model_agents() -> None:
    """Keep the Docker matrix meaningful beyond a single deterministic route."""
    agents = ROOT / "e2e_demo" / "agents"
    for path in (
        agents / "coordinator" / "__init__.py",
        agents / "researcher" / "__init__.py",
        agents / "writer" / "connections.py",
        agents / "omlx_echo" / "__init__.py",
    ):
        assert path.is_file(), f"Missing production fixture component: {path.relative_to(ROOT)}"


def test_compose_exposes_opt_in_audit_and_otlp_configuration() -> None:
    """The packaged service must be able to export its production evidence."""
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "AGENTOMATIC_AUDIT_LOG" in compose
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in compose
