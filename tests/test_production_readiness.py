# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Regression tests for release-blocking defects found by end-to-end testing.

Each test here corresponds to something that was observed failing (or being
unusably noisy) when the platform was actually booted and driven over HTTP,
rather than to a hypothetical from reading code.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from conftest import install_plugin_package
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform


async def _echo(state: dict[str, Any]) -> dict[str, Any]:
    return {"response": "ok", "agent_type": "echo"}


@pytest.fixture
def dual_mounted_platform(tmp_path):
    """A platform whose agent's folder name and manifest slug differ.

    Such an agent is mounted under BOTH names so Studio (which addresses
    agents by slug) does not 404.
    """
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        title="Prod Readiness",
    )
    platform.register_agent(
        manifest=AgentManifest(name="hello", slug="agent-hello", description="Hello"),
        node_fn=_echo,
    )
    return platform


# =====================================================================
# OpenAPI: no duplicate operationIds from the name/slug dual mount
# =====================================================================


def test_openapi_has_no_duplicate_operation_ids(dual_mounted_platform) -> None:
    """Duplicate operationIds break OpenAPI client codegen.

    Mounting each agent under both its folder name and its slug previously
    emitted one ``UserWarning: Duplicate Operation ID`` per route (~205 on a
    small project) and produced a spec generators reject.
    """
    app = dual_mounted_platform.build()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = app.openapi()

    duplicate_warnings = [w for w in caught if "Duplicate Operation ID" in str(w.message)]
    assert not duplicate_warnings, (
        f"{len(duplicate_warnings)} duplicate operationId warning(s): "
        f"{[str(w.message) for w in duplicate_warnings[:3]]}"
    )

    operation_ids = [
        operation["operationId"]
        for path_item in spec["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids)), "operationIds are not unique"


def test_slug_alias_routes_work_but_are_not_documented_twice(dual_mounted_platform) -> None:
    """The slug mount is a compatibility alias: live, but not in the schema."""
    app = dual_mounted_platform.build()
    spec = app.openapi()

    assert "/api/v1/hello/invoke" in spec["paths"], "canonical route must be documented"
    assert "/api/v1/agent-hello/invoke" not in spec["paths"], (
        "the slug alias must not be documented — it doubles the advertised surface"
    )

    with TestClient(app) as client:
        # ...but it must still route, so Studio's slug-based calls keep working.
        assert client.post("/api/v1/agent-hello/invoke", json={"query": "x"}).status_code == 200
        assert client.post("/api/v1/hello/invoke", json={"query": "x"}).status_code == 200


# =====================================================================
# OpenTelemetry console export is opt-in
# =====================================================================


def test_otel_console_export_is_opt_in_by_default(monkeypatch) -> None:
    """Console span export must not default on.

    It previously attached whenever no OTLP endpoint was configured — i.e. for
    most deployments — dumping a full JSON span document to stdout for every
    single HTTP request.
    """
    from agentomatic.observability import telemetry

    monkeypatch.delenv("AGENTOMATIC_OTEL_CONSOLE", raising=False)
    assert telemetry._env_flag("AGENTOMATIC_OTEL_CONSOLE") is False

    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("AGENTOMATIC_OTEL_CONSOLE", truthy)
        assert telemetry._env_flag("AGENTOMATIC_OTEL_CONSOLE") is True, truthy

    for falsy in ("0", "false", "no", "", "off"):
        monkeypatch.setenv("AGENTOMATIC_OTEL_CONSOLE", falsy)
        assert telemetry._env_flag("AGENTOMATIC_OTEL_CONSOLE") is False, falsy


def test_otel_setup_does_not_attach_console_exporter_by_default(monkeypatch) -> None:
    """Without the opt-in env var, no ConsoleSpanExporter is registered."""
    pytest.importorskip("opentelemetry.sdk")
    from agentomatic.observability import telemetry

    if not telemetry.HAS_OTEL:  # pragma: no cover - depends on extras
        pytest.skip("OpenTelemetry not installed")

    monkeypatch.delenv("AGENTOMATIC_OTEL_CONSOLE", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    attached: list[Any] = []
    real_provider_cls = telemetry.TracerProvider

    class _RecordingProvider(real_provider_cls):  # type: ignore[misc, valid-type]
        def add_span_processor(self, processor: Any) -> None:
            attached.append(processor)
            super().add_span_processor(processor)

    monkeypatch.setattr(telemetry, "TracerProvider", _RecordingProvider)
    telemetry.setup_telemetry(app=None, service_name="test-svc")

    assert not attached, (
        "a span processor was attached with no OTLP endpoint and console export "
        "off — this is the per-request stdout span dump regression"
    )


# =====================================================================
# `agentomatic run` can import the project's main.py
# =====================================================================


def test_run_puts_project_dir_on_sys_path_before_uvicorn(monkeypatch, tmp_path) -> None:
    """``uvicorn.run("main:app")`` resolves the import string against sys.path.

    Launched as a console script (``uv run agentomatic run``), the project
    directory is not on sys.path, so importing ``main`` failed outright. The
    run command must add it (and export PYTHONPATH for the --reload child).
    """
    import os
    import sys

    from agentomatic.cli import commands

    (tmp_path / "main.py").write_text("app = object()\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(commands, "_has_project_main_app", lambda *a, **k: True)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    captured: dict[str, Any] = {}

    class _FakeUvicorn:
        @staticmethod
        def run(target: str, **kwargs: Any) -> None:
            captured["target"] = target
            captured["app_dir"] = kwargs.get("app_dir")
            captured["sys_path_0"] = sys.path[0]
            captured["pythonpath"] = os.environ.get("PYTHONPATH", "")

    monkeypatch.setitem(sys.modules, "uvicorn", _FakeUvicorn)

    original_sys_path = list(sys.path)
    try:
        # Invoke the click command's underlying callback directly.
        commands.run.callback(
            agents_dir="agents",
            plugins_dir="plugins",
            endpoints_dir="endpoints",
            ingestion_dir="ingestion",
            stacks_dir="stacks",
            host="127.0.0.1",
            port=8000,
            reload=False,
            title=None,
            log_level="INFO",
            with_ui=False,
            studio=True,
            ssl_certfile=None,
            ssl_keyfile=None,
            require_auth_globally=False,
        )
    finally:
        sys.path[:] = original_sys_path

    project_dir = str(tmp_path)
    assert captured["target"] == "main:app"
    assert captured["app_dir"] == project_dir
    assert captured["sys_path_0"] == project_dir
    assert project_dir in captured["pythonpath"].split(os.pathsep)


# =====================================================================
# Scaffold hygiene: no filesystem paths or dead collectors leaked
# =====================================================================


def test_project_title_uses_only_the_directory_name() -> None:
    """Regression: ``agentomatic new /srv/apps/my_proj`` baked the whole
    filesystem path into the platform title, which is published via
    ``/openapi.json``, ``/.well-known/agent.json`` and ``/studio/info`` —
    leaking the server's directory layout to any caller.
    """
    from agentomatic.cli.project import get_project_files

    main_py = get_project_files("/srv/apps/my_proj")["main.py"]

    assert '"My Proj Platform"' in main_py
    assert "/srv/apps" not in main_py.split("description=")[0]


def test_env_example_does_not_enable_a_collector_that_is_not_there() -> None:
    """An OTLP endpoint set with nothing listening makes the exporter retry
    every span and flood the log with transient-failure warnings.
    """
    from agentomatic.cli.project import get_project_files

    env_example = get_project_files("demo")[".env.example"]

    for line in env_example.splitlines():
        stripped = line.strip()
        if stripped.startswith("OTEL_EXPORTER_OTLP_ENDPOINT="):
            pytest.fail(f"OTLP endpoint enabled by default in .env.example: {stripped!r}")
    # It should still be documented, just commented out.
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in env_example


def test_full_template_agent_keeps_its_generated_endpoints() -> None:
    """A module-level ``router`` in an agent's ``api.py`` REPLACES every
    auto-generated endpoint (/invoke, /chat, /stream, /card, /health). The
    ``full`` template shipped one, so the flagship scaffold produced an agent
    with no way to invoke it.
    """
    from agentomatic.cli.templates import get_template_files

    api_py = get_template_files("full", "sample_agent")["api.py"]

    # The registry keys on the exact name ``router``; anything else is inert.
    assert "\nrouter = APIRouter()" not in api_py
    assert "custom_router = APIRouter()" in api_py
    # The pattern is still demonstrated and explained.
    assert "REPLACES" in api_py


def test_all_extra_contents_match_what_the_docs_claim() -> None:
    """``[all]`` is the advertised "recommended" install, so what it contains
    must stay in sync with the documented list — a user following the docs and
    then hitting a missing dependency is a release defect.
    """
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    extras = pyproject["project"]["optional-dependencies"]
    all_spec = " ".join(extras["all"])
    included = {e.strip() for e in all_spec.split("[", 1)[1].rstrip("]\"' ").split(",")}

    documented = {
        "langgraph",
        "ollama",
        "metrics",
        "db",
        "cli",
        "studio",
        "optimize",
        "telemetry",
        "dotenv",
        "security",
        "swarm",
        "vector",
    }
    assert included == documented, (
        "The `all` extra changed; update docs/getting-started/installation.md "
        f"(added={included - documented}, removed={documented - included})"
    )

    # These are deliberately excluded — vendor SDKs (provider-agnostic
    # principle), an alternative DB driver, and the heavy Chainlit UI.
    for deliberately_excluded in ("openai", "azure", "vertex", "db-postgres", "ui"):
        assert deliberately_excluded in extras, f"{deliberately_excluded} extra vanished"
        assert deliberately_excluded not in included


def test_platform_marks_plugin_loaded_even_if_subclass_forgets_super(tmp_path) -> None:
    """A plugin overriding ``load_model`` without calling ``super()`` used to
    stay ``_is_loaded=False`` forever: /predict answered 503 and /health went
    "degraded", while startup logged "loaded successfully". The platform now
    stamps the flag itself so the footgun cannot produce a dead plugin.
    """
    from fastapi.testclient import TestClient

    from agentomatic import AgentPlatform

    plugins_dir = tmp_path / "plugins"
    source = '''"""Plugin that overrides load_model without calling super()."""
from __future__ import annotations

from pydantic import BaseModel

from agentomatic.plugins import BaseMLPlugin


class Inp(BaseModel):
    text: str


class Out(BaseModel):
    result: str


class ForgetfulPlugin(BaseMLPlugin[Inp, Out]):
    plugin_name = "forgetful"

    async def load_model(self) -> None:
        self.model = object()  # deliberately no: await super().load_model()

    async def predict(self, inputs: Inp) -> Out:
        return Out(result=inputs.text.upper())
'''
    importable = install_plugin_package(plugins_dir, "forgetful", source)

    with importable:
        platform = AgentPlatform(
            agents_dir=tmp_path / "agents",
            plugins_dir=plugins_dir,
            endpoints_dir=tmp_path / "endpoints",
        )
        with TestClient(platform.build()) as client:
            assert client.get("/health").json()["status"] == "healthy"
            response = client.post("/api/v1/plugins/forgetful/predict", json={"text": "hi"})
            assert response.status_code == 200, response.text
            assert response.json()["result"] == "HI"


# =====================================================================
# Log hygiene: one line per request, one separator, one DDL pass
# =====================================================================


def test_log_format_separator_matches_loguru_default() -> None:
    """Lines emitted before ``configure_logging`` installs our sink use loguru's
    built-in format. Using a different separator afterwards produced two
    formats in one log, which breaks log-shipping regexes.
    """
    import inspect

    from agentomatic.core.lifespan import configure_logging

    source = inspect.getsource(configure_logging)
    assert "{line}</cyan> - " in source
    assert "—" not in source, "em dash in the log format: non-ASCII and inconsistent"


def test_platform_run_disables_uvicorn_access_log_when_middleware_logs(tmp_path) -> None:
    """The platform's LoggingMiddleware already logs every request, so leaving
    uvicorn's access log on doubles the volume for the same information.
    """
    from unittest.mock import patch

    from agentomatic import AgentPlatform

    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        enable_logging=True,
    )
    with patch("uvicorn.run") as mock_run:
        platform.run(host="127.0.0.1", port=9999)
    assert mock_run.call_args.kwargs["access_log"] is False

    # An explicit choice from the caller still wins.
    with patch("uvicorn.run") as mock_run:
        platform.run(host="127.0.0.1", port=9999, access_log=True)
    assert mock_run.call_args.kwargs["access_log"] is True


def test_platform_run_keeps_access_log_when_middleware_is_off(tmp_path) -> None:
    """With the middleware disabled, uvicorn's access log is the only record
    of requests — it must not be suppressed.
    """
    from unittest.mock import patch

    from agentomatic import AgentPlatform

    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        enable_logging=False,
    )
    with patch("uvicorn.run") as mock_run:
        platform.run(host="127.0.0.1", port=9999)
    assert "access_log" not in mock_run.call_args.kwargs


@pytest.mark.asyncio
async def test_sqlalchemy_store_initialize_is_idempotent(tmp_path) -> None:
    """Startup can reach ``initialize()`` from several paths (configured store,
    one derived from DATABASE_URL, and a post-connection pass). Re-running the
    DDL each time is wasted round trips and duplicate log lines.
    """
    from loguru import logger

    from agentomatic.storage.sqlalchemy import SQLAlchemyStore

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="INFO")

    store = SQLAlchemyStore(url=f"sqlite+aiosqlite:///{tmp_path / 'x.db'}")
    try:
        await store.initialize()
        assert store._initialized is True
        await store.initialize()
        await store.initialize()
    finally:
        logger.remove(sink_id)
        await store.close()

    created = [m for m in messages if "Database tables created/verified" in m]
    assert len(created) == 1, f"DDL ran {len(created)} times, expected once"
