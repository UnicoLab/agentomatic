"""Multi-resource logs_history coverage (plugin/pipeline/ingestion/endpoint)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from agentomatic import AgentPlatform
from agentomatic.core.manifest import AgentManifest
from agentomatic.endpoints.base import BaseEndpoint
from agentomatic.ingestion import BaseIngestor, IngestionResult
from agentomatic.logs.recorder import InvocationLogRecorder
from agentomatic.logs.runtime import set_invocation_log_recorder
from agentomatic.pipelines.engine import PipelineEngine
from agentomatic.pipelines.models import AgentStepConfig, PipelineConfig, PipelineStatus
from agentomatic.plugins import BaseMLPlugin
from agentomatic.storage.memory import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures / dummies
# ---------------------------------------------------------------------------


class _PluginIn(BaseModel):
    text: str


class _PluginOut(BaseModel):
    label: str


class _DummyPlugin(BaseMLPlugin[_PluginIn, _PluginOut]):
    plugin_name = "sentiment"
    plugin_description = "dummy"
    plugin_version = "0.1.0"

    async def load_model(self) -> None:
        await super().load_model()

    async def predict(self, inputs: _PluginIn) -> _PluginOut:
        return _PluginOut(label="pos" if "good" in inputs.text.lower() else "neg")


class _EpIn(BaseModel):
    text: str


class _EpOut(BaseModel):
    echoed: str


class _EchoEndpoint(BaseEndpoint[_EpIn, _EpOut]):
    endpoint_name = "echo_ep"
    endpoint_description = "echo"
    path = "/echo"

    async def handle(self, request):  # type: ignore[override]
        return _EpOut(echoed=request.text)


class _DocsIngestor(BaseIngestor):
    ingestor_name = "docs"
    ingestor_description = "docs"

    async def ingest(self, request, ctx) -> IngestionResult:
        return IngestionResult(documents=1, chunks=2, upserted=2, collection="kb")


async def _echo_node(state: dict[str, Any]) -> dict[str, Any]:
    query = state.get("current_query") or state.get("query") or ""
    return {"response": f"echo:{query}", "output": {"response": f"echo:{query}"}}


def _platform(tmp_path: Path, *, analysis: bool = True) -> AgentPlatform:
    for name in ("agents", "plugins", "endpoints", "ingestion"):
        (tmp_path / name).mkdir(exist_ok=True)
    store = MemoryStore()
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        ingestion_dir=tmp_path / "ingestion",
        store=store,
        logs_history=True,
        allow_logsllm_analysis=analysis,
        enable_studio=False,
        enable_telemetry=False,
        enable_tasks=False,
    )
    platform.register_agent(
        manifest=AgentManifest(name="echo", slug="echo", description="Echo"),
        node_fn=_echo_node,
    )
    platform._plugin_registry._plugins["sentiment"] = _DummyPlugin()
    platform.register_endpoint(_EchoEndpoint())
    platform.register_ingestor(_DocsIngestor())
    return platform


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


class TestPluginLogs:
    def test_predict_records_and_global_list(self, tmp_path: Path) -> None:
        platform = _platform(tmp_path)
        with TestClient(platform.build()) as client:
            resp = client.post(
                "/api/v1/plugins/sentiment/predict",
                json={"text": "good news"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["label"] == "pos"

            listed = client.get("/api/v1/logs?resource=plugin&name=sentiment")
            assert listed.status_code == 200, listed.text
            body = listed.json()
            assert body["total"] >= 1
            assert body["logs"][0]["resource_type"] == "plugin"
            assert body["logs"][0]["endpoint"] == "predict"
            assert body["logs"][0]["input"]["text"] == "good news"


class TestEndpointLogs:
    def test_handle_records(self, tmp_path: Path) -> None:
        platform = _platform(tmp_path)
        with TestClient(platform.build()) as client:
            resp = client.post(
                "/api/v1/endpoints/echo_ep/echo",
                json={"text": "hi"},
            )
            assert resp.status_code == 200, resp.text

            listed = client.get("/api/v1/logs?resource=endpoint&name=echo_ep")
            assert listed.status_code == 200
            assert listed.json()["total"] >= 1
            assert listed.json()["logs"][0]["endpoint"] == "handle"


class TestIngestionLogs:
    def test_run_records(self, tmp_path: Path) -> None:
        platform = _platform(tmp_path)
        with TestClient(platform.build()) as client:
            resp = client.post(
                "/api/v1/ingestion/docs/run",
                json={"source": "a.pdf", "collection": "kb"},
            )
            assert resp.status_code == 200, resp.text

            listed = client.get("/api/v1/logs?resource=ingestion&name=docs")
            assert listed.status_code == 200
            assert listed.json()["total"] >= 1
            assert listed.json()["logs"][0]["endpoint"] == "run"


class TestPipelineLogs:
    @pytest.mark.asyncio
    async def test_inprocess_agent_step_logs(self) -> None:
        store = MemoryStore()
        recorder = InvocationLogRecorder(store)
        set_invocation_log_recorder(recorder)
        try:
            node_fn = AsyncMock(return_value={"response": "planned"})
            registry = MagicMock()
            agent = MagicMock()
            agent.graph_fn = None
            agent.class_instance = None
            agent.schema_validator = None
            agent.node_fn = node_fn
            registry.get.return_value = agent
            registry.list_names.return_value = ["planner"]

            config = PipelineConfig(
                name="simple_pipe",
                steps=[AgentStepConfig(name="plan", agent="planner")],
            )
            result = await PipelineEngine(config, registry).run({"query": "hello"})
            assert result.status == PipelineStatus.SUCCESS

            step_logs = [
                e
                for e in store._invocation_logs.values()
                if e.get("endpoint") == "pipeline_step"
            ]
            assert len(step_logs) >= 1
            assert step_logs[0]["resource_type"] == "agent"
            assert step_logs[0]["agent_name"] == "planner"
            assert step_logs[0]["metadata"].get("pipeline") == "simple_pipe"
        finally:
            set_invocation_log_recorder(None)

    def test_http_pipeline_run_logs(self, tmp_path: Path) -> None:
        (tmp_path / "agents").mkdir(exist_ok=True)
        pipes_dir = tmp_path / "pipelines"
        pipes_dir.mkdir()
        (pipes_dir / "simple_pipe.yaml").write_text(
            "name: simple_pipe\n"
            "steps:\n"
            "  - name: plan\n"
            "    agent: echo\n"
        )
        platform = _platform(tmp_path)
        with TestClient(platform.build()) as client:
            run = client.post(
                "/api/v1/pipelines/simple_pipe/run",
                json={"input": {"query": "via-http"}},
            )
            assert run.status_code == 200, run.text

            listed = client.get("/api/v1/logs?resource=pipeline&name=simple_pipe")
            assert listed.status_code == 200
            assert listed.json()["total"] >= 1
            assert listed.json()["logs"][0]["endpoint"] == "run"

            # In-process agent step from pipeline also logged
            agent_steps = client.get(
                "/api/v1/logs?resource=agent&name=echo&endpoint=pipeline_step"
            )
            assert agent_steps.status_code == 200
            assert agent_steps.json()["total"] >= 1

    def test_analyze_across_resource_types(self, tmp_path: Path) -> None:
        platform = _platform(tmp_path, analysis=True)
        with TestClient(platform.build()) as client:
            client.post(
                "/api/v1/plugins/sentiment/predict",
                json={"text": "good"},
            )
            resp = client.post(
                "/api/v1/logs/analyze",
                json={"resource": "plugin", "name": "sentiment", "persist": True},
            )
            assert resp.status_code == 200, resp.text
            analysis = resp.json()["analysis"]
            assert analysis["resource_type"] == "plugin"
            assert analysis["resource_name"] == "sentiment"
            assert analysis["status"] in {"healthy", "degraded", "failing", "unknown"}

            latest = client.get(
                "/api/v1/logs/analysis?resource=plugin&name=sentiment"
            )
            assert latest.status_code == 200
            assert latest.json()["analysis"]["id"] == analysis["id"]


class TestAgentRouteBackwardCompat:
    def test_agent_logs_still_work(self, tmp_path: Path) -> None:
        platform = _platform(tmp_path)
        with TestClient(platform.build()) as client:
            inv = client.post("/api/v1/echo/invoke", json={"query": "ping"})
            assert inv.status_code == 200, inv.text

            listed = client.get("/api/v1/echo/logs")
            assert listed.status_code == 200
            assert listed.json()["total"] >= 1
            assert listed.json()["logs"][0]["resource_type"] == "agent"

            # Global filter by agent
            global_list = client.get("/api/v1/logs?resource=agent&name=echo")
            assert global_list.status_code == 200
            assert global_list.json()["total"] >= 1


@pytest.mark.asyncio
async def test_sqlite_multi_resource_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plugin logs survive restart via SQLAlchemyStore."""
    from agentomatic.storage.sqlalchemy import SQLAlchemyStore

    db_path = tmp_path / "multi.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.delenv("AGENTOMATIC_DB_URL", raising=False)

    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        logs_history=True,
        enable_studio=False,
        enable_telemetry=False,
        enable_tasks=False,
    )
    (tmp_path / "agents").mkdir(exist_ok=True)
    (tmp_path / "plugins").mkdir(exist_ok=True)
    platform._plugin_registry._plugins["sentiment"] = _DummyPlugin()

    with TestClient(platform.build()) as client:
        resp = client.post(
            "/api/v1/plugins/sentiment/predict",
            json={"text": "persist"},
        )
        assert resp.status_code == 200, resp.text
        listed = client.get("/api/v1/logs?resource=plugin&name=sentiment")
        assert listed.status_code == 200
        log_id = listed.json()["logs"][0]["id"]
        assert isinstance(platform.store, SQLAlchemyStore)

    store2 = SQLAlchemyStore(url)
    await store2.initialize()
    try:
        rows = await store2.list_invocation_logs(
            agent_name="sentiment",
            resource_type="plugin",
            limit=10,
        )
        assert any(row["id"] == log_id for row in rows)
        assert any(row.get("resource_type") == "plugin" for row in rows)
    finally:
        await store2.close()
