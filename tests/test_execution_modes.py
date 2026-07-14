"""Tests for per-resource execution-mode sugar (/async and /batch).

Verifies that agents, plugins, pipelines, and ingestors all expose consistent
asynchronous and batch execution routes backed by the unified task system.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from agentomatic import AgentPlatform, BaseIngestor, IngestionResult
from agentomatic.core.manifest import AgentManifest


def _poll(client: TestClient, task_id: str, tries: int = 60) -> dict:
    """Poll the unified task API until the task reaches a terminal state."""
    got: dict = {}
    for _ in range(tries):
        got = client.get(f"/api/v1/tasks/{task_id}").json()
        if got["status"] in {"succeeded", "failed", "cancelled"}:
            return got
        time.sleep(0.02)
    return got


# =====================================================================
# Shared fixtures
# =====================================================================


async def _echo_node(state: dict) -> dict:
    return {"response": f"echo:{state.get('current_query', '')}"}


class _DocsIngestor(BaseIngestor):
    ingestor_name = "docs"

    async def ingest(self, request, ctx) -> IngestionResult:
        return IngestionResult(documents=1, chunks=3, upserted=3)


def _platform(tmp_path) -> AgentPlatform:
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        ingestion_dir=tmp_path / "ingestion",
        enable_studio=False,
    )
    platform.register_agent(
        AgentManifest(name="echo", slug="echo", description="Echo", version="1.0.0"),
        node_fn=_echo_node,
    )
    platform.register_ingestor(_DocsIngestor())
    return platform


# =====================================================================
# Agents
# =====================================================================


class TestAgentExecutionModes:
    def test_invoke_async(self, tmp_path):
        with TestClient(_platform(tmp_path).build()) as client:
            resp = client.post("/api/v1/echo/invoke/async", json={"query": "hi"})
            assert resp.status_code == 202
            body = resp.json()
            assert body["target_type"] == "agent"
            assert "links" in body and body["links"]["status"].endswith(body["id"])

            done = _poll(client, body["id"])
            assert done["status"] == "succeeded"
            result = client.get(f"/api/v1/tasks/{body['id']}/result").json()
            assert result["result"]["response"] == "echo:hi"

    def test_invoke_batch(self, tmp_path):
        with TestClient(_platform(tmp_path).build()) as client:
            resp = client.post(
                "/api/v1/echo/invoke/batch",
                json={"inputs": [{"query": "a"}, {"query": "b"}, {"query": "c"}]},
            )
            assert resp.status_code == 202
            task_id = resp.json()["id"]

            done = _poll(client, task_id)
            assert done["status"] == "succeeded"
            result = client.get(f"/api/v1/tasks/{task_id}/result").json()["result"]
            assert isinstance(result, list)
            assert len(result) == 3


# =====================================================================
# Ingestors (batch is new via the shared helper)
# =====================================================================


class TestIngestionExecutionModes:
    def test_run_batch(self, tmp_path):
        with TestClient(_platform(tmp_path).build()) as client:
            resp = client.post(
                "/api/v1/ingestion/docs/run/batch",
                json={"inputs": [{"source": "a.pdf"}, {"source": "b.pdf"}]},
            )
            assert resp.status_code == 202
            task_id = resp.json()["id"]

            done = _poll(client, task_id)
            assert done["status"] == "succeeded"
            result = client.get(f"/api/v1/tasks/{task_id}/result").json()["result"]
            assert len(result) == 2
            assert all(r["chunks"] == 3 for r in result)


# =====================================================================
# Plugins (standalone router wiring)
# =====================================================================


class _PredictIn(BaseModel):
    text: str = Field(...)


class _PredictOut(BaseModel):
    result: str


def _plugin_app():
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from agentomatic.plugins.ml import BaseMLPlugin
    from agentomatic.plugins.registry import PluginRegistry
    from agentomatic.plugins.router import create_plugin_router
    from agentomatic.tasks import TargetType, TaskManager
    from agentomatic.tasks.dispatchers import make_plugin_dispatcher
    from agentomatic.tasks.routes import create_task_router

    class UpperPlugin(BaseMLPlugin[_PredictIn, _PredictOut]):
        plugin_name = "upper"

        async def predict(self, inputs: _PredictIn) -> _PredictOut:
            return _PredictOut(result=inputs.text.upper())

    plugin = UpperPlugin()
    reg = PluginRegistry()
    reg._plugins["upper"] = plugin

    mgr = TaskManager()
    mgr.register_dispatcher(TargetType.PLUGIN, make_plugin_dispatcher(reg))

    @asynccontextmanager
    async def lifespan(_app):
        await plugin.load_model()
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(
        create_plugin_router(plugin, task_manager=mgr, api_prefix="/api/v1"),
        prefix="/api/v1/plugins/upper",
    )
    app.include_router(create_task_router(mgr), prefix="/api/v1/tasks")
    return app


class TestPluginExecutionModes:
    def test_predict_async(self):
        with TestClient(_plugin_app()) as client:
            resp = client.post("/api/v1/plugins/upper/predict/async", json={"text": "hi"})
            assert resp.status_code == 202
            task_id = resp.json()["id"]

            done = _poll(client, task_id)
            assert done["status"] == "succeeded"
            result = client.get(f"/api/v1/tasks/{task_id}/result").json()["result"]
            assert result["result"] == "HI"


# =====================================================================
# Pipelines (standalone router wiring)
# =====================================================================


def _pipeline_app():
    from fastapi import FastAPI

    from agentomatic.core.registry import AgentRegistry
    from agentomatic.pipelines.loader import PipelineLoader
    from agentomatic.pipelines.router import create_pipeline_router
    from agentomatic.tasks import TargetType, TaskManager
    from agentomatic.tasks.dispatchers import make_pipeline_dispatcher
    from agentomatic.tasks.routes import create_task_router

    config = PipelineLoader.from_dict(
        {
            "name": "double",
            "steps": [{"transform": "return {'n': ctx.input.get('n', 0) * 2}", "name": "d"}],
        }
    )
    pipelines = {"double": config}
    registry = AgentRegistry()

    mgr = TaskManager()
    mgr.register_dispatcher(TargetType.PIPELINE, make_pipeline_dispatcher(pipelines, registry))

    app = FastAPI()
    app.include_router(
        create_pipeline_router(pipelines, registry, task_manager=mgr, api_prefix="/api/v1"),
        prefix="/api/v1",
    )
    app.include_router(create_task_router(mgr), prefix="/api/v1/tasks")
    return app


class TestPipelineExecutionModes:
    def test_run_async(self):
        with TestClient(_pipeline_app()) as client:
            resp = client.post("/api/v1/pipelines/double/run/async", json={"input": {"n": 21}})
            assert resp.status_code == 202
            task_id = resp.json()["id"]

            done = _poll(client, task_id)
            assert done["status"] == "succeeded"
            result = client.get(f"/api/v1/tasks/{task_id}/result").json()["result"]
            assert result["output"]["n"] == 42

    def test_run_batch(self):
        with TestClient(_pipeline_app()) as client:
            resp = client.post(
                "/api/v1/pipelines/double/run/batch",
                json={"inputs": [{"n": 1}, {"n": 2}]},
            )
            assert resp.status_code == 202
            task_id = resp.json()["id"]

            done = _poll(client, task_id)
            assert done["status"] == "succeeded"
            result = client.get(f"/api/v1/tasks/{task_id}/result").json()["result"]
            assert len(result) == 2
            assert result[0]["output"]["n"] == 2
            assert result[1]["output"]["n"] == 4
