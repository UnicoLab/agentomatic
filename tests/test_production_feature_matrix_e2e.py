"""One-process production feature matrix exercised through the public HTTP API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import BaseModel, RootModel

from agentomatic import AgentManifest, AgentPlatform
from agentomatic.connections import HttpConnectionConfig, get_connections
from agentomatic.core.agent_invoke import invoke_registered_agent
from agentomatic.core.schemas import SchemaValidator
from agentomatic.endpoints import BaseEndpoint
from agentomatic.ingestion import BaseIngestor, IngestionResult
from agentomatic.plugins import BaseMLPlugin
from agentomatic.storage import MemoryStore


class _ScoreInput(BaseModel):
    """Input accepted by the feature-matrix plugin."""

    text: str


class _ScoreOutput(BaseModel):
    """Output returned by the feature-matrix plugin."""

    score: int


class _RootText(RootModel[str | None]):
    """A non-object request schema must remain valid through all task paths."""


class _RootOutput(BaseModel):
    text: str


class _DocumentInput(BaseModel):
    """A required source proves ingestion task validation is not bypassed."""

    source: str


class _LengthScorer(BaseMLPlugin[_ScoreInput, _ScoreOutput]):
    """Small deterministic plugin that needs no external model service."""

    plugin_name = "length_scorer"
    plugin_description = "Returns the input length."

    async def predict(self, inputs: _ScoreInput) -> _ScoreOutput:
        return _ScoreOutput(score=len(inputs.text))


class _RootEcho(BaseMLPlugin[_RootText, _RootOutput]):
    """Verifies FastAPI/OpenAPI and task validation preserve a RootModel body."""

    plugin_name = "root_echo"

    async def predict(self, inputs: _RootText) -> _RootOutput:
        return _RootOutput(text=inputs.root)


class _EnrichInput(BaseModel):
    """Input accepted by the custom endpoint."""

    text: str


class _EnrichOutput(BaseModel):
    """Output returned by the custom endpoint."""

    text: str


class _Enricher(BaseEndpoint[_EnrichInput, _EnrichOutput]):
    """Deterministic custom endpoint used by the pipeline."""

    endpoint_name = "enricher"
    endpoint_description = "Adds deployment context to a prompt."
    path = "/enrich"

    async def handle(self, request: _EnrichInput) -> _EnrichOutput:
        return _EnrichOutput(text=f"enriched:{request.text}")


class _DocumentIngestor(BaseIngestor[_DocumentInput]):
    """Small ingestor that proves the sync resource and pipeline contracts."""

    ingestor_name = "documents"
    ingestor_description = "Counts one ingested document."

    async def ingest(self, request: Any, _ctx: Any) -> IngestionResult:
        return IngestionResult(documents=1, chunks=1, upserted=1, collection="production")


def _write_pipeline(root: Path) -> None:
    """Write a pipeline that consumes every deployable resource type."""
    pipelines = root / "pipelines"
    pipelines.mkdir()
    (pipelines / "production_flow.yaml").write_text(
        """name: production_flow
input_schema:
  query: str
strict_schema: true
steps:
  - name: ingest
    ingestion: documents
    input:
      source: $.input.query
  - name: enrich
    endpoint: enricher
    input:
      text: $.input.query
  - name: score
    plugin: length_scorer
    input:
      text: $.steps.enrich.text
  - name: root_echo
    plugin: root_echo
    input:
      __root__: $.input.query
  - name: research
    agent: researcher
    input:
      current_query: $.steps.enrich.text
  - name: write
    agent: writer
    input:
      current_query: $.steps.research.response
""",
        encoding="utf-8",
    )


def test_production_feature_matrix_over_authenticated_http(tmp_path: Path) -> None:
    """Run agents, delegation, resources, pipeline, auth, and control plane together."""
    for name in ("agents", "plugins", "endpoints", "ingestion"):
        (tmp_path / name).mkdir()
    _write_pipeline(tmp_path)

    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        ingestion_dir=tmp_path / "ingestion",
        store=MemoryStore(),
        enable_auth=True,
        auth_api_key="production-key",
        enable_studio=True,
        enable_zero_trust=True,
        require_auth_globally=True,
        enable_control_plane=True,
        control_token="control-key",
        connections=[HttpConnectionConfig(name="catalog", base_url="https://catalog.invalid")],
        enable_telemetry=False,
    )

    async def researcher(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": f"research:{state['current_query']}", "agent_type": "researcher"}

    async def writer(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "response": f"writer:{state['current_query']}",
            "agent_type": "writer",
            "connections": get_connections("__platform__").list_names(),
        }

    async def coordinator(state: dict[str, Any]) -> dict[str, Any]:
        target = platform.registry.get("researcher")
        assert target is not None
        delegated = await invoke_registered_agent(
            target,
            {"current_query": state.get("current_query", "")},
        )
        return {
            "response": f"delegated:{delegated['response']}",
            "agent_type": "coordinator",
        }

    async def root_agent(state: dict[str, Any]) -> dict[str, Any]:
        """Expose the exact RootModel body forwarded by every invoke surface."""
        return {
            "response": f"root:{state['__root__']}",
            "root": state["__root__"],
            "agent_type": "root_agent",
        }

    platform.register_agent(
        AgentManifest(name="researcher", slug="researcher", description="Research specialist"),
        node_fn=researcher,
    )
    platform.register_agent(
        AgentManifest(name="writer", slug="writer", description="Writing specialist"),
        node_fn=writer,
    )
    platform.register_agent(
        AgentManifest(
            name="coordinator",
            slug="coordinator",
            description="Delegates to the researcher.",
            delegation_targets=["researcher"],
        ),
        node_fn=coordinator,
    )
    platform.register_agent(
        AgentManifest(
            name="root_agent",
            slug="root_agent",
            description="Accepts a root body.",
            framework="custom",
        ),
        node_fn=root_agent,
        schema_validator=SchemaValidator(request_model=_RootText),
    )
    platform._plugin_registry._plugins["length_scorer"] = _LengthScorer()  # noqa: SLF001
    platform._plugin_registry._plugins["root_echo"] = _RootEcho()  # noqa: SLF001
    platform.register_endpoint(_Enricher())
    platform.register_ingestor(_DocumentIngestor())

    headers = {"X-Api-Key": "production-key"}
    with TestClient(platform.build()) as client:
        assert client.post("/api/v1/researcher/invoke", json={"query": "x"}).status_code == 401

        agents = client.get("/api/v1/agents", headers=headers)
        assert agents.status_code == 200
        assert {"researcher", "writer", "coordinator", "root_agent"} <= set(
            agents.json()["agents"]
        )

        # A root-schema agent accepts the native scalar body over its public
        # route, through the generic Task Board, and through Studio's object
        # envelope. All three paths expose the original value as state.__root__.
        root_agent_invoke = client.post(
            "/api/v1/root_agent/invoke",
            headers=headers,
            json="native agent root",
        )
        assert root_agent_invoke.status_code == 200, root_agent_invoke.text
        assert root_agent_invoke.json()["response"] == "root:native agent root"
        assert root_agent_invoke.json()["output"]["root"] == "native agent root"

        root_agent_task = client.post(
            "/api/v1/tasks",
            headers=headers,
            json={
                "target_type": "agent",
                "target": "root_agent",
                "input": "task agent root",
                "wait": True,
            },
        )
        assert root_agent_task.status_code == 200, root_agent_task.text
        assert root_agent_task.json()["status"] == "succeeded", root_agent_task.json()
        assert root_agent_task.json()["result"]["root"] == "task agent root"

        root_agent_studio = client.post(
            "/studio/agents/root_agent/runs",
            headers=headers,
            json={"agent_input": "studio agent root"},
        )
        assert root_agent_studio.status_code == 200, root_agent_studio.text
        assert root_agent_studio.json()["status"] == "completed", root_agent_studio.json()
        studio_events = root_agent_studio.json()["events"]
        assert any(
            event["event"] == "run_complete"
            and event["data"].get("output", {}).get("output", {}).get("root")
            == "studio agent root"
            for event in studio_events
        ), studio_events

        root_agent_studio_null = client.post(
            "/studio/agents/root_agent/runs",
            headers=headers,
            json={"agent_input": None},
        )
        assert root_agent_studio_null.status_code == 200, root_agent_studio_null.text
        null_events = root_agent_studio_null.json()["events"]
        assert any(
            event["event"] == "run_complete"
            and event["data"].get("output", {}).get("output", {}).get("root") is None
            for event in null_events
        ), null_events

        delegation = client.post(
            "/api/v1/coordinator/invoke",
            headers=headers,
            json={"query": "production brief"},
        )
        assert delegation.status_code == 200, delegation.text
        assert delegation.json()["response"] == "delegated:research:production brief"

        plugin = client.post(
            "/api/v1/plugins/length_scorer/predict",
            headers=headers,
            json={"text": "hello"},
        )
        assert plugin.status_code == 200, plugin.text
        assert plugin.json() == {"score": 5}

        root_plugin = client.post(
            "/api/v1/plugins/root_echo/predict",
            headers=headers,
            json="root payload",
        )
        assert root_plugin.status_code == 200, root_plugin.text
        assert root_plugin.json() == {"text": "root payload"}

        endpoint = client.post(
            "/api/v1/endpoints/enricher/enrich",
            headers=headers,
            json={"text": "hello"},
        )
        assert endpoint.status_code == 200, endpoint.text
        assert endpoint.json() == {"text": "enriched:hello"}

        ingested = client.post(
            "/api/v1/ingestion/documents/run",
            headers=headers,
            json={"source": "brief.md"},
        )
        assert ingested.status_code == 200, ingested.text
        assert ingested.json()["documents"] == 1

        # The generic Task Board must honour the same live schemas as each
        # resource's native route. Invalid work is rejected before it can
        # pollute the durable queue.
        for target_type, target, payload in (
            ("plugin", "length_scorer", {}),
            ("endpoint", "enricher", {}),
            ("ingestion", "documents", {}),
            ("pipeline", "production_flow", {}),
        ):
            rejected = client.post(
                "/api/v1/tasks",
                headers=headers,
                json={"target_type": target_type, "target": target, "input": payload},
            )
            assert rejected.status_code == 422, rejected.text

        rejected_batch = client.post(
            "/api/v1/tasks",
            headers=headers,
            json={
                "target_type": "plugin",
                "target": "length_scorer",
                "batch": [{"text": "valid item"}, {}],
            },
        )
        assert rejected_batch.status_code == 422, rejected_batch.text

        for target_type in ("plugin", "endpoint", "ingestion", "pipeline"):
            missing = client.post(
                "/api/v1/tasks",
                headers=headers,
                json={"target_type": target_type, "target": "missing", "input": {}},
            )
            assert missing.status_code == 404, missing.text
        # The successful root-agent task above is the only queued task so far;
        # rejected submissions must not create additional durable records.
        assert client.get("/api/v1/tasks", headers=headers).json()["total"] == 1

        for target_type, target, payload in (
            ("plugin", "length_scorer", {"text": "hello"}),
            ("endpoint", "enricher", {"text": "hello"}),
            ("ingestion", "documents", {"source": "brief.md"}),
            ("pipeline", "production_flow", {"query": "hello"}),
        ):
            accepted = client.post(
                "/api/v1/tasks",
                headers=headers,
                json={
                    "target_type": target_type,
                    "target": target,
                    "input": payload,
                    "wait": True,
                },
            )
            assert accepted.status_code == 200, accepted.text
            assert accepted.json()["status"] == "succeeded", accepted.json()

        root_task = client.post(
            "/api/v1/tasks",
            headers=headers,
            json={
                "target_type": "plugin",
                "target": "root_echo",
                "input": "task root payload",
                "wait": True,
            },
        )
        assert root_task.status_code == 200, root_task.text
        assert root_task.json()["status"] == "succeeded", root_task.json()

        pipeline = client.post(
            "/api/v1/pipelines/production_flow/run",
            headers=headers,
            json={"input": {"query": "hello"}},
        )
        assert pipeline.status_code == 200, pipeline.text
        assert pipeline.json()["status"] == "success", pipeline.json()
        assert pipeline.json()["output"]["response"] == "writer:research:enriched:hello"
        assert set(pipeline.json()["steps"]) == {
            "ingest",
            "enrich",
            "score",
            "root_echo",
            "research",
            "write",
        }
        assert pipeline.json()["steps"]["root_echo"]["output"] == {"text": "hello"}

        connections = client.get("/api/v1/control/connections", headers=headers)
        assert connections.status_code == 200, connections.text
        assert connections.json()[0]["scope"] == "__platform__"
        storage = connections.json()[0]["connections"]["storage"]
        assert storage["connection"] == "storage"
        assert storage["kind"] == "database"
        assert storage["purpose"] == "persistence"

        storage_probe = client.get(
            "/api/v1/control/connections/__platform__/storage", headers=headers
        )
        assert storage_probe.status_code == 200, storage_probe.text
        assert storage_probe.json()["connection"] == "storage"

        # A concrete response model keeps this operational check discoverable
        # for Studio and any generated client, while still permitting
        # backend-specific diagnostics as additional fields.
        openapi = client.get("/openapi.json", headers=headers)
        assert openapi.status_code == 200
        probe_schema = openapi.json()["paths"]["/api/v1/control/connections/{scope}/{name}"][
            "get"
        ]["responses"]["200"]["content"]["application/json"]["schema"]
        assert probe_schema["$ref"].endswith("/ControlConnectionProbe")

        # The Studio Connection card uses this per-resource probe. It must run
        # only the named connection (not issue a broad status refresh) and
        # return a safe, structured health result.
        connection_probe = client.get(
            "/api/v1/control/connections/__platform__/catalog", headers=headers
        )
        assert connection_probe.status_code == 200, connection_probe.text
        assert connection_probe.json()["connection"] == "catalog"

        denied = client.post("/api/v1/control/agents/researcher/disable", headers=headers)
        assert denied.status_code == 401
        disabled = client.post(
            "/api/v1/control/agents/researcher/disable",
            headers={**headers, "X-Control-Token": "control-key"},
        )
        assert disabled.status_code == 200, disabled.text
        assert (
            client.post(
                "/api/v1/researcher/invoke", headers=headers, json={"query": "x"}
            ).status_code
            == 503
        )
        enabled = client.post(
            "/api/v1/control/agents/researcher/enable",
            headers={**headers, "X-Control-Token": "control-key"},
        )
        assert enabled.status_code == 200, enabled.text
        assert (
            client.post(
                "/api/v1/researcher/invoke", headers=headers, json={"query": "x"}
            ).status_code
            == 200
        )
