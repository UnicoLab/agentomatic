# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for the visual-builder REST API (validate-draft, save, delete).

Covers the stateless draft validation endpoint, YAML persistence
(create/update/delete), the loader round-trip serializer, source-path
discovery, and the platform-level ingestor palette endpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform, BaseIngestor, IngestionResult
from agentomatic.core.registry import AgentRegistry
from agentomatic.pipelines.loader import PipelineLoader, pipeline_to_dict, pipeline_to_yaml
from agentomatic.pipelines.router import create_pipeline_router

# =====================================================================
# Helpers
# =====================================================================


class _DocsIngestor(BaseIngestor):
    ingestor_name = "docs_ingestor"
    ready = False

    async def setup(self) -> None:
        self.ready = True

    async def ingest(self, request, ctx) -> IngestionResult:
        return IngestionResult(documents=1, chunks=1, upserted=1)


def _registry(agents: dict[str, object] | None = None) -> MagicMock:
    """Registry stub that mirrors AgentRegistry.get/list_names."""
    agents = agents or {}
    registry = MagicMock(spec=AgentRegistry)
    registry.get.side_effect = lambda n: agents.get(n)
    registry.list_names.return_value = list(agents)
    return registry


def _pipeline_app(tmp_path: Path, *, agents: dict[str, object] | None = None) -> FastAPI:
    """Build an app with the pipeline router bound to a temp pipelines dir."""
    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir(exist_ok=True)
    registry = _registry(agents)
    router = create_pipeline_router(
        {},
        registry,
        api_prefix="/api/v1",
        pipelines_dir=pipelines_dir,
        pipeline_paths={},
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1", tags=["Pipelines"])
    return app


def _platform(tmp_path: Path) -> AgentPlatform:
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        ingestion_dir=tmp_path / "ingestion",
        enable_studio=False,
    )
    platform.register_ingestor(_DocsIngestor())
    return platform


_DRAFT = {
    "name": "demo",
    "description": "builder demo",
    "steps": [{"name": "double", "transform": "return {'n': ctx.input.get('n', 0) * 2}"}],
}


# =====================================================================
# POST /api/v1/pipelines/validate-draft
# =====================================================================


class TestValidateDraft:
    def test_valid_json_draft(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": _DRAFT})
            assert resp.status_code == 200
            data = resp.json()
            assert data["valid"] is True
            assert data["errors"] == []
            assert data["warnings"] == []
            assert data["step_count"] == 1
            assert data["name"] == "demo"

    def test_valid_yaml_draft(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post(
                "/api/v1/pipelines/validate-draft",
                json={
                    "yaml": "name: demo\nsteps:\n  - name: t\n    transform: 'return {\"ok\": True}'\n"
                },
            )
            assert resp.status_code == 200
            assert resp.json()["valid"] is True

    def test_neither_pipeline_nor_yaml(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={})
            data = resp.json()
            assert data["valid"] is False
            assert any("exactly one" in e for e in data["errors"])

    def test_invalid_yaml_reports_parse_error(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post(
                "/api/v1/pipelines/validate-draft",
                json={"yaml": "name: [unclosed"},
            )
            data = resp.json()
            assert data["valid"] is False
            assert data["errors"]

    def test_unknown_agent_reported(self, tmp_path: Path) -> None:
        draft = {
            "name": "p",
            "steps": [{"name": "s", "agent": "ghost"}],
        }
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": draft})
            data = resp.json()
            assert data["valid"] is False
            assert any("ghost" in e and "registry" in e for e in data["errors"])

    def test_known_agent_passes(self, tmp_path: Path) -> None:
        draft = {"name": "p", "steps": [{"name": "s", "agent": "planner"}]}
        with TestClient(_pipeline_app(tmp_path, agents={"planner": object()})) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": draft})
            assert resp.json()["valid"] is True

    def test_unknown_mapping_root_is_error(self, tmp_path: Path) -> None:
        draft = {
            "name": "p",
            "steps": [
                {
                    "name": "s",
                    "agent": "planner",
                    "input": {"q": "$.unknown.query"},
                }
            ],
        }
        with TestClient(_pipeline_app(tmp_path, agents={"planner": object()})) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": draft})
            data = resp.json()
            assert data["valid"] is False
            assert any("unknown mapping root 'unknown'" in e for e in data["errors"])

    def test_referencing_later_step_is_error(self, tmp_path: Path) -> None:
        draft = {
            "name": "p",
            "steps": [
                {
                    "name": "first",
                    "agent": "planner",
                    "input": {"q": "$.steps.second.result"},
                },
                {"name": "second", "agent": "planner"},
            ],
        }
        with TestClient(_pipeline_app(tmp_path, agents={"planner": object()})) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": draft})
            data = resp.json()
            assert data["valid"] is False
            assert any("runs after" in e for e in data["errors"])

    def test_referencing_missing_step_is_error(self, tmp_path: Path) -> None:
        draft = {
            "name": "p",
            "steps": [{"name": "first", "agent": "planner", "input": {"q": "$.steps.nope.x"}}],
        }
        with TestClient(_pipeline_app(tmp_path, agents={"planner": object()})) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": draft})
            data = resp.json()
            assert data["valid"] is False
            assert any("step 'nope'" in e for e in data["errors"])

    def test_condition_syntax_error(self, tmp_path: Path) -> None:
        draft = {
            "name": "p",
            "steps": [{"name": "t", "transform": "return {'ok': True}", "condition": "len(("}],
        }
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": draft})
            data = resp.json()
            assert data["valid"] is False
            assert any("invalid syntax" in e for e in data["errors"])

    def test_on_error_skip_is_warning(self, tmp_path: Path) -> None:
        draft = {
            "name": "p",
            "steps": [{"name": "t", "transform": "return {'ok': True}", "on_error": "skip"}],
        }
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": draft})
            data = resp.json()
            assert data["valid"] is True
            assert any("on_error=skip" in w for w in data["warnings"])

    def test_transform_without_return_is_warning(self, tmp_path: Path) -> None:
        draft = {
            "name": "p",
            "steps": [{"name": "t", "transform": "ctx.shared['x'] = 1"}],
        }
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": draft})
            data = resp.json()
            assert data["valid"] is True
            assert any("no output" in w for w in data["warnings"])

    def test_collision_with_existing_pipeline_is_warning(self, tmp_path: Path) -> None:
        existing = PipelineLoader.from_dict(_DRAFT)
        pipelines_dir = tmp_path / "pipelines"
        router = create_pipeline_router(
            {"demo": existing},
            _registry(),
            api_prefix="/api/v1",
            pipelines_dir=pipelines_dir,
        )
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        with TestClient(app) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": _DRAFT})
            data = resp.json()
            assert data["valid"] is True
            assert any("already exists" in w for w in data["warnings"])


# =====================================================================
# POST /api/v1/pipelines/{name} (create / update)
# =====================================================================


class TestSavePipeline:
    def test_save_new_pipeline(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/demo", json={"pipeline": _DRAFT})
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "demo"
            assert data["valid"] is True
            assert data["step_count"] == 1

            target = tmp_path / "pipelines" / "demo.yaml"
            assert data["path"] == str(target)
            assert target.exists()

            # Now visible in the list and re-loadable from disk.
            listed = client.get("/api/v1/pipelines").json()
            assert [p["name"] for p in listed] == ["demo"]

            reloaded = PipelineLoader.from_yaml(target)
            assert reloaded.model_dump() == PipelineLoader.from_dict(_DRAFT).model_dump()

    def test_save_via_yaml_body(self, tmp_path: Path) -> None:
        yaml_text = "name: yml_demo\nsteps:\n  - name: t\n    transform: 'return {\"ok\": True}'\n"
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/yml_demo", json={"yaml": yaml_text})
            assert resp.status_code == 200
            assert (tmp_path / "pipelines" / "yml_demo.yaml").exists()

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            first = client.post("/api/v1/pipelines/demo", json={"pipeline": _DRAFT})
            assert first.status_code == 200

            updated = dict(_DRAFT)
            updated["description"] = "v2"
            second = client.post("/api/v1/pipelines/demo", json={"pipeline": updated})
            assert second.status_code == 200

            listed = client.get("/api/v1/pipelines").json()
            assert [p["name"] for p in listed] == ["demo"]
            reloaded = PipelineLoader.from_yaml(tmp_path / "pipelines" / "demo.yaml")
            assert reloaded.description == "v2"

    def test_save_name_mismatch_422(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/other", json={"pipeline": _DRAFT})
            assert resp.status_code == 422
            assert "does not match" in resp.json()["detail"]["message"]

    def test_save_invalid_draft_422(self, tmp_path: Path) -> None:
        draft = {"name": "bad", "steps": [{"name": "s", "agent": "ghost"}]}
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/bad", json={"pipeline": draft})
            assert resp.status_code == 422
            detail = resp.json()["detail"]
            assert any("ghost" in e for e in detail["errors"])

    def test_save_requires_persistence_dir(self, tmp_path: Path) -> None:
        registry = _registry()
        router = create_pipeline_router({}, registry, api_prefix="/api/v1")
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        with TestClient(app) as client:
            resp = client.post("/api/v1/pipelines/demo", json={"pipeline": _DRAFT})
            assert resp.status_code == 400
            assert "not configured" in resp.json()["detail"]["message"]

    def test_save_rejects_unsafe_name(self, tmp_path: Path) -> None:
        draft = {"name": "..evil", "steps": [{"name": "t", "transform": "return {}"}]}
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.post("/api/v1/pipelines/..evil", json={"pipeline": draft})
            assert resp.status_code == 422
            assert "Invalid pipeline name" in resp.json()["detail"]["message"]


# =====================================================================
# DELETE /api/v1/pipelines/{name}
# =====================================================================


class TestDeletePipeline:
    def test_delete_removes_file_and_entry(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            client.post("/api/v1/pipelines/demo", json={"pipeline": _DRAFT})

            resp = client.delete("/api/v1/pipelines/demo")
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True
            assert not (tmp_path / "pipelines" / "demo.yaml").exists()
            assert client.get("/api/v1/pipelines").json() == []

    def test_delete_missing_404(self, tmp_path: Path) -> None:
        with TestClient(_pipeline_app(tmp_path)) as client:
            resp = client.delete("/api/v1/pipelines/nope")
            assert resp.status_code == 404

    def test_delete_without_persistence_dir_400(self, tmp_path: Path) -> None:
        router = create_pipeline_router({}, _registry(), api_prefix="/api/v1")
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        with TestClient(app) as client:
            resp = client.delete("/api/v1/pipelines/demo")
            assert resp.status_code == 400

    def test_delete_agent_folder_pipeline(self, tmp_path: Path) -> None:
        """Pipelines discovered from agents/*/pipeline.yaml delete that file."""
        agent_dir = tmp_path / "agents" / "my_agent"
        agent_dir.mkdir(parents=True)
        source = agent_dir / "pipeline.yaml"
        source.write_text(pipeline_to_yaml(PipelineLoader.from_dict(_DRAFT)), encoding="utf-8")

        registry = _registry()
        paths = PipelineLoader.discover_pipeline_files(tmp_path)
        router = create_pipeline_router(
            PipelineLoader.discover_pipelines(tmp_path),
            registry,
            api_prefix="/api/v1",
            pipelines_dir=tmp_path / "pipelines",
            pipeline_paths=paths,
        )
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        with TestClient(app) as client:
            assert client.get("/api/v1/pipelines").json()[0]["name"] == "demo"
            resp = client.delete("/api/v1/pipelines/demo")
            assert resp.status_code == 200
            assert resp.json()["path"] == str(source.resolve())
            assert not source.exists()


# =====================================================================
# Loader round-trip + path discovery
# =====================================================================


class TestLoaderRoundTrip:
    def test_all_step_types_round_trip(self) -> None:
        raw = {
            "name": "kitchen_sink",
            "description": "everything",
            "version": "2.0.0",
            "input_schema": {"query": "str"},
            "output_schema": {"answer": "str"},
            "defaults": {"lang": "en"},
            "on_error": "continue",
            "strict_schema": True,
            "timeout": 600.0,
            "metadata": {"team": "ml"},
            "steps": [
                {"name": "a", "agent": "planner", "input": {"q": "$.input.query"}},
                {"name": "pl", "plugin": "sentiment"},
                {"name": "ep", "endpoint": "ensemble", "upstreams": ["a"]},
                {"name": "ing", "ingestion": "docs", "retry": {"max_attempts": 2}},
                {"name": "t", "transform": "return {'x': 1}", "condition": "len(ctx.input.query)"},
                {
                    "name": "par",
                    "parallel": {
                        "steps": [{"name": "w", "agent": "web"}, {"name": "k", "agent": "kb"}],
                        "strategy": "first",
                        "max_concurrency": 3,
                    },
                    "on_error": "skip",
                },
                {
                    "name": "m",
                    "map": {
                        "agent": "scorer",
                        "items": "$.steps.a.results",
                        "item_key": "doc",
                        "index_key": "i",
                        "max_concurrency": 8,
                        "strategy": "majority",
                        "retry": {"max_attempts": 3, "backoff": "linear"},
                        "item_timeout": 30.0,
                        "fallback_agent": "backup",
                    },
                    "timeout": 200.0,
                },
                {
                    "name": "lp",
                    "loop": {
                        "step": {"name": "ref", "agent": "refiner"},
                        "max_iterations": 5,
                        "until": "ctx.current.done",
                    },
                },
                {"name": "sub", "sub_pipeline": "other", "input": {"x": "$.shared"}},
                {
                    "name": "fb",
                    "agent": "planner",
                    "fallback_agent": "backup",
                    "rollback": "ctx.shared['u'] = 1",
                },
            ],
        }
        config = PipelineLoader.from_dict(raw)
        dumped = pipeline_to_dict(config)
        assert PipelineLoader.from_dict(dumped).model_dump() == config.model_dump()

    def test_yaml_string_round_trip(self) -> None:
        config = PipelineLoader.from_dict(_DRAFT)
        yaml_text = pipeline_to_yaml(config)
        assert PipelineLoader.from_yaml_string(yaml_text).model_dump() == config.model_dump()

    def test_discover_pipeline_files_maps_names_to_paths(self, tmp_path: Path) -> None:
        pipelines_dir = tmp_path / "pipelines"
        pipelines_dir.mkdir()
        (pipelines_dir / "alpha.yaml").write_text(
            pipeline_to_yaml(PipelineLoader.from_dict(_DRAFT)), encoding="utf-8"
        )
        files = PipelineLoader.discover_pipeline_files(tmp_path)
        assert files == {"demo": (pipelines_dir / "alpha.yaml").resolve()}


# =====================================================================
# Platform-level palette endpoint
# =====================================================================


class TestPlatformIngestors:
    def test_ingestors_list_route(self, tmp_path: Path) -> None:
        with TestClient(_platform(tmp_path).build()) as client:
            resp = client.get("/api/v1/ingestors")
            assert resp.status_code == 200
            names = [ing["name"] for ing in resp.json()]
            assert "docs_ingestor" in names

    def test_builder_endpoints_on_platform(self, tmp_path: Path) -> None:
        """Platform build exposes the builder API even with zero pipelines."""
        with TestClient(_platform(tmp_path).build()) as client:
            resp = client.post("/api/v1/pipelines/validate-draft", json={"pipeline": _DRAFT})
            assert resp.status_code == 200
            assert resp.json()["valid"] is True

            saved = client.post("/api/v1/pipelines/demo", json={"pipeline": _DRAFT})
            assert saved.status_code == 200
            assert (tmp_path / "pipelines" / "demo.yaml").exists()

            assert client.get("/api/v1/pipelines").json()[0]["name"] == "demo"

            deleted = client.delete("/api/v1/pipelines/demo")
            assert deleted.status_code == 200
            assert not (tmp_path / "pipelines" / "demo.yaml").exists()
