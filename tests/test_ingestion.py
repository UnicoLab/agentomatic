"""Tests for the first-class ingestion / RAG ops layer.

Verifies that Agentomatic provides the *packaging* (base class, registry, REST,
task execution, progress) while the user brings the implementation.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentomatic.ingestion import (
    BaseIngestor,
    IngestionRegistry,
    IngestionRequest,
    IngestionResult,
)
from agentomatic.ingestion.context import NullIngestionContext
from agentomatic.providers.embeddings import HashEmbedder, get_embeddings, reset_embeddings

# =====================================================================
# A user-defined ingestor (simulating reuse of an external library)
# =====================================================================


def _fake_pdf_to_markdown(path: str) -> str:
    """Stand-in for an external lib like pymupdf4llm/docling."""
    return f"# {path}\n\nHello world. " * 3


class DocsIngestor(BaseIngestor):
    ingestor_name = "docs"
    ingestor_description = "Reads docs to markdown and 'upserts' them."

    async def setup(self) -> None:
        self.store: list[str] = []

    async def ingest(self, request, ctx) -> IngestionResult:
        md = _fake_pdf_to_markdown(request.source or "unknown")
        chunks = [c for c in md.split(". ") if c.strip()]
        for i, chunk in enumerate(chunks):
            if ctx.cancelled:
                break
            self.store.append(chunk)
            await ctx.report(current=i + 1, total=len(chunks), message="upserting")
        return IngestionResult(
            documents=1,
            chunks=len(chunks),
            upserted=len(chunks),
            collection=request.collection,
        )


class TypedRequest(BaseModel):
    url: str
    top_k: int = 5


class TypedIngestor(BaseIngestor[TypedRequest]):
    ingestor_name = "typed"

    async def ingest(self, request, ctx) -> IngestionResult:
        return IngestionResult(documents=1, output={"url": request.url, "k": request.top_k})


# =====================================================================
# BaseIngestor
# =====================================================================


class TestBaseIngestor:
    async def test_run_returns_result(self):
        ing = DocsIngestor()
        await ing.startup()
        result = await ing.run({"source": "a.pdf", "collection": "kb"})
        assert isinstance(result, IngestionResult)
        assert result.ingestor == "docs"
        assert result.chunks > 0
        assert result.collection == "kb"
        assert result.duration_ms >= 0

    async def test_default_input_schema(self):
        assert DocsIngestor().get_input_schema() is IngestionRequest

    async def test_custom_input_schema(self):
        assert TypedIngestor().get_input_schema() is TypedRequest

    async def test_typed_ingestor_coerces(self):
        ing = TypedIngestor()
        result = await ing.run({"url": "http://x", "top_k": 3})
        assert result.output == {"url": "http://x", "k": 3}

    async def test_info_and_health(self):
        ing = DocsIngestor()
        assert ing.info()["name"] == "docs"
        assert (await ing.health_check())["status"] == "not_ready"
        await ing.startup()
        assert (await ing.health_check())["status"] == "healthy"

    async def test_not_implemented(self):
        class Bare(BaseIngestor):
            ingestor_name = "bare"

        with pytest.raises(NotImplementedError):
            await Bare().ingest(IngestionRequest(), NullIngestionContext())


# =====================================================================
# Registry
# =====================================================================


class TestRegistry:
    def test_register_and_get(self):
        reg = IngestionRegistry()
        reg.register(DocsIngestor())
        assert reg.count == 1
        assert reg.get("docs") is not None
        assert reg.list_names() == ["docs"]

    def test_missing_dir_is_safe(self, tmp_path):
        reg = IngestionRegistry()
        reg.discover(tmp_path / "does_not_exist")
        assert reg.count == 0


# =====================================================================
# Embeddings hardening
# =====================================================================


class TestEmbeddings:
    def test_hash_embedder_deterministic(self):
        emb = HashEmbedder(dimension=32)
        v1 = emb.embed_query("hello world")
        v2 = emb.embed_query("hello world")
        assert v1 == v2
        assert len(v1) == 32

    def test_cache_per_provider(self):
        reset_embeddings()
        a = get_embeddings("hash", dimension=16)
        b = get_embeddings("hash", dimension=16)
        c = get_embeddings("hash", dimension=32)
        assert a is b
        assert a is not c  # different kwargs -> different instance

    def test_embed_documents(self):
        emb = get_embeddings("hash", dimension=8)
        vecs = emb.embed_documents(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 8 for v in vecs)


# =====================================================================
# REST router + task execution
# =====================================================================


def _build_app():
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from agentomatic.ingestion.router import create_ingestion_router
    from agentomatic.tasks import TargetType, TaskManager
    from agentomatic.tasks.dispatchers import make_ingestion_dispatcher
    from agentomatic.tasks.routes import create_task_router

    reg = IngestionRegistry()
    ing = DocsIngestor()
    reg.register(ing)

    mgr = TaskManager()
    mgr.register_dispatcher(TargetType.INGESTION, make_ingestion_dispatcher(reg))

    @asynccontextmanager
    async def lifespan(_app):
        await ing.startup()
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(
        create_ingestion_router(reg, task_manager=mgr, api_prefix="/api/v1"),
        prefix="/api/v1/ingestion",
    )
    app.include_router(create_task_router(mgr), prefix="/api/v1/tasks")
    return app


class TestIngestionRoutes:
    def test_list(self):
        from fastapi.testclient import TestClient

        with TestClient(_build_app()) as client:
            resp = client.get("/api/v1/ingestion")
            assert resp.status_code == 200
            assert resp.json()[0]["name"] == "docs"

    def test_run_sync(self):
        from fastapi.testclient import TestClient

        with TestClient(_build_app()) as client:
            resp = client.post(
                "/api/v1/ingestion/docs/run",
                json={"source": "report.pdf", "collection": "kb"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["chunks"] > 0
            assert body["collection"] == "kb"

    def test_run_async_as_task(self):
        import time

        from fastapi.testclient import TestClient

        with TestClient(_build_app()) as client:
            resp = client.post(
                "/api/v1/ingestion/docs/run/async",
                json={"source": "report.pdf"},
            )
            assert resp.status_code == 202
            task_id = resp.json()["id"]

            # Poll the unified task API until terminal.
            for _ in range(50):
                got = client.get(f"/api/v1/tasks/{task_id}").json()
                if got["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.02)
            assert got["status"] == "succeeded"
            result = client.get(f"/api/v1/tasks/{task_id}/result").json()
            assert result["result"]["chunks"] > 0


# =====================================================================
# Ingestion as a pipeline step
# =====================================================================


class TestIngestionPipelineStep:
    async def test_ingestion_step_runs_and_passes_output(self):
        from agentomatic.core.registry import AgentRegistry
        from agentomatic.pipelines.engine import PipelineEngine
        from agentomatic.pipelines.loader import PipelineLoader

        reg = IngestionRegistry()
        ing = DocsIngestor()
        await ing.startup()
        reg.register(ing)

        config = PipelineLoader.from_dict(
            {
                "name": "ingest_pipeline",
                "steps": [
                    {
                        "ingestion": "docs",
                        "name": "load",
                        "input": {"source": "report.pdf", "collection": "kb"},
                    },
                    {
                        "transform": "return {'total': ctx.current.get('chunks', 0)}",
                        "name": "summarize",
                    },
                ],
            }
        )

        engine = PipelineEngine(config, AgentRegistry(), ingestors=reg)
        assert engine.validate() == []
        result = await engine.run({})
        assert result.succeeded
        assert result.steps["load"].output["chunks"] > 0
        assert result.output["total"] > 0

    async def test_missing_ingestor_fails_validation(self):
        from agentomatic.core.registry import AgentRegistry
        from agentomatic.pipelines.engine import PipelineEngine
        from agentomatic.pipelines.loader import PipelineLoader

        config = PipelineLoader.from_dict(
            {"name": "p", "steps": [{"ingestion": "nope", "name": "x"}]}
        )
        engine = PipelineEngine(config, AgentRegistry(), ingestors=IngestionRegistry())
        errors = engine.validate()
        assert any("nope" in e for e in errors)


# =====================================================================
# Full platform wiring
# =====================================================================


class TestPlatformIntegration:
    def test_platform_wires_ingestion_end_to_end(self, tmp_path):
        import time

        from fastapi.testclient import TestClient

        from agentomatic import AgentPlatform

        platform = AgentPlatform(
            agents_dir=tmp_path / "agents",
            plugins_dir=tmp_path / "plugins",
            endpoints_dir=tmp_path / "endpoints",
            ingestion_dir=tmp_path / "ingestion",
            enable_studio=False,
        )
        platform.register_ingestor(DocsIngestor())
        app = platform.build()

        with TestClient(app) as client:
            # Root index advertises ingestors
            assert client.get("/").json()["ingestors"] == 1

            # Unified health includes the ingestor
            health = client.get("/health").json()
            assert "docs" in health["ingestors"]
            assert health["ingestor_count"] == 1

            # Ingestion REST is mounted
            listed = client.get("/api/v1/ingestion").json()
            assert listed[0]["name"] == "docs"

            # Sync run works
            sync = client.post(
                "/api/v1/ingestion/docs/run",
                json={"source": "a.pdf", "collection": "kb"},
            )
            assert sync.status_code == 200
            assert sync.json()["chunks"] > 0

            # Async run flows through the unified task system
            submitted = client.post(
                "/api/v1/ingestion/docs/run/async",
                json={"source": "a.pdf"},
            )
            assert submitted.status_code == 202
            task_id = submitted.json()["id"]
            for _ in range(50):
                got = client.get(f"/api/v1/tasks/{task_id}").json()
                if got["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.02)
            assert got["status"] == "succeeded"
