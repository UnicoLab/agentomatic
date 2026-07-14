"""Tests for the unified platform status endpoint + HTML dashboard."""

from __future__ import annotations

from agentomatic import AgentPlatform, BaseIngestor, IngestionResult


class _StatusDocsIngestor(BaseIngestor):
    ingestor_name = "status_docs"

    async def setup(self) -> None:
        self.ready = True

    async def ingest(self, request, ctx) -> IngestionResult:
        return IngestionResult(documents=1, chunks=1, upserted=1)


def _platform(tmp_path) -> AgentPlatform:
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
        ingestion_dir=tmp_path / "ingestion",
        enable_studio=False,
    )
    platform.register_ingestor(_StatusDocsIngestor())
    return platform


class TestStatusJson:
    def test_status_payload_shape(self, tmp_path):
        from fastapi.testclient import TestClient

        with TestClient(_platform(tmp_path).build()) as client:
            resp = client.get("/api/v1/status")
            assert resp.status_code == 200
            d = resp.json()

            assert d["status"] in {"healthy", "degraded"}
            assert set(d["platform"]) >= {"name", "version", "uptime_seconds"}
            assert d["platform"]["uptime_seconds"] >= 0

            # All resource sections present
            for key in ("agents", "plugins", "endpoints", "ingestors", "pipelines"):
                assert key in d["resources"]
                assert key in d["summary"]

            # Our ingestor shows up
            assert "status_docs" in d["resources"]["ingestors"]["items"]
            assert d["summary"]["ingestors"]["total"] == 1

    def test_status_includes_task_stats(self, tmp_path):
        from fastapi.testclient import TestClient

        with TestClient(_platform(tmp_path).build()) as client:
            d = client.get("/api/v1/status").json()
            assert d["tasks"]["enabled"] is True
            assert "by_status" in d["tasks"]
            assert d["tasks"]["max_concurrency"] >= 1
            assert "ingestion" in d["tasks"]["supported_targets"]

    def test_root_advertises_status(self, tmp_path):
        from fastapi.testclient import TestClient

        with TestClient(_platform(tmp_path).build()) as client:
            assert client.get("/").json()["status"] == "/status"


class TestStatusHtml:
    def test_dashboard_served(self, tmp_path):
        from fastapi.testclient import TestClient

        with TestClient(_platform(tmp_path).build()) as client:
            resp = client.get("/status")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
            body = resp.text
            assert "Agentomatic" in body
            # The API prefix placeholder is interpolated
            assert "/api/v1/status" in body
            assert "__API_PREFIX__" not in body


class TestTaskManagerStats:
    async def test_stats_snapshot(self):
        from agentomatic.tasks import TaskManager

        mgr = TaskManager(max_concurrency=4)
        stats = await mgr.stats()
        assert stats["total"] == 0
        assert stats["max_concurrency"] == 4
        assert "queued" in stats["by_status"]
        assert stats["running"] == 0
