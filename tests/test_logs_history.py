"""Tests for invocation log history, analyser, and optimization run store."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agentomatic.core.manifest import AgentManifest
from agentomatic.core.platform import AgentPlatform
from agentomatic.logs.analyser import LogAnalyser, LogAnalysisResult
from agentomatic.logs.optimization_store import OptimizationRunStore
from agentomatic.logs.recorder import InvocationLogRecorder, truncate_for_storage
from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.storage.memory import MemoryStore

# ---------------------------------------------------------------------------
# truncate_for_storage
# ---------------------------------------------------------------------------


class TestTruncateForStorage:
    """Size-bounded JSON truncation helpers."""

    def test_short_string_passthrough(self) -> None:
        assert truncate_for_storage("hello") == "hello"

    def test_long_string_truncated(self) -> None:
        value = "x" * 200
        out = truncate_for_storage(value, max_string=50)
        assert isinstance(out, str)
        assert out.startswith("x" * 50)
        assert "truncated" in out

    def test_nested_dict_depth_cap(self) -> None:
        nested: dict[str, Any] = {"a": {"b": {"c": {"d": "deep"}}}}
        out = truncate_for_storage(nested, max_depth=2)
        assert out["a"]["b"] == "<truncated:max_depth>"

    def test_list_item_cap(self) -> None:
        out = truncate_for_storage(list(range(20)), max_list_items=3)
        assert len(out) == 4  # 3 kept + truncation marker
        assert "truncated" in str(out[-1])


# ---------------------------------------------------------------------------
# InvocationLogRecorder
# ---------------------------------------------------------------------------


class TestInvocationLogRecorder:
    """Nil-safe persistence of invoke/chat history."""

    @pytest.mark.asyncio
    async def test_noop_without_store(self) -> None:
        recorder = InvocationLogRecorder(None)
        assert recorder.enabled is False
        assert await recorder.record(agent_name="a", endpoint="invoke") is None

    @pytest.mark.asyncio
    async def test_records_into_memory_store(self) -> None:
        store = MemoryStore()
        recorder = InvocationLogRecorder(store)
        assert recorder.enabled is True

        entry = await recorder.record(
            agent_name="echo",
            endpoint="invoke",
            input_data={"query": "hi"},
            output_data={"response": "hello"},
            metadata={"prompt_version": "v1"},
            duration_ms=12.5,
            status="ok",
        )
        assert entry is not None
        assert entry["agent_name"] == "echo"
        assert entry["resource_type"] == "agent"
        assert entry["resource_name"] == "echo"
        assert entry["endpoint"] == "invoke"
        assert entry["input"]["query"] == "hi"
        assert entry["output"]["response"] == "hello"
        assert entry["status"] == "ok"

        listed = await store.list_invocation_logs(agent_name="echo")
        assert len(listed) == 1
        assert listed[0]["id"] == entry["id"]

    @pytest.mark.asyncio
    async def test_swallows_store_errors(self) -> None:
        store = MemoryStore()
        store.create_invocation_log = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        recorder = InvocationLogRecorder(store)
        assert await recorder.record(agent_name="x", endpoint="chat") is None


# ---------------------------------------------------------------------------
# LogAnalyser
# ---------------------------------------------------------------------------


class TestLogAnalyser:
    """Score / summary / status / recommendations from logs."""

    @pytest.mark.asyncio
    async def test_empty_logs_unknown_status(self) -> None:
        store = MemoryStore()
        analyser = LogAnalyser(store)
        result = await analyser.analyse("ghost")
        assert isinstance(result, LogAnalysisResult)
        assert result.status == "unknown"
        assert result.score is None
        assert result.recommendations
        assert result.analysis_id is not None

        latest = await store.get_latest_log_analysis("ghost")
        assert latest is not None
        assert latest["id"] == result.analysis_id

    @pytest.mark.asyncio
    async def test_heuristic_when_no_llm(self) -> None:
        store = MemoryStore()
        recorder = InvocationLogRecorder(store)
        await recorder.record(
            agent_name="svc",
            endpoint="invoke",
            input_data={"query": "ok"},
            output_data={"response": "fine"},
            status="ok",
            duration_ms=100,
        )
        await recorder.record(
            agent_name="svc",
            endpoint="invoke",
            input_data={"query": "bad"},
            status="error",
            error="timeout",
            duration_ms=9000,
        )

        analyser = LogAnalyser(store, llm=None)
        result = await analyser.analyse("svc")
        assert result.status in {"degraded", "failing", "healthy"}
        assert result.score is not None
        assert 0.0 <= result.score <= 1.0
        assert result.summary
        assert result.recommendations
        assert result.metadata.get("sample_size") == 2
        assert result.metadata.get("heuristic") is True

    @pytest.mark.asyncio
    async def test_mocked_llm_json_result(self) -> None:
        store = MemoryStore()
        await store.create_invocation_log(
            agent_name="bot",
            endpoint="chat",
            input_data={"query": "q"},
            output_data={"response": "a"},
            status="ok",
        )

        llm_payload = {
            "score": 0.82,
            "summary": "Looks healthy with minor latency spikes.",
            "status": "healthy",
            "recommendations": ["Watch p95 latency.", "Add retry on timeouts."],
        }

        with patch(
            "agentomatic.optimize.llm_caller.LLMCaller.call_with_json",
            new_callable=AsyncMock,
            return_value=llm_payload,
        ):
            analyser = LogAnalyser(store, llm="mock/model")
            result = await analyser.analyse("bot", persist=True)

        assert result.score == pytest.approx(0.82)
        assert result.status == "healthy"
        assert "latency" in result.summary.lower() or "Looks healthy" in result.summary
        assert len(result.recommendations) == 2
        assert result.metadata.get("heuristic") is False

    @pytest.mark.asyncio
    async def test_requires_store(self) -> None:
        analyser = LogAnalyser(None)
        with pytest.raises(RuntimeError, match="store"):
            await analyser.analyse("x")


# ---------------------------------------------------------------------------
# OptimizationRunStore
# ---------------------------------------------------------------------------


class TestOptimizationRunStore:
    """Auditable fit/retrain persistence via MemoryStore."""

    @pytest.mark.asyncio
    async def test_save_and_list_run(self) -> None:
        store = MemoryStore()
        run_store = OptimizationRunStore(store)
        assert run_store.enabled is True

        saved = await run_store.save_run(
            experiment_id="exp1",
            agent_name="tuner",
            baseline_score=0.4,
            best_score=0.7,
            prompt_versions={"baseline": "old", "best": "new"},
            score_history=[0.4, 0.55, 0.7],
            learnings=["prefer shorter answers"],
            artefacts={"n_trials": 3},
        )
        assert saved is not None
        assert saved["experiment_id"] == "exp1"

        fetched = await run_store.get(saved["id"])
        assert fetched is not None
        assert fetched["best_score"] == 0.7

        listed = await run_store.list(agent_name="tuner")
        assert len(listed) == 1
        assert listed[0]["learnings"] == ["prefer shorter answers"]

    @pytest.mark.asyncio
    async def test_save_fit_result(self) -> None:
        store = MemoryStore()
        run_store = OptimizationRunStore(store)
        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="best"),
            baseline_config=PromptRuntimeConfig(system_prompt="base"),
            best_score=0.9,
            baseline_score=0.5,
            metric_deltas={"exact": 0.4},
            suggestions=["add examples"],
            agent="fit_agent",
            experiment_id="fit_exp",
            score_history=[0.5, 0.7, 0.9],
            prompt_history=[{"round": 1, "what_worked": ["clarity"]}],
        )
        saved = await run_store.save_fit_result(result)
        assert saved is not None
        assert saved["agent_name"] == "fit_agent"
        assert saved["prompt_versions"]["best"] == "best"
        assert saved["artefacts"]["absolute_improvement"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_noop_without_store(self) -> None:
        run_store = OptimizationRunStore(None)
        assert run_store.enabled is False
        assert await run_store.save_run(experiment_id="e", agent_name="a") is None
        assert await run_store.list() == []


# ---------------------------------------------------------------------------
# Platform REST endpoints
# ---------------------------------------------------------------------------


async def _echo_node(state: dict[str, Any]) -> dict[str, Any]:
    """Minimal agent node used by platform wiring tests."""
    query = state.get("current_query", "")
    return {
        "response": f"echo:{query}",
        "output": {"response": f"echo:{query}"},
        "agent_type": "echo",
    }


def _platform_with_logs(*, analysis: bool = True) -> AgentPlatform:
    store = MemoryStore()
    platform = AgentPlatform(
        agents_dir=".",
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
    return platform


class TestLogsHistoryEndpoints:
    """End-to-end invoke recording + REST list/get/analyze."""

    def test_invoke_records_and_list_logs(self) -> None:
        platform = _platform_with_logs()
        with TestClient(platform.build()) as client:
            inv = client.post("/api/v1/echo/invoke", json={"query": "ping"})
            assert inv.status_code == 200, inv.text

            listed = client.get("/api/v1/echo/logs")
            assert listed.status_code == 200, listed.text
            body = listed.json()
            assert body["total"] >= 1
            assert body["logs"][0]["endpoint"] == "invoke"
            assert body["logs"][0]["input"]["query"] == "ping"

            log_id = body["logs"][0]["id"]
            single = client.get(f"/api/v1/echo/logs/{log_id}")
            assert single.status_code == 200
            assert single.json()["id"] == log_id

    def test_analyze_endpoint_heuristic(self) -> None:
        platform = _platform_with_logs(analysis=True)
        with TestClient(platform.build()) as client:
            client.post("/api/v1/echo/invoke", json={"query": "one"})
            client.post("/api/v1/echo/invoke", json={"query": "two"})

            resp = client.post("/api/v1/echo/logs/analyze", json={"persist": True})
            assert resp.status_code == 200, resp.text
            analysis = resp.json()["analysis"]
            assert analysis["status"] in {"healthy", "degraded", "failing", "unknown"}
            assert "recommendations" in analysis
            assert analysis.get("summary")

            latest = client.get("/api/v1/echo/logs/analysis")
            assert latest.status_code == 200
            assert latest.json()["analysis"]["id"] == analysis["id"]

    def test_analyze_refused_when_flag_off(self) -> None:
        platform = _platform_with_logs(analysis=False)
        with TestClient(platform.build()) as client:
            client.post("/api/v1/echo/invoke", json={"query": "x"})
            refused = client.post("/api/v1/echo/logs/analyze", json={})
            assert refused.status_code == 400
            detail = refused.json()["detail"]
            assert "disabled" in str(detail).lower() or "ALLOW_LOGSLLM" in str(detail)

    def test_logs_disabled_without_flag(self) -> None:
        store = MemoryStore()
        platform = AgentPlatform(
            agents_dir=".",
            store=store,
            logs_history=False,
            allow_logsllm_analysis=False,
            enable_studio=False,
            enable_telemetry=False,
            enable_tasks=False,
        )
        platform.register_agent(
            manifest=AgentManifest(name="echo", slug="echo", description="Echo"),
            node_fn=_echo_node,
        )
        with TestClient(platform.build()) as client:
            client.post("/api/v1/echo/invoke", json={"query": "no-log"})
            listed = client.get("/api/v1/echo/logs")
            assert listed.status_code == 400
            # Recorder must not write when flag is off.
            assert store._invocation_logs == {}


def test_platform_defers_memory_store_until_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a DB URL, logs_history falls back to MemoryStore at lifespan."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AGENTOMATIC_DB_URL", raising=False)
    platform = AgentPlatform(
        agents_dir=".",
        logs_history=True,
        enable_studio=False,
        enable_telemetry=False,
        enable_tasks=False,
    )
    # Deferred — construction must not install MemoryStore (that used to
    # preempt SQLAlchemyStore auto-derive from DATABASE_URL).
    assert platform.store is None
    with TestClient(platform.build()):
        assert platform.store is not None
        assert isinstance(platform.store, MemoryStore)


@pytest.mark.asyncio
async def test_logs_history_uses_sqlite_url_across_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When DATABASE_URL is set, invocation logs persist via SQLAlchemyStore."""
    from agentomatic.storage.sqlalchemy import SQLAlchemyStore

    db_path = tmp_path / "logs.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.delenv("AGENTOMATIC_DB_URL", raising=False)

    platform = AgentPlatform(
        agents_dir=".",
        logs_history=True,
        allow_logsllm_analysis=True,
        enable_studio=False,
        enable_telemetry=False,
        enable_tasks=False,
    )
    platform.register_agent(
        manifest=AgentManifest(name="echo", slug="echo", description="Echo"),
        node_fn=_echo_node,
    )
    with TestClient(platform.build()) as client:
        inv = client.post("/api/v1/echo/invoke", json={"query": "persist-me"})
        assert inv.status_code == 200, inv.text
        listed = client.get("/api/v1/echo/logs")
        assert listed.status_code == 200
        body = listed.json()
        assert body["total"] >= 1
        log_id = body["logs"][0]["id"]
        assert isinstance(platform.store, SQLAlchemyStore)

    # New process / new store instance against the same SQLite file.
    store2 = SQLAlchemyStore(url)
    await store2.initialize()
    try:
        rows = await store2.list_invocation_logs(agent_name="echo", limit=10)
        assert any(row["id"] == log_id for row in rows)
        assert any(
            (row.get("input") or {}).get("query") == "persist-me" for row in rows
        )
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_optimization_runs_survive_sqlite_restart(tmp_path: Path) -> None:
    """Retrain artefacts written to SQLAlchemyStore survive a new store open."""
    from agentomatic.logs.optimization_store import OptimizationRunStore
    from agentomatic.storage.sqlalchemy import SQLAlchemyStore

    url = f"sqlite+aiosqlite:///{tmp_path / 'retrain.db'}"
    store = SQLAlchemyStore(url)
    await store.initialize()
    try:
        run_store = OptimizationRunStore(store)
        saved = await run_store.save_run(
            experiment_id="exp-restart",
            agent_name="tuner",
            baseline_score=0.4,
            best_score=0.75,
            prompt_versions={"baseline": "old", "best": "new"},
            score_history=[0.4, 0.75],
            learnings=["prefer shorter answers"],
            artefacts={"n_trials": 2},
        )
        assert saved is not None
        run_id = saved["id"]
    finally:
        await store.close()

    store2 = SQLAlchemyStore(url)
    await store2.initialize()
    try:
        fetched = await store2.get_optimization_run(run_id)
        assert fetched is not None
        assert fetched["experiment_id"] == "exp-restart"
        assert fetched["best_score"] == pytest.approx(0.75)
        assert fetched["learnings"] == ["prefer shorter answers"]
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_persist_fit_result_via_fit_store() -> None:
    """fit_store.persist_fit_result writes through OptimizationRunStore."""
    from agentomatic.optimize.fit_store import persist_fit_result, set_fit_store

    store = MemoryStore()
    set_fit_store(store)
    try:
        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="best"),
            baseline_config=PromptRuntimeConfig(system_prompt="base"),
            best_score=0.8,
            baseline_score=0.5,
            experiment_id="abc123",
            agent="fitbot",
            score_history=[0.5, 0.8],
            suggestions=["tighten tone"],
            prompt_history=[{"round": 1, "score": 0.8, "what_worked": ["short"]}],
        )
        saved = await persist_fit_result(result)
        assert saved is not None
        runs = await store.list_optimization_runs(agent_name="fitbot")
        assert len(runs) == 1
        assert runs[0]["experiment_id"] == "abc123"
        assert runs[0]["best_score"] == 0.8
    finally:
        set_fit_store(None)
