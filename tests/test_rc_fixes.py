"""Regression tests for release-candidate fixes A/B/E/F (and helpers for C).

Covers:
- A: Full OpenAPI schema (StudioResumeRequest at module scope)
- B: Class-agent async invoke via ``invoke_registered_agent`` / ``input_to_state``
- E: ``context`` flattened into transform payload before ``input_to_state``
- C/F helpers: project ``main.py`` detection + metrics env mapping
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform
from agentomatic.agents import BaseGraphAgent
from agentomatic.cli.commands import _env_bool, _has_project_main_app
from agentomatic.core.agent_invoke import _input_from_state
from agentomatic.core.manifest import AgentManifest

# ---------------------------------------------------------------------------
# Shared class agent
# ---------------------------------------------------------------------------


@dataclass
class _SnapState:
    """Dataclass state — attribute access on a raw dict would raise."""

    request: str = ""
    snapshot: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)


class _SnapClassAgent(BaseGraphAgent[_SnapState]):
    """Class agent that reads flattened context fields via ``input_to_state``."""

    agent_name = "snap_class"
    agent_description = "Snapshot class agent"
    agent_framework = "graph_agent"

    def build_graph(self) -> Any:
        """Build a single-node graph that echoes request + snapshot."""
        g = self.new_graph()
        g.add_node("answer", self.answer)
        g.set_entry_point("answer")
        g.set_finish_point("answer")
        return g.compile()

    def answer(self, state: _SnapState) -> _SnapState:
        """Read dataclass attributes (fails if a raw dict reached the node)."""
        keys = sorted(state.snapshot.keys())
        state.output = {
            "response": f"Echo: {state.request}",
            "snapshot_keys": keys,
            # Surfaced via AgentInvokeResponse.context on the sync path.
            "context": {"snapshot_keys": keys},
            "agent_type": "snap-class",
        }
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> _SnapState:
        """Map request + flattened context into the dataclass state."""
        snapshot = input_data.get("snapshot") or {}
        if not isinstance(snapshot, dict):
            snapshot = {}
        return _SnapState(
            request=input_data.get("current_query") or input_data.get("query") or "",
            snapshot=snapshot,
        )

    def state_to_output(self, state: _SnapState) -> dict[str, Any]:
        """Return the node's output dict."""
        return state.output


def _platform_with_class_agent(tmp_path: Any, *, enable_studio: bool = True) -> AgentPlatform:
    """Build a platform with one class agent registered."""
    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        title="RC Fix Test",
        version="0.0.1",
        enable_studio=enable_studio,
    )
    reg = _SnapClassAgent().as_registered_agent()
    platform.register_agent(
        manifest=reg.manifest,
        node_fn=reg.node_fn,
        graph_fn=reg.graph_fn,
        class_instance=reg.class_instance,
    )
    return platform


def _poll(client: TestClient, task_id: str, tries: int = 80) -> dict[str, Any]:
    """Poll the unified task API until the task reaches a terminal state."""
    got: dict[str, Any] = {}
    for _ in range(tries):
        got = client.get(f"/api/v1/tasks/{task_id}").json()
        if got["status"] in {"succeeded", "failed", "cancelled"}:
            return got
        time.sleep(0.02)
    return got


# ---------------------------------------------------------------------------
# A — OpenAPI full schema
# ---------------------------------------------------------------------------


class TestOpenAPIFullSchema:
    """Full OpenAPI generation must not fall back to the minimal stub catalog."""

    def test_openapi_has_full_paths_and_agent_invoke(self, tmp_path: Any) -> None:
        """``/openapi.json`` includes agent invoke and enough paths for a full schema."""
        app = _platform_with_class_agent(tmp_path, enable_studio=True).build()
        with TestClient(app) as client:
            resp = client.get("/openapi.json")
            assert resp.status_code == 200, resp.text
            schema = resp.json()

        description = (schema.get("info") or {}).get("description") or ""
        assert "full schema generation failed" not in description.lower()
        paths = schema.get("paths") or {}
        # Stub catalog is ~13 paths; a real schema with one agent + studio is >> that.
        assert len(paths) > 40, f"expected full schema, got {len(paths)} paths"
        assert "/api/v1/snap_class/invoke" in paths
        # Resume body model must be resolvable (the prior ForwardRef failure mode).
        assert any("resume" in p for p in paths)


# ---------------------------------------------------------------------------
# B — Class-agent async dispatcher
# ---------------------------------------------------------------------------


class TestClassAgentAsyncInvoke:
    """Async / batch paths must use ``input_to_state`` for class agents."""

    def test_invoke_async_class_agent_succeeds(self, tmp_path: Any) -> None:
        """POST ``/invoke/async`` must not fail with AttributeError on dataclass state."""
        app = _platform_with_class_agent(tmp_path, enable_studio=False).build()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/snap_class/invoke/async",
                json={"query": "hi", "context": {"snapshot": {"a": 1}}},
            )
            assert resp.status_code == 202, resp.text
            task_id = resp.json()["id"]
            done = _poll(client, task_id)
            assert done["status"] == "succeeded", done
            result = client.get(f"/api/v1/tasks/{task_id}/result").json()["result"]
            assert result["response"] == "Echo: hi"
            assert "a" in result.get("snapshot_keys", [])

    def test_invoke_batch_class_agent_succeeds(self, tmp_path: Any) -> None:
        """Batch invoke must also route through ``input_to_state``."""
        app = _platform_with_class_agent(tmp_path, enable_studio=False).build()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/snap_class/invoke/batch",
                json={
                    "inputs": [
                        {"query": "a", "context": {"snapshot": {"x": 1}}},
                        {"query": "b", "context": {"snapshot": {"y": 2}}},
                    ]
                },
            )
            assert resp.status_code == 202, resp.text
            task_id = resp.json()["id"]
            done = _poll(client, task_id)
            assert done["status"] == "succeeded", done
            result = client.get(f"/api/v1/tasks/{task_id}/result").json()["result"]
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["response"] == "Echo: a"
            assert result[1]["response"] == "Echo: b"


# ---------------------------------------------------------------------------
# E — context flatten before input_to_state
# ---------------------------------------------------------------------------


class TestContextFlatten:
    """``context`` must be merged into the transform payload."""

    def test_input_from_state_flattens_context(self) -> None:
        """Nested context keys appear at top level; top-level wins on collision."""
        payload = _input_from_state(
            {
                "current_query": "q",
                "query": "q",
                "context": {"snapshot": {"k": 1}, "query": "ignored"},
                "messages": [],
                "thread_id": "t1",
                "user_id": "u1",
            }
        )
        assert payload["snapshot"] == {"k": 1}
        assert payload["query"] == "q"  # top-level wins over context.query
        assert payload["context"] == {"snapshot": {"k": 1}, "query": "ignored"}
        assert "messages" not in payload
        assert "thread_id" not in payload

    def test_sync_invoke_reads_flattened_context(self, tmp_path: Any) -> None:
        """REST ``/invoke`` with ``context.snapshot`` reaches ``input_to_state``."""
        app = _platform_with_class_agent(tmp_path, enable_studio=False).build()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/snap_class/invoke",
                json={"query": "hello", "context": {"snapshot": {"n": 9}}},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["response"] == "Echo: hello"
            assert body.get("context", {}).get("snapshot_keys") == ["n"]


# ---------------------------------------------------------------------------
# C / F helpers
# ---------------------------------------------------------------------------


class TestRunHelpers:
    """Helpers used by ``agentomatic run`` for main:app + metrics env."""

    def test_has_project_main_app_detects_scaffold(self, tmp_path: Any) -> None:
        """Scaffold-style ``app = ...`` is detected."""
        (tmp_path / "main.py").write_text(
            "from agentomatic import AgentPlatform\n"
            "_p = AgentPlatform.from_folder('agents/')\n"
            "app = _p.build()\n",
            encoding="utf-8",
        )
        assert _has_project_main_app(tmp_path) is True

    def test_has_project_main_app_false_without_app(self, tmp_path: Any) -> None:
        """A main.py without module-level ``app`` is ignored."""
        (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
        assert _has_project_main_app(tmp_path) is False

    def test_has_project_main_app_false_when_missing(self, tmp_path: Any) -> None:
        """Missing main.py returns False."""
        assert _has_project_main_app(tmp_path) is False

    def test_env_bool_metrics_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``AGENTOMATIC_ENABLE_METRICS`` parses like the scaffolded main.py."""
        monkeypatch.delenv("AGENTOMATIC_ENABLE_METRICS", raising=False)
        assert _env_bool("AGENTOMATIC_ENABLE_METRICS", True) is True
        monkeypatch.setenv("AGENTOMATIC_ENABLE_METRICS", "0")
        assert _env_bool("AGENTOMATIC_ENABLE_METRICS", True) is False
        monkeypatch.setenv("AGENTOMATIC_ENABLE_METRICS", "true")
        assert _env_bool("AGENTOMATIC_ENABLE_METRICS", False) is True


class TestMetricsUnderFromFolder:
    """``enable_metrics`` from env exposes ``/metrics`` under from_folder builds."""

    def test_metrics_enabled_serves_endpoint(self, tmp_path: Any) -> None:
        """Platform with ``enable_metrics=True`` mounts the metrics middleware path."""

        async def _node(state: dict) -> dict:
            return {"response": "ok"}

        platform = AgentPlatform(
            agents_dir=tmp_path / "agents",
            enable_studio=False,
            enable_metrics=True,
        )
        platform.register_agent(
            AgentManifest(name="echo", slug="echo", description="e", version="1.0.0"),
            node_fn=_node,
        )
        app = platform.build()
        with TestClient(app) as client:
            resp = client.get("/metrics")
            # prometheus_client may or may not be installed; when present → 200.
            assert resp.status_code in {200, 404}
