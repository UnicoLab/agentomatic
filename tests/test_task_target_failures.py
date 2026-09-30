# pyright: reportMissingParameterType=none
"""A task whose target reports failure must be marked ``failed``.

Studio's Task Board showed ``succeeded`` for a pipeline whose status was
``failed`` and for an endpoint whose every upstream call failed: both return
normally (with the failure inside the result), and the task manager only
looked at whether the dispatcher raised.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform
from agentomatic.tasks import TargetFailedError
from agentomatic.tasks.dispatchers import make_endpoint_dispatcher

FAILING = """
name: failing_pipe
steps:
  - name: boom
    transform: "raise ValueError('upstream exploded')"
  - name: after
    transform: "return {'done': True}"
"""

PARTIAL = """
name: partial_pipe
on_error: continue
steps:
  - name: boom
    transform: "raise ValueError('upstream exploded')"
  - name: after
    transform: "return {'done': True}"
"""


@pytest.fixture
def client(tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    pipelines = tmp_path / "pipelines"
    pipelines.mkdir()
    (pipelines / "failing_pipe.yaml").write_text(FAILING)
    (pipelines / "partial_pipe.yaml").write_text(PARTIAL)
    with TestClient(AgentPlatform(agents_dir=str(agents)).build()) as test_client:
        yield test_client


def _run_task(client: TestClient, pipeline: str) -> dict[str, Any]:
    response = client.post(
        "/api/v1/tasks",
        json={"target_type": "pipeline", "target": pipeline, "input": {}},
    )
    assert response.status_code == 202, response.text
    task_id = response.json()["id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        record = client.get(f"/api/v1/tasks/{task_id}").json()
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(0.05)
    raise AssertionError("task did not finish")


class TestPipelineTasks:
    def test_a_failed_pipeline_fails_its_task_and_keeps_the_details(self, client) -> None:
        record = _run_task(client, "failing_pipe")
        assert record["status"] == "failed"
        assert "Pipeline 'failing_pipe' failed" in record["error"]
        assert "boom" in record["error"]
        assert record["result"]["steps"]["boom"]["status"] == "failed"

    def test_a_partial_pipeline_still_succeeds(self, client) -> None:
        record = _run_task(client, "partial_pipe")
        assert record["status"] == "succeeded"
        assert record["result"]["status"] == "partial"


class TestEndpointTasks:
    async def test_all_upstreams_failing_raises(self) -> None:
        class _Schema:
            @staticmethod
            def model_validate(payload: Any) -> Any:
                return payload

        async def handle(request: Any) -> dict[str, Any]:
            return {"endpoint": "e", "ok": False, "results": [{"ok": False}], "aggregated": None}

        endpoint = SimpleNamespace(get_input_schema=lambda: _Schema, handle=handle)
        registry = SimpleNamespace(get=lambda name: endpoint, list_names=lambda: ["e"])
        ctx = SimpleNamespace(report=_noop)

        with pytest.raises(TargetFailedError) as info:
            await make_endpoint_dispatcher(registry)("e", {}, ctx)  # type: ignore[arg-type]
        assert info.value.result["ok"] is False


async def _noop(**_: Any) -> None:
    return None
