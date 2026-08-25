# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
"""Sub-pipelines must compose the pipelines the platform already serves.

Regression: the platform never passed ``sub_pipelines=`` when building the
pipeline router, so the pool was always empty and *every* ``sub_pipeline``
step failed validation with "Sub-pipeline 'x' not found" — naming a pipeline
that was discovered, served, and runnable at its own route. The step type was
parsed, validated and implemented, and could not be used at all.

Making it work introduces the hazard it never had: a pipeline may now
reference itself, directly or around a longer loop, and pipelines are
editable over HTTP. So cycles are rejected by validation and depth is bounded
at runtime for the case validation could not have seen.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform
from agentomatic.pipelines.engine import MAX_SUB_PIPELINE_DEPTH, PipelineEngine
from agentomatic.pipelines.loader import PipelineLoader

INNER = """
name: inner_flow
steps:
  - name: greet
    agent: echo_agent
    input:
      current_query: $.input.query
"""

OUTER = """
name: outer_flow
steps:
  - name: nested
    sub_pipeline: inner_flow
    input:
      query: $.input.query
  - name: after
    upstreams: [nested]
    transform: "return {'wrapped': True}"
"""

SELF_REFERENTIAL = """
name: loop_a
steps:
  - name: again
    sub_pipeline: loop_a
"""

MUTUAL_A = """
name: mutual_a
steps:
  - name: to_b
    sub_pipeline: mutual_b
"""

MUTUAL_B = """
name: mutual_b
steps:
  - name: to_a
    sub_pipeline: mutual_a
"""


def _platform(tmp_path, *pipelines: str):
    """Build a platform serving one echo agent and the given pipelines."""
    agents = tmp_path / "agents"
    agents.mkdir(exist_ok=True)
    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir(exist_ok=True)
    for text in pipelines:
        name = text.strip().splitlines()[0].split(":", 1)[1].strip()
        (pipelines_dir / f"{name}.yaml").write_text(text)

    platform = AgentPlatform(agents_dir=str(agents))

    async def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": f"echo:{state.get('current_query', '')}"}

    platform.register_agent(
        AgentManifest(name="echo_agent", slug="echo_agent", description="echo"),
        node_fn=echo,
    )
    return platform


class TestDiscoveredPipelinesAreComposable:
    def test_a_sub_pipeline_step_validates(self, tmp_path) -> None:
        """Regression: this returned "Sub-pipeline 'inner_flow' not found"."""
        with TestClient(_platform(tmp_path, INNER, OUTER).build()) as client:
            resp = client.get("/api/v1/pipelines/outer_flow/validate")

            assert resp.status_code == 200, resp.text
            assert resp.json().get("valid") is True, resp.text

    def test_a_sub_pipeline_step_runs(self, tmp_path) -> None:
        with TestClient(_platform(tmp_path, INNER, OUTER).build()) as client:
            resp = client.post(
                "/api/v1/pipelines/outer_flow/run", json={"input": {"query": "hi"}}
            )

            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["status"] == "success", body
            assert body["steps"]["nested"]["status"] == "success"

    def test_the_inner_pipeline_still_serves_its_own_route(self, tmp_path) -> None:
        """Being usable as a sub-pipeline must not remove it from the API."""
        with TestClient(_platform(tmp_path, INNER, OUTER).build()) as client:
            resp = client.post(
                "/api/v1/pipelines/inner_flow/run", json={"input": {"query": "hi"}}
            )

            assert resp.status_code == 200
            assert resp.json()["status"] == "success"


class TestCyclesAreRefused:
    def test_a_self_referencing_pipeline_fails_validation(self, tmp_path) -> None:
        with TestClient(_platform(tmp_path, SELF_REFERENTIAL).build()) as client:
            body = client.get("/api/v1/pipelines/loop_a/validate").json()

            assert body.get("valid") is False, body
            assert any("cycle" in e.lower() for e in body.get("errors", [])), body

    def test_a_mutual_cycle_fails_validation(self, tmp_path) -> None:
        with TestClient(_platform(tmp_path, MUTUAL_A, MUTUAL_B).build()) as client:
            body = client.get("/api/v1/pipelines/mutual_a/validate").json()

            assert body.get("valid") is False, body
            assert any("cycle" in e.lower() for e in body.get("errors", [])), body

    def test_running_a_cycle_is_refused_rather_than_recursing(self, tmp_path) -> None:
        """It must not take the worker down, whatever validation decided."""
        with TestClient(_platform(tmp_path, SELF_REFERENTIAL).build()) as client:
            resp = client.post("/api/v1/pipelines/loop_a/run", json={"input": {}})

            # Either refused up front, or bounded — never unbounded recursion.
            assert resp.status_code in (200, 422)
            if resp.status_code == 200:
                assert resp.json()["status"] == "failed"


class TestDepthIsBoundedAtRuntime:
    @pytest.mark.asyncio
    async def test_depth_beyond_the_cap_fails_the_step(self, tmp_path) -> None:
        """The backstop for a cycle created after the engine was built."""
        inner = PipelineLoader.from_yaml_string(SELF_REFERENTIAL)
        platform = _platform(tmp_path)
        platform.build()
        engine = PipelineEngine(
            inner,
            platform._registry,  # noqa: SLF001 - exercising the guard directly
            {"loop_a": inner},
            depth=MAX_SUB_PIPELINE_DEPTH,
        )

        result = await engine.run({})

        assert not result.succeeded
        assert "nesting exceeded" in (result.error or "").lower()

    def test_the_cap_is_a_sane_bound(self) -> None:
        assert 1 < MAX_SUB_PIPELINE_DEPTH <= 50
