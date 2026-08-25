# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
"""A condition that cannot be evaluated must fail, not silently skip.

Regression: ``_evaluate_condition`` caught every exception and returned
``False``, so a typo, a renamed step, or ``$.`` mapping syntax used where a
``ctx`` expression belongs made the branch *silently never fire* — while the
pipeline still reported ``status: success``. A conditional branch that never
runs, under a green status, is the kind of defect that reaches production and
stays there.

Evaluating falsy is a routing decision and still skips. Failing to evaluate at
all is a defect in the pipeline and now surfaces.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform

GOOD = """
name: good_condition
steps:
  - name: seed
    transform: "return {'n': 4}"
  - name: runs
    upstreams: [seed]
    condition: "ctx.get_step_output('seed').get('n', 0) > 2"
    transform: "return {'branch': 'long'}"
  - name: skipped
    upstreams: [seed]
    condition: "ctx.get_step_output('seed').get('n', 0) <= 2"
    transform: "return {'branch': 'short'}"
"""

#: `$.` is mapping syntax, not a Python expression — a very easy mistake.
MAPPING_SYNTAX = """
name: mapping_syntax_condition
steps:
  - name: seed
    transform: "return {'n': 4}"
  - name: branch
    upstreams: [seed]
    condition: "$.seed.n > 2"
    transform: "return {'branch': 'long'}"
"""

#: Names a step that does not exist.
UNKNOWN_STEP = """
name: unknown_step_condition
steps:
  - name: seed
    transform: "return {'n': 4}"
  - name: branch
    upstreams: [seed]
    condition: "ctx.get_step_output('typo').get('n') > 2"
    transform: "return {'branch': 'long'}"
"""

#: Valid Python, but only discoverable as broken once it runs.
SKIPPABLE = """
name: skippable_condition
steps:
  - name: seed
    transform: "return {'n': 4}"
  - name: branch
    upstreams: [seed]
    on_error: skip
    condition: "ctx.get_step_output('typo').get('n') > 2"
    transform: "return {'branch': 'long'}"
  - name: after
    upstreams: [seed]
    transform: "return {'reached_end': True}"
"""


def _platform(tmp_path, *pipelines: str):
    """Build a platform serving the given pipelines."""
    agents = tmp_path / "agents"
    agents.mkdir(exist_ok=True)
    pipelines_dir = tmp_path / "pipelines"
    pipelines_dir.mkdir(exist_ok=True)
    for text in pipelines:
        name = text.strip().splitlines()[0].split(":", 1)[1].strip()
        (pipelines_dir / f"{name}.yaml").write_text(text)
    platform = AgentPlatform(agents_dir=str(agents))

    async def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": "ok"}

    platform.register_agent(
        AgentManifest(name="echo_agent", slug="echo_agent", description="echo"),
        node_fn=echo,
    )
    return platform


class TestWorkingConditionsStillRoute:
    def test_a_true_branch_runs_and_a_false_branch_skips(self, tmp_path) -> None:
        with TestClient(_platform(tmp_path, GOOD).build()) as client:
            body = client.post("/api/v1/pipelines/good_condition/run", json={"input": {}}).json()

            assert body["status"] == "success", body
            assert body["steps"]["runs"]["status"] == "success"
            assert body["steps"]["skipped"]["status"] == "skipped"
            # A falsy condition skips without an error attached.
            assert not body["steps"]["skipped"].get("error")


class TestBrokenConditionsSurface:
    def test_a_runtime_unevaluable_condition_fails_the_pipeline(self, tmp_path) -> None:
        """Previously: the step was skipped and the pipeline reported success."""
        with TestClient(_platform(tmp_path, UNKNOWN_STEP).build()) as client:
            body = client.post(
                "/api/v1/pipelines/unknown_step_condition/run", json={"input": {}}
            ).json()

            assert body["status"] == "failed", body
            assert body["steps"]["branch"]["status"] == "failed"
            assert "could not be evaluated" in body["steps"]["branch"]["error"]

    def test_the_error_says_what_to_write_instead(self, tmp_path) -> None:
        """`ctx` is the contract; the message has to name it."""
        with TestClient(_platform(tmp_path, UNKNOWN_STEP).build()) as client:
            body = client.post(
                "/api/v1/pipelines/unknown_step_condition/run", json={"input": {}}
            ).json()

            assert "ctx" in body["steps"]["branch"]["error"]

    def test_mapping_syntax_is_refused_before_the_pipeline_runs(self, tmp_path) -> None:
        """`$.` is mapping syntax, so it is a syntax error validation can see."""
        with TestClient(_platform(tmp_path, MAPPING_SYNTAX).build()) as client:
            body = client.get("/api/v1/pipelines/mapping_syntax_condition/validate").json()

            assert body.get("valid") is False, body
            assert any("condition" in e.lower() for e in body.get("errors", [])), body

    def test_running_a_syntax_invalid_condition_is_rejected(self, tmp_path) -> None:
        """Never executed as if the branch simply did not apply."""
        with TestClient(_platform(tmp_path, MAPPING_SYNTAX).build()) as client:
            resp = client.post(
                "/api/v1/pipelines/mapping_syntax_condition/run", json={"input": {}}
            )

            assert resp.status_code == 422, resp.text
            assert "condition" in resp.text.lower()

    def test_on_error_skip_still_lets_the_pipeline_continue(self, tmp_path) -> None:
        """The step's own policy decides — the failure is not forced fatal."""
        with TestClient(_platform(tmp_path, SKIPPABLE).build()) as client:
            body = client.post(
                "/api/v1/pipelines/skippable_condition/run", json={"input": {}}
            ).json()

            assert body["steps"]["branch"]["status"] == "failed"
            assert body["steps"]["after"]["status"] == "success"
