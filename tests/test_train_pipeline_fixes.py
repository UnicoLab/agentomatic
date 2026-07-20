"""Regression tests for train/optimize + execution-mode bugfixes.

Covers:
- system_prompt_override / compiled_config prompt resolution
- honest LLM-as-judge failures (no fabricated mid-scale scores)
- PromptFitterBridge config extraction into compiled_config
- class-agent SSE streaming of per-node frames
- llm_judge metric resolution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentPlatform
from agentomatic.agents import BaseGraphAgent
from agentomatic.agents.optimizers import PromptFitterBridge
from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.optimize.metrics import (
    CompositeMetric,
    ExactMatchMetric,
    LLMJudgeMetric,
    WeightedMetric,
    resolve_metrics,
)


@dataclass
class _PromptState:
    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    used_prompt: str = ""


class _PromptAwareAgent(BaseGraphAgent[_PromptState]):
    """Minimal agent that surfaces the resolved system prompt in output."""

    agent_name = "prompt_aware"
    agent_description = "Prompt resolution test agent"
    agent_framework = "graph_agent"

    def __init__(self) -> None:
        super().__init__()
        self.prompt_manager = SimpleNamespace(
            get_prompt=lambda *a, **k: "BASE_PROMPT_FROM_MANAGER"
        )

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: _PromptState) -> _PromptState:
        prompt = self.resolve_system_prompt()
        state.used_prompt = prompt
        state.output = {
            "response": f"ok:{state.request}",
            "used_prompt": prompt,
        }
        return state

    def input_to_state(self, data: dict[str, Any]) -> _PromptState:
        return _PromptState(request=data.get("current_query") or data.get("query") or "")

    def state_to_output(self, state: _PromptState) -> dict[str, Any]:
        return state.output


# =====================================================================
# Prompt resolution
# =====================================================================


class TestPromptResolution:
    def test_override_beats_prompt_manager(self) -> None:
        agent = _PromptAwareAgent()
        out = agent.transform(
            {
                "query": "hi",
                "system_prompt_override": "OPTIMIZED_PROMPT",
            }
        )
        assert out["used_prompt"] == "OPTIMIZED_PROMPT"

    def test_compiled_config_used_after_fit(self) -> None:
        agent = _PromptAwareAgent()
        agent.compiled_config["system_prompt"] = "FITTED_PROMPT"
        out = agent.transform({"query": "hi"})
        assert out["used_prompt"] == "FITTED_PROMPT"

    def test_prompt_manager_fallback(self) -> None:
        agent = _PromptAwareAgent()
        out = agent.transform({"query": "hi"})
        assert out["used_prompt"] == "BASE_PROMPT_FROM_MANAGER"

    def test_optimize_invoke_applies_override(self, tmp_path) -> None:
        platform = AgentPlatform(
            agents_dir=tmp_path / "agents",
            plugins_dir=tmp_path / "plugins",
            endpoints_dir=tmp_path / "endpoints",
            enable_studio=False,
        )
        reg = _PromptAwareAgent().as_registered_agent()
        platform.register_agent(
            manifest=reg.manifest,
            node_fn=reg.node_fn,
            graph_fn=reg.graph_fn,
            class_instance=reg.class_instance,
        )
        with TestClient(platform.build()) as client:
            resp = client.post(
                "/api/v1/prompt_aware/optimize/invoke",
                json={
                    "query": "test",
                    "system_prompt_override": "REWRITE_V2",
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["response"] == "ok:test"
            meta = body.get("metadata") or {}
            assert meta.get("used_prompt") == "REWRITE_V2" or "REWRITE_V2" in str(body)


# =====================================================================
# PromptFitterBridge config extraction
# =====================================================================


class TestPromptFitterBridgeConfig:
    def test_extract_config_includes_system_prompt_without_attr(self) -> None:
        agent = _PromptAwareAgent()
        result = PromptFitResult(
            best_config=PromptRuntimeConfig(system_prompt="BEST_PROMPT"),
            baseline_config=PromptRuntimeConfig(system_prompt="BASE"),
            best_score=0.9,
            baseline_score=0.5,
        )
        config = PromptFitterBridge._extract_config(agent, result)
        assert config["system_prompt"] == "BEST_PROMPT"


# =====================================================================
# Honest LLM-as-judge
# =====================================================================


class TestHonestJudgeScores:
    @pytest.mark.asyncio
    async def test_llm_judge_failure_is_zero_not_half(self, monkeypatch) -> None:
        async def _boom(*_a, **_k):
            return {}

        monkeypatch.setattr(
            "agentomatic.optimize.llm_types.call_llm_json",
            _boom,
        )
        metric = LLMJudgeMetric(criteria="Is it good?")
        result = await metric.evaluate("q", "r", expected="e")
        assert result.score == 0.0
        assert result.metadata.get("evaluation_failed") is True
        assert "0.5" not in result.reason

    @pytest.mark.asyncio
    async def test_composite_all_failed_marks_evaluation_failed(self, monkeypatch) -> None:
        async def _fail(*_a, **_k):
            from agentomatic.optimize.metrics import EvalResult

            return EvalResult(
                metric_name="x",
                score=0.0,
                reason="failed",
                metadata={"evaluation_failed": True},
            )

        monkeypatch.setattr(LLMJudgeMetric, "evaluate", _fail)
        metric = CompositeMetric(
            metrics=[
                WeightedMetric("j1", LLMJudgeMetric(criteria="a"), weight=0.5),
                WeightedMetric("j2", LLMJudgeMetric(criteria="b"), weight=0.5),
            ]
        )
        result = await metric.evaluate("q", "r")
        assert result.score == 0.0
        assert result.metadata.get("evaluation_failed") is True

    def test_resolve_llm_judge_shorthand(self) -> None:
        metrics = resolve_metrics(["llm_judge:Is it polite?"])
        assert len(metrics) == 1
        assert isinstance(metrics[0], LLMJudgeMetric)
        assert "polite" in metrics[0].criteria


# =====================================================================
# Class-agent streaming
# =====================================================================


class TestClassAgentStreaming:
    def test_invoke_stream_emits_node_frames(self, tmp_path) -> None:
        platform = AgentPlatform(
            agents_dir=tmp_path / "agents",
            plugins_dir=tmp_path / "plugins",
            endpoints_dir=tmp_path / "endpoints",
            enable_studio=False,
        )
        reg = _PromptAwareAgent().as_registered_agent()
        platform.register_agent(
            manifest=reg.manifest,
            node_fn=reg.node_fn,
            graph_fn=reg.graph_fn,
            class_instance=reg.class_instance,
        )
        with TestClient(platform.build()) as client:
            resp = client.post(
                "/api/v1/prompt_aware/invoke/stream",
                json={"query": "streamed"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.text
            assert "[DONE]" in body
            assert "respond" in body  # per-node frame
            assert "ok:streamed" in body
            assert "error" not in body.lower()


# =====================================================================
# Fitter skips failed judge scores
# =====================================================================


class TestFitterEvalHonesty:
    @pytest.mark.asyncio
    async def test_evaluate_config_skips_failed_judge(self, monkeypatch) -> None:
        from agentomatic.optimize.dataset import Dataset
        from agentomatic.optimize.fitter import PromptFitter
        from agentomatic.optimize.runner import RunResult

        fitter = PromptFitter(agent="x", auto_report=False, max_trials=1)

        async def _fake_run_dataset(points, prompt_override=None, concurrency=5):
            return [
                RunResult(query="q1", response="r1", expected="e1"),
                RunResult(query="q2", response="r2", expected="e2"),
            ]

        fitter._runner.run_dataset = _fake_run_dataset  # type: ignore[method-assign]

        call_n = {"n": 0}

        async def _eval(query, response, expected=None, context=None):
            from agentomatic.optimize.metrics import EvalResult

            call_n["n"] += 1
            if call_n["n"] == 1:
                return EvalResult(
                    metric_name="llm_judge",
                    score=0.0,
                    reason="judge down",
                    metadata={"evaluation_failed": True},
                )
            return EvalResult(metric_name="exact", score=1.0, reason="ok")

        metric = ExactMatchMetric()
        monkeypatch.setattr(metric, "evaluate", _eval)

        avg, _dims, details = await fitter._evaluate_config(
            PromptRuntimeConfig(system_prompt="p"),
            Dataset.from_list(
                [
                    {"query": "q1", "expected_answer": "e1"},
                    {"query": "q2", "expected_answer": "e2"},
                ]
            ),
            metric,
        )
        # Only the successful point contributes → avg 1.0, not 0.5.
        assert avg == pytest.approx(1.0)
        assert any(d.get("error") for d in details)
