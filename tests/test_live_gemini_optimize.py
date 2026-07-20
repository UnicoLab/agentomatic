"""Live prompt-optimization smoke against Google Gemini.

Skipped automatically when ``GEMINI_API_KEY`` is unset. Run with::

    export GEMINI_API_KEY=…
    export HIVE_LLM_MODEL=gemini-3.1-flash-lite   # optional
    uv run pytest tests/test_live_gemini_optimize.py -q --override-ini='addopts='
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentomatic.agents import AgentDataset, AgentExample, BaseGraphAgent
from agentomatic.agents.metrics import ContainsTermsMetric, ExactKeyMatchMetric
from agentomatic.agents.optimizers import PromptFitterBridge
from agentomatic.optimize.briefing import looks_like_slm, resolve_rewrite_passes
from agentomatic.optimize.dataset import DataPoint, Dataset
from agentomatic.optimize.fitter import PromptFitter
from agentomatic.optimize.llm_caller import LLMCaller, parse_model_spec
from agentomatic.optimize.metrics import ContainsMetric, LLMJudgeMetric
from agentomatic.optimize.search_space import PromptSearchSpace

_RAW_MODEL = os.getenv("HIVE_LLM_MODEL") or os.getenv(
    "AGENTOMATIC_LIVE_GEMINI_MODEL", "gemini-3.1-flash-lite"
)
MODEL = _RAW_MODEL if _RAW_MODEL.startswith("gemini/") else f"gemini/{_RAW_MODEL}"
_BASELINE_PROMPT = "You are a vague assistant."


def _gemini_configured() -> bool:
    return bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    )


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _gemini_configured(), reason="GEMINI_API_KEY not set"),
]


@dataclass
class _State:
    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class _EchoAgent(BaseGraphAgent[_State]):
    """Deterministic agent: any prompt containing banana → OPT success."""

    agent_name = "live_gemini_echo"
    system_prompt = _BASELINE_PROMPT

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: _State) -> _State:
        prompt = self.resolve_system_prompt(default=self.system_prompt)
        improved = "banana" in prompt.lower()
        marker = "OPT" if improved else "BASE"
        extra = " banana" if improved else ""
        state.output = {
            "response": f"{marker}: answer to {state.request}{extra}",
            "used_prompt": prompt,
        }
        return state

    def input_to_state(self, data: dict[str, Any]) -> _State:
        return _State(request=data.get("current_query") or data.get("query") or "")

    def state_to_output(self, state: _State) -> dict[str, Any]:
        return state.output


def _opt_splits() -> tuple[Dataset, Dataset]:
    train = Dataset(
        points=[
            DataPoint(
                query="capital of france",
                expected_answer="OPT, banana",
                metadata={"split": "train"},
            ),
            DataPoint(
                query="2+2",
                expected_answer="OPT, banana",
                metadata={"split": "train"},
            ),
        ]
    )
    val = Dataset(
        points=[
            DataPoint(
                query="color of sky",
                expected_answer="OPT, banana",
                metadata={"split": "validation"},
            ),
            DataPoint(
                query="hello world",
                expected_answer="OPT, banana",
                metadata={"split": "validation"},
            ),
        ]
    )
    return train, val


def _agent_dataset() -> AgentDataset:
    examples = []
    for i, q in enumerate(["capital of france", "2+2", "color of sky", "hello world"]):
        split = "train" if i < 2 else ("validation" if i == 2 else "test")
        examples.append(
            AgentExample(
                id=f"e{i}",
                input={"current_query": q},
                expected_output={"response": "OPT, banana"},
                split=split,
                metadata={"split": split},
            )
        )
    return AgentDataset(name="live_gemini", examples=examples)


def test_gemini_model_is_llm_multipass_not_slm() -> None:
    assert parse_model_spec(MODEL)[0] == "gemini"
    assert looks_like_slm(MODEL) is False
    assert resolve_rewrite_passes(MODEL) == 2


@pytest.mark.asyncio
async def test_gemini_caller_clean_text_and_json() -> None:
    text = await LLMCaller.call(
        MODEL,
        "Reply with exactly the word OK and nothing else.",
        temperature=0.0,
        max_tokens=16,
    )
    assert "OK" in text.upper()
    data = await LLMCaller.call_with_json(
        MODEL,
        'Return JSON {"score": 0.5, "reason": "x"}.',
        temperature=0.0,
        max_retries=1,
    )
    assert "score" in data


@pytest.mark.asyncio
async def test_gemini_multipass_rewrite_improves() -> None:
    agent = _EchoAgent()
    train, val = _opt_splits()
    fitter = PromptFitter(
        agent="live_gemini_echo",
        task_model=MODEL,
        rewrite_model=MODEL,
        optimizer="rewrite",
        max_trials=4,
        concurrency=1,
        auto_report=False,
        local_agent=agent,
        search_space=PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_model_params=False,
            optimize_few_shot=False,
        ),
        # Explicit: frontier Gemini → draft + self-check
        llm_multipass=True,
        llm_default_passes=2,
    )
    result = await fitter.fit(train, val, ContainsMetric())
    assert result.best_score >= result.baseline_score - 1e-9
    assert result.best_score > result.baseline_score
    assert "banana" in (result.best_config.system_prompt or "").lower()


@pytest.mark.asyncio
async def test_gemini_gepa_holds_or_improves() -> None:
    agent = _EchoAgent()
    train, val = _opt_splits()
    fitter = PromptFitter(
        agent="live_gemini_echo",
        task_model=MODEL,
        rewrite_model=MODEL,
        optimizer="gepa",
        max_trials=4,
        concurrency=1,
        auto_report=False,
        local_agent=agent,
        search_space=PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_model_params=False,
            optimize_few_shot=False,
        ),
    )
    result = await fitter.fit(train, val, ContainsMetric())
    assert result.best_score >= result.baseline_score - 1e-9
    assert result.best_config.system_prompt is not None


@pytest.mark.asyncio
async def test_gemini_llm_judge() -> None:
    judge = LLMJudgeMetric(
        criteria="Is the response short and helpful?",
        model=MODEL,
    )
    r = await judge.evaluate("What is 2+2?", "4", expected="4")
    assert 0.0 <= r.score <= 1.0
    assert not r.metadata.get("evaluation_failed")


def test_gemini_prompt_fitter_bridge_rewrite() -> None:
    agent = _EchoAgent()
    ds = _agent_dataset()
    metrics = [
        ExactKeyMatchMetric(["response"]),
        ContainsTermsMetric(["banana"]),
    ]
    bridge = PromptFitterBridge(
        agent_name="live_gemini_echo",
        task_model=MODEL,
        rewrite_model=MODEL,
        max_trials=4,
        metric=ContainsMetric(),
        optimizer="rewrite",
        search_space=PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_model_params=False,
            optimize_few_shot=False,
        ),
        auto_report=False,
        concurrency=1,
        llm_multipass=True,
    )
    agent.compile(ds, metrics, optimizer=bridge)
    agent.fit(ds, epochs=1, verbose=0, optimize_mode="rewrite", max_trials=4)
    assert getattr(agent, "_last_optimize_status", None) == "ok"
    result = getattr(agent, "_last_fit_result", None)
    assert result is not None
    assert result.best_score > result.baseline_score
    prompt = agent.compiled_config.get("system_prompt") or ""
    assert "banana" in prompt.lower()
