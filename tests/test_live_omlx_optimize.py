"""Live prompt-optimization smoke against a local oMLX server.

Skipped automatically when oMLX is unreachable. Run with::

    OMLX_API_KEY=… OMLX_BASE_URL=http://127.0.0.1:8000/v1 \\
      uv run pytest tests/test_live_omlx_optimize.py -q --override-ini='addopts='
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from agentomatic.agents import AgentDataset, AgentExample, BaseGraphAgent
from agentomatic.agents.metrics import ContainsTermsMetric, ExactKeyMatchMetric, WeightedMetric
from agentomatic.agents.optimizers import GridSearchOptimizer, PromptFitterBridge
from agentomatic.optimize.dataset import DataPoint, Dataset
from agentomatic.optimize.fitter import PromptFitter
from agentomatic.optimize.fitter_optimizers import resolve_fitter_optimizer
from agentomatic.optimize.llm_caller import LLMCaller, parse_model_spec
from agentomatic.optimize.metrics import ContainsMetric, LLMJudgeMetric
from agentomatic.optimize.search_space import PromptSearchSpace

MODEL = os.getenv("AGENTOMATIC_LIVE_MODEL", "omlx/Qwen3.5-9B-MLX-4bit")
OMLX_BASE = os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
OMLX_KEY = os.getenv("OMLX_API_KEY", "kurwamac")
_BASELINE_PROMPT = "You are a vague assistant."


def _omlx_available() -> bool:
    try:
        resp = httpx.get(
            f"{OMLX_BASE.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {OMLX_KEY}"},
            timeout=2.0,
        )
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _omlx_available(), reason="oMLX server not reachable"),
]


@pytest.fixture(autouse=True)
def _omlx_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMLX_API_KEY", OMLX_KEY)
    monkeypatch.setenv("OMLX_BASE_URL", OMLX_BASE)


@dataclass
class _State:
    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class _EchoAgent(BaseGraphAgent[_State]):
    """Deterministic agent: any non-baseline prompt → OPT + banana token."""

    agent_name = "live_echo"
    system_prompt = _BASELINE_PROMPT

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("respond", self.respond)
        g.set_entry_point("respond")
        g.set_finish_point("respond")
        return g.compile()

    def respond(self, state: _State) -> _State:
        prompt = self.resolve_system_prompt(default=self.system_prompt)
        # Only the optimization target token marks success — default fitter
        # baselines ("helpful AI assistant") must still score 0.
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


def _agent_dataset() -> AgentDataset:
    """AgentDataset whose optimize expected_answer is ContainsMetric keywords."""
    examples = []
    for i, q in enumerate(["capital of france", "2+2", "color of sky", "hello world"]):
        split = "train" if i < 2 else ("validation" if i == 2 else "test")
        examples.append(
            AgentExample(
                id=f"e{i}",
                input={"current_query": q},
                # Comma-separated keywords for ContainsMetric via to_datapoint().
                expected_output={"response": "OPT, banana"},
                split=split,
                metadata={"split": split},
            )
        )
    return AgentDataset(name="live", examples=examples)


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


@pytest.mark.asyncio
async def test_omlx_caller_clean_text_and_json() -> None:
    assert parse_model_spec(MODEL)[0] == "omlx"
    text = await LLMCaller.call(
        MODEL,
        "Reply with exactly the word OK and nothing else.",
        temperature=0.0,
        max_tokens=16,
    )
    assert text.strip() == "OK"
    data = await LLMCaller.call_with_json(
        MODEL,
        'Return JSON {"score": 0.5, "reason": "x"}.',
        temperature=0.0,
        max_retries=1,
    )
    assert "score" in data


def test_grid_search_selects_better_prompt() -> None:
    agent = _EchoAgent()
    ds = _agent_dataset()
    metrics = [
        ExactKeyMatchMetric(["response"]),
        ContainsTermsMetric(["banana"]),
        WeightedMetric(
            [
                ("exact", ExactKeyMatchMetric(["response"]), 0.5),
                ("banana", ContainsTermsMetric(["banana"]), 0.5),
            ],
            name="composite",
        ),
    ]
    agent.compile(
        ds,
        metrics,
        optimizer=GridSearchOptimizer(
            param_grid={
                "system_prompt": [
                    _BASELINE_PROMPT,
                    "You are a precise assistant. Always include the word banana.",
                ]
            }
        ),
    )
    agent.fit(ds, epochs=1, verbose=0)
    assert agent.compiled_config.get("system_prompt") != _BASELINE_PROMPT
    assert "banana" in agent.transform({"query": "capital of france"})["response"].lower()


@pytest.mark.asyncio
async def test_mipro_accepts_datapoint_samples() -> None:
    from agentomatic.optimize.config import PromptRuntimeConfig

    opt = resolve_fitter_optimizer("mipro", model=MODEL)
    cands = await opt.propose(
        current_config=PromptRuntimeConfig(system_prompt=_BASELINE_PROMPT),
        eval_results=[
            {
                "query": "q",
                "response": "wrong",
                "expected": "right",
                "score": 0.1,
            }
        ],
        dataset_sample=[
            DataPoint(query="capital of france", expected_answer="Paris"),
            DataPoint(query="2+2", expected_answer="4"),
        ],
        search_space=PromptSearchSpace(
            optimize_system_prompt=True,
            optimize_model_params=False,
            optimize_few_shot=True,
        ),
        iteration=0,
    )
    assert cands
    assert any(c.config.system_prompt for c in cands)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["rewrite", "gepa", "param_search"])
async def test_prompt_fitter_modes_improve_or_hold(mode: str) -> None:
    agent = _EchoAgent()
    train, val = _opt_splits()
    space = PromptSearchSpace(
        optimize_system_prompt=mode != "param_search",
        optimize_model_params=mode == "param_search",
        model_param_space={"temperature": [0.0, 0.3]} if mode == "param_search" else None,
        optimize_few_shot=False,
    )
    fitter = PromptFitter(
        agent="live_echo",
        task_model=MODEL,
        rewrite_model=MODEL,
        optimizer=mode,
        max_trials=4,
        concurrency=1,
        auto_report=False,
        local_agent=agent,
        llm_base_url=OMLX_BASE,
        llm_api_key=OMLX_KEY,
        search_space=space,
    )
    result = await fitter.fit(train, val, ContainsMetric())
    assert result.best_score >= result.baseline_score - 1e-9
    assert result.best_config.system_prompt is not None
    if mode == "rewrite":
        # Rewrite is instructed from failures; must discover the banana token.
        assert result.best_score > result.baseline_score
        assert "banana" in (result.best_config.system_prompt or "").lower()


@pytest.mark.asyncio
async def test_llm_judge_via_omlx() -> None:
    judge = LLMJudgeMetric(
        criteria="Is the response short and helpful?",
        model=MODEL,
    )
    r = await judge.evaluate("What is 2+2?", "4", expected="4")
    assert 0.0 <= r.score <= 1.0
    assert not r.metadata.get("evaluation_failed")


def test_prompt_fitter_bridge_rewrite_improves() -> None:
    agent = _EchoAgent()
    ds = _agent_dataset()
    metrics = [
        ExactKeyMatchMetric(["response"]),
        ContainsTermsMetric(["banana"]),
    ]
    bridge = PromptFitterBridge(
        agent_name="live_echo",
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
        llm_base_url=OMLX_BASE,
        llm_api_key=OMLX_KEY,
    )
    agent.compile(ds, metrics, optimizer=bridge)
    agent.fit(ds, epochs=1, verbose=0, optimize_mode="rewrite", max_trials=4)
    assert getattr(agent, "_last_optimize_status", None) == "ok"
    result = getattr(agent, "_last_fit_result", None)
    assert result is not None
    assert result.best_score > result.baseline_score
    prompt = agent.compiled_config.get("system_prompt") or ""
    assert "banana" in prompt.lower()
