"""End-to-end mechanics that ``train.py`` / ``optimize.py`` templates rely on.

Does **not** rewrite the scaffold scripts — it exercises the same compile →
fit → evaluate → save path (and prompt_only GridSearch / PromptFitterBridge
local runner) so those scripts work with all their moving parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentomatic.agents import AgentDataset, AgentExample, BaseGraphAgent, EarlyStopping
from agentomatic.agents.history import MetricLoss
from agentomatic.agents.metrics import (
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
    ResponseSimilarityMetric,
    WeightedMetric,
)
from agentomatic.agents.optimizers import (
    GridSearchOptimizer,
    NoOpOptimizer,
    PromptFitterBridge,
)
from agentomatic.cli.templates import get_template_files
from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
from agentomatic.optimize.dataset import Dataset
from agentomatic.optimize.metrics import ExactMatchMetric


@dataclass
class _TrainState:
    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class _TrainDemoAgent(BaseGraphAgent[_TrainState]):
    """Mirrors the full-template basic agent surface used by train.py."""

    agent_name = "train_demo"
    agent_description = "Train mechanics demo"
    agent_framework = "graph_agent"

    def __init__(self) -> None:
        super().__init__()
        self.prompt_manager = SimpleNamespace(
            get_prompt=lambda *a, **k: "BASE_PROMPT",
        )

    def _system_prompt(self) -> str:
        return self.resolve_system_prompt(default="BASE_PROMPT")

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("process", self.process)
        g.set_entry_point("process")
        g.set_finish_point("process")
        return g.compile()

    def process(self, state: _TrainState) -> _TrainState:
        prompt = self._system_prompt()
        # Mimic scaffold: include "Result" so ContainsTermsMetric scores.
        # When the optimized prompt is active, echo a distinctive marker so
        # ResponseSimilarity / custom metrics can prefer it.
        marker = "OPTIMIZED" if "precise, detail-oriented" in prompt else "BASE"
        state.output = {
            "response": f"Result for: {state.request} [{marker}]",
            "agent_type": "train_demo",
            "used_prompt": prompt,
        }
        return state

    def input_to_state(self, data: dict[str, Any]) -> _TrainState:
        return _TrainState(
            request=data.get("current_query") or data.get("query") or "",
        )

    def state_to_output(self, state: _TrainState) -> dict[str, Any]:
        return state.output


def _metrics() -> list:
    """Same weighted stack as the full-template train.py."""
    rows = [
        ("exact_response", ExactKeyMatchMetric(["response"]), 0.5),
        ("contains_terms", ContainsTermsMetric(["Result"]), 0.3),
        (
            "has_output",
            CallableMetric(
                "has_output",
                lambda example, pred: 1.0 if pred.get("response") else 0.0,
            ),
            0.2,
        ),
    ]
    individual = [m for _, m, _ in rows]
    return [*individual, WeightedMetric(rows, name="composite")]


def _dataset() -> AgentDataset:
    examples = [
        AgentExample(
            id=f"ex{i}",
            input={"current_query": f"task {i}"},
            expected_output={"response": f"Result for: task {i} [OPTIMIZED]"},
            split="train" if i < 4 else "test",
        )
        for i in range(6)
    ]
    return AgentDataset(name="demo", examples=examples)


# =====================================================================
# Original train.py path (NoOp compile → fit → evaluate → save)
# =====================================================================


class TestTrainPyMechanics:
    def test_noop_train_compile_fit_evaluate_save(self, tmp_path: Path) -> None:
        agent = _TrainDemoAgent()
        dataset = _dataset()
        metrics = _metrics()

        agent.compile(dataset, metrics, optimizer=NoOpOptimizer())
        history = agent.fit(dataset, epochs=1, verbose=0)
        assert history is not None

        report = agent.evaluate(dataset.test, metrics)
        assert report.num_examples == 2
        assert report.pass_rate >= 0.0

        out = tmp_path / "compiled"
        agent.save(str(out))
        assert (out / "config.json").exists()
        assert (out / "metadata.json").exists()

    def test_fit_with_loss_and_early_stopping(self, tmp_path: Path) -> None:
        agent = _TrainDemoAgent()
        dataset = _dataset()
        metrics = _metrics()
        composite = next(m for m in metrics if m.name == "composite")

        agent.compile(
            dataset,
            metrics,
            optimizer=NoOpOptimizer(),
            loss=MetricLoss(composite, name="loss"),
        )
        history = agent.fit(
            dataset,
            epochs=5,
            verbose=0,
            callbacks=[EarlyStopping(monitor="loss", patience=1, mode="min")],
        )
        assert "loss" in history.history
        assert len(history.history["loss"]) >= 1


# =====================================================================
# optimize.py prompt_only GridSearch — must apply system_prompt
# =====================================================================


class TestOptimizePromptOnlyMechanics:
    def test_grid_search_applies_system_prompt_via_compiled_config(self) -> None:
        agent = _TrainDemoAgent()
        dataset = _dataset()
        # Similarity against expected [OPTIMIZED] output gives GridSearch a
        # real signal (template ContainsTerms alone would tie).
        metrics = [
            ResponseSimilarityMetric(),
            ExactKeyMatchMetric(["response"]),
        ]

        optimizer = GridSearchOptimizer(
            param_grid={
                "system_prompt": [
                    "You are a helpful assistant.",
                    "You are a precise, detail-oriented assistant.",
                ],
            },
            max_examples=4,
        )
        agent.compile(dataset, metrics, optimizer=optimizer)
        agent.fit(dataset, epochs=1, verbose=0)

        assert "system_prompt" in agent.compiled_config
        # Best prompt should be the detail-oriented one (higher similarity).
        assert "precise, detail-oriented" in agent.compiled_config["system_prompt"]

        out = agent.transform({"current_query": "task 0"})
        assert out["used_prompt"] == agent.compiled_config["system_prompt"]
        assert "[OPTIMIZED]" in out["response"]


# =====================================================================
# PromptFitterBridge local runner (no HTTP server)
# =====================================================================


class TestPromptFitterBridgeLocal:
    def test_bridge_wires_local_agent_and_stores_system_prompt(self) -> None:
        agent = _TrainDemoAgent()

        class _StubFitter:
            name = "stub"

            async def fit(self, trainset, valset, metric, testset=None):
                return PromptFitResult(
                    best_config=PromptRuntimeConfig(
                        system_prompt="You are a precise, detail-oriented assistant.",
                    ),
                    baseline_config=PromptRuntimeConfig(system_prompt="BASE_PROMPT"),
                    best_score=0.9,
                    baseline_score=0.4,
                )

        bridge = PromptFitterBridge(agent_name="train_demo", fitter=_StubFitter())
        dataset = _dataset()
        config = bridge.optimize(agent, dataset, _metrics())
        assert config.get("system_prompt")
        assert agent._last_optimize_status == "ok"  # noqa: SLF001

        # Simulate what fit() does with the returned config.
        agent.compiled_config.update(config)
        out = agent.transform({"query": "hello"})
        assert "precise, detail-oriented" in out["used_prompt"]

    @pytest.mark.asyncio
    async def test_local_runner_applies_prompt_override(self) -> None:
        from agentomatic.optimize.runner import AgentRunner

        agent = _TrainDemoAgent()
        from agentomatic.optimize.fitter import _wrap_local_agent

        runner = AgentRunner(agent="train_demo", agent_callable=_wrap_local_agent(agent))
        result = await runner.run_single(
            "hello",
            prompt_override="You are a precise, detail-oriented assistant.",
        )
        assert result.error is None
        assert "[OPTIMIZED]" in result.response


# =====================================================================
# PromptFitter evaluate path (local) + loss signal
# =====================================================================


class TestFitterLocalEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_config_uses_local_agent_and_scores(self) -> None:
        from agentomatic.optimize.fitter import PromptFitter

        agent = _TrainDemoAgent()
        fitter = PromptFitter(
            agent="train_demo",
            auto_report=False,
            max_trials=1,
            local_agent=agent,
        )

        score_base, _, _ = await fitter._evaluate_config(
            PromptRuntimeConfig(system_prompt="BASE_PROMPT"),
            Dataset.from_list(
                [{"query": "task 0", "expected_answer": "Result for: task 0 [OPTIMIZED]"}]
            ),
            ExactMatchMetric(fuzzy=True),
        )
        score_opt, _, _ = await fitter._evaluate_config(
            PromptRuntimeConfig(
                system_prompt="You are a precise, detail-oriented assistant.",
            ),
            Dataset.from_list(
                [{"query": "task 0", "expected_answer": "Result for: task 0 [OPTIMIZED]"}]
            ),
            ExactMatchMetric(fuzzy=True),
        )
        assert score_opt > score_base


# =====================================================================
# Template contract: train.py still the original scaffold, agents resolve
# =====================================================================


class TestTemplateContract:
    def test_full_train_py_uses_local_prompt_fitter(self) -> None:
        files = get_template_files("full", "demo_agent")
        train = files["train.py"]
        assert "TrainCliSettings" in train
        assert "to_train_config" in train
        assert "train_and_report" in train
        assert "import argparse" not in train
        assert "optimizer" in train

    def test_agent_templates_use_resolve_system_prompt(self) -> None:
        for tmpl in ("basic", "full", "rag", "chatbot"):
            files = get_template_files(tmpl, "demo_agent")
            agent_py = files["agent.py"]
            assert "resolve_system_prompt" in agent_py, tmpl

    def test_response_similarity_metric_scores_expected(self) -> None:
        metric = ResponseSimilarityMetric()
        ex = AgentExample(
            id="1",
            input={"query": "q"},
            expected_output={"response": "hello world"},
        )
        assert metric.score(ex, {"response": "hello world"}) == pytest.approx(1.0)
        assert metric.score(ex, {"response": ""}) == 0.0
        assert metric.score(ex, {"response": "goodbye"}) < 0.5
