# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""End-to-end tests proving the new optimize + LangChain stack works.

Runs fully offline (no HTTP LLM server): local echo agents, ExactMatchMetric,
and a noop/echo fitter optimizer. Covers the production paths users hit:

* ``agent.fit(test_cases)`` via :class:`OptimizerMixin`
* :class:`PromptFitter` + ML callbacks + experiment tracker
* presets → fitter kwargs
* ``create_train_cli`` / ``_do_train``
* LangChain adapter + scaffolded langchain agent
* evals discovery + diversity selection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentomatic.optimize.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    ScoreThreshold,
    TemperatureScheduler,
)
from agentomatic.optimize.config import PromptCandidate, PromptRuntimeConfig
from agentomatic.optimize.dataset import DataPoint, Dataset
from agentomatic.optimize.fitter import PromptFitter
from agentomatic.optimize.fitter_optimizers import BaseFitterOptimizer
from agentomatic.optimize.metrics import ExactMatchMetric
from agentomatic.optimize.optimizer_mixin import FitResult, OptimizerMixin
from agentomatic.optimize.presets import Presets, to_fitter_kwargs
from agentomatic.optimize.runner import AgentRunner

# =====================================================================
# Shared fixtures / helpers
# =====================================================================


@dataclass
class EchoOptimizer(BaseFitterOptimizer):
    """Proposes one slightly-tweaked candidate — no LLM needed."""

    name: str = "echo"

    async def propose(
        self,
        current_config: Any,
        eval_results: Any,
        dataset_sample: Any,
        search_space: Any,
        iteration: int = 0,
        context: Any = None,
    ) -> list[PromptCandidate]:
        return [
            PromptCandidate(
                name=f"echo_{iteration:03d}",
                config=PromptRuntimeConfig(
                    system_prompt=(current_config.system_prompt or "base") + f" #{iteration}",
                    model_params=dict(current_config.model_params or {}),
                ),
                source="echo",
            )
        ]


@dataclass
class NoopOptimizer(BaseFitterOptimizer):
    """Produces no candidates — forces early plateau / callback stop paths."""

    name: str = "noop"

    async def propose(
        self,
        current_config: Any,
        eval_results: Any,
        dataset_sample: Any,
        search_space: Any,
        iteration: int = 0,
        context: Any = None,
    ) -> list[PromptCandidate]:
        return []


class EchoAgent(OptimizerMixin):
    """Minimal local agent: returns ``query.upper()``."""

    agent_name = "echo_bot"
    agent_description = "E2E echo agent"
    system_prompt: str = "You echo the user query in uppercase."

    def transform(self, input_data: dict[str, Any]) -> dict[str, Any]:
        query = str(input_data.get("current_query", ""))
        return {"response": query.upper(), "current_query": query}


def _cases() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(input="hello", expected_output="HELLO"),
        SimpleNamespace(input="world", expected_output="WORLD"),
        SimpleNamespace(input="foo", expected_output="FOO"),
        SimpleNamespace(input="bar", expected_output="BAR"),
    ]


def _dataset() -> Dataset:
    return Dataset(
        points=[
            DataPoint(query="hello", expected_answer="HELLO"),
            DataPoint(query="world", expected_answer="WORLD"),
            DataPoint(query="foo", expected_answer="FOO"),
            DataPoint(query="bar", expected_answer="BAR"),
        ]
    )


# =====================================================================
# 1. OptimizerMixin.fit — primary UX
# =====================================================================


class TestOptimizerMixinFitE2E:
    def test_fit_returns_fit_result_and_tracks_history(self, tmp_path: Path) -> None:
        agent = EchoAgent()
        result = agent.fit(
            _cases(),
            max_iterations=2,
            target_score=0.99,
            strategy="iterative_refinement",
            metric=ExactMatchMetric(fuzzy=False),
            optimizer=EchoOptimizer(),
            callbacks=[
                EarlyStopping(patience=3),
                ScoreThreshold(threshold=0.99),
                ProgressLogger(),
            ],
            experiment_dir=str(tmp_path / "fit"),
            auto_report=False,
            drain_seconds=0,
        )

        assert isinstance(result, FitResult)
        assert result.agent_name == "echo_bot"
        assert result.final_score >= 0.99  # perfect echo match
        assert result.improved or result.final_score >= result.initial_score
        assert result.best_prompt
        assert agent.get_optimized_prompt() == result.best_prompt
        assert agent.last_fit_result is result
        assert len(agent.get_optimization_history()) == 1
        assert "echo_bot" in result.summary()

    def test_fit_score_threshold_callback_stops_early(self, tmp_path: Path) -> None:
        agent = EchoAgent()
        stopper = ScoreThreshold(threshold=0.5)
        result = agent.fit(
            _cases(),
            max_iterations=8,
            target_score=0.5,
            metric=ExactMatchMetric(fuzzy=False),
            optimizer=EchoOptimizer(),
            callbacks=[stopper],
            experiment_dir=str(tmp_path / "stop"),
            auto_report=False,
            drain_seconds=0,
            patience=8,
        )
        assert result.final_score >= 0.5
        assert stopper.context.stop_requested is True


# =====================================================================
# 2. PromptFitter + callbacks + experiment tracker
# =====================================================================


class TestPromptFitterCallbacksTrackerE2E:
    @pytest.mark.asyncio
    async def test_fitter_callbacks_tracker_checkpoint(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "ckpts"
        tracker_dir = tmp_path / "exp"
        early = EarlyStopping(patience=1, min_delta=0.01)
        sched = TemperatureScheduler(initial_temperature=0.8, decay_rate=0.5)
        ckpt = ModelCheckpoint(save_dir=str(ckpt_dir), save_best_only=False)

        async def echo_fn(query, *, prompt_override, context, invoke):
            return query.upper()

        fitter = PromptFitter(
            agent="e2e_tracker",
            optimizer=NoopOptimizer(),
            max_trials=8,
            patience=1,
            experiment_dir=str(tracker_dir),
            auto_report=False,
            drain_seconds=0,
            callbacks=[early, sched, ckpt, ProgressLogger()],
            baseline_system_prompt="baseline prompt",
        )
        fitter._runner = AgentRunner(agent="e2e_tracker", agent_callable=echo_fn)

        result = await fitter.fit(_dataset(), _dataset(), ExactMatchMetric(fuzzy=False))

        assert result.baseline_score >= 0.99
        assert result.best_score >= 0.99
        assert result.early_stop_reason or early.context.stop_requested
        # Temperature scheduler actually set a temperature (not forced 0.7 default).
        assert sched.context.current_temperature is not None
        assert sched.context.current_temperature == pytest.approx(0.8)

        # Experiment tracker wrote a visible row (completed OR stopped).
        from agentomatic.optimize.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=str(tracker_dir / "experiments.db"))
        rows = tracker.get_experiments(agent_name="e2e_tracker")
        assert rows, "early-stopped / completed experiments must be listed"
        assert rows[0]["status"] in {"completed", "stopped"}
        assert rows[0]["initial_score"] is not None
        assert float(rows[0]["initial_score"]) >= 0.99

    @pytest.mark.asyncio
    async def test_default_callbacks_do_not_force_temperature(self, tmp_path: Path) -> None:
        from agentomatic.optimize.callbacks import default_callbacks
        from agentomatic.optimize.events import CallbackManager

        mgr = CallbackManager(default_callbacks(patience=2, target_score=0.9))
        assert mgr.current_temperature() is None

        async def echo_fn(query, *, prompt_override, context, invoke):
            # Capture model_params temperature if injected
            params = (invoke or {}).get("model_params") or {}
            echo_fn.last_temp = params.get("temperature")  # type: ignore[attr-defined]
            return query.upper()

        echo_fn.last_temp = "unset"  # type: ignore[attr-defined]

        fitter = PromptFitter(
            agent="e2e_temp",
            optimizer=NoopOptimizer(),
            max_trials=2,
            patience=1,
            experiment_dir=str(tmp_path),
            auto_report=False,
            drain_seconds=0,
            callbacks=default_callbacks(patience=1, target_score=0.99),
            baseline_system_prompt="p",
        )
        fitter._runner = AgentRunner(agent="e2e_temp", agent_callable=echo_fn)
        await fitter.fit(_dataset(), _dataset(), ExactMatchMetric(fuzzy=False))
        # Without TemperatureScheduler, fitter must not inject 0.7.
        assert echo_fn.last_temp != 0.7  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_temperature_scheduler_does_not_override_eval_temp(self, tmp_path: Path) -> None:
        """TemperatureScheduler must not break deterministic eval (temp=0.0)."""
        temps: list[float | None] = []

        async def echo_fn(query, *, prompt_override, context, invoke):
            params = (invoke or {}).get("model_params") or {}
            temps.append(params.get("temperature"))
            return query.upper()

        sched = TemperatureScheduler(initial_temperature=0.8, decay_rate=0.5)
        fitter = PromptFitter(
            agent="e2e_temp_sched",
            optimizer=NoopOptimizer(),
            max_trials=4,
            patience=1,
            experiment_dir=str(tmp_path),
            auto_report=False,
            drain_seconds=0,
            callbacks=[sched],
            baseline_system_prompt="p",
        )
        fitter._runner = AgentRunner(agent="e2e_temp_sched", agent_callable=echo_fn)
        await fitter.fit(_dataset(), _dataset(), ExactMatchMetric(fuzzy=False))

        assert sched.context.current_temperature is not None
        assert temps, "agent should have been invoked"
        assert all(t == 0.0 for t in temps), temps

    @pytest.mark.asyncio
    async def test_completed_fit_marks_experiment_completed(self, tmp_path: Path) -> None:
        """Full-budget runs must be status=completed, not stopped.

        Baseline returns wrong answers; EchoOptimizer's longer prompt unlocks
        correct answers so the final round improves and the loop exits
        naturally (not via patience early-stop).
        """
        fitter = PromptFitter(
            agent="e2e_complete",
            optimizer=EchoOptimizer(),
            max_trials=4,  # 1 round
            patience=5,
            experiment_dir=str(tmp_path),
            auto_report=False,
            drain_seconds=0,
            callbacks=[],
            baseline_system_prompt="baseline",
        )

        async def gated_echo(query, *, prompt_override, context, invoke):
            # EchoOptimizer appends " #N" — only then return the exact answer.
            if prompt_override and "#" in prompt_override:
                return query.upper()
            return "WRONG"

        fitter._runner = AgentRunner(agent="e2e_complete", agent_callable=gated_echo)
        result = await fitter.fit(_dataset(), _dataset(), ExactMatchMetric(fuzzy=False))
        assert result.improved
        assert "completed all" in (result.early_stop_reason or "")

        from agentomatic.optimize.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(db_path=str(tmp_path / "experiments.db"))
        rows = tracker.get_experiments(agent_name="e2e_complete")
        assert rows
        assert rows[0]["status"] == "completed"


# =====================================================================
# 3. Presets → real PromptFitter
# =====================================================================


class TestPresetsE2E:
    @pytest.mark.asyncio
    async def test_preset_kwargs_drive_full_fit(self, tmp_path: Path) -> None:
        preset = Presets.for_quick()
        kwargs = to_fitter_kwargs(
            preset,
            optimizer=EchoOptimizer(),
            experiment_dir=str(tmp_path),
            auto_report=False,
            drain_seconds=0,
        )
        # for_quick max_iterations=2 rounds → 8 trial budget
        assert kwargs["max_trials"] == 8
        assert kwargs["optimizer"]  # echo optimizer instance

        async def echo_fn(query, *, prompt_override, context, invoke):
            return query.upper()

        fitter = PromptFitter(agent="preset_bot", local_agent=None, **kwargs)
        fitter._runner = AgentRunner(agent="preset_bot", agent_callable=echo_fn)
        result = await fitter.fit(_dataset(), _dataset(), ExactMatchMetric(fuzzy=False))
        assert result.best_score >= 0.99
        cfg = preset.to_config(system_prompt="hello")
        assert cfg.system_prompt == "hello"
        assert cfg.model_params["temperature"] == preset.temperature


# =====================================================================
# 4. train_cli end-to-end (Click + _do_train)
# =====================================================================


class TestTrainCliE2E:
    def test_click_train_group_and_do_train(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentomatic.optimize.train_cli import _do_train, create_train_cli

        agent = EchoAgent()
        cli = create_train_cli(agent, test_cases_fn=_cases, description="E2E train")
        assert hasattr(cli, "commands")
        assert "train" in cli.commands
        assert "evaluate" in cli.commands
        assert "experiments" in cli.commands

        # Patch PromptFitter inside _do_train to use echo optimizer + exact match.
        real_fitter_cls = PromptFitter

        class PatchedFitter(real_fitter_cls):
            def __init__(self, *a: Any, **kw: Any) -> None:
                kw["optimizer"] = EchoOptimizer()
                kw["auto_report"] = False
                kw["drain_seconds"] = 0
                kw.setdefault("experiment_dir", str(tmp_path))
                super().__init__(*a, **kw)
                self._runner = AgentRunner(
                    agent=kw.get("agent") or a[0] if a else "x",
                    agent_callable=lambda q, **_: (
                        asyncio.sleep(0, result=str(q).upper()) if False else None
                    ),
                )

                async def _echo(query, *, prompt_override=None, context=None, invoke=None):
                    return str(query).upper()

                self._runner = AgentRunner(agent=self.agent, agent_callable=_echo)

            async def fit(self, trainset, valset, metric, testset=None):
                # Force ExactMatchMetric so no LLM judge is needed.
                return await super().fit(trainset, valset, ExactMatchMetric(fuzzy=False), testset)

        monkeypatch.setattr("agentomatic.optimize.fitter.PromptFitter", PatchedFitter)

        result = asyncio.run(
            _do_train(
                agent=agent,
                test_cases=_cases(),
                model="ollama/mistral:7b",
                iterations=2,
                target=0.9,
                strategy="bootstrap_few_shot",
                verbose=False,
                output_dir=str(tmp_path),
            )
        )
        assert isinstance(result, FitResult)
        assert result.final_score >= 0.9
        assert result.strategy == "bootstrap_few_shot"


# =====================================================================
# 5. LangChain adapter + scaffold template
# =====================================================================


class TestLangChainAdapterE2E:
    def test_scaffold_langchain_agent_runs_with_message_history(self) -> None:
        from agentomatic.cli.templates import get_template_files

        files = get_template_files("langchain", "e2e_chat")
        assert "agent.py" in files
        assert "dict_to_messages" in files["agent.py"]
        assert "serialize_messages" in files["agent.py"]

        # Execute the generated agent module in-process.
        ns: dict[str, Any] = {}
        exec(compile(files["agent.py"], "<e2e_chat>", "exec"), ns)
        AgentCls = ns["E2EChatAgent"]
        agent = AgentCls(llm=None)
        state = agent.input_to_state(
            {
                "current_query": "Hi there",
                "messages": [
                    {"role": "user", "content": "earlier"},
                    {"role": "assistant", "content": "hi"},
                ],
            }
        )
        # Must not raise (dict_to_messages accepts list).
        out_state = agent.chat(state)
        assert out_state.response
        assert isinstance(out_state.messages, list)
        assert out_state.messages[0]["role"] in ("user", "human")
        output = agent.state_to_output(out_state)
        assert "response" in output

    @pytest.mark.asyncio
    async def test_agent_adapter_round_trip(self) -> None:
        from agentomatic.langchain_adapter import (
            AgentAdapter,
            dict_to_messages,
            messages_to_dict,
            serialize_messages,
        )

        class LCStyle:
            agent_name = "lc_e2e"

            async def ainvoke(self, payload, config=None):
                msgs = payload.get("messages") or []
                last = msgs[-1] if msgs else {"content": ""}
                content = getattr(last, "content", None) or last.get("content", "")
                return {
                    "messages": list(msgs)
                    + [SimpleNamespace(content=f"ECHO:{content}", type="ai")],
                }

        adapted = AgentAdapter(LCStyle())
        result = await adapted.atransform({"current_query": "ping", "messages": []})
        assert "response" in result or "messages" in result

        # Sync path must work even if a loop is running (uses run_sync).
        sync_result = adapted.transform({"current_query": "pong", "messages": []})
        assert isinstance(sync_result, dict)

        msgs = dict_to_messages([{"role": "user", "content": "x"}])
        assert serialize_messages(msgs)[0]["content"] == "x"
        assert "messages" in messages_to_dict(msgs)


# =====================================================================
# 6. Agent detect + evals discovery + diversity
# =====================================================================


class TestDiscoveryAndDiversityE2E:
    def test_evals_discovery_and_diversity(self, tmp_path: Path) -> None:
        from agentomatic.optimize.agent_detect import AgentType, Evaluator, detect_agent_type
        from agentomatic.optimize.diversity_selector import DiversitySelector
        from agentomatic.optimize.evals_discovery import discover_agent_evals

        agents_dir = tmp_path / "agents" / "demo"
        agents_dir.mkdir(parents=True)
        (agents_dir / "evals.py").write_text(
            "def get_test_cases():\n"
            "    return [type('C', (), {'input': 'a', 'expected_output': 'A',"
            " 'category': 'short'})()]\n"
            "THRESHOLDS = {'answer_relevancy': 0.8}\n"
            "def get_custom_metrics():\n"
            "    return []\n",
            encoding="utf-8",
        )

        found = discover_agent_evals(str(tmp_path / "agents"))
        assert "demo" in found
        assert found["demo"]["thresholds"]["answer_relevancy"] == 0.8

        agent = EchoAgent()
        assert detect_agent_type(agent) == AgentType.STATELESS
        evaluator = Evaluator.for_agent(agent)
        assert "answer_relevancy" in evaluator.metrics

        cases = [
            SimpleNamespace(input="short", expected_output="S", category="a"),
            SimpleNamespace(input="x" * 80, expected_output="M", category="b"),
            SimpleNamespace(input="y" * 250, expected_output="L", category="c"),
            DataPoint(query="q1", expected_answer="A1"),
            DataPoint(query="q2", expected_answer="A2"),
        ]
        selected = DiversitySelector().select_diverse(cases, num_select=3)
        assert len(selected) == 3

    @pytest.mark.asyncio
    async def test_evaluator_evaluate_offline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentomatic.optimize.agent_detect import Evaluator

        agent = EchoAgent()
        evaluator = Evaluator.for_agent(agent)

        # Patch PromptFitter used inside evaluate to avoid LLM judge.
        real = PromptFitter

        class OfflineFitter(real):
            def __init__(self, *a: Any, **kw: Any) -> None:
                kw["optimizer"] = NoopOptimizer()
                kw["auto_report"] = False
                kw["drain_seconds"] = 0
                kw["experiment_dir"] = str(tmp_path)
                kw["baseline_system_prompt"] = "echo"
                super().__init__(*a, **kw)

                async def _echo(query, *, prompt_override=None, context=None, invoke=None):
                    return str(query).upper()

                self._runner = AgentRunner(agent=self.agent, agent_callable=_echo)

            async def _evaluate_config(self, config, dataset, metric):
                return await super()._evaluate_config(
                    config, dataset, ExactMatchMetric(fuzzy=False)
                )

        monkeypatch.setattr("agentomatic.optimize.fitter.PromptFitter", OfflineFitter)
        summary = await evaluator.evaluate(agent, _cases(), model="ollama/mistral:7b")
        assert summary["n_cases"] == 4
        assert summary["mean_score"] >= 0.99
        assert summary["passed"] is True


# =====================================================================
# 7. Full stack smoke: fit → tracker → version control → JSON extract
# =====================================================================


class TestFullStackSmokeE2E:
    def test_fit_then_inspect_tracker_and_helpers(self, tmp_path: Path) -> None:
        from agentomatic.optimize.experiment_tracker import ExperimentTracker
        from agentomatic.optimize.json_extractor import JSONExtractor
        from agentomatic.optimize.prompt_version_control import PromptVersionControl

        agent = EchoAgent()
        exp_dir = tmp_path / "stack"
        result = agent.fit(
            _cases(),
            max_iterations=2,
            metric=ExactMatchMetric(fuzzy=False),
            optimizer=EchoOptimizer(),
            callbacks=[EarlyStopping(patience=2), ScoreThreshold(threshold=0.99)],
            experiment_dir=str(exp_dir),
            auto_report=False,
            drain_seconds=0,
        )
        assert result.final_score >= 0.99

        tracker = ExperimentTracker(db_path=str(exp_dir / "experiments.db"))
        experiments = tracker.get_experiments(agent_name="echo_bot")
        assert experiments
        best = tracker.get_best_experiment("echo_bot")
        assert best is not None

        pvc = PromptVersionControl(agent_name="echo_bot", save_dir=str(tmp_path / "pvc"))
        pvc.add_version(result.best_prompt or "v1", score=result.final_score)
        pvc.add_version((result.best_prompt or "v1") + " v2", score=result.final_score + 0.01)
        best = pvc.get_best()
        assert best is not None
        assert best.score >= result.final_score

        extracted = JSONExtractor().extract('Sure: ```json {"ok": true, "n": 1}```')
        assert extracted["ok"] is True
        assert extracted["n"] == 1
