# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for the Keras-style agent training lifecycle (Phase 6).

Covers the History record, Callback / EarlyStopping hooks, the Loss
abstraction, the epoch-aware fit() loop, and the PromptFitterBridge wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentomatic.agents.base import BaseGraphAgent
from agentomatic.agents.decorators import agent_node
from agentomatic.agents.history import (
    CallableLoss,
    Callback,
    EarlyStopping,
    History,
    MetricLoss,
    resolve_loss,
)
from agentomatic.agents.optimizers import NoOpOptimizer, PromptFitterBridge
from agentomatic.agents.types import AgentDataset, AgentExample

# ---------------------------------------------------------------------------
# Test agent + fixtures
# ---------------------------------------------------------------------------


@dataclass
class EchoState:
    query: str = ""
    response: str = ""


class EchoAgent(BaseGraphAgent[EchoState]):
    agent_name = "echo"
    system_prompt = "default"

    @agent_node(entrypoint=True, finish=True)
    def respond(self, state: EchoState) -> EchoState:
        state.response = state.query.upper()
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> EchoState:
        return EchoState(query=input_data.get("query", ""))

    def state_to_output(self, state: EchoState) -> dict[str, Any]:
        return {"response": state.response}


class _Accuracy:
    """Exact-match on the 'response' key (0/1)."""

    name = "accuracy"

    def score(self, example: AgentExample, prediction: dict[str, Any]) -> float:
        expected = (example.expected_output or {}).get("response")
        return 1.0 if prediction.get("response") == expected else 0.0


def _dataset() -> AgentDataset:
    return AgentDataset(
        name="echo_ds",
        examples=[
            AgentExample(id="1", input={"query": "hi"}, expected_output={"response": "HI"}),
            AgentExample(id="2", input={"query": "yo"}, expected_output={"response": "YO"}),
        ],
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    def test_record_and_access(self):
        h = History(params={"epochs": 2})
        h.record(0, {"loss": 0.5, "accuracy": 0.6})
        h.record(1, {"loss": 0.3, "accuracy": 0.8})
        assert h.epoch == [0, 1]
        assert h["loss"] == [0.5, 0.3]
        assert h.final("accuracy") == 0.8
        assert "loss" in h

    def test_best_min_and_max(self):
        h = History()
        h.record(0, {"loss": 0.5, "acc": 0.6})
        h.record(1, {"loss": 0.2, "acc": 0.9})
        h.record(2, {"loss": 0.4, "acc": 0.7})
        assert h.best("loss", mode="min") == (1, 0.2)
        assert h.best("acc", mode="max") == (1, 0.9)
        assert h.best("missing") is None

    def test_to_dict_and_summary(self):
        h = History(params={"epochs": 1})
        h.record(0, {"loss": 0.1})
        d = h.to_dict()
        assert d["history"]["loss"] == [0.1]
        assert d["epoch"] == [0]
        assert "History" in h.summary()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class TestLoss:
    def test_metric_loss(self):
        loss = MetricLoss(_Accuracy())
        ex = AgentExample(input={"query": "hi"}, expected_output={"response": "HI"})
        assert loss.compute(ex, {"response": "HI"}) == 0.0
        assert loss.compute(ex, {"response": "NO"}) == 1.0

    def test_callable_loss(self):
        loss = CallableLoss(lambda ex, pred: 0.25, name="fixed")
        assert loss.name == "fixed"
        assert loss.compute(AgentExample(), {}) == 0.25

    def test_resolve_loss(self):
        assert resolve_loss(None) is None
        assert isinstance(resolve_loss(_Accuracy()), MetricLoss)
        assert isinstance(resolve_loss(lambda e, p: 0.0), CallableLoss)
        existing = MetricLoss(_Accuracy())
        assert resolve_loss(existing) is existing


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_history(self):
        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()], loss=_Accuracy())
        history = agent.fit(epochs=3, verbose=0)

        assert isinstance(history, History)
        assert history is agent.history
        assert len(history.epoch) == 3
        assert history["accuracy"] == [1.0, 1.0, 1.0]
        assert history["loss"] == [0.0, 0.0, 0.0]

    def test_fit_accepts_optimize_config(self):
        """fit() should accept search_space / mode toggles without recompile."""
        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()])

        class CapturingOptimizer:
            def __init__(self) -> None:
                self.seen: dict | None = None

            def optimize(self, agent, dataset, metrics):
                self.seen = getattr(agent, "_fit_optimize_options", None)
                return {"system_prompt": "tuned"}

        opt = CapturingOptimizer()
        history = agent.fit(
            epochs=1,
            verbose=0,
            optimizer=opt,
            optimize_mode="param_search",
            optimize_prompt=False,
            optimize_params=True,
            model_param_space={"temperature": [0.0, 0.5]},
            max_trials=3,
        )
        assert opt.seen is not None
        assert opt.seen["optimizer"] == "param_search"
        assert opt.seen["max_trials"] == 3
        assert opt.seen["optimize_params"] is True
        assert history.params["optimize"]["optimizer"] == "param_search"
        assert (
            agent.system_prompt == "tuned" or agent.compiled_config.get("system_prompt") == "tuned"
        )

    def test_fit_coerce_search_space_dict(self):
        agent = EchoAgent()
        space = agent._coerce_search_space(
            {"optimize_system_prompt": True, "optimize_model_params": False},
            optimize_params=True,
            model_param_space={"temperature": [0.1]},
        )
        assert space is not None
        assert space.optimize_model_params is True
        assert space.model_param_space["temperature"] == [0.1]

    def test_fit_uses_compiled_dataset_by_default(self):
        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()])
        history = agent.fit(verbose=0)
        assert history.params["train_size"] == 2

    def test_fit_with_validation_data(self):
        agent = EchoAgent()
        agent.compile(metrics=[_Accuracy()], loss=_Accuracy())
        train = [AgentExample(input={"query": "hi"}, expected_output={"response": "HI"})]
        val = [AgentExample(input={"query": "bye"}, expected_output={"response": "BYE"})]
        history = agent.fit(train, epochs=1, verbose=0, validation_data=val)

        assert "val_accuracy" in history
        assert "val_loss" in history
        assert history["val_accuracy"] == [1.0]

    def test_fit_runs_optimizer_each_epoch(self):
        agent = EchoAgent()
        calls = {"n": 0}

        class CountingOptimizer:
            def optimize(self, agent, dataset, metrics):
                calls["n"] += 1
                return {}

        agent.compile(_dataset(), metrics=[_Accuracy()], optimizer=CountingOptimizer())
        agent.fit(epochs=3, verbose=0)
        assert calls["n"] == 3

    def test_fit_applies_optimizer_config(self):
        agent = EchoAgent()

        class PromptOptimizer:
            def optimize(self, agent, dataset, metrics):
                return {"system_prompt": "tuned"}

        agent.compile(_dataset(), metrics=[_Accuracy()], optimizer=PromptOptimizer())
        agent.fit(epochs=1, verbose=0)
        assert agent.system_prompt == "tuned"

    def test_fit_no_metrics_no_loss(self):
        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[], optimizer=NoOpOptimizer())
        history = agent.fit(epochs=2, verbose=0)
        # Optimizer active → History starts with a pre-fit baseline row (-1).
        assert len(history.epoch) == 3
        assert history.epoch[0] == -1


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_callback_hooks_fire(self):
        events: list[str] = []

        class Recorder(Callback):
            def on_train_begin(self, logs=None):
                events.append("train_begin")

            def on_epoch_begin(self, epoch, logs=None):
                events.append(f"epoch_begin:{epoch}")

            def on_epoch_end(self, epoch, logs=None):
                events.append(f"epoch_end:{epoch}")

            def on_train_end(self, logs=None):
                events.append("train_end")

        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()])
        agent.fit(epochs=2, verbose=0, callbacks=[Recorder()])

        assert events[0] == "train_begin"
        assert events[-1] == "train_end"
        assert "epoch_begin:0" in events
        assert "epoch_end:1" in events

    def test_callback_receives_logs(self):
        seen: list[dict[str, float]] = []

        class LogGrabber(Callback):
            def on_epoch_end(self, epoch, logs=None):
                seen.append(dict(logs or {}))

        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()], loss=_Accuracy())
        agent.fit(epochs=1, verbose=0, callbacks=[LogGrabber()])

        assert seen and "accuracy" in seen[0] and "loss" in seen[0]

    def test_early_stopping_halts(self):
        agent = EchoAgent()
        # Constant loss (deterministic agent) → no improvement after epoch 0.
        agent.compile(_dataset(), metrics=[_Accuracy()], loss=_Accuracy())
        es = EarlyStopping(monitor="loss", mode="min", patience=1)
        history = agent.fit(epochs=10, verbose=0, callbacks=[es])

        assert agent.stop_training is True
        assert es.stopped_epoch is not None
        assert len(history.epoch) < 10


# ---------------------------------------------------------------------------
# PromptFitterBridge
# ---------------------------------------------------------------------------


class TestPromptFitterBridge:
    def test_bridge_runs_injected_fitter_and_applies_config(self):
        class _Cfg:
            system_prompt = "optimized"

            def to_dict(self):
                return {
                    "system_prompt": self.system_prompt,
                    "user_template": "ignored",
                    "model_choice": "m",
                }

        class _Result:
            best_config = _Cfg()

        class _FakeFitter:
            def __init__(self):
                self.called = False

            async def fit(self, trainset, valset, metric, testset=None):
                self.called = True
                return _Result()

        fitter = _FakeFitter()
        bridge = PromptFitterBridge(fitter=fitter, metric=object())

        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()], optimizer=bridge)
        agent.fit(epochs=1, verbose=0)

        assert fitter.called is True
        assert agent.system_prompt == "optimized"
        assert agent._last_fit_result is not None

    def test_bridge_empty_dataset_returns_noop(self):
        bridge = PromptFitterBridge(fitter=object(), metric=object())
        agent = EchoAgent()
        empty = AgentDataset(name="empty", examples=[])
        config = bridge.optimize(agent, empty, [])
        assert config == {}

    def test_bridge_surfaces_skip_status(self):
        """P2-1: a skipped optimize records a structured status on the agent."""
        bridge = PromptFitterBridge(fitter=object(), metric=object())
        agent = EchoAgent()
        empty = AgentDataset(name="empty", examples=[])
        bridge.optimize(agent, empty, [])
        assert agent._last_optimize_status == "skipped: empty dataset"


# ---------------------------------------------------------------------------
# save() / load() — what does and does not round-trip
# ---------------------------------------------------------------------------


class TestMetricsDoNotSurviveSaveLoad:
    """A metric is a live object, so ``load()`` cannot bring it back.

    That is the design, not a bug — but the error a loaded agent raised
    ("Pass metrics=... to evaluate() or compile(metrics=...) first") reads as
    wrong advice to somebody who compiled with metrics, saved, and loaded.
    The message now names the metrics the save recorded and says why they are
    gone.
    """

    def test_a_loaded_agent_reports_what_it_was_compiled_with(self, tmp_path) -> None:
        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()])
        agent.save(tmp_path / "saved")

        restored = EchoAgent()
        restored.load(tmp_path / "saved")

        with pytest.raises(ValueError) as exc:
            restored.evaluate(_dataset())

        message = str(exc.value)
        assert "accuracy" in message, message
        assert "save()/load()" in message, message

    def test_supplying_metrics_again_works(self, tmp_path) -> None:
        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()])
        agent.save(tmp_path / "saved")

        restored = EchoAgent()
        restored.load(tmp_path / "saved")

        report = restored.evaluate(_dataset(), metrics=[_Accuracy()])

        assert report.scores["accuracy"] == 1.0

    def test_recompiling_restores_the_full_loop(self, tmp_path) -> None:
        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()])
        agent.save(tmp_path / "saved")

        restored = EchoAgent()
        restored.load(tmp_path / "saved")
        restored.compile(_dataset(), metrics=[_Accuracy()])

        assert restored.evaluate(_dataset()).scores["accuracy"] == 1.0

    def test_an_agent_never_compiled_gets_the_plain_message(self) -> None:
        """No saved metric names to quote — keep the original advice."""
        with pytest.raises(ValueError) as exc:
            EchoAgent().evaluate(_dataset())

        message = str(exc.value)
        assert "compile(metrics=...)" in message
        assert "save()/load()" not in message

    def test_the_tuned_config_and_histories_do_survive(self, tmp_path) -> None:
        agent = EchoAgent()
        agent.compile(_dataset(), metrics=[_Accuracy()])
        agent.evaluate(_dataset())
        agent.save(tmp_path / "saved")

        restored = EchoAgent()
        restored.load(tmp_path / "saved")

        assert restored.compiled_metadata["dataset_size"] == 2
        assert restored.compiled_metadata["metrics"] == ["accuracy"]
        assert restored.evaluation_history
        assert restored.evaluation_history[-1].scores["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# PromptFitterBridge — where the first epoch's baseline comes from
# ---------------------------------------------------------------------------


class TestTheFirstEpochStartsFromTheAgentsOwnPrompt:
    """Nothing is compiled yet on epoch one — do not fall back to a generic.

    ``_build_fitter`` seeded ``baseline_system_prompt`` only from
    ``compiled_config``, which is empty until a fit succeeds. The first epoch
    therefore measured its baseline against the fitter's own default and
    rewrote *that*, silently ignoring the prompt the author wrote on the
    agent.
    """

    @staticmethod
    def _baseline_seen(agent) -> str | None:
        captured: dict[str, Any] = {}

        class _Probe(PromptFitterBridge):
            def _build_fitter(self, agent, name):  # noqa: ANN001
                import agentomatic.optimize.fitter as fitter_mod

                real = fitter_mod.PromptFitter

                class _Capture(real):  # type: ignore[misc, valid-type]
                    def __init__(self, *a, **kw):
                        captured["baseline"] = kw.get("baseline_system_prompt")
                        raise RuntimeError("stop here — only the kwargs matter")

                fitter_mod.PromptFitter = _Capture
                try:
                    return super()._build_fitter(agent, name)
                except RuntimeError:
                    return None
                finally:
                    fitter_mod.PromptFitter = real

        _Probe()._build_fitter(agent, "probe")
        return captured.get("baseline")

    def test_the_instance_prompt_is_used_when_nothing_is_compiled(self) -> None:
        agent = EchoAgent()
        agent.system_prompt = "Answer only in French."

        assert self._baseline_seen(agent) == "Answer only in French."

    def test_a_compiled_prompt_still_wins(self) -> None:
        """Later epochs must compound, not restart from the original."""
        agent = EchoAgent()
        agent.system_prompt = "Answer only in French."
        agent.compiled_config = {"system_prompt": "Tuned by a previous epoch."}

        assert self._baseline_seen(agent) == "Tuned by a previous epoch."

    def test_an_agent_with_no_prompt_at_all_passes_none(self) -> None:
        """No prompt anywhere is a real state — let the fitter decide."""

        class _Bare(EchoAgent):
            system_prompt = ""

        assert self._baseline_seen(_Bare()) is None
