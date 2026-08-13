# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Keras-style optimization proofs: every fitter optimizer lowers loss per epoch.

Part 1 (offline): the public Keras-style API surface from the docs, the
``PromptFitterBridge`` config-application fixes (model params + few-shot +
output contract), the fit() baseline row, and save() with optimize knobs.

Part 2 (live omlx): ``agent.compile() → fit() → evaluate() → save()`` for
every registered fitter optimizer mode (``rewrite``, ``gepa_like``,
``mipro_like``, ``few_shot_bootstrap``, ``param_search``) — asserting the
Keras-style ``loss`` starts at the pre-fit baseline and never increases
across epochs, and the compiled config actually contains the improvements.

Run (live part requires the local oMLX / OpenAI-compatible server)::

    OMLX_API_KEY=… OMLX_BASE_URL=http://127.0.0.1:8000/v1 \\
      uv run pytest tests/test_live_omlx_keras_optimize.py -q --override-ini='addopts='
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

# Make the showcase importable from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Public API only — the exact Keras-style imports from the docs ──────
from agentomatic import (  # noqa: E402
    BaseGraphAgent,
    EarlyStopping,
    GridSearchOptimizer,
    History,
    MetricLoss,
    PromptFitterBridge,
)
from agentomatic.optimize import (  # noqa: E402
    PromptFitResult,
    PromptRuntimeConfig,
    PromptSearchSpace,
)
from examples.keras_optimize_showcase.agent import (  # noqa: E402
    MARKER,
    MARKER2,
    MARKER3,
    MARKER4,
    MarkerAgent,
)
from examples.keras_optimize_showcase.dataset import build_dataset  # noqa: E402
from examples.keras_optimize_showcase.train import (  # noqa: E402
    CURRICULUM_START,
    MAX_DIFFICULTY,
    CurriculumCallback,
    build_metrics,
    search_space_for,
)

MODEL = os.getenv("AGENTOMATIC_LIVE_MODEL", "omlx/Qwen3.5-9B-MLX-4bit")
OMLX_BASE = os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
OMLX_KEY = os.getenv("OMLX_API_KEY", "kurwamac")
ALL_FIT_OPTIMIZERS = ["rewrite", "gepa_like", "mipro_like", "few_shot_bootstrap", "param_search"]


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


pytestmark = pytest.mark.skipif(not _omlx_available(), reason="oMLX server not reachable")


@pytest.fixture(autouse=True)
def _omlx_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMLX_API_KEY", OMLX_KEY)
    monkeypatch.setenv("OMLX_BASE_URL", OMLX_BASE)


# =========================================================================
# Part 1 — offline: public API + engine fixes
# =========================================================================


class TestPublicKerasImports:
    """The docs' Keras-style imports must work from the top-level package."""

    def test_docs_import_line(self) -> None:
        from agentomatic import BaseGraphAgent as _B
        from agentomatic import EarlyStopping as _E
        from agentomatic import History as _H
        from agentomatic import MetricLoss as _ML

        assert _B is BaseGraphAgent
        assert _E is EarlyStopping
        assert _H is History
        assert _ML is MetricLoss

    def test_metrics_exported_top_level(self) -> None:
        import agentomatic

        for name in (
            "CallableMetric",
            "ContainsTermsMetric",
            "ResponseSimilarityMetric",
            "ExactKeyMatchMetric",
            "WeightedMetric",
            "MetricLoss",
            "GridSearchOptimizer",
            "NoOpOptimizer",
            "PromptFitterBridge",
        ):
            assert hasattr(agentomatic, name), f"agentomatic.{name} missing"


class TestBridgeConfigExtraction:
    """PromptFitterBridge must apply model params / few-shot / output contract."""

    @staticmethod
    def _result_with(
        *,
        system_prompt: str = "best prompt",
        model_params: dict[str, Any] | None = None,
        few_shot: list[dict[str, Any]] | None = None,
        output_contract: str | None = None,
    ) -> Any:
        cfg = PromptRuntimeConfig(
            system_prompt=system_prompt,
            model_params=model_params or {},
            few_shot_examples=few_shot or [],
            output_contract=output_contract,
            model_choice=None,
        )
        return type("R", (), {"best_config": cfg})()

    def test_model_params_flattened(self) -> None:
        result = self._result_with(model_params={"temperature": 0.0, "max_tokens": 512})
        config = PromptFitterBridge._extract_config(None, result)
        assert config["system_prompt"] == "best prompt"
        assert config["temperature"] == 0.0
        assert config["max_tokens"] == 512

    def test_few_shot_and_output_contract_carried(self) -> None:
        fs = [{"query": "q", "response": "OPT banana"}]
        result = self._result_with(few_shot=fs, output_contract="json")
        config = PromptFitterBridge._extract_config(None, result)
        assert config["few_shot_examples"] == fs
        assert config["output_contract"] == "json"

    def test_empty_and_none_skipped(self) -> None:
        result = self._result_with(model_params={"temperature": None, "top_p": 0.9})
        config = PromptFitterBridge._extract_config(None, result)
        assert "temperature" not in config
        assert config["top_p"] == 0.9


class TestMarkerAgentSignals:
    """Baseline is fully bad; every optimizable knob moves a distinct signal."""

    def test_baseline_fails_everything(self) -> None:
        agent = MarkerAgent()
        out = agent.transform({"query": "q"})
        assert out["response"].startswith("BASE")
        assert out["banana_ok"] is False
        assert out["strawberry_ok"] is False
        assert out["blueberry_ok"] is False
        assert out["kiwi_ok"] is False
        assert out["temp_ok"] is False
        assert out["temp_rungs"] == 0

    def test_prompt_marker_signals(self) -> None:
        agent = MarkerAgent()
        agent.compiled_config["system_prompt"] = f"mention {MARKER}"
        out = agent.transform({"query": "q"})
        assert out["banana_ok"] is True
        assert out["strawberry_ok"] is False

        agent.compiled_config["system_prompt"] = (
            f"mention {MARKER} and {MARKER2} and {MARKER3} and {MARKER4}"
        )
        out = agent.transform({"query": "q"})
        assert out["banana_ok"] is True
        assert out["strawberry_ok"] is True
        assert out["blueberry_ok"] is True
        assert out["kiwi_ok"] is True
        # At difficulty 1 only the first required signal is emitted.
        assert "banana" in out["response"] and "kiwi" not in out["response"]

    def test_curriculum_gates_emitted_markers(self) -> None:
        agent = MarkerAgent()
        agent.difficulty = 3
        agent.compiled_config["system_prompt"] = (
            f"mention {MARKER} and {MARKER2} and {MARKER3} and {MARKER4}"
        )
        out = agent.transform({"query": "q"})
        assert "banana" in out["response"]
        assert "strawberry" in out["response"]
        assert "blueberry" in out["response"]
        assert "kiwi" not in out["response"]  # not required yet
        assert out["n_satisfied"] == 3

    def test_temperature_rungs(self) -> None:
        agent = MarkerAgent()
        for temp, expected_rungs in [(0.7, 0), (0.5, 1), (0.3, 2), (0.15, 3), (0.0, 3)]:
            agent.temperature = temp
            out = agent.transform({"query": "q"})
            assert out["temp_rungs"] == expected_rungs, temp
        assert out["temp_ok"] is True

    def test_few_shot_block_counts_as_grounding(self) -> None:
        agent = MarkerAgent()
        agent.few_shot_examples = [{"query": "marker?", "response": f"OPT: answer {MARKER}"}]
        out = agent.transform({"query": "q"})
        assert out["banana_ok"] is True

    def test_fitter_injected_temperature_wins(self) -> None:
        agent = MarkerAgent()
        out = agent.transform({"query": "q", "temperature": 0.1})
        assert out["temp_ok"] is True
        assert out["temp_rungs"] == 3


class TestEpochDiffCallback:
    """EpochDiffCallback prints and records per-epoch loss + prompt changes."""

    @staticmethod
    def _fixed_optimizer(config: dict[str, Any]):
        class Fixed:
            def optimize(self, agent, dataset, metrics):
                return dict(config)

        return Fixed()

    def test_records_loss_and_prompt_diff(self) -> None:
        from agentomatic import EpochDiffCallback

        agent = MarkerAgent()
        dataset = build_dataset()
        metrics, loss, _ = build_metrics()
        agent.compile(dataset, metrics=metrics, loss=loss)
        cb = EpochDiffCallback(epochs=2)
        agent.fit(
            dataset,
            epochs=2,
            verbose=0,
            optimizer=self._fixed_optimizer(
                {"system_prompt": f"improved prompt with {MARKER}", "temperature": 0.1}
            ),
            callbacks=[cb],
        )
        assert len(cb.per_epoch) == 2
        first = cb.per_epoch[0]
        assert first["loss"] < 1.0
        diff = first["changes"]["prompt_diff"]
        assert any(line.startswith("+") and "improved" in line for line in diff)
        assert any(line.startswith("-") and "vague" in line for line in diff)
        assert first["changes"]["params"]["temperature"]["new"] == 0.1
        # Second epoch: nothing changed → empty diff, recorded loss.
        second = cb.per_epoch[1]
        assert second["changes"] == {"prompt_diff": [], "params": {}}
        assert second["loss"] == first["loss"]

    def test_exported_top_level(self) -> None:
        import agentomatic

        assert hasattr(agentomatic, "EpochDiffCallback")
        assert agentomatic.EpochDiffCallback is not None


class TestFitBaselineRowAndPersistence:
    """fit() records a pre-optimization baseline; save() survives optimize knobs."""

    def _fixed_optimizer(self, config: dict[str, Any]):
        class Fixed:
            def optimize(self, agent, dataset, metrics):
                return dict(config)

        return Fixed()

    def test_baseline_row_recorded_with_optimizer(self) -> None:
        agent = MarkerAgent()
        dataset = build_dataset()
        metrics, loss, _ = build_metrics()
        agent.compile(dataset, metrics=metrics, loss=loss)
        history = agent.fit(
            dataset,
            epochs=2,
            verbose=0,
            optimizer=self._fixed_optimizer({"system_prompt": f"be precise, mention {MARKER}"}),
        )
        assert history.epoch == [-1, 0, 1]
        # Baseline row = the deliberately-bad starting state.
        assert history["loss"][0] == 1.0
        assert history["banana"][0] == 0.0
        # After the optimizer's config is applied, banana is found.
        assert history["banana"][1] == 1.0
        assert history["loss"][1] < history["loss"][0]
        best = history.best("loss", mode="min")
        assert best is not None and best[0] >= 0

    def test_no_baseline_row_without_optimizer(self) -> None:
        agent = MarkerAgent()
        dataset = build_dataset()
        metrics, loss, _ = build_metrics()
        agent.compile(dataset, metrics=metrics, loss=loss)
        history = agent.fit(dataset, epochs=2, verbose=0)
        assert history.epoch == [0, 1]

    def test_save_after_fit_with_search_space_knobs(self, tmp_path: Path) -> None:
        """Regression: History.params held a PromptSearchSpace → save() crashed."""
        agent = MarkerAgent()
        dataset = build_dataset()
        metrics, loss, _ = build_metrics()
        agent.compile(dataset, metrics=metrics, loss=loss)
        agent.fit(
            dataset,
            epochs=1,
            verbose=0,
            optimizer=self._fixed_optimizer({"temperature": 0.0}),
            search_space=PromptSearchSpace(optimize_model_params=True),
            optimize_mode="param_search",
            optimize_prompt=False,
            optimize_params=True,
            max_trials=2,
        )
        out = tmp_path / "compiled"
        agent.save(out)  # must not raise
        restored = MarkerAgent()
        restored.load(out)
        assert restored.compiled_config == agent.compiled_config
        assert restored.temperature == 0.0


# =========================================================================
# Part 2 — live omlx: every optimizer mode lowers loss per epoch
# =========================================================================


def _fit_mode(
    mode: str, *, epochs: int = 2, max_trials: int = 4, seed: int = 42
) -> tuple[MarkerAgent, History]:
    """Run one Keras-style fit for ``mode``; returns (agent, History)."""
    dataset = build_dataset(seed=seed)
    agent = MarkerAgent()
    metrics, loss, fit_metric = build_metrics()

    bridge = PromptFitterBridge(
        agent_name=agent.agent_name,
        task_model=MODEL,
        rewrite_model=MODEL,
        metric=fit_metric,
        optimizer=mode,
        max_trials=max_trials,
        search_space=search_space_for(mode),
        concurrency=1,
        auto_report=False,
        experiment_dir=".optimize",
        llm_base_url=OMLX_BASE,
        llm_api_key=OMLX_KEY,
        min_absolute_improvement=0.001,
        patience=1,
    )
    agent.compile(dataset, metrics=metrics, loss=loss, optimizer=bridge)
    curriculum = CurriculumCallback(
        max_difficulty=MAX_DIFFICULTY.get(mode, 4),
        start=CURRICULUM_START.get(mode, 1),
    )
    history = agent.fit(
        dataset,
        epochs=epochs,
        verbose=0,
        validation_data=dataset.validation,
        callbacks=[curriculum, EarlyStopping(monitor="val_loss", patience=1, mode="min")],
        optimize_mode=mode,
        search_space=search_space_for(mode),
        max_trials=max_trials,
    )
    return agent, history


@pytest.mark.parametrize("mode", ALL_FIT_OPTIMIZERS)
def test_keras_fit_loss_decreases_per_epoch(mode: str) -> None:
    """Keras-style loss starts at baseline, never increases, and the fitted
    config actually lands on the agent (prompt + params + few-shot)."""
    agent, history = _fit_mode(mode)

    assert "loss" in history and "val_loss" in history, mode
    losses = history["loss"]
    val_losses = history["val_loss"]

    # Epoch -1 is the pre-fit baseline: the deliberately-bad state.
    assert losses[0] == pytest.approx(1.0, abs=1e-9), f"{mode}: baseline loss wrong"
    # Loss is non-increasing across epochs (optimizer never regresses).
    for i in range(len(losses) - 1):
        assert losses[i + 1] <= losses[i] + 1e-9, (
            f"{mode}: loss increased epoch {i} → {i + 1}: {losses[i]:.4f} → {losses[i + 1]:.4f}"
        )
        assert val_losses[i + 1] <= val_losses[i] + 1e-9, f"{mode}: val_loss increased"
    # And strictly better than the baseline by the end.
    assert losses[-1] < losses[0], f"{mode}: loss did not improve: {losses}"

    best = history.best("val_loss", mode="min")
    assert best is not None and best[1] == pytest.approx(val_losses[-1], abs=1e-6)

    # ── the fitted config actually landed on the agent ────────────────
    assert getattr(agent, "_last_optimize_status", None) == "ok", mode
    result: PromptFitResult | None = getattr(agent, "_last_fit_result", None)
    assert result is not None, mode
    assert result.best_score >= result.baseline_score - 1e-9, mode

    if mode == "param_search":
        # Params are found and applied to the agent.
        assert agent.compiled_config.get("temperature", agent.temperature) <= 0.3, mode
        assert history["temp_ok"][-1] == 1.0, mode
    elif mode == "few_shot_bootstrap":
        # Few-shot examples are applied and the primary marker is found.
        assert agent.compiled_config.get("few_shot_examples") or agent.few_shot_examples, mode
        assert history["banana"][-1] == 1.0, mode
    else:
        # Prompt modes discover at least the deterministic marker.
        assert history["banana"][-1] == 1.0, mode
        prompt = agent.compiled_config.get("system_prompt") or agent.system_prompt
        assert (
            MARKER in prompt.lower() or agent.compiled_config.get("few_shot_examples")
        ), f"{mode}: banana missing from applied config: {prompt[:120]}"


@pytest.mark.asyncio
async def test_keras_full_workflow_from_docs(tmp_path: Path) -> None:
    """The exact docs snippet: transform → compile → fit → best → evaluate → save."""
    dataset = build_dataset(seed=7)
    agent = MarkerAgent()

    # transform before any optimization
    result = agent.transform({"query": "What is the budget?"})
    assert result["response"].startswith("BASE")

    metrics, loss, fit_metric = build_metrics()
    bridge = PromptFitterBridge(
        agent_name=agent.agent_name,
        task_model=MODEL,
        rewrite_model=MODEL,
        metric=fit_metric,
        optimizer="rewrite",
        max_trials=4,
        search_space=search_space_for("rewrite"),
        concurrency=1,
        auto_report=False,
        experiment_dir=".optimize",
        llm_base_url=OMLX_BASE,
        llm_api_key=OMLX_KEY,
        min_absolute_improvement=0.001,
        patience=1,
    )
    agent.compile(dataset=dataset, metrics=metrics, loss=loss, optimizer=bridge)

    history = agent.fit(
        dataset,
        epochs=2,
        verbose=0,
        validation_data=dataset.validation,
        callbacks=[EarlyStopping(monitor="val_loss", patience=1, mode="min")],
        optimize_mode="rewrite",
        search_space=search_space_for("rewrite"),
        max_trials=4,
    )
    assert isinstance(history, History)
    assert history is agent.history
    best = history.best("val_loss", mode="min")
    assert best is not None
    assert best[1] < 1.0

    report = agent.evaluate(dataset.test, metrics)
    assert report.scores["quality"] > 0.0  # at least one marker discovered

    out = tmp_path / "compiled" / "v1"
    agent.save(out)
    assert (out / "config.json").exists()
    assert (out / "fit_history.json").exists()

    restored = MarkerAgent()
    restored.load(out)
    assert restored.compiled_config == agent.compiled_config
    improved = restored.transform({"query": "What is the budget?"})
    assert improved["response"].startswith(("OPT", "PARTIAL"))


def test_grid_search_optimizer_improves_params() -> None:
    """GridSearchOptimizer picks the cool temperature from the grid."""
    agent = MarkerAgent()
    dataset = build_dataset()
    metrics, loss, _ = build_metrics()
    agent.compile(
        dataset,
        metrics=metrics,
        loss=loss,
        optimizer=GridSearchOptimizer({"temperature": [0.7, 0.4, 0.2, 0.0]}),
    )
    history = agent.fit(dataset, epochs=1, verbose=0)
    assert agent.compiled_config.get("temperature", 0.7) <= 0.3
    assert history["temp_ok"][-1] == 1.0
