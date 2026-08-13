# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for the new optimisation modules added from the Agents inspiration project."""

from __future__ import annotations

import pytest

# =====================================================================
# callbacks
# =====================================================================


@pytest.mark.asyncio
async def test_callback_manager_honours_stop_requested() -> None:
    from agentomatic.optimize.callbacks import EarlyStopping
    from agentomatic.optimize.events import CallbackManager, EventData, OptimizationEvent

    es = EarlyStopping(patience=1, min_delta=0.01)
    mgr = CallbackManager([es])
    await mgr.emit(
        OptimizationEvent.FIT_START,
        EventData(agent="bot", total_rounds=5, prompt="p0"),
    )
    await mgr.emit(
        OptimizationEvent.ROUND_START,
        EventData(round_idx=0, total_rounds=5, best_score=0.5, prompt="p0"),
    )
    await mgr.emit(
        OptimizationEvent.ROUND_END,
        EventData(round_idx=0, score=0.5, best_score=0.5, prompt="p0"),
    )
    assert mgr.stop_requested() is False
    await mgr.emit(
        OptimizationEvent.ROUND_START,
        EventData(round_idx=1, total_rounds=5, best_score=0.5, prompt="p0"),
    )
    await mgr.emit(
        OptimizationEvent.ROUND_END,
        EventData(round_idx=1, score=0.5, best_score=0.5, prompt="p0"),
    )
    assert mgr.stop_requested() is True


def test_early_stopping_triggers_after_patience() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, EarlyStopping

    es = EarlyStopping(patience=2, min_delta=0.01)
    ctx = CallbackContext(agent_name="test", total_iterations=10)
    es.on_train_begin(ctx)

    # First score sets baseline
    ctx.current_score = 0.5
    es.on_iteration_end(ctx)
    assert ctx.stop_requested is False
    assert ctx.no_improvement_count == 0

    # Same score — wait count increases
    ctx.current_score = 0.5
    es.on_iteration_end(ctx)
    assert ctx.stop_requested is False
    assert ctx.no_improvement_count == 1

    # Still no improvement — should trigger
    ctx.current_score = 0.5
    es.on_iteration_end(ctx)
    assert ctx.stop_requested is True
    assert ctx.no_improvement_count == 2


def test_early_stopping_resets_on_improvement() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, EarlyStopping

    es = EarlyStopping(patience=3, min_delta=0.005)
    ctx = CallbackContext(agent_name="test", total_iterations=10)
    es.on_train_begin(ctx)

    ctx.current_score = 0.5
    es.on_iteration_end(ctx)
    ctx.current_score = 0.5  # no improvement
    es.on_iteration_end(ctx)
    assert ctx.no_improvement_count == 1

    ctx.current_score = 0.6  # improvement!
    es.on_iteration_end(ctx)
    assert ctx.no_improvement_count == 0
    assert ctx.stop_requested is False


def test_score_threshold_reached() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, ScoreThreshold

    st = ScoreThreshold(threshold=0.85, mode="max")
    ctx = CallbackContext(agent_name="test")
    ctx.current_score = 0.90
    st.on_iteration_end(ctx)
    assert ctx.stop_requested is True


def test_score_threshold_not_reached() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, ScoreThreshold

    st = ScoreThreshold(threshold=0.85, mode="max")
    ctx = CallbackContext(agent_name="test")
    ctx.current_score = 0.50
    st.on_iteration_end(ctx)
    assert ctx.stop_requested is False


def test_nan_stopping_triggers() -> None:
    import math

    from agentomatic.optimize.callbacks import CallbackContext, NaNStopping

    ns = NaNStopping(max_consecutive_nan=2, validate_output=False, nan_rollback=False)
    ctx = CallbackContext(agent_name="test")
    ns.on_train_begin(ctx)

    ctx.current_score = math.nan
    ns.on_evaluation_end(ctx)
    assert ctx.stop_requested is False

    ctx.current_score = math.nan
    ns.on_evaluation_end(ctx)
    assert ctx.stop_requested is True


def test_nan_stopping_ignores_zero_scores() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, NaNStopping

    ns = NaNStopping(max_consecutive_nan=1, validate_output=True, nan_rollback=False)
    ctx = CallbackContext(agent_name="test")
    ns.on_train_begin(ctx)
    ctx.current_score = 0.0
    ns.on_evaluation_end(ctx)
    assert ctx.stop_requested is False


def test_nan_stopping_valid_scores_ok() -> None:
    import math

    from agentomatic.optimize.callbacks import CallbackContext, NaNStopping

    ns = NaNStopping(max_consecutive_nan=2, validate_output=False)
    ctx = CallbackContext(agent_name="test")
    ns.on_train_begin(ctx)

    ctx.current_score = math.nan
    ns.on_evaluation_end(ctx)
    ctx.current_score = 0.5  # valid — resets counter
    ns.on_evaluation_end(ctx)
    ctx.current_score = math.nan
    ns.on_evaluation_end(ctx)
    assert ctx.stop_requested is False  # only 1 consecutive


def test_temperature_scheduler_exponential() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, TemperatureScheduler

    ts = TemperatureScheduler(
        initial_temperature=0.7,
        min_temperature=0.1,
        decay_rate=0.9,
        decay_type="exponential",
    )
    ctx = CallbackContext(agent_name="test")
    # Round 1 keeps the initial temperature (exponent 0).
    ctx.current_iteration = 1
    ts.on_iteration_begin(ctx)
    assert ctx.current_temperature == pytest.approx(0.7)

    ctx.current_iteration = 5  # exponent = 4
    ts.on_iteration_begin(ctx)
    expected = max(0.1, 0.7 * (0.9**4))
    assert ctx.current_temperature == pytest.approx(expected, rel=1e-3)


def test_temperature_scheduler_linear() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, TemperatureScheduler

    ts = TemperatureScheduler(
        initial_temperature=0.7,
        min_temperature=0.3,
        decay_type="linear",
    )
    ctx = CallbackContext(agent_name="test", total_iterations=10)
    ctx.current_iteration = 5  # step=4 → ratio 0.4
    ts.on_iteration_begin(ctx)
    assert ctx.current_temperature == pytest.approx(0.54)


def test_model_checkpoint_creates_file(tmp_path) -> None:  # noqa: ARG001
    import shutil

    from agentomatic.optimize.callbacks import CallbackContext, ModelCheckpoint

    save_dir = tmp_path / "ckpts"
    mc = ModelCheckpoint(save_dir=str(save_dir), save_best_only=False, save_freq=1)
    ctx = CallbackContext(
        agent_name="test",
        current_iteration=1,
        current_score=0.72,
        current_prompt="You are a helpful assistant.",
        best_score=0.72,
    )
    mc.on_train_begin(ctx)
    mc.on_iteration_end(ctx)

    ckpts = list(save_dir.glob("ckpt_*.json"))
    assert len(ckpts) >= 1
    # Cleanup
    shutil.rmtree(save_dir, ignore_errors=True)


def test_plateau_stopping_reduces_temperature() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, PlateauStopping

    ps = PlateauStopping(patience=2, factor=0.5, min_temperature=0.1)
    ctx = CallbackContext(agent_name="test", current_temperature=0.7)
    ps.on_train_begin(ctx)

    ctx.current_score = 0.5
    ps.on_iteration_end(ctx)  # sets best
    ctx.current_score = 0.5  # same
    ps.on_iteration_end(ctx)  # wait=1
    ctx.current_score = 0.5  # same
    ps.on_iteration_end(ctx)  # triggers
    assert ctx.current_temperature == pytest.approx(0.35)

    # Floor check
    ctx.current_temperature = 0.15
    ctx.current_score = 0.5
    ps.on_iteration_end(ctx)
    ctx.current_score = 0.5
    ps.on_iteration_end(ctx)
    assert ctx.current_temperature == pytest.approx(0.1)  # floor


def test_default_callbacks_count() -> None:
    from agentomatic.optimize.callbacks import default_callbacks

    cbs = default_callbacks(patience=3, target_score=0.85)
    assert (
        len(cbs) == 5
    )  # EarlyStopping, ModelCheckpoint, ScoreThreshold, NaNStopping, ProgressLogger


def test_progress_logger(capsys) -> None:
    from agentomatic.optimize.callbacks import CallbackContext, ProgressLogger

    pl = ProgressLogger(show_prompt_diff=False)
    ctx = CallbackContext(agent_name="test", total_iterations=3)
    pl.on_train_begin(ctx)
    ctx.current_score = 0.65
    ctx.best_score = 0.70
    ctx.current_iteration = 1
    pl.on_iteration_end(ctx)
    pl.on_train_end(ctx)
    # Should not raise


# =====================================================================
# presets
# =====================================================================


def test_presets_exist() -> None:
    from agentomatic.optimize.presets import Preset, Presets

    local = Presets.for_local()
    assert isinstance(local, Preset)
    assert local.name == "local"
    assert local.max_iterations == 5

    quality = Presets.for_quality()
    assert quality.name == "quality"
    assert quality.max_iterations == 10

    quick = Presets.for_quick()
    assert quick.name == "quick"
    assert quick.max_iterations == 2


def test_preset_custom_model() -> None:
    from agentomatic.optimize.presets import Preset, Presets

    p = Presets.for_model("ollama/llama3:70b", iterations=7)
    assert isinstance(p, Preset)
    assert p.model == "ollama/llama3:70b"
    assert p.max_iterations == 7


def test_presets_all() -> None:
    from agentomatic.optimize.presets import Presets

    all_presets = Presets.all()
    assert len(all_presets) == 3


def test_to_fitter_kwargs() -> None:
    from agentomatic.optimize.presets import Presets, to_fitter_kwargs

    preset = Presets.for_local()
    kwargs = to_fitter_kwargs(preset)
    assert kwargs["task_model"] == "ollama/mistral:7b"
    assert kwargs["rewrite_model"] == "ollama/mistral:7b"
    # max_iterations=5 rounds × 4 candidates/round → trial budget 20
    assert kwargs["max_trials"] == 20
    assert kwargs["patience"] == 2
    assert kwargs["optimizer"] == "gepa_like"
    # Explicit max_trials keeps PromptFitter budget semantics (no multiply).
    assert to_fitter_kwargs(preset, max_trials=7)["max_trials"] == 7
    # Must be safe for PromptFitter(**kwargs)
    from agentomatic.optimize.fitter import PromptFitter

    PromptFitter(agent="test", **kwargs)


def test_preset_to_config() -> None:
    from agentomatic.optimize.config import PromptRuntimeConfig
    from agentomatic.optimize.presets import Presets

    cfg = Presets.for_local().to_config(system_prompt="Hello")
    assert isinstance(cfg, PromptRuntimeConfig)
    assert cfg.system_prompt == "Hello"
    assert cfg.model_params["temperature"] == 0.7


# =====================================================================
# agent_detect
# =====================================================================


def test_detect_stateless() -> None:
    from agentomatic.optimize.agent_detect import AgentType, detect_agent_type

    class Agent:
        pass

    assert detect_agent_type(Agent()) == AgentType.STATELESS


def test_detect_rag() -> None:
    from agentomatic.optimize.agent_detect import AgentType, detect_agent_type

    class Agent:
        retriever = "retriever"

    assert detect_agent_type(Agent()) == AgentType.RAG


def test_detect_tool_using() -> None:
    from agentomatic.optimize.agent_detect import AgentType, detect_agent_type

    class Agent:
        tools = ["tool1", "tool2"]

    assert detect_agent_type(Agent()) == AgentType.TOOL_USING


def test_detect_conversational() -> None:
    from agentomatic.optimize.agent_detect import AgentType, detect_agent_type

    class Agent:
        memory = {}
        enable_long_term_memory = True

    assert detect_agent_type(Agent()) == AgentType.CONVERSATIONAL


def test_detect_deep_agent() -> None:
    from agentomatic.optimize.agent_detect import AgentType, detect_agent_type

    class Agent:
        subagents = ["a", "b"]

    assert detect_agent_type(Agent()) == AgentType.DEEP_AGENT


def test_deep_agent_takes_priority_over_tools() -> None:
    from agentomatic.optimize.agent_detect import AgentType, detect_agent_type

    class Agent:
        subagents = ["a"]
        tools = ["t1"]

    # Deep agent is checked first
    assert detect_agent_type(Agent()) == AgentType.DEEP_AGENT


def test_evaluator_for_agent() -> None:
    from agentomatic.optimize.agent_detect import AgentType, Evaluator

    class Agent:
        tools = ["t1"]

    e = Evaluator.for_agent(Agent())
    assert e.agent_type == AgentType.TOOL_USING
    assert "tool_call_accuracy" in e.metrics
    assert "tool_selection" in e.metrics


def test_list_available_metrics() -> None:
    from agentomatic.optimize.agent_detect import list_available_metrics

    all_metrics = list_available_metrics()
    assert len(all_metrics) > 5
    assert "answer_relevancy" in all_metrics


def test_get_metrics_for_agent_type() -> None:
    from agentomatic.optimize.agent_detect import AgentType, get_metrics_for_agent_type

    rag_metrics = get_metrics_for_agent_type(AgentType.RAG)
    assert "ragas_faithfulness" in rag_metrics
    assert "ragas_context_precision" in rag_metrics

    stateless_metrics = get_metrics_for_agent_type(AgentType.STATELESS)
    assert "answer_relevancy" in stateless_metrics


def test_metric_presets_key_coverage() -> None:
    from agentomatic.optimize.agent_detect import METRIC_PRESETS, AgentType

    assert AgentType.STATELESS in METRIC_PRESETS
    assert AgentType.RAG in METRIC_PRESETS
    assert AgentType.TOOL_USING in METRIC_PRESETS
    assert AgentType.CONVERSATIONAL in METRIC_PRESETS
    assert AgentType.DEEP_AGENT in METRIC_PRESETS
    assert AgentType.CUSTOM in METRIC_PRESETS


# =====================================================================
# experiment_tracker
# =====================================================================


@pytest.mark.asyncio
async def test_experiment_tracker_lifecycle() -> None:
    from agentomatic.optimize.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(":memory:")
    eid = await tracker.start_experiment("testbot", strategy="iterative")
    assert eid

    await tracker.log_iteration(eid, 1, 0.60, "prompt version 1")
    await tracker.log_iteration(eid, 2, 0.75, "prompt version 2", metrics={"relevancy": 0.85})

    await tracker.end_experiment(eid, final_score=0.75)
    exp = tracker.get_experiment(eid)
    assert exp is not None
    assert exp["best_score"] == 0.75
    assert exp["total_iterations"] == 2
    assert exp["status"] == "completed"


@pytest.mark.asyncio
async def test_experiment_tracker_get_best() -> None:
    from agentomatic.optimize.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(":memory:")

    e1 = await tracker.start_experiment("bot", model="m1")
    await tracker.log_iteration(e1, 1, 0.5)
    await tracker.end_experiment(e1, 0.5, best_score=0.5)

    e2 = await tracker.start_experiment("bot", model="m2")
    await tracker.log_iteration(e2, 1, 0.9)
    await tracker.end_experiment(e2, 0.9, best_score=0.9)

    best = tracker.get_best_experiment("bot")
    assert best is not None
    assert best["best_score"] == 0.9


@pytest.mark.asyncio
async def test_experiment_tracker_get_best_includes_stopped() -> None:
    from agentomatic.optimize.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(":memory:")

    e1 = await tracker.start_experiment("bot", model="m1")
    await tracker.log_iteration(e1, 1, 0.5)
    await tracker.end_experiment(e1, 0.5, best_score=0.5, status="completed")

    e2 = await tracker.start_experiment("bot", model="m2")
    await tracker.log_iteration(e2, 1, 0.95)
    await tracker.end_experiment(e2, 0.95, best_score=0.95, status="stopped")

    best = tracker.get_best_experiment("bot")
    assert best is not None
    assert best["id"] == e2
    assert best["best_score"] == 0.95


@pytest.mark.asyncio
async def test_experiment_tracker_iterations() -> None:
    from agentomatic.optimize.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(":memory:")
    eid = await tracker.start_experiment("bot")
    for i in range(3):
        await tracker.log_iteration(eid, i + 1, 0.5 + i * 0.1)
    await tracker.end_experiment(eid, 0.7)

    iters = tracker.get_iterations(eid)
    assert len(iters) == 3
    assert iters[0]["score"] == 0.5
    assert iters[2]["score"] == 0.7


# =====================================================================
# diversity_selector
# =====================================================================


def test_diversity_selector_basic() -> None:
    from agentomatic.optimize.diversity_selector import DiversitySelector

    class Case:
        def __init__(self, inp, cat="general"):
            self.input = inp
            self.category = cat

    cases = [
        Case("short query"),
        Case("a somewhat longer query that goes on" * 3),
        Case("x" * 250),
        Case("hello", "greeting"),
        Case("goodbye", "greeting"),
        Case("compute the sum", "math"),
    ]

    ds = DiversitySelector()
    selected = ds.select_diverse(cases, num_select=3)
    assert len(selected) <= 3
    assert len(selected) >= 2


def test_diversity_selector_fewer_than_target() -> None:
    from agentomatic.optimize.diversity_selector import DiversitySelector

    class Case:
        def __init__(self, inp):
            self.input = inp
            self.category = "general"

    cases = [Case("a"), Case("b")]
    ds = DiversitySelector()
    selected = ds.select_diverse(cases, num_select=10)
    assert len(selected) == 2  # all returned


def test_diversity_estimator() -> None:
    from agentomatic.optimize.diversity_selector import DiversitySelector

    class Case:
        def __init__(self, inp, cat="general"):
            self.input = inp
            self.category = cat

    diverse = [
        Case("short", "a"),
        Case("medium " * 10, "b"),
        Case("long " * 50, "c"),
    ]
    uniform = [Case("same") for _ in range(5)]
    score_div = DiversitySelector.estimate_diversity(diverse)
    score_uni = DiversitySelector.estimate_diversity(uniform)
    assert score_div > score_uni


# =====================================================================
# json_extractor
# =====================================================================


def test_json_extract_direct() -> None:
    from agentomatic.optimize.json_extractor import JSONExtractor

    ext = JSONExtractor()
    result = ext.extract('{"ok": true, "value": 42}')
    assert result == {"ok": True, "value": 42}


def test_json_extract_code_block() -> None:
    from agentomatic.optimize.json_extractor import JSONExtractor

    ext = JSONExtractor()
    result = ext.extract('```json\n{"key": "val"}\n```')
    assert result == {"key": "val"}


def test_json_extract_embedded() -> None:
    from agentomatic.optimize.json_extractor import JSONExtractor

    ext = JSONExtractor()
    result = ext.extract('Sure! Here is your data: {"items": [1,2,3]}')
    assert result == {"items": [1, 2, 3]}


def test_json_extract_empty() -> None:
    from agentomatic.optimize.json_extractor import JSONExtractor

    ext = JSONExtractor()
    assert ext.extract("") == {}
    assert ext.extract("no json here") == {}


def test_json_extract_repair() -> None:
    from agentomatic.optimize.json_extractor import JSONExtractor

    ext = JSONExtractor()
    result = ext.extract('{"a": 1, "b": 2,}')
    assert result == {"a": 1, "b": 2}


def test_json_extract_list() -> None:
    from agentomatic.optimize.json_extractor import JSONExtractor

    ext = JSONExtractor()
    result = ext.extract_list("[1, 2, 3]")
    assert result == [1, 2, 3]


def test_json_extract_code_block_list() -> None:
    from agentomatic.optimize.json_extractor import JSONExtractor

    ext = JSONExtractor()
    result = ext.extract_list('```json\n[{"a": 1}]\n```')
    assert result == [{"a": 1}]


def test_convenience_extract_json() -> None:
    from agentomatic.optimize.json_extractor import extract_json

    result = extract_json('{"ok": true}')
    assert result == {"ok": True}


# =====================================================================
# prompt_version_control
# =====================================================================


def test_pvc_add_and_get_best() -> None:
    from agentomatic.optimize.prompt_version_control import PromptVersionControl

    pvc = PromptVersionControl("test_agent")
    pvc.add_version("prompt A", score=0.5)
    pvc.add_version("prompt B", score=0.9)
    pvc.add_version("prompt C", score=0.7)

    best = pvc.get_best()
    assert best is not None
    assert best.score == 0.9
    assert best.prompt == "prompt B"


def test_pvc_rollback() -> None:
    from agentomatic.optimize.prompt_version_control import PromptVersionControl

    pvc = PromptVersionControl("test_agent")
    pvc.add_version("v1", score=0.5)
    pvc.add_version("v2", score=0.7)
    pvc.add_version("v3", score=0.6)

    rolled = pvc.rollback(1)
    assert rolled is not None
    assert rolled.prompt == "v2"

    current = pvc.get_current()
    assert current is not None
    assert current.prompt == "v2"


def test_pvc_score_history() -> None:
    from agentomatic.optimize.prompt_version_control import PromptVersionControl

    pvc = PromptVersionControl("test_agent")
    pvc.add_version("a", score=0.3)
    pvc.add_version("b", score=0.7)
    pvc.add_version("c", score=0.5)

    history = pvc.score_history()
    assert history == [0.3, 0.7, 0.5]
    assert pvc.improvement() == pytest.approx(0.4)


def test_pvc_save_and_load(tmp_path) -> None:
    from agentomatic.optimize.prompt_version_control import PromptVersionControl

    save_dir = str(tmp_path / "results")
    pvc = PromptVersionControl("test_agent", save_dir=save_dir)
    pvc.add_version("prompt X", score=0.88)
    pvc.save()

    loaded = PromptVersionControl.load("test_agent", save_dir=save_dir)
    assert loaded.version_count == 1
    best = loaded.get_best()
    assert best is not None
    assert best.score == 0.88


def test_pvc_empty() -> None:
    from agentomatic.optimize.prompt_version_control import PromptVersionControl

    pvc = PromptVersionControl("empty")
    assert pvc.get_best() is None
    assert pvc.get_current() is None
    assert pvc.score_history() == []
    assert pvc.improvement() == 0.0


def test_pvc_max_versions_pruning() -> None:
    from agentomatic.optimize.prompt_version_control import PromptVersionControl

    pvc = PromptVersionControl("test_agent", max_versions=3)
    for i in range(10):
        pvc.add_version(f"v{i}", score=float(i) / 10)
    assert pvc.version_count == 3


# =====================================================================
# settings
# =====================================================================


def test_optimizer_settings_defaults() -> None:
    from agentomatic.optimize.settings import (
        AgentTypeEnum,
        OptimizationStrategy,
        OptimizerSettings,
    )

    s = OptimizerSettings()
    assert s.trainer_model == "ollama/mistral:7b"
    assert s.max_iterations == 5
    assert s.target_score == 0.85
    assert s.strategy == OptimizationStrategy.ITERATIVE_REFINEMENT
    assert s.agent_type == AgentTypeEnum.AUTO


def test_optimizer_settings_presets() -> None:
    from agentomatic.optimize.settings import (
        OptimizationStrategy,
        OptimizerSettings,
    )

    local = OptimizerSettings.for_local()
    assert local.trainer_model == "ollama/mistral:7b"
    assert local.max_iterations == 5

    quality = OptimizerSettings.for_quality()
    assert quality.trainer_model == "openai/gpt-4o"
    assert quality.max_iterations == 10
    assert quality.target_score == 0.90

    quick = OptimizerSettings.for_quick()
    assert quick.max_iterations == 2
    assert quick.strategy == OptimizationStrategy.BOOTSTRAP_FEW_SHOT


def test_optimizer_settings_custom() -> None:
    from agentomatic.optimize.settings import OptimizationStrategy, OptimizerSettings

    s = OptimizerSettings(
        trainer_model="ollama/llama3:70b",
        max_iterations=7,
        strategy=OptimizationStrategy.COMBINED,
        target_score=0.95,
        verbose=False,
    )
    assert s.trainer_model == "ollama/llama3:70b"
    assert s.max_iterations == 7
    assert s.strategy == OptimizationStrategy.COMBINED
    assert s.verbose is False


def test_optimizer_settings_display(capsys) -> None:
    from agentomatic.optimize.settings import OptimizerSettings

    s = OptimizerSettings.for_local()
    s.display()
    captured = capsys.readouterr()
    assert "OptimizerSettings" in captured.out


def test_generate_env_template() -> None:
    from agentomatic.optimize.settings import generate_env_template

    tmpl = generate_env_template()
    assert "OPTIMIZER_TRAINER_MODEL" in tmpl
    assert "OPTIMIZER_MAX_ITERATIONS" in tmpl
    assert "OPTIMIZER_STRATEGY" in tmpl


def test_validate_model_string() -> None:
    from agentomatic.optimize.settings import validate_model_string

    assert validate_model_string("ollama/mistral:7b") is True
    assert validate_model_string("openai/gpt-4o") is True
    assert validate_model_string("invalid") is False


# =====================================================================
# optimizer_mixin
# =====================================================================


def test_fit_result_summary() -> None:
    from agentomatic.optimize.optimizer_mixin import FitResult

    r = FitResult(
        agent_name="test_bot",
        initial_score=0.42,
        final_score=0.78,
        improvement=0.36,
        iterations=5,
        improved=True,
    )
    summary = r.summary()
    assert "test_bot" in summary
    assert "0.420" in summary
    assert "0.780" in summary
    assert "+0.360" in summary


def test_optimizer_mixin_has_fit() -> None:
    from agentomatic.optimize.optimizer_mixin import OptimizerMixin

    assert hasattr(OptimizerMixin, "fit")
    assert callable(OptimizerMixin.fit)
    # Async extension alias
    assert hasattr(OptimizerMixin, "optimize_prompts")
    assert callable(OptimizerMixin.optimize_prompts)


def test_optimizer_mixin_fit_empty_cases() -> None:
    from agentomatic.optimize.optimizer_mixin import FitResult, OptimizerMixin

    class Agent(OptimizerMixin):
        agent_name = "empty_bot"

    result = Agent().fit([])
    assert isinstance(result, FitResult)
    assert result.improved is False


def test_callback_manager_temperature_default_none() -> None:
    from agentomatic.optimize.callbacks import EarlyStopping
    from agentomatic.optimize.events import CallbackManager

    mgr = CallbackManager([EarlyStopping(patience=2)])
    assert mgr.current_temperature() is None


def test_detect_agent_type_bool_is_deep_agent() -> None:
    from types import SimpleNamespace

    from agentomatic.optimize.agent_detect import AgentType, detect_agent_type

    agent = SimpleNamespace(capabilities=SimpleNamespace(is_deep_agent=False))
    assert detect_agent_type(agent) == AgentType.STATELESS


def test_stopped_experiments_are_listed(tmp_path) -> None:
    import asyncio

    from agentomatic.optimize.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(db_path=str(tmp_path / "e.db"))

    async def _run() -> None:
        eid = await tracker.start_experiment("bot", strategy="gepa", model="m")
        await tracker.end_experiment(eid, final_score=0.8, status="stopped")

    asyncio.run(_run())
    rows = tracker.get_experiments()
    assert len(rows) == 1
    assert rows[0]["status"] == "stopped"


def test_optimizer_mixin_accessors() -> None:
    from agentomatic.optimize.optimizer_mixin import OptimizerMixin

    mixin = OptimizerMixin()
    assert mixin.get_optimized_prompt() is None
    assert mixin.get_optimization_history() == []
    assert mixin.last_fit_result is None
    mixin.reset_optimization()  # should not raise


def test_fit_result_from_prompt_fit_result() -> None:
    from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
    from agentomatic.optimize.optimizer_mixin import FitResult

    result = PromptFitResult(
        best_config=PromptRuntimeConfig(system_prompt="best"),
        baseline_config=PromptRuntimeConfig(system_prompt="base"),
        best_score=0.9,
        baseline_score=0.4,
        score_history=[0.4, 0.7, 0.9],
        agent="bot",
        optimizer_name="gepa_like",
    )
    wrapped = FitResult.from_prompt_fit_result(result)
    assert wrapped.agent_name == "bot"
    assert wrapped.initial_score == 0.4
    assert wrapped.final_score == 0.9
    assert wrapped.iterations == 3
    assert wrapped.improved is True
    assert wrapped.best_prompt == "best"
    assert wrapped.strategy == "gepa_like"


def test_fit_result_iterations_prefer_prompt_history() -> None:
    """Baseline-seeded score_history must not inflate round count."""
    from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
    from agentomatic.optimize.optimizer_mixin import FitResult

    result = PromptFitResult(
        best_config=PromptRuntimeConfig(system_prompt="best"),
        baseline_config=PromptRuntimeConfig(system_prompt="base"),
        best_score=0.9,
        baseline_score=0.4,
        score_history=[0.4, 0.7, 0.9],  # baseline + 2 rounds
        prompt_history=[
            {"round_idx": 0, "score": 0.7},
            {"round_idx": 1, "score": 0.9},
        ],
        agent="bot",
    )
    wrapped = FitResult.from_prompt_fit_result(result)
    assert wrapped.iterations == 2


@pytest.mark.asyncio
async def test_wrap_local_agent_restores_none_system_prompt() -> None:
    """Prompt override must not stick when the agent started with None."""
    from agentomatic.optimize.fitter import _wrap_local_agent

    class Agent:
        system_prompt = None

        def transform(self, data):
            return {"response": data.get("current_query", "")}

    agent = Agent()
    fn = _wrap_local_agent(agent)
    await fn("hello", prompt_override="OVERRIDE")
    assert agent.system_prompt is None


def test_progress_logger_delta_with_zero_best_score() -> None:
    from agentomatic.optimize.callbacks import CallbackContext, ProgressLogger

    pl = ProgressLogger()
    ctx = CallbackContext(agent_name="t", best_score=0.0, current_score=0.5, current_iteration=1)
    pl.on_train_begin(ctx)
    pl.on_iteration_end(ctx)  # must not treat best_score=0.0 as missing


# =====================================================================
# train_cli
# =====================================================================


def test_create_train_cli_returns_click_group() -> None:
    from agentomatic.optimize.train_cli import create_train_cli

    class FakeAgent:
        agent_name = "test_agent"
        agent_description = "A test agent"

    cli = create_train_cli(FakeAgent(), description="Test CLI")
    assert cli is not None
    assert hasattr(cli, "__call__")


def test_create_train_cli_with_test_cases_fn() -> None:
    from agentomatic.optimize.train_cli import create_train_cli

    class FakeAgent:
        agent_name = "test_agent"

    def get_cases():
        return []

    cli = create_train_cli(FakeAgent(), test_cases_fn=get_cases)
    assert cli is not None


def test_train_cli_do_train_builds_valid_fitter() -> None:
    """_do_train must construct PromptFitter without invalid kwargs."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig
    from agentomatic.optimize.train_cli import _do_train

    fake_result = PromptFitResult(
        best_config=PromptRuntimeConfig(system_prompt="best"),
        baseline_config=PromptRuntimeConfig(system_prompt="base"),
        best_score=0.9,
        baseline_score=0.5,
        score_history=[0.5, 0.9],
        agent="cli_bot",
        optimizer_name="gepa_like",
    )

    with patch("agentomatic.optimize.fitter.PromptFitter") as Fitter:
        inst = MagicMock()
        inst.fit = AsyncMock(return_value=fake_result)
        Fitter.return_value = inst
        cases = [SimpleNamespace(input="q", expected_output="a")]
        result = asyncio.run(
            _do_train(
                agent=SimpleNamespace(agent_name="cli_bot"),
                test_cases=cases,
                model="ollama/mistral:7b",
                iterations=3,
                target=0.9,
                strategy="bootstrap_few_shot",
                verbose=True,
                output_dir=".optimize-test",
            )
        )
        kwargs = Fitter.call_args.kwargs
        assert "verbose" not in kwargs
        assert kwargs["optimizer"] == "few_shot"
        # iterations=3 rounds × 4 candidates/round
        assert kwargs["max_trials"] == 12
        assert result.final_score == 0.9


# =====================================================================
# evals_discovery
# =====================================================================


def test_evals_discovery_empty_dir(tmp_path) -> None:
    from agentomatic.optimize.evals_discovery import discover_agent_evals, list_agents_with_evals

    # Empty dir — no agents
    discovered = discover_agent_evals(str(tmp_path))
    assert discovered == {}
    assert list_agents_with_evals(str(tmp_path)) == []


def test_evals_discovery_finds_agent(tmp_path) -> None:
    from agentomatic.optimize.evals_discovery import (
        discover_agent_evals,
        get_agent_test_cases,
        get_agent_thresholds,
        list_agents_with_evals,
    )

    # Create a mock agent folder with evals.py
    agent_dir = tmp_path / "test_bot"
    agent_dir.mkdir()
    evals_file = agent_dir / "evals.py"
    evals_file.write_text("""
AGENT_NAME = "test_bot"
AGENT_DESCRIPTION = "A test bot"
THRESHOLDS = {"AnswerRelevancyMetric": 0.5}

def get_test_cases():
    class FakeCase:
        def __init__(self, inp, exp):
            self.input = inp
            self.expected_output = exp
            self.context = []
    return [FakeCase("hello", "hi there"), FakeCase("bye", "goodbye")]

def get_custom_metrics():
    return ["custom_metric_1"]
""")

    discovered = discover_agent_evals(str(tmp_path))
    assert "test_bot" in discovered
    assert discovered["test_bot"]["agent_name"] == "test_bot"
    assert discovered["test_bot"]["description"] == "A test bot"
    assert discovered["test_bot"]["thresholds"] == {"AnswerRelevancyMetric": 0.5}
    assert len(discovered["test_bot"]["test_cases"]) == 2
    assert discovered["test_bot"]["custom_metrics"] == ["custom_metric_1"]

    # Accessor functions
    cases = get_agent_test_cases("test_bot", str(tmp_path))
    assert len(cases) == 2
    assert cases[0].input == "hello"

    thresholds = get_agent_thresholds("test_bot", str(tmp_path))
    assert thresholds == {"AnswerRelevancyMetric": 0.5}

    names = list_agents_with_evals(str(tmp_path))
    assert names == ["test_bot"]


def test_evals_discovery_no_evals_file(tmp_path) -> None:
    from agentomatic.optimize.evals_discovery import discover_agent_evals

    agent_dir = tmp_path / "no_evals_bot"
    agent_dir.mkdir()
    # No evals.py — should not be discovered
    discovered = discover_agent_evals(str(tmp_path))
    assert "no_evals_bot" not in discovered


def test_evals_discovery_skips_underscore_dirs(tmp_path) -> None:
    from agentomatic.optimize.evals_discovery import discover_agent_evals

    hidden_dir = tmp_path / "__pycache__"
    hidden_dir.mkdir()
    discovered = discover_agent_evals(str(tmp_path))
    assert "__pycache__" not in discovered


def test_generate_pytest_params(tmp_path) -> None:
    from agentomatic.optimize.evals_discovery import generate_pytest_params

    agent_dir = tmp_path / "bot"
    agent_dir.mkdir()
    (agent_dir / "evals.py").write_text("""
AGENT_NAME = "bot"
def get_test_cases():
    class Case:
        pass
    return [Case(), Case(), Case()]
""")

    params = generate_pytest_params(str(tmp_path))
    assert len(params) == 3
    assert params[0][0] == "bot"
    assert params[0][1] is not None
    assert params[0][2] == "bot_case_0"


# =====================================================================
# langchain_adapter
# =====================================================================


def test_dict_state_to_messages() -> None:
    from agentomatic.langchain_adapter import dict_to_messages

    state = {"current_query": "What is Python?"}
    msgs = dict_to_messages(state)
    assert len(msgs) >= 1
    last = msgs[-1]
    if hasattr(last, "content"):
        assert "Python" in str(last.content)
    else:
        assert "Python" in str(last.get("content", ""))


def test_dict_state_to_messages_with_history() -> None:
    from agentomatic.langchain_adapter import dict_to_messages

    state = {
        "current_query": "New question",
        "messages": [
            {"role": "user", "content": "Previous"},
            {"role": "ai", "content": "Previous answer"},
        ],
    }
    msgs = dict_to_messages(state)
    assert len(msgs) >= 3


def test_messages_to_dict_state() -> None:
    from agentomatic.langchain_adapter import dict_to_messages, messages_to_dict

    orig = {"current_query": "Hello", "thread_id": "t1"}
    msgs = dict_to_messages(orig)
    restored = messages_to_dict(msgs, orig)
    assert "current_query" in restored
    assert "messages" in restored
    assert len(restored["messages"]) >= 1


def test_make_config() -> None:
    from agentomatic.langchain_adapter import make_config

    cfg = make_config()
    assert "recursion_limit" in cfg
    assert cfg["recursion_limit"] == 25

    cfg2 = make_config(thread_id="abc123", tags=["test"])
    assert cfg2["configurable"]["thread_id"] == "abc123"
    assert cfg2["tags"] == ["test"]


def test_inject_config() -> None:
    from agentomatic.langchain_adapter import inject_config

    state = {"current_query": "test"}
    result = inject_config(state, thread_id="t42")
    assert "runnable_config" in result
    assert result["runnable_config"]["configurable"]["thread_id"] == "t42"
    assert result["current_query"] == "test"


def test_is_chain() -> None:
    from agentomatic.langchain_adapter import is_chain

    class FakeChain:
        def invoke(self, x):
            return x

        async def ainvoke(self, x):
            return x

        def stream(self, x):
            yield x

    assert is_chain(FakeChain()) is True
    assert is_chain("not a chain") is False


def test_extract_system_prompt_plain_string() -> None:
    from agentomatic.langchain_adapter import extract_system_prompt

    assert extract_system_prompt("You are helpful.") == "You are helpful."
    assert extract_system_prompt(None, default="fallback") == "fallback"


def test_extract_system_prompt_attribute() -> None:
    from agentomatic.langchain_adapter import extract_system_prompt

    class WithPrompt:
        system_prompt = "You are a bot."

    assert extract_system_prompt(WithPrompt()) == "You are a bot."

    class WithMessage:
        system_message = "Different attr."

    assert extract_system_prompt(WithMessage()) == "Different attr."


def test_inject_system_prompt_plain() -> None:
    from agentomatic.langchain_adapter import inject_system_prompt

    class FakeTemplate:
        def __init__(self):
            self.system_prompt = "Old prompt"

    t = FakeTemplate()
    t2 = inject_system_prompt(t, "New prompt")
    assert t2 is t


# =====================================================================
# Pydantic settings
# =====================================================================


def test_pydantic_settings_defaults() -> None:
    from agentomatic.optimize.settings import OptimizerPydanticSettings

    s = OptimizerPydanticSettings()
    assert s.trainer_model == "ollama/mistral:7b"
    assert s.max_iterations == 5
    assert s.verbose is True


def test_pydantic_settings_presets() -> None:
    from agentomatic.optimize.settings import OptimizerPydanticSettings

    local = OptimizerPydanticSettings.for_local()
    assert local.max_iterations == 5
    assert local.trainer_model == "ollama/mistral:7b"

    quality = OptimizerPydanticSettings.for_quality()
    assert quality.max_iterations == 10
    assert quality.target_score == 0.90
    assert quality.strategy == "combined"

    quick = OptimizerPydanticSettings.for_quick()
    assert quick.max_iterations == 2
    assert quick.verbose is False


def test_pydantic_settings_properties() -> None:
    from agentomatic.optimize.settings import OptimizerPydanticSettings

    s = OptimizerPydanticSettings(eval_metrics="relevancy,geval,faithfulness")
    assert s.eval_metrics_list == ["relevancy", "geval", "faithfulness"]

    s2 = OptimizerPydanticSettings(callbacks="early_stopping,checkpoint")
    assert s2.callbacks_list == ["early_stopping", "checkpoint"]


def test_pydantic_settings_display(capsys) -> None:
    from agentomatic.optimize.settings import OptimizerPydanticSettings

    s = OptimizerPydanticSettings.for_local()
    s.display()
    captured = capsys.readouterr()
    assert "OptimizerPydanticSettings" in captured.out


def test_pydantic_settings_to_dict() -> None:
    from agentomatic.optimize.settings import OptimizerPydanticSettings

    s = OptimizerPydanticSettings(trainer_model="test/model")
    d = s.to_dict()
    assert d["trainer_model"] == "test/model"
    assert "max_iterations" in d
