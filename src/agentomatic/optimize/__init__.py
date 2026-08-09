"""Prompt optimization engine — like model.fit() but for prompts.

Install: ``pip install agentomatic[optimize]``

Usage::

    from agentomatic.optimize import PromptOptimizer, Dataset

    optimizer = PromptOptimizer(
        agent="my_agent",
        metrics=["answer_relevancy"],
        rewrite_llm="ollama/llama3:70b",    # powerful for rewriting
        eval_llm="ollama/mistral:7b",       # fast for evaluation
    )
    result = await optimizer.optimize(
        dataset=Dataset.from_jsonl("qa.jsonl"),
        max_iterations=10,
    )
    result.apply()  # Saves optimized prompt to prompts.json

Synthetic data::

    from agentomatic.optimize import DataSynthesizer

    synth = DataSynthesizer(model="ollama/mistral:7b")
    dataset = await synth.generate(
        description="Support assistant",
        n_samples=50,
        categories=["billing", "delivery", "refunds"],
    )
"""

from __future__ import annotations

# ── LangChain adapter (re-exported from core) ────────────────────────
from agentomatic.langchain_adapter import (  # noqa: F811
    AgentAdapter,
    adapt_langgraph_agent,
    collect_stream,
    dict_to_messages,
    extract_system_prompt,
    inject_config,
    inject_system_prompt,
    is_chain,
    make_config,
    messages_to_dict,
    resolve_prompt,
    serialize_messages,
    tools_to_names,
    wrap_chain_as_async_node,
    wrap_chain_as_node,
)

# ── New: Agent-type detection ─────────────────────────────────────────
from agentomatic.optimize.agent_detect import (
    METRIC_PRESETS,
    AgentType,
    Evaluator,
    detect_agent_type,
    get_metrics_for_agent_type,
    list_available_metrics,
)
from agentomatic.optimize.algorithm import (
    FitterAlgorithm,
    OptimizationAlgorithm,
    as_algorithm,
)
from agentomatic.optimize.algorithms.apo import APOOptimizer
from agentomatic.optimize.briefing import (
    briefing_limits_for,
    build_full_optimization_briefing,
    extract_prompt_text,
    looks_like_slm,
    multipass_refine_prompt,
    refine_style_for,
    resolve_rewrite_passes,
)

# ── New: ML-style callbacks ───────────────────────────────────────────
from agentomatic.optimize.callbacks import (
    Callback,
    CallbackContext,
    EarlyStopping,
    ModelCheckpoint,
    NaNStopping,
    PlateauStopping,
    ProgressLogger,
    ScoreThreshold,
    TemperatureScheduler,
    default_callbacks,
)
from agentomatic.optimize.cli_settings import EvalCliSettings, TrainCliSettings

# ── PromptFitter API (ML-like prompt/config optimisation) ────────────
from agentomatic.optimize.config import (
    ParamDelta,
    PromptCandidate,
    PromptFitResult,
    PromptRuntimeConfig,
)
from agentomatic.optimize.dataset import DataPoint, Dataset

# ── Deployment-first API ─────────────────────────────────────────────
from agentomatic.optimize.deployment import (
    DeploymentRecommendation,
    RolloutConfig,
    build_deployment_recommendation,
)

# ── New: Diversity selector ───────────────────────────────────────────
from agentomatic.optimize.diversity_selector import (
    DiversityConfig,
    DiversitySelector,
)
from agentomatic.optimize.eval_api import (
    EvalConfig,
    EvaluateResult,
    evaluate_and_report,
    print_eval_result,
    resolve_eval_dataset_path,
    run_eval,
    run_evaluate,
    select_examples,
)
from agentomatic.optimize.eval_contract import EvalContract

# ── New: Evals auto-discovery ──────────────────────────────────────
from agentomatic.optimize.evals_discovery import (
    discover_agent_evals,
    generate_pytest_params,
    get_agent_metrics,
    get_agent_test_cases,
    get_agent_thresholds,
    list_agents_with_evals,
)

# ── New: Experiment tracker ───────────────────────────────────────────
from agentomatic.optimize.experiment_tracker import (
    ExperimentTracker,
    get_tracker,
    reset_tracker,
)
from agentomatic.optimize.failure_analysis import (
    DimensionAnalyzer,
    FailureCluster,
    FailureClusterer,
)
from agentomatic.optimize.feedback_dataset import (
    dataset_from_feedback_collector,
    dataset_from_feedback_jsonl,
    feedback_records_to_dataset,
)
from agentomatic.optimize.fitter import PromptFitter
from agentomatic.optimize.fitter_optimizers import (
    BaseFitterOptimizer,
    FewShotBootstrapOptimizer,
    GEPALikeOptimizer,
    MIPROLikeOptimizer,
    ParamSearchOptimizer,
    RewriteOptimizer,
)

# ── New: JSON extractor ───────────────────────────────────────────────
from agentomatic.optimize.json_extractor import JSONExtractor, extract_json
from agentomatic.optimize.judges import (
    JudgeCalibrationSet,
    LocalJudgeMetric,
    MultiJudgePanel,
)
from agentomatic.optimize.learning import (
    EpochLearning,
    check_generalization,
    synthesize_epoch_learning,
)

# ── Pluggable LLM type system ────────────────────────────────────────
from agentomatic.optimize.llm_types import (
    LLMCallable,
    LLMSpec,
    call_llm,
    call_llm_json,
)
from agentomatic.optimize.loop import (
    AVAILABLE_STRATEGIES,
    LoopResult,
    PromptOptimizationLoop,
    StepResult,
    contains_score,
    keyword_overlap,
)
from agentomatic.optimize.metrics import (
    BaseMetric,
    CompositeMetric,
    ContainsMetric,
    CostMetric,
    CustomMetric,
    DeepEvalMetric,
    DeterministicMetric,
    ExactMatchMetric,
    GEvalMetric,
    LatencyMetric,
    LLMJudgeMetric,
    MetricResult,
    RedTeamMetric,
    WeightedMetric,
    resolve_metrics,
)
from agentomatic.optimize.optimizer import OptimizationResult, PromptOptimizer

# ── New: OptimizerMixin ──────────────────────────────────────────────
from agentomatic.optimize.optimizer_mixin import (
    FitResult,
    OptimizerMixin,
)

# ── New: Presets ──────────────────────────────────────────────────────
from agentomatic.optimize.presets import Preset, Presets, to_fitter_kwargs

# ── New: Prompt version control ───────────────────────────────────────
from agentomatic.optimize.prompt_version_control import (
    PromptVersion,
    PromptVersionControl,
)
from agentomatic.optimize.report import (
    generate_eval_report,
    generate_fit_report,
    generate_html_report,
)
from agentomatic.optimize.resources import ResourceBundle, ResourceRegistry
from agentomatic.optimize.reward import (
    FeedbackRewardAdapter,
    MetricRewardAdapter,
    RewardProtocol,
    resolve_reward_adapter,
)
from agentomatic.optimize.rollout import (
    RewardSignal,
    Rollout,
    RolloutSpan,
    RolloutTraceStore,
    rollout_from_run_result,
)
from agentomatic.optimize.search_space import PromptSearchSpace, load_search_space

# ── New: Pydantic-style settings ──────────────────────────────────────
from agentomatic.optimize.settings import (
    AgentTypeEnum,
    AugmentationMethod,
    CallbackType,
    EvalMetric,
    OptimizerPydanticSettings,
    OptimizerSettings,
    generate_env_template,
    show_available_options,
    validate_model_string,
)
from agentomatic.optimize.settings import (
    OptimizationStrategy as OptStrategy,
)
from agentomatic.optimize.strategies import (
    MIPRO,
    BootstrapRandomSearch,
    ChainOfThought,
    EnsembleOptimizer,
    FewShotBootstrap,
    IterativeRewrite,
    OptimizationStrategy,
)
from agentomatic.optimize.structured_metrics import (
    make_structured_fit_metric,
    structured_composite_score,
)
from agentomatic.optimize.synthesizer import (
    DataSynthesizer,
    augment_dataset,
    generate_dataset,
    generate_from_docs,
    red_team,
)
from agentomatic.optimize.trace_adapter import (
    CritiqueExperiment,
    TraceToCritiqueContext,
    TraceToMessages,
)
from agentomatic.optimize.train_api import (
    CompiledAgent,
    TrainConfig,
    TrainResult,
    build_default_metrics,
    compile_agent,
    default_search_space,
    evaluate_agent,
    fit_agent,
    load_data,
    prepare_dataset,
    print_train_result,
    run_train,
    run_training,
    train_and_report,
)

# ── New: Auto-generated training CLI ─────────────────────────────────
from agentomatic.optimize.train_cli import create_train_cli

__all__ = [
    # ── Callbacks ────────────────────────────────────────────────
    "Callback",
    "CallbackContext",
    "EarlyStopping",
    "OptimizeCallback",
    "OptimizeEarlyStopping",
    "ModelCheckpoint",
    "NaNStopping",
    "PlateauStopping",
    "ProgressLogger",
    "ScoreThreshold",
    "TemperatureScheduler",
    "default_callbacks",
    # ── New: Presets ──────────────────────────────────────────────
    "Preset",
    "Presets",
    "to_fitter_kwargs",
    # ── New: Agent-type detection ─────────────────────────────────
    "AgentType",
    "Evaluator",
    "METRIC_PRESETS",
    "detect_agent_type",
    "get_metrics_for_agent_type",
    "list_available_metrics",
    # ── New: Experiment tracker ───────────────────────────────────
    "ExperimentTracker",
    "get_tracker",
    "reset_tracker",
    # ── New: Diversity selector ───────────────────────────────────
    "DiversityConfig",
    "DiversitySelector",
    # ── New: JSON extractor ───────────────────────────────────────
    "JSONExtractor",
    "extract_json",
    # ── New: Prompt version control ───────────────────────────────
    "PromptVersion",
    "PromptVersionControl",
    # ── New: OptimizerMixin ───────────────────────────────────────
    "FitResult",
    "OptimizerMixin",
    # ── New: Auto-generated training CLI ──────────────────────────
    "create_train_cli",
    # ── New: Pydantic-style settings ──────────────────────────────
    "AgentTypeEnum",
    "AugmentationMethod",
    "CallbackType",
    "EvalMetric",
    "OptimizerPydanticSettings",
    "OptimizerSettings",
    "OptStrategy",
    "generate_env_template",
    "show_available_options",
    "validate_model_string",
    # ── New: Evals auto-discovery ───────────────────────────────
    "discover_agent_evals",
    "generate_pytest_params",
    "get_agent_metrics",
    "get_agent_test_cases",
    "get_agent_thresholds",
    "list_agents_with_evals",
    # ── LangChain adapter ───────────────────────────────────────
    "AgentAdapter",
    "adapt_langgraph_agent",
    "collect_stream",
    "dict_to_messages",
    "extract_system_prompt",
    "inject_config",
    "inject_system_prompt",
    "is_chain",
    "make_config",
    "messages_to_dict",
    "serialize_messages",
    "resolve_prompt",
    "tools_to_names",
    "wrap_chain_as_async_node",
    "wrap_chain_as_node",
    # ── Pluggable LLM type system ─────────────────────────────────
    "LLMSpec",
    "LLMCallable",
    "call_llm",
    "call_llm_json",
    # Core — local-first optimization loop
    "PromptOptimizationLoop",
    "LoopResult",
    "StepResult",
    "AVAILABLE_STRATEGIES",
    # Core — HTTP-based optimizer
    "PromptOptimizer",
    "OptimizationResult",
    "Dataset",
    "DataPoint",
    # Built-in scorers
    "keyword_overlap",
    "contains_score",
    # Metrics
    "BaseMetric",
    "ContainsMetric",
    "CustomMetric",
    "DeepEvalMetric",
    "ExactMatchMetric",
    "GEvalMetric",
    "LLMJudgeMetric",
    "RedTeamMetric",
    # Strategies
    "OptimizationStrategy",
    "IterativeRewrite",
    "FewShotBootstrap",
    "ChainOfThought",
    "MIPRO",
    "BootstrapRandomSearch",
    "EnsembleOptimizer",
    # Synthesis & Red Team
    "DataSynthesizer",
    "generate_dataset",
    "augment_dataset",
    "generate_from_docs",
    "red_team",
    # Reports
    "generate_html_report",
    "generate_eval_report",
    # High-level train API (thin train.py scripts + staged Keras-like)
    "TrainConfig",
    "TrainResult",
    "CompiledAgent",
    "run_train",
    "run_training",
    "train_and_report",
    "print_train_result",
    "load_data",
    "prepare_dataset",
    "build_default_metrics",
    "default_search_space",
    "compile_agent",
    "fit_agent",
    "evaluate_agent",
    "make_structured_fit_metric",
    "structured_composite_score",
    # High-level eval API (thin eval.py scripts)
    "EvalConfig",
    "EvaluateResult",
    "evaluate_and_report",
    "print_eval_result",
    "run_eval",
    "run_evaluate",
    "select_examples",
    "resolve_eval_dataset_path",
    # CLI/env settings for flat train/eval scripts
    "TrainCliSettings",
    "EvalCliSettings",
    # ── PromptFitter API ──────────────────────────────────────────
    "PromptFitter",
    "PromptRuntimeConfig",
    "PromptCandidate",
    "PromptFitResult",
    "ParamDelta",
    "PromptSearchSpace",
    "load_search_space",
    # Unified algorithm surface (Lightning-inspired)
    "OptimizationAlgorithm",
    "FitterAlgorithm",
    "as_algorithm",
    "ResourceBundle",
    "ResourceRegistry",
    "Rollout",
    "RolloutSpan",
    "RewardSignal",
    "RolloutTraceStore",
    "rollout_from_run_result",
    "TraceToMessages",
    "TraceToCritiqueContext",
    "CritiqueExperiment",
    "RewardProtocol",
    "MetricRewardAdapter",
    "FeedbackRewardAdapter",
    "resolve_reward_adapter",
    "feedback_records_to_dataset",
    "dataset_from_feedback_collector",
    "dataset_from_feedback_jsonl",
    # Multi-pass briefing (SLM + LLM)
    "looks_like_slm",
    "refine_style_for",
    "briefing_limits_for",
    "resolve_rewrite_passes",
    "build_full_optimization_briefing",
    "multipass_refine_prompt",
    "extract_prompt_text",
    # PromptFitter metrics
    "MetricResult",
    "CompositeMetric",
    "DeterministicMetric",
    "WeightedMetric",
    "resolve_metrics",
    # Judges
    "LocalJudgeMetric",
    "MultiJudgePanel",
    "JudgeCalibrationSet",
    # Fitter optimizers
    "BaseFitterOptimizer",
    "RewriteOptimizer",
    "FewShotBootstrapOptimizer",
    "MIPROLikeOptimizer",
    "GEPALikeOptimizer",
    "ParamSearchOptimizer",
    "APOOptimizer",
    # Failure analysis
    "FailureClusterer",
    "FailureCluster",
    "DimensionAnalyzer",
    # Epoch learning + generalization
    "EpochLearning",
    "check_generalization",
    "synthesize_epoch_learning",
    # ── Deployment-first API ───────────────────────────────────────
    "EvalContract",
    "DeploymentRecommendation",
    "RolloutConfig",
    "build_deployment_recommendation",
    "LatencyMetric",
    "CostMetric",
    "generate_fit_report",
]

# Aliases for Optimize* prefixed names
OptimizeCallback = Callback
OptimizeEarlyStopping = EarlyStopping
