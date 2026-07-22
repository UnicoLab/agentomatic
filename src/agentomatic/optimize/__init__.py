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

from agentomatic.optimize.briefing import (
    briefing_limits_for,
    build_full_optimization_briefing,
    extract_prompt_text,
    looks_like_slm,
    multipass_refine_prompt,
    refine_style_for,
    resolve_rewrite_passes,
)

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
from agentomatic.optimize.eval_api import (
    EvalConfig,
    EvaluateResult,
    evaluate_and_report,
    resolve_eval_dataset_path,
    run_eval,
    run_evaluate,
    select_examples,
)
from agentomatic.optimize.eval_contract import EvalContract
from agentomatic.optimize.failure_analysis import (
    DimensionAnalyzer,
    FailureCluster,
    FailureClusterer,
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
from agentomatic.optimize.report import (
    generate_eval_report,
    generate_fit_report,
    generate_html_report,
)
from agentomatic.optimize.search_space import PromptSearchSpace, load_search_space
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
from agentomatic.optimize.train_api import (
    TrainConfig,
    TrainResult,
    load_data,
    prepare_dataset,
    print_train_result,
    run_train,
    run_training,
    train_and_report,
)

__all__ = [
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
    # High-level train API (thin train.py scripts)
    "TrainConfig",
    "TrainResult",
    "run_train",
    "run_training",
    "train_and_report",
    "print_train_result",
    "load_data",
    "prepare_dataset",
    "make_structured_fit_metric",
    "structured_composite_score",
    # High-level eval API (thin eval.py scripts)
    "EvalConfig",
    "EvaluateResult",
    "evaluate_and_report",
    "run_eval",
    "run_evaluate",
    "select_examples",
    "resolve_eval_dataset_path",
    # ── PromptFitter API ──────────────────────────────────────────
    "PromptFitter",
    "PromptRuntimeConfig",
    "PromptCandidate",
    "PromptFitResult",
    "ParamDelta",
    "PromptSearchSpace",
    "load_search_space",
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
