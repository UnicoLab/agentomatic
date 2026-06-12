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
        description="HR policy assistant",
        n_samples=50,
        categories=["leave", "benefits", "expenses"],
    )
"""

from __future__ import annotations

from agentomatic.optimize.dataset import DataPoint, Dataset
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
    ContainsMetric,
    CustomMetric,
    DeepEvalMetric,
    ExactMatchMetric,
    GEvalMetric,
    LLMJudgeMetric,
    RedTeamMetric,
)
from agentomatic.optimize.optimizer import OptimizationResult, PromptOptimizer
from agentomatic.optimize.report import generate_html_report
from agentomatic.optimize.strategies import (
    MIPRO,
    BootstrapRandomSearch,
    ChainOfThought,
    EnsembleOptimizer,
    FewShotBootstrap,
    IterativeRewrite,
    OptimizationStrategy,
)
from agentomatic.optimize.synthesizer import (
    DataSynthesizer,
    augment_dataset,
    generate_dataset,
    generate_from_docs,
    red_team,
)

__all__ = [
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
]
