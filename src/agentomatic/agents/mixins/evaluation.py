"""Evaluation mixin — run metrics over a dataset and produce reports.

Calls ``self.transform()`` on each example, scores with provided
metrics, and aggregates into an ``EvaluationReport``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from ..types import (
        AgentDataset,
        AgentExample,
        EvaluationReport,
        Metric,
    )


class EvaluationMixin:
    """Mixin for evaluating an agent against a dataset.

    Expects the host class to implement a ``transform()`` method
    that accepts an input dict and returns a prediction dict.

    Attributes:
        evaluation_history: List of past ``EvaluationReport`` objects.

    Example::

        report = agent.evaluate(dataset, metrics=[accuracy])
        print(report.summary())
    """

    evaluation_history: list[EvaluationReport]

    def evaluate(
        self,
        dataset: AgentDataset | Sequence[AgentExample],
        metrics: Sequence[Metric],
    ) -> EvaluationReport:
        """Run evaluation over a dataset.

        For each example, calls ``self.transform(example.input)``
        and scores the prediction against all provided metrics.
        Errors are caught per-example so that one failure does not
        abort the entire evaluation.

        Args:
            dataset: An ``AgentDataset`` or list of
                ``AgentExample`` objects to evaluate.
            metrics: Sequence of ``Metric`` instances to score
                predictions.

        Returns:
            An ``EvaluationReport`` with per-example results and
            aggregated scores.
        """
        from ..types import (
            EvaluationReport as _EvaluationReport,
        )
        from ..types import (
            ExampleResult,
        )

        # Resolve examples list
        examples: Sequence[AgentExample]
        dataset_name = "dataset"
        if hasattr(dataset, "examples"):
            examples = dataset.examples  # type: ignore[union-attr]
            dataset_name = getattr(dataset, "name", dataset_name)
        else:
            examples = dataset

        agent_name: str = getattr(self, "agent_name", type(self).__name__)
        results: list[ExampleResult] = []

        for example in examples:
            start = time.perf_counter()
            try:
                prediction: dict[str, Any] = self.transform(  # type: ignore[attr-defined]
                    example.input,
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = (time.perf_counter() - start) * 1000
                logger.warning(
                    "Evaluation error on '{}': {}",
                    example.id,
                    exc,
                )
                results.append(
                    ExampleResult(
                        example_id=example.id,
                        error=str(exc),
                        duration_ms=elapsed,
                    )
                )
                continue

            elapsed = (time.perf_counter() - start) * 1000

            # Score with all metrics
            scores: dict[str, float] = {}
            for metric in metrics:
                try:
                    scores[metric.name] = metric.score(example, prediction)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Metric '{}' failed on '{}': {}",
                        metric.name,
                        example.id,
                        exc,
                    )
                    scores[metric.name] = 0.0

            results.append(
                ExampleResult(
                    example_id=example.id,
                    prediction=prediction,
                    scores=scores,
                    duration_ms=elapsed,
                )
            )

        # Aggregate scores across examples
        agg_scores: dict[str, float] = {}
        for metric in metrics:
            metric_scores = [r.scores.get(metric.name, 0.0) for r in results if r.error is None]
            if metric_scores:
                agg_scores[metric.name] = sum(metric_scores) / len(metric_scores)
            else:
                agg_scores[metric.name] = 0.0

        report = _EvaluationReport(
            agent_name=agent_name,
            dataset_name=dataset_name,
            scores=agg_scores,
            example_results=results,
        )

        # Persist to history
        if not hasattr(self, "evaluation_history"):
            self.evaluation_history = []
        self.evaluation_history.append(report)

        return report
