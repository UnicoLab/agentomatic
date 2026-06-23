"""MVP evaluation metrics for class-owned graph agents.

Provides simple, pluggable metrics that implement the ``Metric`` protocol.

Example::

    from agentomatic.agents.metrics import ExactKeyMatchMetric

    metric = ExactKeyMatchMetric(["summary", "risks", "next_steps"])
    score = metric.score(example, prediction)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .types import AgentExample

# ---------------------------------------------------------------------------
# ExactKeyMatchMetric
# ---------------------------------------------------------------------------


class ExactKeyMatchMetric:
    """Check whether prediction contains required keys.

    Scores the fraction of required keys present in the prediction.

    Args:
        required_keys: Keys that must be present in the output.
        name: Optional metric name.

    Example::

        metric = ExactKeyMatchMetric(["summary", "risks"])
        score = metric.score(example, {"summary": "...", "risks": [...]})
        # score == 1.0
    """

    def __init__(
        self,
        required_keys: Sequence[str],
        name: str = "exact_key_match",
    ) -> None:
        self.name = name
        self.required_keys = list(required_keys)

    def score(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
    ) -> float:
        """Score based on fraction of required keys present."""
        if not self.required_keys:
            return 1.0
        found = sum(1 for k in self.required_keys if k in prediction)
        return found / len(self.required_keys)


# ---------------------------------------------------------------------------
# ContainsTermsMetric
# ---------------------------------------------------------------------------


class ContainsTermsMetric:
    """Check whether output text contains expected terms.

    Searches for terms in all string values of the prediction dict.

    Args:
        required_terms: Terms to search for.
        case_sensitive: Whether to do case-sensitive matching.
        name: Optional metric name.
    """

    def __init__(
        self,
        required_terms: Sequence[str],
        *,
        case_sensitive: bool = False,
        name: str = "contains_terms",
    ) -> None:
        self.name = name
        self.required_terms = list(required_terms)
        self.case_sensitive = case_sensitive

    def score(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
    ) -> float:
        """Score based on fraction of terms found in output."""
        if not self.required_terms:
            return 1.0

        # Flatten all string values into one searchable text
        text = self._extract_text(prediction)
        if not self.case_sensitive:
            text = text.lower()

        found = 0
        for term in self.required_terms:
            search_term = term if self.case_sensitive else term.lower()
            if search_term in text:
                found += 1

        return found / len(self.required_terms)

    def _extract_text(self, data: Any) -> str:
        """Recursively extract all text from a data structure."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return " ".join(self._extract_text(v) for v in data.values())
        if isinstance(data, (list, tuple)):
            return " ".join(self._extract_text(item) for item in data)
        return str(data)


# ---------------------------------------------------------------------------
# CallableMetric
# ---------------------------------------------------------------------------


class CallableMetric:
    """Wrap an arbitrary scoring function as a Metric.

    Args:
        name: Metric name.
        fn: Scoring function ``(example, prediction) -> float``.

    Example::

        metric = CallableMetric(
            "custom",
            lambda ex, pred: 1.0 if pred.get("ok") else 0.0,
        )
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[AgentExample, dict[str, Any]], float],
    ) -> None:
        self.name = name
        self._fn = fn

    def score(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
    ) -> float:
        """Score using the wrapped function."""
        return self._fn(example, prediction)


# ---------------------------------------------------------------------------
# OptimizeMetricAdapter
# ---------------------------------------------------------------------------


class OptimizeMetricAdapter:
    """Adapt an ``agentomatic.optimize.BaseMetric`` to the ``Metric`` protocol.

    This bridges the existing optimization metrics to the new
    class-agent evaluation system.

    Args:
        optimize_metric: An instance of ``optimize.BaseMetric``.
        name: Optional override name.
    """

    def __init__(
        self,
        optimize_metric: Any,
        name: str | None = None,
    ) -> None:
        self._metric = optimize_metric
        self.name = name or getattr(optimize_metric, "name", "adapted_metric")

    def score(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
    ) -> float:
        """Score by bridging to the optimize metric's interface."""
        # optimize.BaseMetric expects RunResult-like objects
        try:
            from agentomatic.optimize.runner import RunResult

            run_result = RunResult(
                query=example.input.get("query", ""),
                response=prediction.get("response", str(prediction)),
                expected=example.expected_output.get("response")
                if example.expected_output
                else None,
            )
            result = self._metric.evaluate(run_result)
            return float(getattr(result, "score", result))
        except Exception:
            return 0.0
