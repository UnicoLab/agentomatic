"""Tests for agent evaluation metrics."""

from __future__ import annotations

from typing import Any

import pytest

from agentomatic.agents.metrics import (
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
)
from agentomatic.agents.types import AgentExample, Metric

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_example(
    expected_output: dict[str, Any] | None = None,
) -> AgentExample:
    """Create a test example with optional expected output."""
    return AgentExample(
        id="test_ex",
        input={"query": "test"},
        expected_output=expected_output,
    )


# ===========================================================================
# ExactKeyMatchMetric tests
# ===========================================================================


class TestExactKeyMatchMetric:
    """Test ExactKeyMatchMetric scoring."""

    def test_scores_1_when_all_keys_present(self):
        """Score should be 1.0 when all required keys are in prediction."""
        metric = ExactKeyMatchMetric(["summary", "risks"])
        example = _make_example({"summary": "x", "risks": ["r1"]})
        score = metric.score(
            example,
            {"summary": "got it", "risks": ["r1", "r2"]},
        )
        assert score == 1.0

    def test_scores_0_when_all_keys_missing(self):
        """Score should be 0.0 when no required keys are in prediction."""
        metric = ExactKeyMatchMetric(["summary", "risks"])
        example = _make_example({"summary": "x", "risks": []})
        score = metric.score(example, {"other": "value"})
        assert score == 0.0

    def test_scores_partial_when_some_keys_present(self):
        """Score should be fractional when only some keys present."""
        metric = ExactKeyMatchMetric(["a", "b", "c", "d"])
        example = _make_example()
        score = metric.score(example, {"a": 1, "b": 2})
        assert score == pytest.approx(0.5)

    def test_empty_required_keys_returns_1(self):
        """Score should be 1.0 when no required keys specified."""
        metric = ExactKeyMatchMetric([])
        example = _make_example()
        score = metric.score(example, {"any": "value"})
        assert score == 1.0

    def test_custom_name(self):
        """Metric should use custom name when provided."""
        metric = ExactKeyMatchMetric(["a"], name="custom_match")
        assert metric.name == "custom_match"


# ===========================================================================
# ContainsTermsMetric tests
# ===========================================================================


class TestContainsTermsMetric:
    """Test ContainsTermsMetric scoring."""

    def test_scores_based_on_term_presence(self):
        """Score should reflect fraction of terms found."""
        metric = ContainsTermsMetric(["python", "machine learning"])
        example = _make_example()
        score = metric.score(
            example,
            {"text": "Python is great for machine learning"},
        )
        assert score == 1.0

    def test_partial_term_match(self):
        """Score should be partial when not all terms found."""
        metric = ContainsTermsMetric(["python", "rust", "go"])
        example = _make_example()
        score = metric.score(example, {"text": "I love Python"})
        assert score == pytest.approx(1.0 / 3.0)

    def test_case_insensitive_by_default(self):
        """Terms should match case-insensitively by default."""
        metric = ContainsTermsMetric(["PYTHON"])
        example = _make_example()
        score = metric.score(example, {"text": "python is great"})
        assert score == 1.0

    def test_empty_terms_returns_1(self):
        """Score should be 1.0 when no terms specified."""
        metric = ContainsTermsMetric([])
        example = _make_example()
        score = metric.score(example, {"anything": "here"})
        assert score == 1.0


# ===========================================================================
# CallableMetric tests
# ===========================================================================


class TestCallableMetric:
    """Test CallableMetric wrapping arbitrary functions."""

    def test_wraps_arbitrary_function(self):
        """CallableMetric should delegate to the wrapped function."""

        def my_scorer(
            ex: AgentExample,
            pred: dict[str, Any],
        ) -> float:
            return 1.0 if "result" in pred else 0.0

        metric = CallableMetric("has_result", my_scorer)
        example = _make_example()

        assert metric.score(example, {"result": "yes"}) == 1.0
        assert metric.score(example, {"other": "no"}) == 0.0

    def test_custom_name(self):
        """CallableMetric should store custom name."""
        metric = CallableMetric(
            "custom",
            lambda ex, pred: 0.5,
        )
        assert metric.name == "custom"


# ===========================================================================
# Metric protocol compliance
# ===========================================================================


class TestMetricProtocol:
    """Test that all metrics satisfy the Metric protocol."""

    def test_exact_key_match_satisfies_protocol(self):
        """ExactKeyMatchMetric should be a Metric."""
        metric = ExactKeyMatchMetric(["key"])
        assert isinstance(metric, Metric)

    def test_contains_terms_satisfies_protocol(self):
        """ContainsTermsMetric should be a Metric."""
        metric = ContainsTermsMetric(["term"])
        assert isinstance(metric, Metric)

    def test_callable_metric_satisfies_protocol(self):
        """CallableMetric should be a Metric."""
        metric = CallableMetric("test", lambda ex, pred: 0.0)
        assert isinstance(metric, Metric)
