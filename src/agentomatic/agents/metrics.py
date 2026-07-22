"""MVP evaluation metrics for class-owned graph agents.

Provides simple, pluggable metrics that implement the ``Metric`` protocol.

Example::

    from agentomatic.agents.metrics import ExactKeyMatchMetric

    metric = ExactKeyMatchMetric(["summary", "risks", "next_steps"])
    score = metric.score(example, prediction)
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Sequence
from typing import Any

from .types import AgentExample

# ---------------------------------------------------------------------------
# ResponseSimilarityMetric
# ---------------------------------------------------------------------------


class ResponseSimilarityMetric:
    """Fuzzy-match ``prediction["response"]`` against expected output text.

    Uses :class:`difflib.SequenceMatcher` ratio in ``[0, 1]``. Returns
    ``0.0`` when ground truth is missing (honest — never fabricates a
    mid-scale score).

    Args:
        name: Optional metric name.
        fuzzy: When ``False``, require exact (case-insensitive) equality.
    """

    def __init__(self, name: str = "response_similarity", *, fuzzy: bool = True) -> None:
        self.name = name
        self.fuzzy = fuzzy

    def score(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
    ) -> float:
        """Score response text against ``expected_output``."""
        expected = self._expected_text(example)
        if not expected:
            return 0.0
        actual = str(prediction.get("response", "") or "").strip()
        if not actual:
            return 0.0
        if not self.fuzzy:
            return 1.0 if actual.lower() == expected.lower() else 0.0
        return difflib.SequenceMatcher(None, actual.lower(), expected.lower()).ratio()

    @staticmethod
    def _expected_text(example: AgentExample) -> str:
        expected = example.expected_output
        if expected is None:
            return ""
        if isinstance(expected, str):
            return expected.strip()
        if isinstance(expected, dict):
            for key in ("response", "answer", "output", "text"):
                val = expected.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return ""
        return str(expected).strip()


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
# WeightedMetric — composite of multiple metrics with per-metric weights
# ---------------------------------------------------------------------------


class WeightedMetric:
    """Weighted average of multiple ``Metric``-protocol scorers.

    Accepts metrics as either ``(name, metric, weight)`` triples or
    ``(metric, weight)`` pairs (the metric's ``name`` attribute is used
    if the leading name is omitted).  Sub-scores are averaged using
    weight normalisation so weights do not have to sum to ``1.0``.

    Example::

        metric = WeightedMetric(
            [
                ("exact_response", ExactKeyMatchMetric(["response"]), 0.5),
                ("contains_terms", ContainsTermsMetric(["Result"]), 0.3),
                ("has_output", CallableMetric(
                    "has_output",
                    lambda ex, pred: 1.0 if pred.get("response") else 0.0,
                ), 0.2),
            ],
            name="quality",
        )
        score = metric.score(example, prediction)

    Args:
        metrics: Iterable of ``(name, metric, weight)`` triples or
            ``(metric, weight)`` pairs.
        name: Optional composite metric name (default ``"weighted"``).
        last_component_scores: Populated by :meth:`score` with the last
            evaluation's per-component sub-scores for debugging.

    Raises:
        ValueError: If *metrics* is empty or total weight is not positive.
    """

    def __init__(
        self,
        metrics: Sequence[Any],
        *,
        name: str = "weighted",
    ) -> None:
        if not metrics:
            raise ValueError("WeightedMetric requires at least one component metric")

        components: list[tuple[str, Any, float]] = []
        for entry in metrics:
            if isinstance(entry, (list, tuple)):
                if len(entry) == 3:
                    comp_name, comp_metric, comp_weight = entry
                elif len(entry) == 2:
                    comp_metric, comp_weight = entry
                    comp_name = getattr(comp_metric, "name", None) or "component"
                else:
                    raise ValueError(
                        "WeightedMetric entries must be (name, metric, weight) "
                        "or (metric, weight) tuples"
                    )
            else:
                comp_metric = entry
                comp_name = getattr(comp_metric, "name", None) or "component"
                comp_weight = 1.0

            if not hasattr(comp_metric, "score"):
                raise TypeError(
                    f"Component '{comp_name}' does not implement the Metric.score protocol"
                )
            components.append((str(comp_name), comp_metric, float(comp_weight)))

        total_weight = sum(w for _, _, w in components)
        if total_weight <= 0:
            raise ValueError("WeightedMetric total weight must be positive")

        self.name = name
        self._components = components
        self._total_weight = total_weight
        self.last_component_scores: dict[str, float] = {}

    def score(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
    ) -> float:
        """Return the weight-normalised average across all components.

        Component failures (exceptions) are recorded as ``0.0`` sub-scores
        so a single misbehaving metric never blocks evaluation.
        """
        weighted_sum = 0.0
        component_scores: dict[str, float] = {}

        for comp_name, comp_metric, comp_weight in self._components:
            try:
                raw = comp_metric.score(example, prediction)
            except Exception:
                raw = 0.0
            component_scores[comp_name] = float(raw)
            weighted_sum += float(raw) * comp_weight

        self.last_component_scores = component_scores
        return weighted_sum / self._total_weight


# ---------------------------------------------------------------------------
# OptimizeMetricAdapter
# ---------------------------------------------------------------------------


class OptimizeMetricAdapter:
    """Adapt an ``agentomatic.optimize.BaseMetric`` to the ``Metric`` protocol.

    This bridges the existing optimization metrics (e.g. ``LocalJudgeMetric``,
    ``LLMJudgeMetric``, ``CompositeMetric``) to the class-agent evaluation
    system, which expects a synchronous ``score(example, prediction) -> float``
    interface.

    The adapter correctly:
    - Extracts ``query`` / ``expected`` from the ``AgentExample``
    - Serialises the ``prediction`` dict to a JSON string as ``response``
    - Awaits the async ``.evaluate()`` call via
      :func:`agentomatic.async_utils.run_sync` (persistent loop; works
      inside an existing event loop via a worker thread)

    Args:
        optimize_metric: An instance of ``optimize.BaseMetric``.
        name: Optional override name.

    Example::

        from agentomatic.agents import OptimizeMetricAdapter
        from agentomatic.optimize import LocalJudgeMetric

        judge = LocalJudgeMetric(model="openai/my-local-model", criteria="...")
        adapter = OptimizeMetricAdapter(judge, name="judge")
        score = adapter.score(example, prediction)   # float in [0, 1]
    """

    def __init__(
        self,
        optimize_metric: Any,
        name: str | None = None,
    ) -> None:
        self._metric = optimize_metric
        self.name = name or getattr(optimize_metric, "name", "adapted_metric")
        self.last_result: Any | None = None
        """Most recent ``EvalResult`` from :meth:`score` (for report rationales)."""

    def score(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
    ) -> float:
        """Score by calling the wrapped optimize metric synchronously.

        Converts ``AgentExample`` + ``prediction`` to the
        ``(query, response, expected)`` string tuple expected by
        ``optimize.BaseMetric.evaluate()``, awaits the async result,
        and returns the ``score`` as a ``float``.

        The rich ``EvalResult`` (reason / motivation / dimensions) is
        stashed on :attr:`last_result` so ``agent.evaluate`` can attach
        judge rationales to ``ExampleResult.metadata``.
        """
        import json as _json

        from agentomatic.async_utils import run_sync

        self.last_result = None

        # Prefer AgentExample.to_datapoint() so judge query / expected / context
        # match the fit path (question over meta-query, rich expected, snapshot).
        query = ""
        expected: str | None = None
        context: list[str] | None = None
        to_dp = getattr(example, "to_datapoint", None)
        if callable(to_dp):
            try:
                dp = to_dp()
                query = str(getattr(dp, "query", "") or "")
                expected = getattr(dp, "expected_answer", None)
                ctx = getattr(dp, "context", None) or []
                context = [str(c) for c in ctx if c] or None
            except Exception:  # noqa: BLE001
                dp = None
        else:
            dp = None

        if not query:
            inp = example.input
            if hasattr(inp, "get"):
                query = str(
                    inp.get("question")
                    or inp.get("current_query")
                    or inp.get("query")
                    or inp.get("request")
                    or ""
                )
            else:
                query = str(inp)

        if expected is None:
            exp_out = getattr(example, "expected_output", None)
            expected = (
                _json.dumps(exp_out, ensure_ascii=False)
                if isinstance(exp_out, dict)
                else (str(exp_out) if exp_out is not None else None)
            )

        if context is None:
            inp = getattr(example, "input", None) or {}
            raw_ctx = None
            if hasattr(inp, "get"):
                raw_ctx = inp.get("context")
            if isinstance(raw_ctx, dict) and raw_ctx:
                context = [_json.dumps(raw_ctx, ensure_ascii=False)]
            elif isinstance(raw_ctx, list) and raw_ctx:
                context = [str(c) for c in raw_ctx if c]

        # --- serialise prediction as response string ---
        if isinstance(prediction, dict):
            response = str(
                prediction.get("response")
                or prediction.get("answer")
                or _json.dumps(prediction, ensure_ascii=False)
            )
        else:
            response = str(prediction)

        # --- run evaluate() synchronously ---
        # Wrap the entire call (including coro creation) so that metrics whose
        # evaluate() raises synchronously (non-coroutine callables that throw)
        # are also handled gracefully. Use persistent-loop run_sync so OpenAI
        # async clients survive across fit → evaluate.
        try:
            result = run_sync(self._metric.evaluate(query, response, expected, context))
        except Exception:
            return 0.5  # judge unavailable — neutral, don't crash training

        self.last_result = result
        if (getattr(result, "metadata", None) or {}).get("evaluation_failed"):
            return 0.0
        return float(getattr(result, "score", 0.5))
