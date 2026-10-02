"""MVP evaluation metrics for class-owned graph agents.

Provides simple, pluggable metrics that implement the ``Metric`` protocol.

Example::

    from agentomatic.agents.metrics import ExactKeyMatchMetric

    metric = ExactKeyMatchMetric(["summary", "risks", "next_steps"])
    score = metric.score(example, prediction)
"""

from __future__ import annotations

import difflib
import weakref
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from loguru import logger

from .types import AgentExample

#: Per-metric cache of successful judge results, keyed by what was judged.
#: The same judge is routinely wrapped several times (as a compiled metric,
#: inside a WeightedMetric, inside the loss); without this every wrapper paid
#: for its own LLM call on the same (question, answer, reference).
_RESULT_CACHE: weakref.WeakKeyDictionary[Any, OrderedDict[tuple[Any, ...], Any]] = (
    weakref.WeakKeyDictionary()
)
_RESULT_CACHE_SIZE = 4096

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
        for key in ("response", "answer", "output", "text"):
            val = expected.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""


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

            try:
                comp_metric = as_agent_metric(comp_metric)
            except TypeError as exc:
                raise TypeError(f"Component '{comp_name}': {exc}") from exc
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
            except Exception as exc:  # noqa: BLE001 - see docstring
                logger.warning(f"WeightedMetric component '{comp_name}' failed: {exc}")
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

    The adapter speaks **both** metric protocols, so the same object can be
    used as a ``compile(metrics=[...])`` entry, inside ``agents.WeightedMetric``
    / ``MetricLoss``, *and* as the ``metric=`` fit objective of
    ``PromptFitterBridge`` / ``PromptFitter`` (which call the async
    :meth:`evaluate`).

    A failed evaluation (judge unreachable, unparsable reply) scores ``0.0``
    and is logged and counted in :attr:`failures` — never a made-up
    mid-scale score that would hide the outage from the optimizer.

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
        if not callable(getattr(optimize_metric, "evaluate", None)):
            raise TypeError(
                f"OptimizeMetricAdapter wraps an optimize metric with an async "
                f"`evaluate(query, response, expected, context)`; got "
                f"{type(optimize_metric).__name__}. Class-agent metrics with "
                "`score(example, prediction)` can be used directly."
            )
        self._metric = optimize_metric
        self.name = name or getattr(optimize_metric, "name", "adapted_metric")
        self.last_result: Any | None = None
        """Most recent ``EvalResult`` from :meth:`score` (for report rationales)."""
        self.failures = 0
        """Number of evaluations that failed and were scored ``0.0``."""

    @property
    def optimize_metric(self) -> Any:
        """The wrapped ``optimize.BaseMetric``."""
        return self._metric

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> Any:
        """Delegate to the wrapped metric's async ``evaluate`` (fit objective path).

        Returns:
            The wrapped metric's ``EvalResult``.
        """
        return await self._metric.evaluate(query, response, expected, context)

    async def _evaluate_as_run(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
        query: str,
        response: str,
        expected: str | None,
        context: list[str] | None,
    ) -> Any:
        """Evaluate with the example exposed as the current scoring run.

        Inside ``PromptFitter`` every metric sees the run being scored (the
        example's metadata and the structured output), so a nested
        class-agent metric — e.g. one reading ``metadata["must_include"]``
        inside a ``CompositeMetric`` — scores the same during ``evaluate()``
        as it did while candidates were selected.
        """
        from agentomatic.optimize.metrics import scoring_run
        from agentomatic.optimize.runner import RunResult

        # The agent inputs ride in ``metadata.invoke``, as they do in fit().
        raw_input = getattr(example, "input", None)
        invoke = (
            {
                k: v
                for k, v in raw_input.items()
                if k not in {"query", "request", "question", "current_query"}
            }
            if isinstance(raw_input, dict)
            else {}
        )
        run = RunResult(
            query=query,
            response=response,
            expected=expected,
            context=list(context or []),
            metadata={
                "example": {
                    **({"invoke": invoke} if invoke else {}),
                    **(getattr(example, "metadata", None) or {}),
                    "id": getattr(example, "id", ""),
                    "split": getattr(example, "split", ""),
                },
                "example_tags": list(getattr(example, "tags", None) or []),
                "output": dict(prediction),
            },
        )
        with scoring_run(run):
            return await self._metric.evaluate(query, response, expected, context)

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
            from agentomatic.optimize.dataset import normalize_context

            inp = getattr(example, "input", None) or {}
            raw_ctx = inp.get("context") if hasattr(inp, "get") else None
            context = normalize_context(raw_ctx) or None
        if context is None:
            # No reference documents in the example: judge groundedness
            # against what the agent says it retrieved (as PromptFitter does).
            from agentomatic.optimize.runner import retrieval_from_output

            context = retrieval_from_output(prediction) or None

        # --- the text judged: the same rule PromptFitter scores with ---
        # (structured ``output`` dict as JSON, else ``response``/``answer``),
        # so a metric reports on exactly what candidates were selected on.
        from agentomatic.optimize.runner import _response_text

        if isinstance(prediction.get("output"), dict) and prediction.get("output"):
            response = _response_text(prediction)
        else:
            response = str(
                prediction.get("response")
                or prediction.get("answer")
                or _json.dumps(prediction, ensure_ascii=False)
            )

        # --- run evaluate() synchronously (cached per metric) ---
        key = (getattr(example, "id", ""), query, response, expected, tuple(context or ()))
        try:
            cache = _RESULT_CACHE.setdefault(self._metric, OrderedDict())
        except TypeError:  # not weak-referenceable: no caching
            cache = OrderedDict()
        result = cache.get(key)
        if result is None:
            # Wrap the entire call (including coro creation) so that metrics
            # whose evaluate() raises synchronously are handled too. Use the
            # persistent-loop run_sync so async clients survive fit → evaluate.
            try:
                result = run_sync(
                    self._evaluate_as_run(example, prediction, query, response, expected, context)
                )
            except Exception as exc:  # noqa: BLE001 - one failed judge call must not abort fit
                self.failures += 1
                logger.warning(f"Metric '{self.name}' failed ({exc}); scoring this example 0.0")
                return 0.0
            if not (getattr(result, "metadata", None) or {}).get("evaluation_failed"):
                cache[key] = result
                if len(cache) > _RESULT_CACHE_SIZE:
                    cache.popitem(last=False)

        self.last_result = result
        if (getattr(result, "metadata", None) or {}).get("evaluation_failed"):
            self.failures += 1
            return 0.0
        return float(getattr(result, "score", 0.0) or 0.0)


def as_agent_metric(metric: Any) -> Any:
    """Coerce any supported metric into the class-agent ``Metric`` protocol.

    ``compile(metrics=[...])``, ``evaluate(metrics=[...])``,
    ``agents.WeightedMetric`` and ``MetricLoss`` call the sync
    ``score(example, prediction)``. An optimize metric (``LocalJudgeMetric``,
    ``LLMJudgeMetric``, ``ExactMatchMetric``, …) only has the async
    ``evaluate`` — it is wrapped in :class:`OptimizeMetricAdapter` so it can be
    passed anywhere without manual wrapping.

    Args:
        metric: A class-agent metric or an optimize metric.

    Returns:
        An object with ``name`` and ``score(example, prediction) -> float``.

    Raises:
        TypeError: When ``metric`` implements neither protocol.
    """
    if callable(getattr(metric, "score", None)):
        return metric
    if callable(getattr(metric, "evaluate", None)):
        return OptimizeMetricAdapter(metric)
    raise TypeError(
        f"{type(metric).__name__} is not a usable metric: implement "
        "`score(example, prediction) -> float` (agentomatic.agents.Metric) or "
        "`async evaluate(query, response, expected=None, context=None)` "
        "(agentomatic.optimize.BaseMetric)."
    )
