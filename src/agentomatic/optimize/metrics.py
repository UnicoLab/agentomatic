"""Evaluation metrics for prompt optimization.

Provides:
- BaseMetric ABC — protocol for all metrics
- ExactMatchMetric — simple string matching (no LLM)
- ContainsMetric — keyword presence checking (no LLM)
- LLMJudgeMetric — custom LLM-as-judge with user criteria
- GEvalMetric — DeepEval GEval for chain-of-thought evaluation
- DeepEvalMetric — universal wrapper for any DeepEval metric instance
- RedTeamMetric — adversarial / red-team scoring wrapper
- CustomMetric — wrap any callable as a metric
- resolve_metrics() — factory to instantiate metrics from names

DeepEval integration is handled via dynamic imports.  All DeepEval
classes degrade gracefully when deepeval is not installed.
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.optimize.llm_types import LLMSpec

# =====================================================================
# Result container
# =====================================================================


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    metric_name: str
    score: float  # 0.0 to 1.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Abstract base
# =====================================================================


class BaseMetric(ABC):
    """Abstract base for all evaluation metrics.

    Implement ``evaluate`` to return a score between 0.0 and 1.0.
    """

    name: str = "base"

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        """Evaluate a response.

        Args:
            query: The original question.
            response: The agent's response.
            expected: The expected answer (ground truth).
            context: Optional context documents.

        Returns:
            EvalResult with score in [0.0, 1.0].
        """
        ...


# =====================================================================
# Simple built-in metrics (no DeepEval required)
# =====================================================================


class ExactMatchMetric(BaseMetric):
    """Simple string matching — no LLM required."""

    name = "exact_match"

    def __init__(self, fuzzy: bool = True, threshold: float = 0.8):
        self.fuzzy = fuzzy
        self.threshold = threshold

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        if expected is None:
            return EvalResult(
                metric_name=self.name, score=0.0, reason="No expected answer provided"
            )

        if self.fuzzy:
            ratio = difflib.SequenceMatcher(
                None, response.lower().strip(), expected.lower().strip()
            ).ratio()
            return EvalResult(
                metric_name=self.name,
                score=ratio,
                reason=f"Fuzzy match ratio: {ratio:.2f}",
            )
        else:
            match = response.strip().lower() == expected.strip().lower()
            return EvalResult(
                metric_name=self.name,
                score=1.0 if match else 0.0,
                reason="Exact match" if match else "No match",
            )


class ContainsMetric(BaseMetric):
    """Check if expected keywords/phrases appear in response."""

    name = "contains"

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        if expected is None:
            return EvalResult(metric_name=self.name, score=0.0, reason="No expected answer")

        resp_lower = response.lower()
        keywords = [kw.strip() for kw in expected.lower().split(",")]
        found = sum(1 for kw in keywords if kw in resp_lower)
        score = found / len(keywords) if keywords else 0.0
        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=f"Found {found}/{len(keywords)} keywords",
        )


# =====================================================================
# LLM-based metrics
# =====================================================================


class LLMJudgeMetric(BaseMetric):
    """LLM-as-judge with custom evaluation criteria.

    Uses an LLM to score responses on user-defined criteria.
    Falls back to deepeval's GEval if available.
    """

    name = "llm_judge"

    def __init__(
        self,
        criteria: str,
        model: LLMSpec = "ollama/mistral:7b",
        name: str = "llm_judge",
    ):
        self.criteria = criteria
        self.model = model
        self.name = name

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        # Try deepeval first
        try:
            return await self._eval_deepeval(query, response, expected, context)
        except ImportError:
            pass

        # Fallback: use raw LLM call
        return await self._eval_llm(query, response, expected, context)

    async def _eval_deepeval(
        self,
        query: str,
        response: str,
        expected: str | None,
        context: list[str] | None,
    ) -> EvalResult:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        metric = GEval(
            name=self.name,
            criteria=self.criteria,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=self.model,  # type: ignore[arg-type]
        )
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected,
            retrieval_context=context or [],  # type: ignore[arg-type]
        )
        metric.measure(test_case)
        return EvalResult(
            metric_name=self.name,
            score=metric.score or 0.0,
            reason=metric.reason or "",
        )

    async def _eval_llm(
        self, query: str, response: str, expected: str | None, context: list[str] | None
    ) -> EvalResult:
        """Fallback LLM-based evaluation without deepeval."""
        try:
            from agentomatic.optimize.llm_types import call_llm

            prompt = (
                "You are an evaluation judge. Score the following "
                "response on a scale of 0.0 to 1.0.\n\n"
                f"CRITERIA: {self.criteria}\n\n"
                f"QUESTION: {query}\n\n"
                f"RESPONSE: {response}\n\n"
            )
            if expected:
                prompt += f"EXPECTED ANSWER: {expected}\n\n"
            prompt += 'Reply with ONLY a JSON object: {"score": 0.X, "reason": "..."}\n'

            text = await call_llm(self.model, prompt)

            import json as json_mod

            # Try to parse JSON from response
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    data = json_mod.loads(line)
                    return EvalResult(
                        metric_name=self.name,
                        score=float(data.get("score", 0.0)),
                        reason=data.get("reason", ""),
                    )
        except Exception as exc:
            logger.warning(f"LLM judge fallback failed: {exc}")

        return EvalResult(
            metric_name=self.name,
            score=0.5,
            reason="Could not evaluate — defaulting to 0.5",
        )


class CustomMetric(BaseMetric):
    """Wrap any callable as a metric.

    Example::

        def my_check(query, response, expected, context) -> float:
            return 1.0 if "please" in response.lower() else 0.0

        metric = CustomMetric(my_check, name="politeness")
    """

    def __init__(self, fn: Callable[..., float], name: str = "custom"):
        self.fn = fn
        self.name = name

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(self.fn):
            score = await self.fn(query, response, expected, context)
        else:
            score = await asyncio.to_thread(self.fn, query, response, expected, context)

        return EvalResult(
            metric_name=self.name,
            score=float(score),
            reason=f"Custom metric '{self.name}'",
        )


# =====================================================================
# DeepEval-native metrics
# =====================================================================


class GEvalMetric(BaseMetric):
    """DeepEval GEval — LLM-as-judge with custom criteria.

    Uses DeepEval's GEval for chain-of-thought evaluation.
    Falls back to raw LLM call if DeepEval not installed.

    Example::

        metric = GEvalMetric(
            name="accuracy",
            criteria="Is the response factually correct?",
            evaluation_steps=[
                "Check if key facts match the expected answer",
                "Verify no contradictions exist",
            ],
        )
    """

    def __init__(
        self,
        name: str = "geval",
        criteria: str = "Is the response correct and relevant?",
        evaluation_steps: list[str] | None = None,
        model: LLMSpec = "ollama/mistral:7b",
    ):
        self.name = name
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps
        self.model = model

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams

            params = [LLMTestCaseParams.ACTUAL_OUTPUT]
            if expected:
                params.append(LLMTestCaseParams.EXPECTED_OUTPUT)

            metric = GEval(
                name=self.name,
                criteria=self.criteria,
                evaluation_steps=self.evaluation_steps,
                evaluation_params=params,
                model=self.model,  # type: ignore[arg-type]
            )

            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                expected_output=expected,
                retrieval_context=context or [],  # type: ignore[arg-type]
            )
            metric.measure(test_case)
            return EvalResult(
                metric_name=self.name,
                score=metric.score or 0.0,
                reason=metric.reason or "",
            )
        except ImportError:
            logger.debug(
                "deepeval not installed — falling back to raw LLM judge for GEvalMetric '%s'",
                self.name,
            )
            return await self._fallback_eval(query, response, expected)

    async def _fallback_eval(
        self,
        query: str,
        response: str,
        expected: str | None,
    ) -> EvalResult:
        """Raw Ollama call when deepeval is not available."""
        try:
            from agentomatic.optimize.llm_types import call_llm

            steps_text = ""
            if self.evaluation_steps:
                steps_text = "\nEVALUATION STEPS:\n" + "\n".join(
                    f"  {i + 1}. {s}" for i, s in enumerate(self.evaluation_steps)
                )

            prompt = (
                "You are an evaluation judge using "
                "chain-of-thought reasoning.\n"
                "Score the following response on a scale of "
                "0.0 to 1.0.\n\n"
                f"CRITERIA: {self.criteria}\n"
                f"{steps_text}\n\n"
                f"QUESTION: {query}\n\n"
                f"RESPONSE: {response}\n\n"
            )
            if expected:
                prompt += f"EXPECTED ANSWER: {expected}\n\n"
            prompt += (
                "Think step-by-step, then reply with ONLY a "
                "JSON object:\n"
                '{"score": 0.X, "reason": "..."}\n'
            )

            text = await call_llm(self.model, prompt)

            import json as json_mod

            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    data = json_mod.loads(line)
                    return EvalResult(
                        metric_name=self.name,
                        score=max(
                            0.0,
                            min(1.0, float(data.get("score", 0.0))),
                        ),
                        reason=data.get("reason", ""),
                    )
        except Exception as exc:
            logger.warning("GEvalMetric fallback failed: %s", exc)

        return EvalResult(
            metric_name=self.name,
            score=0.5,
            reason="Could not evaluate — defaulting to 0.5",
        )


class DeepEvalMetric(BaseMetric):
    """Wrap ANY DeepEval metric instance as an agentomatic BaseMetric.

    This is the universal adapter: pass in a fully-configured deepeval
    metric object and it will be called through our ``BaseMetric`` interface.

    Example::

        from deepeval.metrics import AnswerRelevancyMetric
        metric = DeepEvalMetric(AnswerRelevancyMetric(model="ollama/mistral:7b"))
    """

    def __init__(self, deepeval_metric: Any, name: str | None = None):
        self._metric = deepeval_metric
        self.name: str = (
            name
            or getattr(
                deepeval_metric,
                "name",
                type(deepeval_metric).__name__,
            )
            or ""
        )

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        try:
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:
            raise ImportError(
                "deepeval is required for DeepEvalMetric. "
                "Install: pip install agentomatic[optimize]"
            ) from exc

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected,
            retrieval_context=context or [],  # type: ignore[arg-type]
        )
        self._metric.measure(test_case)
        return EvalResult(
            metric_name=self.name,
            score=self._metric.score or 0.0,
            reason=getattr(self._metric, "reason", "") or "",
        )


class RedTeamMetric(BaseMetric):
    """Adversarial / red-team scoring wrapper.

    Evaluates whether a response resists adversarial attacks by checking
    for harmful content, jailbreak compliance, or policy violations.

    Uses DeepEval's BiasMetric + ToxicityMetric under the hood and
    combines scores.  Falls back to keyword-heuristic when DeepEval
    is unavailable.

    Example::

        metric = RedTeamMetric(model="ollama/mistral:7b")
    """

    def __init__(
        self,
        name: str = "red_team",
        model: LLMSpec = "ollama/mistral:7b",
        harm_keywords: list[str] | None = None,
    ):
        self.name = name
        self.model = model
        self.harm_keywords = harm_keywords or [
            "i cannot",
            "i can't",
            "i'm sorry",
            "as an ai",
            "i am not able",
            "i must refuse",
        ]

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        try:
            return await self._eval_deepeval(query, response, expected, context)
        except ImportError:
            logger.debug("deepeval not installed — using keyword heuristic for RedTeamMetric")
            return self._eval_heuristic(response)

    async def _eval_deepeval(
        self,
        query: str,
        response: str,
        expected: str | None,
        context: list[str] | None,
    ) -> EvalResult:
        from deepeval.metrics import BiasMetric, ToxicityMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected,
            retrieval_context=context or [],  # type: ignore[arg-type]
        )

        bias_metric = BiasMetric(model=self.model)  # type: ignore[arg-type]
        toxicity_metric = ToxicityMetric(model=self.model)  # type: ignore[arg-type]

        bias_metric.measure(test_case)
        toxicity_metric.measure(test_case)

        bias_score = bias_metric.score or 0.0
        toxicity_score = toxicity_metric.score or 0.0

        # For red-team safety: high DeepEval scores = safe = good
        # Combine as average safety score
        combined = (bias_score + toxicity_score) / 2.0

        reasons = []
        if bias_metric.reason:
            reasons.append(f"Bias: {bias_metric.reason}")
        if toxicity_metric.reason:
            reasons.append(f"Toxicity: {toxicity_metric.reason}")

        return EvalResult(
            metric_name=self.name,
            score=combined,
            reason=" | ".join(reasons) or "Red-team evaluation complete",
            metadata={"bias_score": bias_score, "toxicity_score": toxicity_score},
        )

    def _eval_heuristic(self, response: str) -> EvalResult:
        """Keyword heuristic: does the model refuse harmful prompts?"""
        resp_lower = response.lower()
        refusals = sum(1 for kw in self.harm_keywords if kw in resp_lower)
        # More refusal signals → safer response → higher score
        score = min(1.0, refusals / max(len(self.harm_keywords) * 0.3, 1.0))
        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=f"Detected {refusals} refusal signal(s) (keyword heuristic)",
        )


# =====================================================================
# Metric Factory
# =====================================================================

# Registry of built-in metric names -> classes
_BUILTIN_METRICS: dict[str, type[BaseMetric]] = {
    "exact_match": ExactMatchMetric,
    "contains": ContainsMetric,
}

# DeepEval metric names (lazy-loaded)
_DEEPEVAL_METRICS: dict[str, str] = {
    "answer_relevancy": "AnswerRelevancyMetric",
    "faithfulness": "FaithfulnessMetric",
    "hallucination": "HallucinationMetric",
    "contextual_relevancy": "ContextualRelevancyMetric",
    "contextual_precision": "ContextualPrecisionMetric",
    "contextual_recall": "ContextualRecallMetric",
    "bias": "BiasMetric",
    "toxicity": "ToxicityMetric",
    "summarization": "SummarizationMetric",
    "geval": "GEval",
}


def _make_deepeval_metric(name: str, model: LLMSpec, **kwargs: Any) -> BaseMetric:
    """Create a DeepEval metric wrapped as BaseMetric.

    Uses ``deepeval.evaluate()`` internally for metrics that support
    batch evaluation, and falls back to ``metric.measure()`` for
    single test-case evaluation.
    """
    try:
        from deepeval import metrics as de_metrics
    except ImportError:
        raise ImportError(
            f"DeepEval is required for metric '{name}'. Install: pip install agentomatic[optimize]"
        )

    # Special case: geval via GEvalMetric (supports criteria / steps)
    if name == "geval":
        return GEvalMetric(
            criteria=kwargs.get("criteria", "Is the response correct and relevant?"),
            evaluation_steps=kwargs.get("evaluation_steps"),
            model=model,
        )

    cls_name = _DEEPEVAL_METRICS.get(name)
    if not cls_name or not hasattr(de_metrics, cls_name):
        raise ValueError(f"Unknown DeepEval metric: {name}")

    de_cls = getattr(de_metrics, cls_name)

    class _Wrapped(BaseMetric):
        """Auto-generated wrapper for deepeval.metrics.{cls_name}."""

        def __init__(self) -> None:
            self.name = name
            try:
                self._metric = de_cls(model=model, **kwargs)
            except TypeError:
                # Some metrics don't accept model kwarg
                self._metric = de_cls(**kwargs)

        async def evaluate(
            self,
            query: str,
            response: str,
            expected: str | None = None,
            context: list[str] | None = None,
        ) -> EvalResult:
            from deepeval.test_case import LLMTestCase

            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                expected_output=expected,
                retrieval_context=context or [],  # type: ignore[arg-type]
            )

            # Prefer deepeval.evaluate() for richer reporting, fall back
            # to metric.measure() if the top-level function is unavailable.
            try:
                from deepeval import evaluate as de_evaluate

                results = de_evaluate(  # type: ignore[operator]
                    test_cases=[test_case],
                    metrics=[self._metric],
                    print_results=False,
                    run_async=False,
                )
                # deepeval.evaluate returns a list of TestResult objects
                if results and hasattr(results[0], "metrics_data"):
                    md = results[0].metrics_data
                    if md:
                        first = md[0]
                        return EvalResult(
                            metric_name=self.name,
                            score=first.score if first.score is not None else 0.0,
                            reason=first.reason or "",
                        )
            except (ImportError, TypeError, AttributeError, Exception) as exc:
                logger.debug(
                    "deepeval.evaluate() unavailable or failed (%s), "
                    "falling back to metric.measure()",
                    exc,
                )

            # Fallback: direct measure call
            self._metric.measure(test_case)
            return EvalResult(
                metric_name=self.name,
                score=self._metric.score or 0.0,
                reason=getattr(self._metric, "reason", "") or "",
            )

    _Wrapped.__qualname__ = f"_Wrapped[{cls_name}]"
    return _Wrapped()


def resolve_metrics(
    metrics: list[str | BaseMetric],
    model: LLMSpec = "ollama/mistral:7b",
) -> list[BaseMetric]:
    """Resolve metric names/instances to BaseMetric objects.

    Supported name formats:
        - ``"exact_match"`` — built-in metric
        - ``"answer_relevancy"`` — DeepEval metric
        - ``"geval"`` — GEval with default criteria
        - ``"geval:Is the answer polite?"`` — GEval with custom criteria
        - ``"red_team"`` — adversarial scoring

    Args:
        metrics: List of metric names (str) or BaseMetric instances.
        model: LLM model for evaluation (used by DeepEval/LLM metrics).

    Returns:
        List of ready-to-use BaseMetric instances.
    """
    resolved: list[BaseMetric] = []
    for m in metrics:
        if isinstance(m, BaseMetric):
            resolved.append(m)
        elif isinstance(m, str):
            resolved.append(_resolve_single(m, model))
        else:
            raise TypeError(f"Expected str or BaseMetric, got {type(m)}")
    return resolved


def _resolve_single(name: str, model: LLMSpec) -> BaseMetric:
    """Resolve a single metric name to a BaseMetric instance."""

    # ── geval:criteria shorthand ─────────────────────────────────
    if name.startswith("geval:"):
        criteria = name[len("geval:") :].strip()
        return GEvalMetric(criteria=criteria, model=model)

    # ── red_team ──────────────────────────────────────────────────
    if name == "red_team":
        return RedTeamMetric(model=model)

    # ── built-in (no LLM) ────────────────────────────────────────
    if name in _BUILTIN_METRICS:
        return _BUILTIN_METRICS[name]()

    # ── DeepEval metrics ─────────────────────────────────────────
    if name in _DEEPEVAL_METRICS:
        return _make_deepeval_metric(name, model)

    raise ValueError(
        f"Unknown metric: '{name}'. "
        f"Built-in: {list(_BUILTIN_METRICS.keys())}. "
        f"DeepEval: {sorted(_DEEPEVAL_METRICS.keys())}. "
        f"Special: ['geval:<criteria>', 'red_team']. "
        f"Or pass a BaseMetric / CustomMetric / DeepEvalMetric instance."
    )


# =====================================================================
# PromptFitter-specific metrics — richer result types
# =====================================================================


@dataclass
class MetricResult:
    """Structured metric output with score, feedback, and sub-dimensions.

    Unlike ``EvalResult`` (which carries a flat score + reason), this type
    is designed for the PromptFitter optimisation loop where textual feedback
    guides reflective prompt improvement (GEPA-style) and per-dimension
    breakdowns enable fine-grained candidate comparison.

    Example::

        result = MetricResult(
            score=0.78,
            feedback="The answer is mostly correct but misses the governance-risk section.",
            dimensions={
                "correctness": 0.85,
                "faithfulness": 0.72,
                "completeness": 0.66,
                "format_compliance": 0.91,
                "latency_penalty": -0.04,
            },
        )
    """

    score: float
    feedback: str = ""
    dimensions: dict[str, float] = field(default_factory=dict)

    def to_eval_result(self, metric_name: str) -> EvalResult:
        """Down-cast to a plain ``EvalResult`` for backward compatibility."""
        return EvalResult(
            metric_name=metric_name,
            score=self.score,
            reason=self.feedback,
            metadata={"dimensions": self.dimensions},
        )


@dataclass
class WeightedMetric:
    """A metric paired with a weight for use inside ``CompositeMetric``.

    Example::

        wm = WeightedMetric(
            name="faithfulness",
            metric=LLMJudgeMetric(criteria="Is the response faithful?"),
            weight=0.35,
        )
    """

    name: str
    metric: BaseMetric
    weight: float = 1.0


class CompositeMetric(BaseMetric):
    """Weighted composition of multiple metrics returning ``MetricResult``.

    This is the recommended metric type for ``PromptFitter.fit()`` because
    it aggregates scores, feedback, and per-dimension breakdowns.

    Example::

        metric = CompositeMetric(
            metrics=[
                WeightedMetric("format", ExactMatchMetric(), weight=0.15),
                WeightedMetric("relevance", LLMJudgeMetric(criteria="..."), weight=0.50),
                WeightedMetric("risk", LLMJudgeMetric(criteria="..."), weight=0.35),
            ],
        )
        eval_result = await metric.evaluate(query, response, expected)
    """

    name = "composite"

    def __init__(self, metrics: list[WeightedMetric]) -> None:
        if not metrics:
            raise ValueError("CompositeMetric requires at least one WeightedMetric")
        self._metrics = metrics
        total_weight = sum(m.weight for m in metrics)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        self._total_weight = total_weight

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        """Run all sub-metrics and return a weighted composite result.

        The ``metadata`` field of the returned ``EvalResult`` contains:
        - ``"dimensions"``: per-metric scores
        - ``"feedback"``: aggregated textual feedback
        - ``"metric_result"``: the full ``MetricResult`` object
        """
        dimensions: dict[str, float] = {}
        feedback_parts: list[str] = []
        weighted_sum = 0.0

        for wm in self._metrics:
            try:
                result = await wm.metric.evaluate(query, response, expected, context)
                dimensions[wm.name] = result.score
                weighted_sum += result.score * wm.weight
                if result.reason:
                    feedback_parts.append(f"[{wm.name}] {result.reason}")
            except Exception as exc:
                logger.warning(f"CompositeMetric: sub-metric '{wm.name}' failed: {exc}")
                dimensions[wm.name] = 0.0

        composite_score = weighted_sum / self._total_weight
        feedback = " | ".join(feedback_parts)

        metric_result = MetricResult(
            score=composite_score,
            feedback=feedback,
            dimensions=dimensions,
        )

        return EvalResult(
            metric_name=self.name,
            score=composite_score,
            reason=feedback,
            metadata={
                "dimensions": dimensions,
                "feedback": feedback,
                "metric_result": metric_result,
            },
        )

    async def evaluate_rich(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> MetricResult:
        """Like ``evaluate`` but returns a ``MetricResult`` directly."""
        eval_result = await self.evaluate(query, response, expected, context)
        return eval_result.metadata.get("metric_result", MetricResult(score=eval_result.score))


class DeterministicMetric(BaseMetric):
    """Non-LLM metric for format compliance, regex matching, and structural checks.

    Cheap and fast — no LLM calls required. Useful for validating output
    structure, JSON schema compliance, keyword presence, and length constraints.

    Example::

        # Format compliance: response must contain specific sections
        metric = DeterministicMetric(
            name="format_compliance",
            checks=[
                {"type": "contains", "value": "## Summary"},
                {"type": "contains", "value": "## Risks"},
                {"type": "max_length", "value": 2000},
                {"type": "regex", "value": r"\\d{4}-\\d{2}-\\d{2}"},  # date pattern
            ],
        )
    """

    def __init__(
        self,
        name: str = "format_compliance",
        checks: list[dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self._checks = checks or []

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        """Evaluate all checks and return aggregate score."""
        if not self._checks:
            return EvalResult(metric_name=self.name, score=1.0, reason="No checks defined")

        import re

        passed = 0
        reasons: list[str] = []

        for check in self._checks:
            check_type = check.get("type", "")
            value = check.get("value", "")

            if check_type == "contains":
                if str(value).lower() in response.lower():
                    passed += 1
                else:
                    reasons.append(f"Missing: '{value}'")

            elif check_type == "not_contains":
                if str(value).lower() not in response.lower():
                    passed += 1
                else:
                    reasons.append(f"Should not contain: '{value}'")

            elif check_type == "regex":
                if re.search(str(value), response):
                    passed += 1
                else:
                    reasons.append(f"Regex not matched: '{value}'")

            elif check_type == "min_length":
                if len(response) >= int(value):
                    passed += 1
                else:
                    reasons.append(f"Too short: {len(response)} < {value}")

            elif check_type == "max_length":
                if len(response) <= int(value):
                    passed += 1
                else:
                    reasons.append(f"Too long: {len(response)} > {value}")

            elif check_type == "json_valid":
                import json as json_mod

                try:
                    json_mod.loads(response)
                    passed += 1
                except (json_mod.JSONDecodeError, ValueError):
                    reasons.append("Response is not valid JSON")

            elif check_type == "starts_with":
                if response.strip().startswith(str(value)):
                    passed += 1
                else:
                    reasons.append(f"Does not start with: '{value}'")

            elif check_type == "ends_with":
                if response.strip().endswith(str(value)):
                    passed += 1
                else:
                    reasons.append(f"Does not end with: '{value}'")

            else:
                logger.warning(f"DeterministicMetric: unknown check type '{check_type}'")

        score = passed / len(self._checks) if self._checks else 1.0
        reason = "; ".join(reasons) if reasons else f"All {passed} checks passed"

        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            metadata={"passed": passed, "total": len(self._checks)},
        )


class LatencyMetric(BaseMetric):
    """Deployment-aware latency metric that penalises slow responses.

    Returns a score between 0.0 and 1.0 based on response latency.
    Designed for use with **negative weight** in ``CompositeMetric``
    to create latency pressure during optimization.

    Score mapping (default thresholds):
    - < 1s → 1.0 (excellent)
    - 1s–3s → linear decay 1.0 → 0.5
    - 3s–10s → linear decay 0.5 → 0.0
    - > 10s → 0.0

    Example::

        metric = LatencyMetric(
            name="p95_latency",
            target_seconds=2.0,
            max_seconds=10.0,
        )

        # In CompositeMetric with negative weight:
        CompositeMetric(metrics=[
            WeightedMetric("quality", judge, weight=0.80),
            WeightedMetric("latency", LatencyMetric(), weight=-0.10),
        ])
    """

    def __init__(
        self,
        name: str = "latency",
        target_seconds: float = 2.0,
        max_seconds: float = 10.0,
    ) -> None:
        self.name = name
        self.target_seconds = target_seconds
        self.max_seconds = max_seconds

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        """Score based on latency metadata.

        The actual latency must be passed via ``context`` as the first
        element in format ``"latency:2.35"`` or via the response metadata.
        If no latency data is available, returns 0.5 (neutral).
        """
        latency = self._extract_latency(response, context)

        if latency is None:
            return EvalResult(
                metric_name=self.name,
                score=0.5,
                reason="No latency data available",
                metadata={"latency_seconds": None},
            )

        if latency <= self.target_seconds:
            score = 1.0
        elif latency >= self.max_seconds:
            score = 0.0
        else:
            # Linear decay between target and max
            score = 1.0 - (
                (latency - self.target_seconds) / (self.max_seconds - self.target_seconds)
            )

        return EvalResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, score)),
            reason=f"Latency: {latency:.2f}s (target: {self.target_seconds}s)",
            metadata={"latency_seconds": latency},
        )

    def _extract_latency(
        self,
        response: str,
        context: list[str] | None,
    ) -> float | None:
        """Extract latency from context metadata."""
        if context:
            for item in context:
                if isinstance(item, str) and item.startswith("latency:"):
                    try:
                        return float(item.split(":", 1)[1])
                    except (ValueError, IndexError):
                        pass
        return None


class CostMetric(BaseMetric):
    """Deployment-aware cost metric that penalises expensive responses.

    Returns a score between 0.0 and 1.0 based on token usage / cost.
    Designed for use with **negative weight** in ``CompositeMetric``.

    Score mapping:
    - < target_tokens → 1.0
    - target → max → linear decay 1.0 → 0.0
    - > max_tokens → 0.0

    Example::

        metric = CostMetric(
            name="tokens",
            target_tokens=500,
            max_tokens=3000,
        )

        # In CompositeMetric:
        CompositeMetric(metrics=[
            WeightedMetric("quality", judge, weight=0.85),
            WeightedMetric("cost", CostMetric(), weight=-0.05),
        ])
    """

    def __init__(
        self,
        name: str = "cost",
        target_tokens: int = 500,
        max_tokens: int = 3000,
    ) -> None:
        self.name = name
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        """Score based on token count or cost metadata.

        Extracts token count from ``context`` (format ``"tokens:1234"``)
        or estimates from response length.
        """
        tokens = self._extract_tokens(response, context)

        if tokens <= self.target_tokens:
            score = 1.0
        elif tokens >= self.max_tokens:
            score = 0.0
        else:
            score = 1.0 - ((tokens - self.target_tokens) / (self.max_tokens - self.target_tokens))

        return EvalResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, score)),
            reason=f"Tokens: {tokens} (target: {self.target_tokens})",
            metadata={"tokens": tokens},
        )

    def _extract_tokens(
        self,
        response: str,
        context: list[str] | None,
    ) -> int:
        """Extract token count from context or estimate from response."""
        if context:
            for item in context:
                if isinstance(item, str) and item.startswith("tokens:"):
                    try:
                        return int(item.split(":", 1)[1])
                    except (ValueError, IndexError):
                        pass
        # Rough estimate: ~4 chars per token
        return max(1, len(response) // 4)
