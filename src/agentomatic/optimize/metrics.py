"""Evaluation metrics for prompt optimization.

Provides:
- BaseMetric ABC — protocol for all metrics
- ExactMatchMetric — simple string matching (no LLM)
- LLMJudgeMetric — custom LLM-as-judge with user criteria
- CustomMetric — wrap any callable as a metric
- resolve_metrics() — factory to instantiate metrics from names

DeepEval integration is handled via dynamic imports.
"""
from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    metric_name: str
    score: float  # 0.0 to 1.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


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
            return EvalResult(metric_name=self.name, score=0.0,
                              reason="No expected answer provided")

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
            return EvalResult(metric_name=self.name, score=0.0,
                              reason="No expected answer")

        resp_lower = response.lower()
        keywords = [kw.strip() for kw in expected.lower().split(",")]
        found = sum(1 for kw in keywords if kw in resp_lower)
        score = found / len(keywords) if keywords else 0.0
        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=f"Found {found}/{len(keywords)} keywords",
        )


class LLMJudgeMetric(BaseMetric):
    """LLM-as-judge with custom evaluation criteria.

    Uses an LLM to score responses on user-defined criteria.
    Falls back to deepeval's GEval if available.
    """

    name = "llm_judge"

    def __init__(
        self,
        criteria: str,
        model: str = "ollama/mistral:7b",
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
        self, query: str, response: str,
        expected: str | None, context: list[str] | None,
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
            model=self.model,
        )
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected,
            retrieval_context=context or [],
        )
        metric.measure(test_case)
        return EvalResult(
            metric_name=self.name,
            score=metric.score or 0.0,
            reason=metric.reason or "",
        )

    async def _eval_llm(self, query: str, response: str,
                        expected: str | None, context: list[str] | None) -> EvalResult:
        """Fallback LLM-based evaluation without deepeval."""
        try:
            import httpx
            prompt = (
                f"You are an evaluation judge. Score the following response on a scale of 0.0 to 1.0.\n\n"
                f"CRITERIA: {self.criteria}\n\n"
                f"QUESTION: {query}\n\n"
                f"RESPONSE: {response}\n\n"
            )
            if expected:
                prompt += f"EXPECTED ANSWER: {expected}\n\n"
            prompt += "Reply with ONLY a JSON object: {\"score\": 0.X, \"reason\": \"...\"}\n"

            # Try Ollama API
            async with httpx.AsyncClient(timeout=30) as client:
                model_name = self.model.replace("ollama/", "") if self.model.startswith("ollama/") else self.model
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    import json as json_mod
                    text = resp.json().get("response", "")
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
            metric_name=self.name, score=0.5,
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
            score = await asyncio.to_thread(
                self.fn, query, response, expected, context
            )

        return EvalResult(
            metric_name=self.name,
            score=float(score),
            reason=f"Custom metric '{self.name}'",
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
_DEEPEVAL_METRICS = {
    "answer_relevancy",
    "faithfulness",
    "hallucination",
    "contextual_relevancy",
    "contextual_precision",
    "contextual_recall",
    "bias",
    "toxicity",
}


def _make_deepeval_metric(name: str, model: str) -> BaseMetric:
    """Create a DeepEval metric wrapped as BaseMetric."""
    try:
        from deepeval import metrics as de_metrics
    except ImportError:
        raise ImportError(
            f"DeepEval is required for metric '{name}'. "
            "Install: pip install agentomatic[optimize]"
        )

    # Map names to deepeval classes
    mapping = {
        "answer_relevancy": "AnswerRelevancyMetric",
        "faithfulness": "FaithfulnessMetric",
        "hallucination": "HallucinationMetric",
        "contextual_relevancy": "ContextualRelevancyMetric",
        "contextual_precision": "ContextualPrecisionMetric",
        "contextual_recall": "ContextualRecallMetric",
        "bias": "BiasMetric",
        "toxicity": "ToxicityMetric",
    }

    cls_name = mapping.get(name)
    if not cls_name or not hasattr(de_metrics, cls_name):
        raise ValueError(f"Unknown DeepEval metric: {name}")

    de_cls = getattr(de_metrics, cls_name)

    class _Wrapped(BaseMetric):
        def __init__(self):
            self.name = name
            self._metric = de_cls(model=model)

        async def evaluate(self, query, response, expected=None, context=None):
            from deepeval.test_case import LLMTestCase
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                expected_output=expected,
                retrieval_context=context or [],
            )
            self._metric.measure(test_case)
            return EvalResult(
                metric_name=self.name,
                score=self._metric.score or 0.0,
                reason=self._metric.reason or "",
            )

    return _Wrapped()


def resolve_metrics(
    metrics: list[str | BaseMetric],
    model: str = "ollama/mistral:7b",
) -> list[BaseMetric]:
    """Resolve metric names/instances to BaseMetric objects.

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
            if m in _BUILTIN_METRICS:
                resolved.append(_BUILTIN_METRICS[m]())
            elif m in _DEEPEVAL_METRICS:
                resolved.append(_make_deepeval_metric(m, model))
            else:
                raise ValueError(
                    f"Unknown metric: '{m}'. "
                    f"Built-in: {list(_BUILTIN_METRICS.keys())}. "
                    f"DeepEval: {sorted(_DEEPEVAL_METRICS)}. "
                    f"Or pass a BaseMetric/CustomMetric instance."
                )
        else:
            raise TypeError(f"Expected str or BaseMetric, got {type(m)}")
    return resolved
