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
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    from agentomatic.optimize.llm_types import LLMSpec


def _prefer_raw_llm_judge(model: Any) -> bool:
    """Return True when DeepEval should be skipped for *model*.

    Local OpenAI-compatible servers (oMLX, vLLM, …) and explicit ``omlx/``
    specs are not routable by DeepEval; attempting that path burns long
    retries before the honest raw-LLM fallback runs.
    """
    if not isinstance(model, str):
        return True
    lowered = model.lower()
    if lowered.startswith(("omlx/", "gemini/")):
        return True
    if os.getenv("OMLX_BASE_URL") or os.getenv("OMLX_API_KEY"):
        if lowered.startswith("openai/") or "/" not in model:
            return True
    base = (os.getenv("OPENAI_BASE_URL") or "").lower()
    if base and "api.openai.com" not in base and lowered.startswith("openai/"):
        return True
    return False


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


#: Keys a judge may use for its overall score (first match wins).
_JUDGE_SCORE_KEYS = ("overall_score", "score", "overall", "final_score", "rating")

_FRACTION = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
_PERCENT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*%\s*$")


def coerce_judge_score(value: Any) -> float | None:
    """Normalise a judge's score to ``[0, 1]`` — or ``None`` when unusable.

    Small local judges routinely ignore "score 0.0–1.0" and answer on another
    scale. Clamping ``7`` (out of 10) or ``85`` (out of 100) to ``1.0`` made
    every such reply a perfect score; they are rescaled instead:

    * numbers in ``[0, 1]`` are kept; ``(1, 10]`` are divided by 10 and
      ``(10, 100]`` by 100; anything else is unusable;
    * strings: ``"0.8"``, ``"8/10"``, ``"80%"``;
    * ``{"score": x}`` objects are unwrapped.

    Args:
        value: The raw score from the judge's JSON.

    Returns:
        The normalised score, or ``None`` (treat as a failed evaluation).
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, dict):
        for key in ("score", "value", "overall_score"):
            if key in value:
                return coerce_judge_score(value[key])
        return None
    if isinstance(value, str):
        text = value.strip()
        fraction = _FRACTION.match(text)
        if fraction:
            num, den = float(fraction.group(1)), float(fraction.group(2))
            return coerce_judge_score(num / den) if den > 0 else None
        percent = _PERCENT.match(text)
        if percent:
            return coerce_judge_score(float(percent.group(1)) / 100.0)
        try:
            value = float(text)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:  # NaN or negative
        return None
    if number <= 1.0:
        return number
    if number <= 10.0:
        return number / 10.0
    if number <= 100.0:
        return number / 100.0
    return None


#: Header that opens the plain answer inside a rich expected reference.
_EXPECTED_ANSWER_HEADER = re.compile(
    r"^[ \t]*##[ \t]*Expected answer[ \t]*$", re.MULTILINE | re.IGNORECASE
)
#: Any other section header in such a reference, which ends the answer.
_ANY_SECTION_HEADER = re.compile(r"^[ \t]*##[ \t]*\S", re.MULTILINE)


def plain_expected(expected: str | None) -> str | None:
    """Return the literal ground-truth answer inside an expected reference.

    ``AgentExample.to_datapoint`` builds a *judge-facing* reference — judge
    guidance, a rubric, an "## Expected answer" section, the structured
    output as JSON. An LLM judge reads all of that. A deterministic metric
    cannot: comparing a response against markdown headers scores near zero no
    matter how right the answer is, so ``fit()`` over an ``AgentDataset``
    reported "no improvement" forever, whatever the optimizer proposed.

    Plain strings pass through untouched, so a hand-written dataset behaves
    exactly as before.

    Args:
        expected: The expected value as the dataset carries it.

    Returns:
        Just the answer text, or ``expected`` when there is no such section.
    """
    if not expected:
        return expected
    opener = _EXPECTED_ANSWER_HEADER.search(expected)
    if opener is None:
        return expected
    rest = expected[opener.end() :].lstrip("\n")
    nxt = _ANY_SECTION_HEADER.search(rest)
    body = rest[: nxt.start()] if nxt else rest
    stripped = "\n".join(line.strip() for line in body.splitlines()).strip()
    return stripped or expected


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
        expected = plain_expected(expected) or expected

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

        expected = plain_expected(expected) or expected
        resp_lower = response.lower()
        keywords = [kw.strip() for kw in expected.lower().split(",") if kw.strip()]
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
        temperature: float = 0.0,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.criteria = criteria
        self.model = model
        self.name = name
        #: Endpoint for this judge only (else LLMCaller defaults / env).
        self.base_url = base_url
        self.api_key = api_key
        # Default 0.0 for reproducible scoring across epochs (avoids 0.33↔0.67
        # oscillation from sampling noise at temp>0).
        self.temperature = temperature

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        # Skip deepeval for local OpenAI-compatible / oMLX specs — DeepEval
        # cannot route those models and its retries burn minutes before
        # falling through to the raw-LLM judge.
        if self.base_url or _prefer_raw_llm_judge(self.model):
            return await self._eval_llm(query, response, expected, context)

        # Try deepeval first; any failure (missing key, bad model, …)
        # falls through to the honest raw-LLM judge path.
        try:
            return await self._eval_deepeval(query, response, expected, context)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("deepeval LLMJudge path failed, using raw LLM: {}", exc)

        return await self._eval_llm(query, response, expected, context)

    async def _eval_deepeval(
        self,
        query: str,
        response: str,
        expected: str | None,
        context: list[str] | None,
    ) -> EvalResult:
        from deepeval import test_case as deepeval_test_case
        from deepeval.metrics import GEval

        evaluation_params_type = getattr(deepeval_test_case, "SingleTurnParams", None)
        if evaluation_params_type is None:  # pragma: no cover - older DeepEval compatibility
            evaluation_params_type = deepeval_test_case.LLMTestCaseParams

        metric = GEval(
            name=self.name,
            criteria=self.criteria,
            evaluation_params=[
                evaluation_params_type.INPUT,
                evaluation_params_type.ACTUAL_OUTPUT,
                # Without it GEval never looked at the reference answer.
                *([evaluation_params_type.EXPECTED_OUTPUT] if expected else []),
            ],
            model=self.model,  # type: ignore[arg-type]
        )
        test_case = deepeval_test_case.LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected,
            retrieval_context=context or [],  # type: ignore[arg-type]
        )
        metric.measure(test_case)
        if metric.score is None:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=metric.reason or "DeepEval returned no score",
                metadata={"evaluation_failed": True},
            )
        return EvalResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, float(metric.score))),
            reason=metric.reason or "",
        )

    async def _eval_llm(
        self, query: str, response: str, expected: str | None, context: list[str] | None
    ) -> EvalResult:
        """Fallback LLM-based evaluation without deepeval."""
        try:
            from agentomatic.optimize.llm_types import call_llm_json

            prompt = (
                "You are an evaluation judge. Score the following "
                "response on a scale of 0.0 to 1.0.\n\n"
                f"CRITERIA: {self.criteria}\n\n"
                f"QUESTION: {query}\n\n"
                f"RESPONSE: {response}\n\n"
            )
            if expected:
                prompt += f"EXPECTED ANSWER: {expected}\n\n"
            if context:
                prompt += f"CONTEXT:\n{chr(10).join(context[:3])}\n\n"
            labels = scoring_example_labels(has_context=bool(context))
            if labels:
                prompt += f"EXAMPLE INPUTS, METADATA AND TAGS: {labels}\n\n"
            prompt += 'Reply with ONLY a JSON object: {"score": 0.X, "reason": "..."}\n'

            if isinstance(self.model, str):
                from agentomatic.optimize.llm_caller import LLMCaller

                data = await LLMCaller.call_with_json(
                    self.model,
                    prompt,
                    temperature=self.temperature,
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            else:
                data = await call_llm_json(self.model, prompt, temperature=self.temperature)
            raw = next(
                (data[k] for k in _JUDGE_SCORE_KEYS if isinstance(data, dict) and k in data),
                None,
            )
            score = coerce_judge_score(raw)
            if score is None:
                return EvalResult(
                    metric_name=self.name,
                    score=0.0,
                    reason=f"LLM judge returned no usable score ({raw!r})",
                    metadata={"evaluation_failed": True},
                )
            return EvalResult(
                metric_name=self.name,
                score=score,
                reason=str(data.get("reason") or data.get("feedback") or ""),
                metadata={"raw_score": raw},
            )
        except Exception as exc:
            logger.warning(f"LLM judge fallback failed: {exc}")
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=f"LLM judge evaluation failed: {exc}",
                metadata={"evaluation_failed": True},
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

        value = float(score) if score is not None else float("nan")
        if value != value or value in (float("inf"), float("-inf")):
            # NaN compares False with everything: one NaN baseline used to
            # block every candidate from ever being promoted.
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=f"Custom metric '{self.name}' returned {score!r}",
                metadata={"evaluation_failed": True},
            )
        return EvalResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, value)),
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
        temperature: float = 0.0,
    ):
        self.name = name
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps
        self.model = model
        self.temperature = temperature

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        if _prefer_raw_llm_judge(self.model):
            return await self._fallback_eval(query, response, expected)

        # Try deepeval first; any failure (missing key, bad model, …)
        # falls through to the honest raw-LLM judge path.
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams

            # INPUT too: judging relevance without the question is guesswork.
            params = [
                getattr(LLMTestCaseParams, "INPUT"),
                getattr(LLMTestCaseParams, "ACTUAL_OUTPUT"),
            ]
            if expected:
                params.append(getattr(LLMTestCaseParams, "EXPECTED_OUTPUT"))

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
            if metric.score is None:
                return EvalResult(
                    metric_name=self.name,
                    score=0.0,
                    reason=metric.reason or "DeepEval returned no score",
                    metadata={"evaluation_failed": True},
                )
            return EvalResult(
                metric_name=self.name,
                score=max(0.0, min(1.0, float(metric.score))),
                reason=metric.reason or "",
            )
        except ImportError:
            logger.debug(
                "deepeval not installed — falling back to raw LLM judge for GEvalMetric '{}'",
                self.name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "deepeval GEval path failed, using raw LLM: {}",
                exc,
            )

        return await self._fallback_eval(query, response, expected)

    async def _fallback_eval(
        self,
        query: str,
        response: str,
        expected: str | None,
    ) -> EvalResult:
        """Raw LLM call when deepeval is not available."""
        try:
            from agentomatic.optimize.llm_types import call_llm_json

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

            data = await call_llm_json(self.model, prompt, temperature=self.temperature)
            raw = next(
                (data[k] for k in _JUDGE_SCORE_KEYS if isinstance(data, dict) and k in data),
                None,
            )
            score = coerce_judge_score(raw)
            if score is None:
                return EvalResult(
                    metric_name=self.name,
                    score=0.0,
                    reason=f"GEval evaluation failed — no usable score ({raw!r})",
                    metadata={"evaluation_failed": True},
                )
            return EvalResult(
                metric_name=self.name,
                score=score,
                reason=str(data.get("reason", "")),
                metadata={"raw_score": raw},
            )
        except Exception as exc:
            logger.warning("GEvalMetric fallback failed: {}", exc)
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=f"GEval evaluation failed: {exc}",
                metadata={"evaluation_failed": True},
            )


#: DeepEval metrics where a HIGHER score is WORSE (``success = score <= threshold``).
_LOWER_IS_BETTER_DEEPEVAL = frozenset({"BiasMetric", "ToxicityMetric", "HallucinationMetric"})


def _higher_is_better(deepeval_metric: Any, score: float, flag: bool | None = None) -> float:
    """Orient a DeepEval score so higher always means better (optimizers maximise)."""
    lower_is_better = (
        not flag
        if flag is not None
        else type(deepeval_metric).__name__ in _LOWER_IS_BETTER_DEEPEVAL
    )
    score = max(0.0, min(1.0, float(score)))
    return 1.0 - score if lower_is_better else score


class DeepEvalMetric(BaseMetric):
    """Wrap ANY DeepEval metric instance as an agentomatic BaseMetric.

    Scores are oriented so **higher is better**: DeepEval's bias, toxicity and
    hallucination metrics (where a higher score is worse) are inverted
    automatically — maximising them used to push the optimizer toward toxic
    or hallucinated answers. Override with ``higher_is_better=``.

    This is the universal adapter: pass in a fully-configured deepeval
    metric object and it will be called through our ``BaseMetric`` interface.

    Example::

        from deepeval.metrics import AnswerRelevancyMetric
        metric = DeepEvalMetric(AnswerRelevancyMetric(model="ollama/mistral:7b"))
    """

    def __init__(
        self,
        deepeval_metric: Any,
        name: str | None = None,
        *,
        higher_is_better: bool | None = None,
    ):
        self._metric = deepeval_metric
        self._higher_is_better = higher_is_better
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
        raw = getattr(self._metric, "score", None)
        if raw is None:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=getattr(self._metric, "reason", "") or "DeepEval returned no score",
                metadata={"evaluation_failed": True},
            )
        return EvalResult(
            metric_name=self.name,
            score=_higher_is_better(self._metric, raw, self._higher_is_better),
            reason=getattr(self._metric, "reason", "") or "",
            metadata={"raw_score": raw},
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

        # DeepEval bias/toxicity: HIGHER = worse. Safety is the complement
        # (averaging them as "safety" used to rate toxic answers as safest).
        combined = 1.0 - (bias_score + toxicity_score) / 2.0

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
                from deepeval import evaluate as _de_evaluate

                # mypy resolves ``deepeval.evaluate`` as a module and basedpyright
                # as a function; cast to Any keeps both happy while preserving
                # the legacy (<4) kwargs for older deepeval versions.
                de_evaluate = cast(Any, _de_evaluate)
                results = de_evaluate(
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
                        if first.score is None:
                            return EvalResult(
                                metric_name=self.name,
                                score=0.0,
                                reason=first.reason or "DeepEval returned no score",
                                metadata={"evaluation_failed": True},
                            )
                        return EvalResult(
                            metric_name=self.name,
                            score=_higher_is_better(self._metric, first.score),
                            reason=first.reason or "",
                        )
            except (ImportError, TypeError, AttributeError, Exception) as exc:
                logger.debug(
                    "deepeval.evaluate() unavailable or failed ({}), falling back to measure()",
                    exc,
                )

            # Fallback: direct measure call
            self._metric.measure(test_case)
            raw = getattr(self._metric, "score", None)
            if raw is None:
                return EvalResult(
                    metric_name=self.name,
                    score=0.0,
                    reason=getattr(self._metric, "reason", "") or "DeepEval returned no score",
                    metadata={"evaluation_failed": True},
                )
            return EvalResult(
                metric_name=self.name,
                score=_higher_is_better(self._metric, raw),
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
        metrics: List of metric names (str) or metric instances — a
            ``BaseMetric``, an ``OptimizeMetricAdapter`` or a class-agent
            ``score(example, prediction)`` metric.
        model: LLM model for evaluation (used by DeepEval/LLM metrics).

    Returns:
        List of ready-to-use BaseMetric instances.
    """
    resolved: list[BaseMetric] = []
    for m in metrics:
        m_any: Any = m
        if isinstance(m_any, str):
            resolved.append(_resolve_single(m_any, model))
        else:
            # Instances of either metric protocol (see ``as_optimize_metric``).
            resolved.append(as_optimize_metric(m_any))
    return resolved


def _resolve_single(name: str, model: LLMSpec) -> BaseMetric:
    """Resolve a single metric name to a BaseMetric instance."""

    # ── geval:criteria / llm_judge:criteria shorthand ────────────
    if name.startswith("geval:"):
        criteria = name[len("geval:") :].strip()
        return GEvalMetric(criteria=criteria, model=model)

    if name.startswith("llm_judge:"):
        criteria = name[len("llm_judge:") :].strip() or "Is the response correct and helpful?"
        return LLMJudgeMetric(criteria=criteria, model=model)

    if name == "llm_judge":
        return LLMJudgeMetric(
            criteria="Is the response correct, complete, and helpful?",
            model=model,
        )

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
        f"Special: ['geval:<criteria>', 'llm_judge', 'llm_judge:<criteria>', 'red_team']. "
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

    def __post_init__(self) -> None:
        # Accept class-agent ``score()`` metrics and ``OptimizeMetricAdapter``
        # too: CompositeMetric awaits ``metric.evaluate`` on every component.
        self.metric = as_optimize_metric(self.metric)


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

    def __init__(self, metrics: list[WeightedMetric], *, name: str = "composite") -> None:
        if not metrics:
            raise ValueError("CompositeMetric requires at least one WeightedMetric")
        negative = [m.name for m in metrics if m.weight < 0]
        if negative:
            # Every metric scores higher-is-better (LatencyMetric / CostMetric
            # included), so a negative weight *rewards* the worse outcome.
            raise ValueError(
                f"CompositeMetric weights must be >= 0 (got negative for {negative}); "
                "all metrics are higher-is-better — give efficiency terms a small "
                "positive weight instead."
            )
        self.name = name
        self._metrics = metrics
        total_weight = sum(m.weight for m in metrics)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        self._total_weight = total_weight

    @property
    def metrics(self) -> list[WeightedMetric]:
        """Weighted sub-metrics that make up this composite."""
        return self._metrics

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
        failed: list[str] = []

        for wm in self._metrics:
            try:
                result = await wm.metric.evaluate(query, response, expected, context)
                if (result.metadata or {}).get("evaluation_failed"):
                    failed.append(wm.name)
                    dimensions[wm.name] = 0.0
                    if result.reason:
                        feedback_parts.append(f"[{wm.name}] {result.reason}")
                    continue
                dimensions[wm.name] = result.score
                weighted_sum += result.score * wm.weight
                if result.reason:
                    feedback_parts.append(f"[{wm.name}] {result.reason}")
            except Exception as exc:
                logger.warning(f"CompositeMetric: sub-metric '{wm.name}' failed: {exc}")
                dimensions[wm.name] = 0.0
                failed.append(wm.name)

        feedback = " | ".join(feedback_parts)
        failed_count = len(failed)
        failed_weight = sum(wm.weight for wm in self._metrics if wm.name in failed)
        # A failed component counts as 0 against the FULL weight. Dropping it
        # from the denominator made a composite *rise* when its judge failed
        # (judge 0.8 + format 1.0 → 0.83 working, 1.00 with the judge down).
        # When the failed share is the majority, the result is not a
        # measurement at all.
        if failed_count == len(self._metrics) or failed_weight * 2 >= self._total_weight:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason=feedback or "All composite sub-metrics failed",
                metadata={
                    "dimensions": dimensions,
                    "feedback": feedback,
                    "evaluation_failed": True,
                },
            )

        composite_score = weighted_sum / self._total_weight
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
                "failed_components": failed,
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

    def score(self, example: Any, prediction: Any) -> float:
        """Sync bridge for the ``agents.Metric`` protocol.

        Lets ``CompositeMetric`` sit directly in ``compile(metrics=...)``,
        ``agents.WeightedMetric`` or ``MetricLoss``. Delegates to
        :class:`~agentomatic.agents.OptimizeMetricAdapter`, so the question,
        response text and reference are extracted exactly as everywhere else
        (this used to be a second, divergent bridge that judged the label and
        the whole prediction JSON).

        Args:
            example: An ``AgentExample``.
            prediction: The agent's output dict.

        Returns:
            Composite score in ``[0, 1]`` (``0.0`` when the evaluation failed).
        """
        from agentomatic.agents.metrics import OptimizeMetricAdapter

        return OptimizeMetricAdapter(self, name=self.name).score(example, prediction)


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

    Returns a score between 0.0 and 1.0 based on response latency —
    **higher is better** (faster), like every other metric. Give it a small
    *positive* weight in ``CompositeMetric`` to add latency pressure.

    Latency comes from ``context`` (``"latency:2.35"``) or, inside
    ``PromptFitter``, from the measured duration of the agent call.

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

        # In CompositeMetric — positive weight, higher = faster:
        CompositeMetric(metrics=[
            WeightedMetric("quality", judge, weight=0.90),
            WeightedMetric("latency", LatencyMetric(), weight=0.10),
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

        Latency is read from ``context`` (``"latency:2.35"``), else from the
        run being scored by ``PromptFitter``. Without either, the evaluation
        is reported as failed (it used to return a neutral 0.5 that silently
        diluted every composite).
        """
        latency = self._extract_latency(response, context)
        if latency is None:
            run = current_scoring_run()
            duration_ms = getattr(run, "duration_ms", None) if run is not None else None
            if duration_ms:
                latency = float(duration_ms) / 1000.0

        if latency is None:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="No latency data available",
                metadata={"latency_seconds": None, "evaluation_failed": True},
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
                if item.startswith("latency:"):
                    try:
                        return float(item.split(":", 1)[1])
                    except (ValueError, IndexError):
                        pass
        return None


class CostMetric(BaseMetric):
    """Deployment-aware cost metric that penalises expensive responses.

    Returns a score between 0.0 and 1.0 based on token usage / cost —
    **higher is better** (cheaper). Give it a small *positive* weight in
    ``CompositeMetric``.

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
            WeightedMetric("quality", judge, weight=0.95),
            WeightedMetric("cost", CostMetric(), weight=0.05),
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
                if item.startswith("tokens:"):
                    try:
                        return int(item.split(":", 1)[1])
                    except (ValueError, IndexError):
                        pass
        # Rough estimate: ~4 chars per token
        return max(1, len(response) // 4)


# =====================================================================
# Protocol bridge: agents.Metric (sync ``score``) ⇄ optimize.BaseMetric
# =====================================================================

#: The run being scored by ``PromptFitter`` (set around each ``evaluate``
#: call). Lets example-aware metrics reach the dataset metadata and the
#: agent's structured output, which the string-based ``evaluate`` signature
#: cannot carry. ``None`` outside a fitter scoring pass.
_SCORING_RUN: ContextVar[Any] = ContextVar("agentomatic_scoring_run", default=None)


@contextmanager
def scoring_run(run: Any) -> Iterator[None]:
    """Expose ``run`` (a ``RunResult``) to metrics evaluated in this block.

    Args:
        run: The run result being scored.

    Yields:
        Nothing; the run is visible via :func:`current_scoring_run`.
    """
    token = _SCORING_RUN.set(run)
    try:
        yield
    finally:
        _SCORING_RUN.reset(token)


def current_scoring_run() -> Any:
    """Return the ``RunResult`` currently being scored, if any."""
    return _SCORING_RUN.get()


#: Character budget for the example labels shown to an LLM judge.
_JUDGE_LABELS_CHARS = 1500


def scoring_example_labels(*, has_context: bool) -> str:
    """The scored example's other inputs, metadata and tags, for a judge prompt.

    A judge already receives the question, the answer, the expected reference
    and the context documents; this adds what else the example was asked
    with (e.g. a customer tier, a locale, required facts, tags) so the
    answer is judged in that light.

    Args:
        has_context: The judge already lists the context documents, so the
            ``context`` input is left out.

    Returns:
        Compact JSON, or ``""`` when no run is being scored or it has none.
    """
    import json

    from agentomatic.optimize.dataset import example_context

    meta = getattr(current_scoring_run(), "metadata", None) or {}
    example = meta.get("example")
    view = example_context(
        example if isinstance(example, dict) else None, None, meta.get("example_tags")
    )
    inputs = dict(view.pop("inputs", None) or {})
    if has_context:
        inputs.pop("context", None)
    if inputs:
        view = {"inputs": inputs, **view}
    if not view:
        return ""
    text = json.dumps(view, ensure_ascii=False, default=str)
    if len(text) > _JUDGE_LABELS_CHARS:
        text = text[: _JUDGE_LABELS_CHARS - 1] + "…"
    return text


def _decode_json_dict(text: str | None) -> dict[str, Any] | None:
    """Return ``text`` parsed as a JSON object, or ``None`` when it is not one."""
    import json

    if not text:
        return None
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


_STRUCTURED_HEADER = re.compile(
    r"^[ \t]*##[ \t]*Expected structured output[ \t]*$", re.MULTILINE | re.IGNORECASE
)


def _expected_structured(expected: str | None) -> dict[str, Any] | None:
    """Recover ``expected_output`` from an ``AgentExample.to_datapoint`` reference."""
    if not expected:
        return None
    header = _STRUCTURED_HEADER.search(expected)
    if header is None:
        return None
    rest = expected[header.end() :]
    following = _ANY_SECTION_HEADER.search(rest)
    return _decode_json_dict(rest[: following.start()] if following else rest)


class ScoreMetricAdapter(BaseMetric):
    """Run a sync ``score(example, prediction)`` metric as an optimize metric.

    Class-agent metrics (:mod:`agentomatic.agents.metrics` — e.g.
    ``ExactKeyMatchMetric``, ``ContainsTermsMetric``, ``agents.WeightedMetric``)
    implement ``score(example, prediction) -> float``. ``PromptFitter`` scores
    candidates through the async ``evaluate(query, response, expected,
    context)`` protocol instead, so handing it such a metric used to fail with
    ``AttributeError: ... has no attribute 'evaluate'``. This adapter rebuilds
    an :class:`~agentomatic.agents.types.AgentExample` and a prediction dict
    and calls ``score`` in a worker thread. Inside ``PromptFitter`` the
    rebuilt example carries the dataset metadata (``must_include``, rubric
    hints, …) and the prediction is the agent's full structured output, so a
    class-agent metric scores exactly as it does in ``agent.evaluate``.

    You rarely need it directly — :func:`as_optimize_metric` applies it
    wherever an optimize metric is required.

    Args:
        metric: Object with ``score(example, prediction) -> float``.
        name: Optional override name (defaults to ``metric.name``).
    """

    def __init__(self, metric: Any, name: str | None = None) -> None:
        self.metric = metric
        self.name = name or str(getattr(metric, "name", None) or "score_metric")

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        """Score ``response`` with the wrapped sync metric."""
        import asyncio

        from agentomatic.agents.types import AgentExample

        run = current_scoring_run()
        run_meta = dict(getattr(run, "metadata", None) or {})
        example_meta = dict(run_meta.get("example") or {})

        answer = plain_expected(expected)
        expected_output = _expected_structured(expected) or _decode_json_dict(answer)
        if expected_output is None and answer is not None:
            expected_output = {"response": answer}
        # Rebuild the example the metric would see in ``agent.evaluate()``:
        # the agent inputs (``metadata.invoke`` — the original ``context``
        # included) and the tags, not only the question.
        from agentomatic.optimize.dataset import INTERNAL_INVOKE_KEYS

        invoke = example_meta.get("invoke")
        inputs = (
            {k: v for k, v in invoke.items() if k not in INTERNAL_INVOKE_KEYS}
            if isinstance(invoke, dict)
            else {}
        )
        if context and "context" not in inputs:
            inputs["context"] = list(context)
        example = AgentExample(
            id=str(example_meta.get("id") or "fit"),
            input={
                **inputs,
                "current_query": query,
                "query": query,
                "question": query,
            },
            expected_output=expected_output,
            metadata=example_meta,
            tags=[str(t) for t in run_meta.get("example_tags") or []],
            split=str(example_meta.get("split") or "validation"),
        )
        output = run_meta.get("output")
        prediction = (
            dict(output)
            if isinstance(output, dict)
            else (_decode_json_dict(response) or {"response": response})
        )
        score = await asyncio.to_thread(self.metric.score, example, prediction)
        return EvalResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, float(score))),
            reason=f"{type(self.metric).__name__}.score",
        )


def as_optimize_metric(metric: Any) -> BaseMetric:
    """Coerce any supported metric into an optimize :class:`BaseMetric`.

    Accepts, in order:

    1. An ``OptimizeMetricAdapter`` — unwrapped to the optimize metric it
       carries (keeps ``CompositeMetric`` per-dimension tracking intact).
    2. A :class:`BaseMetric` (or any object with an async ``evaluate``).
    3. A class-agent metric with ``score(example, prediction)`` — wrapped in
       :class:`ScoreMetricAdapter`.

    Args:
        metric: The metric to coerce.

    Returns:
        A metric exposing ``async evaluate(query, response, expected, context)``.

    Raises:
        TypeError: When ``metric`` implements neither protocol.
    """
    import inspect

    from agentomatic.agents.metrics import OptimizeMetricAdapter

    if isinstance(metric, OptimizeMetricAdapter):
        return as_optimize_metric(metric.optimize_metric)
    if isinstance(metric, BaseMetric):
        return metric
    if inspect.iscoroutinefunction(getattr(metric, "evaluate", None)):
        return cast(BaseMetric, metric)
    if callable(getattr(metric, "score", None)):
        return ScoreMetricAdapter(metric)
    raise TypeError(
        f"{type(metric).__name__} is not a usable metric: implement "
        "`async evaluate(query, response, expected=None, context=None) -> EvalResult` "
        "(agentomatic.optimize.BaseMetric) or "
        "`score(example, prediction) -> float` (agentomatic.agents.Metric)."
    )
