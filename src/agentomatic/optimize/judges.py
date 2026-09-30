"""LLM-as-judge metrics for the PromptFitter optimisation loop.

Provides structured evaluation that returns ``MetricResult`` with score,
textual feedback, and per-dimension breakdowns — the key ingredient for
GEPA-style reflective prompt improvement.

Classes
-------
- **LocalJudgeMetric** — single SLM judge with rich feedback
- **MultiJudgePanel** — parallel multi-judge with aggregation
- **JudgeCalibrationSet** — human-labeled set for validating judge reliability
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.optimize.metrics import (
    _JUDGE_SCORE_KEYS,
    BaseMetric,
    EvalResult,
    MetricResult,
    coerce_judge_score,
)

if TYPE_CHECKING:
    from agentomatic.optimize.llm_types import LLMSpec


def _as_list(value: Any) -> list[str]:
    """Coerce a judge field that should be a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _judge_failed(result: MetricResult) -> bool:
    """True when a judge's ``MetricResult`` marks a failed evaluation."""
    flag = getattr(result, "_judge_failed", None)
    if flag is not None:
        return bool(flag)
    return (result.feedback or "").startswith("Judge evaluation failed:")


# =====================================================================
# Local SLM Judge
# =====================================================================


class LocalJudgeMetric(BaseMetric):
    """LLM-as-judge that returns rich ``MetricResult`` with feedback.

    Unlike ``LLMJudgeMetric`` (which returns a flat score), this metric
    produces textual feedback and per-dimension scores that GEPA-style
    optimisers can use for reflective prompt improvement.

    Example::

        judge = LocalJudgeMetric(
            name="scope_completeness",
            model="ollama/qwen2.5:7b",
            criteria="Evaluate whether the scoping response covers all project dimensions.",
            dimensions=["completeness", "specificity", "risk_coverage"],
            weight=0.35,
        )
        result = await judge.evaluate(query, response, expected)
    """

    def __init__(
        self,
        name: str = "local_judge",
        model: LLMSpec = "ollama/qwen2.5:7b",
        criteria: str = "Evaluate the quality and correctness of the response.",
        dimensions: list[str] | None = None,
        weight: float = 1.0,
        temperature: float = 0.0,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.criteria = criteria
        self.dimensions = dimensions or ["correctness", "completeness", "relevance"]
        #: Informational only — weight a judge inside ``CompositeMetric`` /
        #: ``WeightedMetric`` (this value is not read by the judge itself).
        self.weight = weight
        #: Endpoint for this judge only (else ``LLMCaller`` defaults / env).
        self.base_url = base_url
        self.api_key = api_key
        # Default 0.0 for reproducible scoring across epochs (reduces 0.33↔0.67
        # oscillation from sampling noise at temp>0).
        self.temperature = temperature

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        """Evaluate response and return rich result with feedback.

        The ``metadata`` field contains the full ``MetricResult`` under
        the ``"metric_result"`` key.
        """
        metric_result = await self.evaluate_rich(query, response, expected, context)
        failed = bool(getattr(metric_result, "_judge_failed", False))
        extras = getattr(metric_result, "_judge_extras", {}) or {}
        return EvalResult(
            metric_name=self.name,
            score=metric_result.score,
            reason=metric_result.feedback,
            metadata={
                "dimensions": metric_result.dimensions,
                "metric_result": metric_result,
                "evaluation_failed": failed,
                "motivation": extras.get("motivation", ""),
                "what_worked": extras.get("what_worked", []),
                "what_failed": extras.get("what_failed", []),
                "improvement_hints": extras.get("improvement_hints", []),
                "score_source": extras.get("score_source", ""),
                "raw_score": extras.get("raw_score"),
            },
        )

    async def evaluate_rich(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> MetricResult:
        """Evaluate with full ``MetricResult`` output."""
        from agentomatic.optimize.llm_caller import LLMCaller

        dimensions_list = "\n".join(f"  - {d}: score (0.0–1.0)" for d in self.dimensions)

        prompt = (
            "You are an expert evaluation judge for prompt optimization.\n"
            "Your job is to score the response AND produce extensive, actionable "
            "motivation so a rewriter can learn what to change next.\n\n"
            f"## Evaluation Criteria\n{self.criteria}\n\n"
            f"## Dimensions to Score\n{dimensions_list}\n\n"
            f"## Query\n{query}\n\n"
            f"## AI Response\n{response}\n\n"
        )
        if expected:
            prompt += (
                "## Expected / Quality Reference (Ground Truth)\n"
                "Use this as the quality bar. If it contains a 'quality contract' "
                "or schema flags, judge semantic fulfillment — not literal key names.\n"
                f"{expected}\n\n"
            )
        if context:
            ctx_block = "\n".join(str(c)[:800] for c in context[:5])
            prompt += f"## Context Documents\n{ctx_block}\n\n"

        prompt += (
            "## Instructions\n"
            "1. Score each dimension from 0.0 to 1.0 with justification.\n"
            "2. Write extensive motivation: what was correct, what was wrong, "
            "and why the score is not higher.\n"
            "3. Give concrete improvement hints the prompt rewriter can use "
            "(general rules, NOT hardcoding this exact query).\n"
            "4. Return a JSON object with this exact structure:\n"
            "{\n"
            '  "overall_score": 0.X,\n'
            '  "feedback": "Concise summary of strengths and weaknesses",\n'
            '  "motivation": "Extensive justification for the overall score",\n'
            '  "what_worked": ["..."],\n'
            '  "what_failed": ["..."],\n'
            '  "improvement_hints": ["generalizable prompt fixes..."],\n'
            '  "dimensions": {\n'
        )
        for d in self.dimensions:
            prompt += f'    "{d}": 0.X,\n'
        prompt += "  }\n}\n\nReturn ONLY the JSON object.\n"

        try:
            data = await LLMCaller.call_with_json(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            if not isinstance(data, dict) or not data:
                raise ValueError("Judge returned no JSON object")

            # Per-dimension scores, each parsed on its own (one malformed
            # dimension used to discard a perfectly good overall score).
            dims: dict[str, float] = {}
            raw_dims = data.get("dimensions") or data.get("scores") or {}
            if isinstance(raw_dims, dict):
                lowered = {str(k).lower(): v for k, v in raw_dims.items()}
                for d in self.dimensions:
                    value = coerce_judge_score(lowered.get(d.lower()))
                    if value is not None:
                        dims[d] = value

            raw_overall = next((data[k] for k in _JUDGE_SCORE_KEYS if k in data), None)
            overall = coerce_judge_score(raw_overall)
            score_source = "overall"
            if overall is None and dims:
                # Small models often return only the dimension scores.
                overall = sum(dims.values()) / len(dims)
                score_source = "dimension_mean"
            if overall is None:
                raise ValueError(
                    f"Judge reply has no usable score (keys: {sorted(data)[:8]}, "
                    f"overall={raw_overall!r})"
                )

            feedback = str(data.get("feedback", ""))
            motivation = str(data.get("motivation", "")).strip()
            if motivation and motivation not in feedback:
                feedback = f"{feedback}\n\nMotivation: {motivation}".strip()
            # Missing dimensions inherit the overall score (not a fabricated 0.5).
            for d in self.dimensions:
                dims.setdefault(d, overall)

            result = MetricResult(score=overall, feedback=feedback, dimensions=dims)
            # Attach rich judge fields for epoch learnings / rewrite briefings.
            rich_meta = {
                "motivation": motivation,
                "what_worked": _as_list(data.get("what_worked")),
                "what_failed": _as_list(data.get("what_failed")),
                "improvement_hints": _as_list(data.get("improvement_hints")),
                "score_source": score_source,
                "raw_score": raw_overall,
            }
            # MetricResult has no metadata field — stash via dynamic attributes
            # read by LocalJudgeMetric.evaluate() / MultiJudgePanel.
            result._judge_extras = rich_meta  # type: ignore[attr-defined]
            result._judge_failed = False  # type: ignore[attr-defined]
            return result

        except Exception as exc:
            logger.warning(f"LocalJudgeMetric '{self.name}' failed: {exc}")
            # Honest failure — never fabricate a mid-scale score.
            failed = MetricResult(
                score=0.0,
                feedback=f"Judge evaluation failed: {exc}",
                dimensions={d: 0.0 for d in self.dimensions},
            )
            failed._judge_failed = True  # type: ignore[attr-defined]
            return failed


# =====================================================================
# Multi-Judge Panel
# =====================================================================


class MultiJudgePanel(BaseMetric):
    """Run several judges in parallel and aggregate their scores.

    A panel reduces the variance and self-preference of a single judge
    ("mixture of experts"). Judges that fail are dropped; each remaining
    judge keeps *its own* weight.

    Aggregations: ``"average"`` (weighted mean, default), ``"median"``
    (robust to one outlier judge — ``"majority"`` is an alias), ``"min"``
    (pessimistic: every judge must be satisfied), ``"max"``.

    Example::

        panel = MultiJudgePanel(
            judges=[
                LocalJudgeMetric(name="judge_qwen", model="ollama/qwen2.5:7b"),
                LocalJudgeMetric(name="judge_llama", model="ollama/llama3.1:8b"),
            ],
            aggregation="average",
            weights=[2.0, 1.0],
        )
    """

    name = "multi_judge_panel"

    _AGGREGATIONS = ("average", "median", "min", "max")

    def __init__(
        self,
        judges: list[LocalJudgeMetric],
        aggregation: str = "average",
        weights: list[float] | None = None,
        *,
        name: str = "multi_judge_panel",
    ) -> None:
        if not judges:
            raise ValueError("MultiJudgePanel requires at least one judge")
        aggregation = {"majority": "median", "majority_vote": "median"}.get(
            aggregation, aggregation
        )
        if aggregation not in self._AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. Choose from {list(self._AGGREGATIONS)}."
            )
        self.name = name
        self._judges = judges
        self._aggregation = aggregation
        self._weights = list(weights) if weights is not None else [1.0] * len(judges)
        if len(self._weights) != len(judges):
            raise ValueError("Number of weights must match number of judges")
        if any(w < 0 for w in self._weights) or sum(self._weights) <= 0:
            raise ValueError("Judge weights must be >= 0 with a positive total")

    async def evaluate(
        self,
        query: str,
        response: str,
        expected: str | None = None,
        context: list[str] | None = None,
    ) -> EvalResult:
        """Run all judges in parallel and aggregate."""
        tasks = [judge.evaluate_rich(query, response, expected, context) for judge in self._judges]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Keep each judge's weight attached to its own result: dropping failed
        # judges used to shift the weights onto the wrong judges.
        usable: list[tuple[MetricResult, float, str]] = []
        failures: list[str] = []
        for judge, weight, r in zip(self._judges, self._weights, results, strict=True):
            judge_name = getattr(judge, "name", "judge")
            if isinstance(r, BaseException):
                logger.warning(f"MultiJudgePanel: judge '{judge_name}' raised: {r}")
                failures.append(judge_name)
            elif isinstance(r, MetricResult) and not _judge_failed(r):
                usable.append((r, weight, judge_name))
            else:
                failures.append(judge_name)

        if not usable or sum(w for _, w, _ in usable) <= 0:
            return EvalResult(
                metric_name=self.name,
                score=0.0,
                reason="All judges failed" if not usable else "All usable judges have weight 0",
                metadata={"evaluation_failed": True, "failed_judges": failures},
            )

        score, dimensions = self._aggregate(usable)
        feedback_parts = [f"[{name}] {r.feedback}" for r, _, name in usable if r.feedback]
        combined_feedback = " | ".join(feedback_parts)
        metric_result = MetricResult(
            score=score,
            feedback=combined_feedback,
            dimensions=dimensions,
        )
        return EvalResult(
            metric_name=self.name,
            score=score,
            reason=combined_feedback,
            metadata={
                "individual_scores": {name: r.score for r, _, name in usable},
                "failed_judges": failures,
                "aggregation": self._aggregation,
                "dimensions": dimensions,
                "metric_result": metric_result,
                "evaluation_failed": False,
            },
        )

    def _aggregate(
        self, usable: list[tuple[MetricResult, float, str]]
    ) -> tuple[float, dict[str, float]]:
        """Aggregate scores (per ``aggregation``) and dimensions (weighted mean)."""
        import statistics

        scores = [r.score for r, _, _ in usable]
        if self._aggregation == "median":
            score = float(statistics.median(scores))
        elif self._aggregation == "min":
            score = min(scores)
        elif self._aggregation == "max":
            score = max(scores)
        else:
            total_w = sum(w for _, w, _ in usable)
            score = sum(r.score * w for r, w, _ in usable) / total_w

        dimensions: dict[str, float] = {}
        keys = {k for r, _, _ in usable for k in r.dimensions}
        for key in keys:
            pairs = [(r.dimensions[key], w) for r, w, _ in usable if key in r.dimensions]
            weight = sum(w for _, w in pairs)
            if weight > 0:
                dimensions[key] = sum(v * w for v, w in pairs) / weight
        return score, dimensions


# =====================================================================
# Judge Calibration
# =====================================================================


@dataclass(slots=True)
class CalibrationPair:
    """Single human-labeled preference pair for judge calibration.

    Example::

        pair = CalibrationPair(
            query="What are the project risks?",
            response_a="The risks include...",
            response_b="Based on the analysis, key risks are...",
            human_preference="b",
            reason="B is grounded and complete.",
        )
    """

    query: str
    response_a: str
    response_b: str
    human_preference: str  # "a", "b", or "tie"
    reason: str = ""


@dataclass(slots=True)
class JudgeCalibrationSet:
    """Human-labeled preference pairs for validating judge reliability.

    Use this to ensure your local SLM judges agree with human evaluators
    before trusting them in the optimisation loop.

    Example::

        calibration = JudgeCalibrationSet(pairs=[
            CalibrationPair(
                query="...", response_a="...", response_b="...",
                human_preference="b", reason="B is more complete.",
            ),
        ])
        agreement = await calibration.calibrate(judge)
        if agreement < 0.7:
            logger.warning("Judge has low agreement with humans!")
    """

    pairs: list[CalibrationPair] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    async def calibrate(self, judge: LocalJudgeMetric) -> float:
        """Test judge agreement with human preferences.

        For each pair, evaluates both responses and checks whether
        the judge ranks them in the same order as the human.

        Returns:
            Agreement rate between 0.0 and 1.0.
        """
        if not self.pairs:
            logger.warning("Empty calibration set — returning 1.0")
            return 1.0

        agreements = 0

        for pair in self.pairs:
            try:
                result_a = await judge.evaluate_rich(
                    query=pair.query,
                    response=pair.response_a,
                )
                result_b = await judge.evaluate_rich(
                    query=pair.query,
                    response=pair.response_b,
                )

                if pair.human_preference == "a":
                    if result_a.score > result_b.score:
                        agreements += 1
                elif pair.human_preference == "b":
                    if result_b.score > result_a.score:
                        agreements += 1
                elif pair.human_preference == "tie":
                    if abs(result_a.score - result_b.score) < 0.1:
                        agreements += 1

            except Exception as exc:
                logger.warning(f"Calibration pair failed: {exc}")

        rate = agreements / len(self.pairs)
        logger.info(
            f"Judge '{judge.name}' calibration: {agreements}/{len(self.pairs)} "
            f"agreements ({rate:.0%})"
        )
        return rate

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> JudgeCalibrationSet:
        """Create from a list of dictionaries.

        Each dict should have: query, response_a, response_b,
        human_preference, and optionally reason.
        """
        pairs = [
            CalibrationPair(
                query=item["query"],
                response_a=item["response_a"],
                response_b=item["response_b"],
                human_preference=item["human_preference"],
                reason=item.get("reason", ""),
            )
            for item in items
        ]
        return cls(pairs=pairs)

    @classmethod
    def from_jsonl(cls, path: str) -> JudgeCalibrationSet:
        """Load from JSONL file."""
        import json

        pairs: list[CalibrationPair] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                pairs.append(
                    CalibrationPair(
                        query=data["query"],
                        response_a=data["response_a"],
                        response_b=data["response_b"],
                        human_preference=data["human_preference"],
                        reason=data.get("reason", ""),
                    )
                )
        return cls(pairs=pairs)
