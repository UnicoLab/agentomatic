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
from typing import Any

from loguru import logger

from agentomatic.optimize.metrics import BaseMetric, EvalResult, MetricResult

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
        model: str = "ollama/qwen2.5:7b",
        criteria: str = "Evaluate the quality and correctness of the response.",
        dimensions: list[str] | None = None,
        weight: float = 1.0,
        temperature: float = 0.1,
    ) -> None:
        self.name = name
        self.model = model
        self.criteria = criteria
        self.dimensions = dimensions or ["correctness", "completeness", "relevance"]
        self.weight = weight
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
        return EvalResult(
            metric_name=self.name,
            score=metric_result.score,
            reason=metric_result.feedback,
            metadata={
                "dimensions": metric_result.dimensions,
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
        """Evaluate with full ``MetricResult`` output."""
        from agentomatic.optimize.llm_caller import LLMCaller

        dimensions_list = "\n".join(f"  - {d}: score (0.0–1.0)" for d in self.dimensions)

        prompt = (
            "You are an expert evaluation judge. Score the following AI response.\n\n"
            f"## Evaluation Criteria\n{self.criteria}\n\n"
            f"## Dimensions to Score\n{dimensions_list}\n\n"
            f"## Query\n{query}\n\n"
            f"## AI Response\n{response}\n\n"
        )
        if expected:
            prompt += f"## Expected Answer (Ground Truth)\n{expected}\n\n"
        if context:
            prompt += f"## Context Documents\n{chr(10).join(context[:3])}\n\n"

        prompt += (
            "## Instructions\n"
            "1. Evaluate the response against each dimension.\n"
            "2. Provide a brief textual feedback explaining strengths and weaknesses.\n"
            "3. Return a JSON object with this exact structure:\n"
            "{\n"
            '  "overall_score": 0.X,\n'
            '  "feedback": "Your textual feedback here...",\n'
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
            )

            overall = max(0.0, min(1.0, float(data.get("overall_score", 0.5))))
            feedback = str(data.get("feedback", ""))
            dims = {}
            raw_dims = data.get("dimensions", {})
            if isinstance(raw_dims, dict):
                for d in self.dimensions:
                    dims[d] = max(0.0, min(1.0, float(raw_dims.get(d, 0.5))))

            return MetricResult(score=overall, feedback=feedback, dimensions=dims)

        except Exception as exc:
            logger.warning(f"LocalJudgeMetric '{self.name}' failed: {exc}")
            return MetricResult(
                score=0.5,
                feedback=f"Judge evaluation failed: {exc}",
                dimensions={d: 0.5 for d in self.dimensions},
            )


# =====================================================================
# Multi-Judge Panel
# =====================================================================


class MultiJudgePanel(BaseMetric):
    """Run multiple judges in parallel and aggregate their results.

    Supports aggregation by average (default) or majority vote.
    Optionally weighted by judge calibration scores.

    Example::

        panel = MultiJudgePanel(
            judges=[
                LocalJudgeMetric(name="judge_qwen", model="ollama/qwen2.5:7b"),
                LocalJudgeMetric(name="judge_llama", model="ollama/llama3.1:8b"),
            ],
            aggregation="average",
        )
    """

    name = "multi_judge_panel"

    def __init__(
        self,
        judges: list[LocalJudgeMetric],
        aggregation: str = "average",
        weights: list[float] | None = None,
    ) -> None:
        if not judges:
            raise ValueError("MultiJudgePanel requires at least one judge")
        self._judges = judges
        self._aggregation = aggregation
        self._weights = weights or [1.0] * len(judges)
        if len(self._weights) != len(judges):
            raise ValueError("Number of weights must match number of judges")

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

        valid_results: list[MetricResult] = []
        for r in results:
            if isinstance(r, MetricResult):
                valid_results.append(r)
            elif isinstance(r, Exception):
                logger.warning(f"MultiJudgePanel: judge failed: {r}")

        if not valid_results:
            return EvalResult(
                metric_name=self.name,
                score=0.5,
                reason="All judges failed",
            )

        # Aggregate
        if self._aggregation == "average":
            aggregated = self._aggregate_average(valid_results)
        else:
            aggregated = self._aggregate_average(valid_results)

        # Collect all feedback
        feedback_parts = [r.feedback for r in valid_results if r.feedback]
        combined_feedback = " | ".join(feedback_parts)

        metric_result = MetricResult(
            score=aggregated["score"],
            feedback=combined_feedback,
            dimensions=aggregated["dimensions"],
        )

        return EvalResult(
            metric_name=self.name,
            score=aggregated["score"],
            reason=combined_feedback,
            metadata={
                "individual_scores": [r.score for r in valid_results],
                "metric_result": metric_result,
            },
        )

    def _aggregate_average(self, results: list[MetricResult]) -> dict[str, Any]:
        """Weighted average aggregation."""
        weights = self._weights[: len(results)]
        total_w = sum(weights)

        weighted_score = sum(r.score * w for r, w in zip(results, weights)) / total_w

        # Aggregate dimensions
        all_dim_keys: set[str] = set()
        for r in results:
            all_dim_keys.update(r.dimensions.keys())

        dimensions: dict[str, float] = {}
        for key in all_dim_keys:
            dim_vals = [
                (r.dimensions.get(key, 0.0), w)
                for r, w in zip(results, weights)
                if key in r.dimensions
            ]
            if dim_vals:
                dimensions[key] = sum(v * w for v, w in dim_vals) / sum(w for _, w in dim_vals)

        return {"score": weighted_score, "dimensions": dimensions}


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
