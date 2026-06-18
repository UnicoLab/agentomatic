"""Failure analysis and clustering for the PromptFitter optimisation loop.

Provides semantic grouping of evaluation failures and dimensional
comparison between prompt candidates, enabling actionable improvement
recommendations.

Classes
-------
- **FailureClusterer** — groups failures into semantic clusters via LLM
- **DimensionAnalyzer** — compares metric dimensions across candidates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# =====================================================================
# Failure Clustering
# =====================================================================


@dataclass(slots=True)
class FailureCluster:
    """A group of semantically similar failures with deployment recommendations.

    Example::

        cluster = FailureCluster(
            label="missing_risk_section",
            description="Agent omits risk analysis from scoping responses",
            count=7,
            representative_examples=[...],
            suggested_fix="Add explicit instruction to always include risk analysis.",
            affected_params=["prompt.output_contract", "rag.top_k"],
            expected_metric_gain={"faithfulness": 0.18, "completeness": 0.12},
        )
    """

    label: str
    description: str
    count: int
    representative_examples: list[dict[str, Any]] = field(default_factory=list)
    suggested_fix: str = ""
    severity: float = 0.0  # 0.0 to 1.0, higher = more impactful

    # ── Deployment-first fields ───────────────────────────────────
    affected_params: list[str] = field(
        default_factory=list
    )  # e.g. ["rag.top_k", "tool_policy.force_retrieval"]
    expected_metric_gain: dict[str, float] = field(
        default_factory=dict
    )  # e.g. {"faithfulness": +0.18}


class FailureClusterer:
    """Groups evaluation failures into semantic clusters using an LLM.

    Analyses failure patterns across the evaluation set and produces
    actionable clusters with representative examples and suggested fixes.

    Example::

        clusterer = FailureClusterer(model="ollama/qwen2.5:7b")
        clusters = await clusterer.cluster(failures)
        for c in clusters:
            print(f"{c.label}: {c.count} failures — {c.suggested_fix}")
    """

    def __init__(
        self,
        model: str = "ollama/qwen2.5:7b",
        max_clusters: int = 8,
        max_examples_per_cluster: int = 3,
    ) -> None:
        self.model = model
        self.max_clusters = max_clusters
        self.max_examples_per_cluster = max_examples_per_cluster

    async def cluster(
        self,
        failures: list[dict[str, Any]],
    ) -> list[FailureCluster]:
        """Group failures into semantic clusters.

        Args:
            failures: List of dicts with at least ``query``, ``response``,
                ``expected`` (optional), ``avg_score``, and ``feedback``
                or ``details`` fields.

        Returns:
            List of ``FailureCluster`` objects sorted by severity.
        """
        if not failures:
            return []

        # For small failure sets, skip LLM and do keyword-based clustering
        if len(failures) <= 3:
            return self._simple_cluster(failures)

        try:
            return await self._llm_cluster(failures)
        except Exception as exc:
            logger.warning(f"LLM clustering failed, falling back to simple: {exc}")
            return self._simple_cluster(failures)

    async def _llm_cluster(
        self,
        failures: list[dict[str, Any]],
    ) -> list[FailureCluster]:
        """Use LLM to identify semantic failure patterns."""
        from agentomatic.optimize.llm_caller import LLMCaller

        # Build failure summary for the LLM
        failure_items = []
        for i, f in enumerate(failures[:20]):  # Limit to 20 for context window
            item = f"Failure {i + 1}:\n"
            item += f"  Query: {str(f.get('query', ''))[:150]}\n"
            item += f"  Response: {str(f.get('response', ''))[:200]}\n"
            if f.get("expected"):
                item += f"  Expected: {str(f['expected'])[:150]}\n"
            item += f"  Score: {f.get('avg_score', 0):.2f}\n"
            # Extract feedback from details or direct field
            feedback = f.get("feedback", "")
            if not feedback and f.get("details"):
                reasons = [d.get("reason", "") for d in f["details"] if d.get("reason")]
                feedback = "; ".join(reasons)
            if feedback:
                item += f"  Feedback: {feedback[:200]}\n"
            failure_items.append(item)

        failures_text = "\n".join(failure_items)

        prompt = (
            "You are an expert AI evaluation analyst. Analyse these evaluation failures "
            "and group them into semantic clusters.\n\n"
            f"## Failures ({len(failures)} total, showing up to 20)\n\n"
            f"{failures_text}\n\n"
            f"## Instructions\n"
            f"1. Identify {self.max_clusters} or fewer distinct failure patterns.\n"
            f"2. For each cluster, provide a label, description, count, and suggested fix.\n"
            f"3. Return a JSON array of clusters.\n\n"
            "Return ONLY a JSON array like:\n"
            "[\n"
            "  {\n"
            '    "label": "short_snake_case_label",\n'
            '    "description": "What goes wrong in these cases",\n'
            '    "count": 5,\n'
            '    "suggested_fix": "Add instruction to the prompt to...",\n'
            '    "severity": 0.8,\n'
            '    "affected_params": ["prompt.output_contract", "rag.top_k"],\n'
            '    "expected_metric_gain": {"faithfulness": 0.15}\n'
            "  }\n"
            "]\n"
        )

        data = await LLMCaller.call_with_json(
            model=self.model,
            prompt=prompt,
            temperature=0.2,
        )

        # Handle both dict (wrapped) and list responses
        clusters_data: list[dict[str, Any]] = []
        if isinstance(data, list):
            clusters_data = data  # type: ignore[assignment]
        elif isinstance(data, dict):
            # Might be wrapped: {"clusters": [...]} or single cluster
            if "clusters" in data:
                clusters_data = data["clusters"]
            elif "label" in data:
                clusters_data = [data]

        clusters: list[FailureCluster] = []
        for cd in clusters_data[: self.max_clusters]:
            cluster = FailureCluster(
                label=str(cd.get("label", "unknown")),
                description=str(cd.get("description", "")),
                count=int(cd.get("count", 1)),
                suggested_fix=str(cd.get("suggested_fix", "")),
                severity=max(0.0, min(1.0, float(cd.get("severity", 0.5)))),
                affected_params=cd.get("affected_params", []),
                expected_metric_gain=cd.get("expected_metric_gain", {}),
            )
            # Attach representative examples
            cluster.representative_examples = [
                f
                for f in failures
                if any(
                    keyword in str(f.get("query", "")).lower()
                    or keyword in str(f.get("response", "")).lower()
                    for keyword in cluster.label.lower().replace("_", " ").split()
                )
            ][: self.max_examples_per_cluster]

            clusters.append(cluster)

        clusters.sort(key=lambda c: c.severity, reverse=True)
        return clusters

    def _simple_cluster(
        self,
        failures: list[dict[str, Any]],
    ) -> list[FailureCluster]:
        """Simple keyword-based clustering without LLM."""
        # Group by score range
        severe = [f for f in failures if f.get("avg_score", 0) < 0.3]
        moderate = [f for f in failures if 0.3 <= f.get("avg_score", 0) < 0.5]

        clusters: list[FailureCluster] = []
        if severe:
            clusters.append(
                FailureCluster(
                    label="severe_failures",
                    description=f"Responses scoring below 0.3 ({len(severe)} cases)",
                    count=len(severe),
                    representative_examples=severe[: self.max_examples_per_cluster],
                    suggested_fix="Major prompt revision needed for these query types.",
                    severity=0.9,
                )
            )
        if moderate:
            clusters.append(
                FailureCluster(
                    label="moderate_failures",
                    description=f"Responses scoring 0.3–0.5 ({len(moderate)} cases)",
                    count=len(moderate),
                    representative_examples=moderate[: self.max_examples_per_cluster],
                    suggested_fix="Targeted improvements for partial failures.",
                    severity=0.5,
                )
            )

        return clusters


# =====================================================================
# Dimension Analysis
# =====================================================================


@dataclass(slots=True)
class DimensionComparison:
    """Comparison of a single metric dimension across baseline and candidate.

    Example::

        comp = DimensionComparison(
            dimension="faithfulness",
            baseline_score=0.54,
            candidate_score=0.78,
            absolute_delta=0.24,
            decision="keep",
        )
    """

    dimension: str
    baseline_score: float
    candidate_score: float
    absolute_delta: float
    decision: str  # "keep", "accept_if_above_threshold", "reject", "watch"


class DimensionAnalyzer:
    """Analyses which metric dimensions improved/regressed across candidates.

    Produces the comparison table used for candidate acceptance decisions.

    Example::

        analyzer = DimensionAnalyzer(
            critical_threshold=0.4,
            regression_tolerance=0.05,
        )
        comparisons = analyzer.compare(baseline_dims, candidate_dims)
        table = analyzer.format_table(comparisons)
    """

    def __init__(
        self,
        critical_threshold: float = 0.4,
        regression_tolerance: float = 0.05,
    ) -> None:
        self.critical_threshold = critical_threshold
        self.regression_tolerance = regression_tolerance

    def compare(
        self,
        baseline: dict[str, float],
        candidate: dict[str, float],
    ) -> list[DimensionComparison]:
        """Compare dimensions between baseline and candidate.

        Args:
            baseline: Per-dimension scores for baseline config.
            candidate: Per-dimension scores for candidate config.

        Returns:
            List of ``DimensionComparison`` with decisions.
        """
        all_dims = sorted(set(baseline.keys()) | set(candidate.keys()))
        comparisons: list[DimensionComparison] = []

        for dim in all_dims:
            b_score = baseline.get(dim, 0.0)
            c_score = candidate.get(dim, 0.0)
            delta = c_score - b_score

            # Decision logic
            if delta >= 0.0:
                decision = "keep"
            elif abs(delta) <= self.regression_tolerance:
                decision = "accept_if_above_threshold"
            elif c_score < self.critical_threshold:
                decision = "reject"
            else:
                decision = "watch"

            comparisons.append(
                DimensionComparison(
                    dimension=dim,
                    baseline_score=round(b_score, 4),
                    candidate_score=round(c_score, 4),
                    absolute_delta=round(delta, 4),
                    decision=decision,
                )
            )

        return comparisons

    def should_accept(
        self,
        comparisons: list[DimensionComparison],
        min_composite_delta: float = 0.05,
        composite_baseline: float = 0.0,
        composite_candidate: float = 0.0,
    ) -> tuple[bool, str]:
        """Decide whether to accept a candidate based on dimension comparisons.

        Returns:
            Tuple of (accept: bool, reason: str).
        """
        composite_delta = composite_candidate - composite_baseline

        # Check: composite improvement threshold
        if composite_delta < min_composite_delta:
            return False, (
                f"Composite improvement {composite_delta:+.3f} is below "
                f"threshold {min_composite_delta}"
            )

        # Check: no critical regressions
        rejections = [c for c in comparisons if c.decision == "reject"]
        if rejections:
            dims = ", ".join(c.dimension for c in rejections)
            return False, f"Critical regression in: {dims}"

        return True, f"Accepted with composite delta {composite_delta:+.3f}"

    def format_table(self, comparisons: list[DimensionComparison]) -> str:
        """Format comparisons as a readable text table.

        Returns:
            Multi-line string table.
        """
        header = (
            f"{'Dimension':<25} {'Baseline':>10} {'Candidate':>10} {'Delta':>10} {'Decision':>10}"
        )
        sep = "─" * len(header)
        lines = [sep, header, sep]

        for c in comparisons:
            delta_str = f"{c.absolute_delta:+.4f}"
            lines.append(
                f"{c.dimension:<25} {c.baseline_score:>10.4f} "
                f"{c.candidate_score:>10.4f} {delta_str:>10} "
                f"{c.decision:>10}"
            )

        lines.append(sep)
        return "\n".join(lines)
