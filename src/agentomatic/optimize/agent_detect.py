"""Auto-detect agent type for evaluation metric selection.

Infers the agent's operational mode (stateless, RAG, tool-using, etc.)
from its attributes and selects appropriate evaluation metrics without
manual configuration.

Example::

    from agentomatic.optimize.agent_detect import AgentType, detect_agent_type, Evaluator

    agent_type = detect_agent_type(my_agent)
    # → AgentType.TOOL_USING

    evaluator = Evaluator.for_agent(my_agent)
    print(evaluator.metrics)
    # → ["tool_call_accuracy", "tool_selection", "answer_relevancy"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# =====================================================================
# Agent Types
# =====================================================================


class AgentType(StrEnum):
    """Categories of agent behaviour for metric selection."""

    STATELESS = "stateless"
    """One-shot Q&A agent — no memory, no tools."""

    RAG = "rag"
    """Retrieval-Augmented Generation — has a retriever / vector store."""

    TOOL_USING = "tool_using"
    """Agent that calls tools / functions."""

    CONVERSATIONAL = "conversational"
    """Stateful chatbot with memory and conversation history."""

    DEEP_AGENT = "deep_agent"
    """Multi-step / hierarchical agent with sub-agents or planning."""

    CUSTOM = "custom"
    """User-defined agent that doesn't fit other categories."""


# =====================================================================
# Metric Presets per Agent Type
# =====================================================================

METRIC_PRESETS: dict[AgentType, list[str]] = {
    AgentType.STATELESS: [
        "answer_relevancy",
        "geval",
        "faithfulness",
    ],
    AgentType.RAG: [
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        "ragas_context_precision",
        "ragas_context_recall",
        "hallucination",
    ],
    AgentType.TOOL_USING: [
        "tool_call_accuracy",
        "tool_selection",
        "answer_relevancy",
        "geval",
    ],
    AgentType.CONVERSATIONAL: [
        "answer_relevancy",
        "toxicity",
        "geval",
    ],
    AgentType.DEEP_AGENT: [
        "task_completion",
        "step_efficiency",
        "goal_accuracy",
        "answer_relevancy",
    ],
    AgentType.CUSTOM: [
        "answer_relevancy",
        "geval",
        "faithfulness",
    ],
}


# =====================================================================
# Detection
# =====================================================================


def detect_agent_type(agent: Any) -> AgentType:
    """Determine the agent type by inspecting its attributes.

    Detection heuristics (checked in order):

    1. Has ``subagents`` or ``planner`` → :attr:`AgentType.DEEP_AGENT`
    2. Has ``retriever`` / ``vector_store`` / ``knowledge_base`` →
       :attr:`AgentType.RAG`
    3. Has non-empty ``tools`` attribute → :attr:`AgentType.TOOL_USING`
    4. Has ``memory`` or ``enable_long_term_memory`` →
       :attr:`AgentType.CONVERSATIONAL`
    5. Fallback → :attr:`AgentType.STATELESS`

    Args:
        agent: Any agent object.

    Returns:
        The detected :class:`AgentType`.
    """
    # Deep agent check
    if _has_attr(agent, "subagents") or _has_attr(agent, "planner"):
        return AgentType.DEEP_AGENT

    # RAG check
    if (
        _has_attr(agent, "retriever")
        or _has_attr(agent, "vector_store")
        or _has_attr(agent, "knowledge_base")
        or _has_attr(agent, "rag_config")
    ):
        return AgentType.RAG

    # Tool-using check
    tools = _get_attr(agent, "tools", default=None)
    if tools is not None and len(tools) > 0:
        return AgentType.TOOL_USING
    # Also check for tool-related attrs
    if _has_attr(agent, "tools_by_name") and getattr(agent, "tools_by_name", {}):
        return AgentType.TOOL_USING

    # Conversational check
    if (
        _has_attr(agent, "memory")
        or _attr_is_true(agent, "enable_long_term_memory")
        or _attr_is_true(agent, "enable_memory")
    ):
        return AgentType.CONVERSATIONAL

    # Capabilities-based check
    caps = _get_attr(agent, "capabilities", default=None)
    if caps is not None:
        if _attr_is_true(caps, "enable_long_term_memory") or _attr_is_true(caps, "enable_memory"):
            return AgentType.CONVERSATIONAL
        deep_flag = getattr(caps, "is_deep_agent", False)
        if callable(deep_flag):
            try:
                deep_flag = deep_flag()
            except Exception:
                deep_flag = False
        if _attr_is_true(caps, "is_deep_agent") or bool(deep_flag):
            return AgentType.DEEP_AGENT

    return AgentType.STATELESS


def get_metrics_for_agent_type(agent_type: AgentType) -> list[str]:
    """Return the recommended metric list for an agent type.

    Args:
        agent_type: An :class:`AgentType` value.

    Returns:
        List of metric name strings.
    """
    return METRIC_PRESETS.get(agent_type, METRIC_PRESETS[AgentType.STATELESS])


def list_available_metrics(agent_type: AgentType | None = None) -> list[str]:
    """List all metrics, optionally filtered by agent type.

    Args:
        agent_type: Optional filter.  Returns all metrics if ``None``.

    Returns:
        Sorted list of metric name strings.
    """
    if agent_type is not None:
        return sorted(METRIC_PRESETS.get(agent_type, []))
    all_metrics: set[str] = set()
    for metrics in METRIC_PRESETS.values():
        all_metrics.update(metrics)
    return sorted(all_metrics)


# =====================================================================
# Evaluator Factory
# =====================================================================


@dataclass
class Evaluator:
    """Evaluator configured for a specific agent type.

    Auto-selects appropriate metrics based on agent type detection.

    Example::

        evaluator = Evaluator.for_agent(my_agent)
        # evaluator.agent_type → AgentType.TOOL_USING
        # evaluator.metrics  → ["tool_call_accuracy", "tool_selection", ...]
    """

    agent_type: AgentType
    metrics: list[str] = field(default_factory=list)
    threshold: float = 0.7
    custom_criteria: str = ""

    @classmethod
    def for_agent(cls, agent: Any, **kwargs: Any) -> Evaluator:
        """Auto-detect agent type and create an evaluator.

        Args:
            agent: The agent instance to evaluate.
            kwargs: Additional keyword arguments forwarded to the
                constructor (e.g. ``threshold=0.8``).

        Returns:
            Configured :class:`Evaluator`.
        """
        agent_type = detect_agent_type(agent)
        metrics = get_metrics_for_agent_type(agent_type)
        return cls(agent_type=agent_type, metrics=metrics, **kwargs)

    @classmethod
    def for_rag(cls, **kwargs: Any) -> Evaluator:
        """Create an evaluator pre-configured for RAG agents."""
        return cls(
            agent_type=AgentType.RAG,
            metrics=METRIC_PRESETS[AgentType.RAG],
            **kwargs,
        )

    @classmethod
    def for_tools(cls, **kwargs: Any) -> Evaluator:
        """Create an evaluator pre-configured for tool-using agents."""
        return cls(
            agent_type=AgentType.TOOL_USING,
            metrics=METRIC_PRESETS[AgentType.TOOL_USING],
            **kwargs,
        )

    @classmethod
    def for_deep_agent(cls, **kwargs: Any) -> Evaluator:
        """Create an evaluator pre-configured for deep agents."""
        return cls(
            agent_type=AgentType.DEEP_AGENT,
            metrics=METRIC_PRESETS[AgentType.DEEP_AGENT],
            **kwargs,
        )

    @classmethod
    def for_stateless(cls, **kwargs: Any) -> Evaluator:
        """Create an evaluator pre-configured for stateless agents."""
        return cls(
            agent_type=AgentType.STATELESS,
            metrics=METRIC_PRESETS[AgentType.STATELESS],
            **kwargs,
        )

    def to_config(self) -> dict[str, Any]:
        """Convert to a dict suitable for optimise config creation."""
        return {
            "agent_type": self.agent_type.value,
            "metrics": list(self.metrics),
            "threshold": self.threshold,
        }

    async def evaluate(
        self,
        agent: Any,
        test_cases: list[Any],
        *,
        model: str = "ollama/mistral:7b",
    ) -> dict[str, Any]:
        """Run a lightweight evaluation of *agent* on *test_cases*.

        Uses an LLM judge over the auto-selected metric list. Returns a
        summary dict with per-case scores and an aggregate mean.

        Args:
            agent: Agent instance (local callable / BaseGraphAgent).
            test_cases: Cases with ``input``/``query`` and
                ``expected_output``/``expected_answer``.
            model: Judge model (LiteLLM format).

        Returns:
            ``{"agent_type", "metrics", "mean_score", "n_cases", "passed",
            "scores"}``.
        """
        from agentomatic.optimize.dataset import DataPoint, Dataset
        from agentomatic.optimize.fitter import PromptFitter
        from agentomatic.optimize.metrics import LLMJudgeMetric

        points: list[DataPoint] = []
        for case in test_cases:
            query = getattr(case, "input", None) or getattr(case, "query", None) or ""
            expected = (
                getattr(case, "expected_output", None)
                or getattr(case, "expected_answer", None)
                or ""
            )
            points.append(
                DataPoint(
                    query=str(query),
                    expected_answer=str(expected),
                    context=list(getattr(case, "context", []) or []),
                )
            )
        dataset = Dataset(points=points)
        agent_name = str(getattr(agent, "agent_name", "agent"))
        criteria = self.custom_criteria or (
            f"Evaluate the response for a {self.agent_type.value} agent. "
            f"Consider: {', '.join(self.metrics)}."
        )
        metric = LLMJudgeMetric(name="evaluator", criteria=criteria, model=model)
        fitter = PromptFitter(
            agent=agent_name,
            local_agent=agent,
            task_model=model,
            rewrite_model=model,
            max_trials=1,
            patience=1,
            auto_report=False,
        )
        # Evaluate baseline only (no optimisation rounds).
        config = fitter._load_baseline_config()
        score, dims, details = await fitter._evaluate_config(config, dataset, metric)
        scores = [float(d.get("avg_score", 0.0) or 0.0) for d in details]
        return {
            "agent_type": self.agent_type.value,
            "metrics": list(self.metrics),
            "dimensions": dict(dims),
            "mean_score": float(score),
            "n_cases": len(points),
            "passed": float(score) >= self.threshold,
            "threshold": self.threshold,
            "scores": scores,
        }


# =====================================================================
# Helpers
# =====================================================================


def _has_attr(obj: Any, name: str) -> bool:
    """Check if *obj* has a non-None, non-empty attribute *name*."""
    try:
        val = getattr(obj, name, None)
        return val is not None and val != [] and val != {}
    except Exception:
        return False


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Safely get an attribute, returning *default* on any error."""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _attr_is_true(obj: Any, name: str) -> bool:
    """Check if attribute *name* is truthy."""
    try:
        return bool(getattr(obj, name, False))
    except Exception:
        return False
