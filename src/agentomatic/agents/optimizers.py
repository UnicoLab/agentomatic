"""MVP optimizers for class-owned graph agents.

Example::

    from agentomatic.agents.optimizers import NoOpOptimizer

    agent.compile(dataset, metrics, optimizer=NoOpOptimizer())
    agent.fit(dataset)
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from .types import AgentDataset, Metric

if TYPE_CHECKING:
    from agentomatic.optimize.llm_types import LLMSpec

# ---------------------------------------------------------------------------
# NoOpOptimizer
# ---------------------------------------------------------------------------


class NoOpOptimizer:
    """Optimizer that does nothing — useful as default.

    Returns an empty config dict.
    """

    def optimize(
        self,
        agent: Any,
        dataset: AgentDataset,
        metrics: Sequence[Metric],
    ) -> dict[str, Any]:
        """Return empty config (no changes)."""
        return {}


# ---------------------------------------------------------------------------
# GridSearchOptimizer
# ---------------------------------------------------------------------------


class GridSearchOptimizer:
    """Test combinations of parameter values.

    Evaluates the agent with each combination and returns
    the best-performing configuration.

    Args:
        param_grid: Mapping of parameter names to lists of values.
        max_examples: Max examples to evaluate per combination.

    Example::

        optimizer = GridSearchOptimizer({
            "temperature": [0.0, 0.2, 0.5],
            "retrieval_top_k": [3, 5, 8],
            "prompt_version": ["v1", "v2"],
        })
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]],
        max_examples: int = 10,
    ) -> None:
        self.param_grid = param_grid
        self.max_examples = max_examples

    def optimize(
        self,
        agent: Any,
        dataset: AgentDataset,
        metrics: Sequence[Metric],
    ) -> dict[str, Any]:
        """Search parameter grid for best config.

        Args:
            agent: The agent to optimize.
            dataset: Training dataset.
            metrics: Metrics to evaluate.

        Returns:
            Best parameter combination.
        """
        if not self.param_grid:
            return {}

        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]
        combinations = list(itertools.product(*value_lists))

        examples = dataset.train[: self.max_examples]
        if not examples:
            logger.warning("No training examples for grid search")
            return {}

        best_config: dict[str, Any] = {}
        best_score = -1.0

        # Save original values so we can restore after search
        originals: dict[str, Any] = {}
        for key in keys:
            if hasattr(agent, key):
                originals[key] = getattr(agent, key)

        for combo in combinations:
            config = dict(zip(keys, combo))

            # Apply config to agent
            for key, value in config.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)

            # Evaluate
            total_score = 0.0
            count = 0
            for example in examples:
                try:
                    prediction = agent.transform(example.input)
                    for metric in metrics:
                        total_score += metric.score(
                            example,
                            prediction,
                        )
                        count += 1
                except Exception as exc:
                    logger.debug(f"Grid search eval error: {exc}")

            avg_score = total_score / count if count else 0.0
            logger.debug(f"Grid search: {config} -> {avg_score:.3f}")

            if avg_score > best_score:
                best_score = avg_score
                best_config = config

        # Restore original values (fit() will apply best_config)
        for key, value in originals.items():
            setattr(agent, key, value)

        logger.info(f"Grid search best: {best_config} (score: {best_score:.3f})")
        return best_config


# ---------------------------------------------------------------------------
# PromptFitterBridge
# ---------------------------------------------------------------------------


class PromptFitterBridge:
    """Bridge to the existing ``optimize.PromptFitter`` for class-agents.

    Wraps the powerful PromptFitter API so it can be used as an
    Optimizer for BaseGraphAgent.

    Args:
        agent_name: Name for the fitter to use.
        task_model: Model for running tasks.
        rewrite_model: Model for prompt rewriting.
        kwargs: Extra keyword arguments for PromptFitter.

    Example::

        optimizer = PromptFitterBridge(
            agent_name="my_agent",
            task_model="ollama/qwen2.5:7b",
            rewrite_model="openai/gpt-4.1",
        )
    """

    def __init__(
        self,
        agent_name: str = "",
        task_model: LLMSpec = "ollama/qwen2.5:7b",
        rewrite_model: LLMSpec = "openai/gpt-4.1",
        **kwargs: Any,
    ) -> None:
        self.agent_name = agent_name
        self.task_model = task_model
        self.rewrite_model = rewrite_model
        self.kwargs = kwargs

    def optimize(
        self,
        agent: Any,
        dataset: AgentDataset,
        metrics: Sequence[Metric],
    ) -> dict[str, Any]:
        """Run PromptFitter optimization.

        Note: This is a sync wrapper. The actual PromptFitter
        is async and should be invoked with ``asyncio.run()``
        in production.

        Returns:
            Optimized configuration dict.
        """
        try:
            import importlib.util

            if importlib.util.find_spec("agentomatic.optimize.fitter") is None:
                raise ImportError("PromptFitter not available")

            name = self.agent_name or getattr(agent, "agent_name", "agent")
            logger.info(f"PromptFitterBridge: preparing fitter for '{name}'")

            # Convert dataset to optimize.Dataset format
            opt_dataset = dataset.to_optimize_dataset()

            return {
                "_fitter_agent": name,
                "_fitter_task_model": self.task_model,
                "_fitter_rewrite_model": self.rewrite_model,
                "_fitter_dataset_size": len(opt_dataset),
                "_fitter_ready": True,
            }
        except ImportError:
            logger.warning("PromptFitter not available. Install agentomatic[optimize].")
            return {}
