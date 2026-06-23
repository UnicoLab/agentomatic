"""Optimization mixin — compile strategy and fit with an optimizer.

``compile()`` stores the optimization strategy (dataset, metrics,
optimizer). ``fit()`` executes the optimizer and stores results
in ``compiled_config``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from typing import Self

    from ..types import AgentDataset, Metric, Optimizer


class OptimizationMixin:
    """Mixin for agent optimization workflows.

    Provides a two-phase API:
    1. ``compile()`` — register the optimization strategy.
    2. ``fit()`` — execute the optimizer and apply results.

    Attributes:
        compiled_config: Configuration produced by the optimizer.
        compiled_metadata: Metadata about the compilation
            (dataset name, metrics, optimizer class).

    Example::

        agent.compile(dataset, metrics=[accuracy], optimizer=opt)
        agent.fit(dataset)
    """

    compiled_config: dict[str, Any]
    compiled_metadata: dict[str, Any]

    def compile(
        self,
        dataset: AgentDataset,
        metrics: Sequence[Metric],
        optimizer: Optimizer | None = None,
    ) -> Self:
        """Store the optimization strategy.

        Does not run the optimizer — use ``fit()`` for that.

        Args:
            dataset: Training dataset.
            metrics: Metrics used to evaluate the agent.
            optimizer: Optional optimizer instance.

        Returns:
            Self for method chaining.
        """
        if not hasattr(self, "compiled_config"):
            self.compiled_config = {}
        if not hasattr(self, "compiled_metadata"):
            self.compiled_metadata = {}

        self.compiled_metadata = {
            "dataset_name": getattr(dataset, "name", "unknown"),
            "metric_names": [m.name for m in metrics],
            "optimizer": (type(optimizer).__name__ if optimizer is not None else None),
        }

        # Store references for fit()
        self._compile_dataset = dataset
        self._compile_metrics = list(metrics)
        self._compile_optimizer = optimizer

        logger.info(
            "Compiled agent with metrics={}, optimizer={}",
            self.compiled_metadata["metric_names"],
            self.compiled_metadata["optimizer"],
        )

        return self  # type: ignore[return-value]

    def fit(
        self,
        dataset: AgentDataset | None = None,
    ) -> Self:
        """Run the optimizer and apply results.

        Uses the dataset from ``compile()`` if none is provided.
        Stores optimizer output in ``compiled_config`` and
        invalidates the cached graph.

        Args:
            dataset: Optional override dataset. Falls back to the
                dataset provided in ``compile()``.

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If ``compile()`` was not called first or
                no optimizer was set.
        """
        optimizer: Optimizer | None = getattr(self, "_compile_optimizer", None)
        if optimizer is None:
            raise RuntimeError("No optimizer set. Call compile(optimizer=...) first.")

        resolved_dataset: AgentDataset | None = dataset or getattr(self, "_compile_dataset", None)
        if resolved_dataset is None:
            raise RuntimeError("No dataset available. Pass a dataset to fit() or compile().")

        metrics: list[Metric] = getattr(self, "_compile_metrics", [])

        logger.info("Running optimizer: {}", type(optimizer).__name__)
        config = optimizer.optimize(self, resolved_dataset, metrics)

        if not hasattr(self, "compiled_config"):
            self.compiled_config = {}
        self.compiled_config.update(config)

        logger.info(
            "Optimization complete. Config keys: {}",
            list(self.compiled_config.keys()),
        )

        # Invalidate cached graph so it rebuilds with new config
        if hasattr(self, "invalidate_graph"):
            self.invalidate_graph()  # type: ignore[attr-defined]

        return self  # type: ignore[return-value]
