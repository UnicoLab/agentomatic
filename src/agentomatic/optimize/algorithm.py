"""Unified optimisation algorithm surface (Agent Lightning–inspired).

Collapses the dual ``PromptOptimizer`` / ``PromptFitter`` entry points
behind a single ABC while keeping existing Keras ``compile`` / ``fit``
APIs stable via :class:`FitterAlgorithm`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from agentomatic.optimize.config import PromptFitResult
from agentomatic.optimize.dataset import Dataset
from agentomatic.optimize.metrics import BaseMetric, CompositeMetric
from agentomatic.optimize.resources import ResourceRegistry
from agentomatic.optimize.rollout import RolloutTraceStore

if TYPE_CHECKING:
    from agentomatic.optimize.fitter import PromptFitter


class OptimizationAlgorithm(ABC):
    """Strategy that improves an agent configuration from datasets.

    Subclasses implement :meth:`run` (async). The trainer / CLI / Keras
    bridge should prefer this ABC over calling fitter internals directly.
    """

    name: str = "algorithm"

    def __init__(self) -> None:
        self.resources = ResourceRegistry()
        self.trace_store = RolloutTraceStore()

    @abstractmethod
    async def run(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        *,
        metric: CompositeMetric | BaseMetric | None = None,
        test_dataset: Dataset | None = None,
        **kwargs: Any,
    ) -> PromptFitResult:
        """Execute the algorithm and return a fit result."""


class FitterAlgorithm(OptimizationAlgorithm):
    """Adapter that runs :class:`PromptFitter` behind the algorithm ABC."""

    name = "fitter"

    def __init__(self, fitter: PromptFitter) -> None:
        super().__init__()
        self.fitter = fitter
        # Share registries so fitter eval can publish resources / traces.
        self.fitter._resource_registry = self.resources  # noqa: SLF001
        self.fitter._trace_store = self.trace_store  # noqa: SLF001

    async def run(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        *,
        metric: CompositeMetric | BaseMetric | None = None,
        test_dataset: Dataset | None = None,
        **kwargs: Any,
    ) -> PromptFitResult:
        """Delegate to :meth:`PromptFitter.fit`."""
        if metric is None:
            raise ValueError("FitterAlgorithm.run() requires metric=")
        val = val_dataset if val_dataset is not None else train_dataset
        return await self.fitter.fit(
            train_dataset,
            val,
            metric,
            testset=test_dataset,
            **kwargs,
        )


def as_algorithm(fitter: PromptFitter) -> FitterAlgorithm:
    """Wrap a :class:`PromptFitter` as an :class:`OptimizationAlgorithm`."""
    return FitterAlgorithm(fitter)
