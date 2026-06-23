"""Agent mixins — composable building blocks for class-owned agents.

Re-exports all mixins for convenient single-import usage::

    from agentomatic.agents.mixins import (
        GraphExecutionMixin,
        DatasetMixin,
        EvaluationMixin,
        OptimizationMixin,
        SerializationMixin,
        ObservabilityMixin,
    )
"""

from __future__ import annotations

from .dataset import DatasetMixin
from .evaluation import EvaluationMixin
from .graph_execution import GraphExecutionMixin
from .observability import ObservabilityMixin
from .optimization import OptimizationMixin
from .serialization import SerializationMixin

__all__ = [
    "DatasetMixin",
    "EvaluationMixin",
    "GraphExecutionMixin",
    "ObservabilityMixin",
    "OptimizationMixin",
    "SerializationMixin",
]
