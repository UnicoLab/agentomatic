"""Agentomatic Pipelines — composition DSL for multi-agent workflows.

Three ways to define pipelines:

1. **YAML** — declarative, minimal config::

    # pipeline.yaml
    name: qa_pipeline
    steps:
      - agent: researcher
      - agent: writer

2. **Builder** — fluent Python API::

    from agentomatic.pipelines import Pipeline

    pipeline = Pipeline("qa").step("researcher").step("writer").to_config()

3. **Flow** — decorator-based Python class::

    from agentomatic.pipelines import Flow, start, listen

    class QAFlow(Flow):
        @start()
        async def research(self, data):
            return await self.agent("researcher").run(data)

        @listen(research)
        async def write(self, research_output):
            return await self.agent("writer").run(research_output)
"""

from __future__ import annotations

from .builder import Pipeline
from .context import PipelineContext
from .engine import PipelineEngine
from .loader import PipelineLoader
from .models import (
    AgentStepConfig,
    ErrorPolicy,
    LoopStepConfig,
    OutputMapping,
    ParallelStepConfig,
    PipelineConfig,
    PipelineResult,
    RetryConfig,
    StepConfigUnion,
    StepResult,
    SubPipelineStepConfig,
    TransformStepConfig,
)
from .router import create_pipeline_router

__all__ = [
    "AgentStepConfig",
    "ErrorPolicy",
    "LoopStepConfig",
    "OutputMapping",
    "ParallelStepConfig",
    "Pipeline",
    "PipelineConfig",
    "PipelineContext",
    "PipelineEngine",
    "PipelineLoader",
    "PipelineResult",
    "RetryConfig",
    "StepConfigUnion",
    "StepResult",
    "SubPipelineStepConfig",
    "TransformStepConfig",
    "create_pipeline_router",
]

# Lazy imports for Flow decorators (avoid import errors if
# the flow module has issues)
try:
    from .flow import Flow, listen, router, start

    __all__ += ["Flow", "listen", "router", "start"]
except ImportError:
    pass
