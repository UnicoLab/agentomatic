"""Pydantic models for pipeline configuration and results.

Defines the data structures for pipeline steps, configuration,
and execution results.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StepType(StrEnum):
    """Discriminator for step types."""

    AGENT = "agent"
    ENDPOINT = "endpoint"
    PLUGIN = "plugin"
    INGESTION = "ingestion"
    TRANSFORM = "transform"
    CONDITION = "condition"
    PARALLEL = "parallel"
    MAP = "map"
    LOOP = "loop"
    SUB_PIPELINE = "sub_pipeline"


class ErrorPolicy(StrEnum):
    """How to handle step failures."""

    FAIL = "fail"
    SKIP = "skip"
    RETRY = "retry"
    FALLBACK = "fallback"


class ParallelStrategy(StrEnum):
    """How to aggregate parallel step results."""

    ALL = "all"
    FIRST = "first"
    MAJORITY = "majority"


class StepStatus(StrEnum):
    """Execution status of a step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class PipelineStatus(StrEnum):
    """Execution status of a pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Retry configuration for a step."""

    max_attempts: int = Field(3, ge=1, le=10)
    backoff: Literal["fixed", "linear", "exponential"] = "exponential"
    base_delay: float = Field(1.0, ge=0.1, le=60.0)


# ---------------------------------------------------------------------------
# Step configurations
# ---------------------------------------------------------------------------


class InputMapping(BaseModel):
    """Maps pipeline context values to step inputs.

    Keys are the target field names.  Values are ``$`` expressions
    (e.g. ``$.input.query``) or literal values.
    """

    mappings: dict[str, Any] = Field(default_factory=dict)

    def __bool__(self) -> bool:  # noqa: D105
        return bool(self.mappings)

    def __getitem__(self, key: str) -> Any:  # noqa: D105
        return self.mappings[key]

    def items(self):  # noqa: ANN201, D102
        return self.mappings.items()


class OutputMapping(BaseModel):
    """Maps step outputs to pipeline context.

    Keys are the target context field names.  Values are ``$`` expressions
    or literal values.
    """

    mappings: dict[str, Any] = Field(default_factory=dict)

    def __bool__(self) -> bool:  # noqa: D105
        return bool(self.mappings)

    def __getitem__(self, key: str) -> Any:  # noqa: D105
        return self.mappings[key]

    def items(self):  # noqa: ANN201, D102
        return self.mappings.items()


class AgentStepConfig(BaseModel):
    """Configuration for an agent invocation step."""

    step_type: Literal[StepType.AGENT] = StepType.AGENT
    name: str
    agent: str  # Agent name in registry
    input: InputMapping = Field(default_factory=InputMapping)
    output: OutputMapping = Field(default_factory=OutputMapping)
    condition: str | None = None
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    fallback_agent: str | None = None
    retry: RetryConfig | None = None
    timeout: float = Field(30.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    rollback: str | None = Field(
        default=None,
        description="Python compensation code run (reverse order) if the pipeline "
        "rolls back. Has `ctx` and `output` (this step's output) in scope.",
    )


class PluginStepConfig(BaseModel):
    """Configuration for an ML-plugin invocation step.

    Calls a registered plugin's ``predict`` and stores its output in the
    pipeline context so downstream steps can consume the prediction.
    """

    step_type: Literal[StepType.PLUGIN] = StepType.PLUGIN
    name: str
    plugin: str  # Plugin name in the plugin registry
    input: InputMapping = Field(default_factory=InputMapping)
    output: OutputMapping = Field(default_factory=OutputMapping)
    condition: str | None = None
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    retry: RetryConfig | None = None
    timeout: float = Field(30.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    rollback: str | None = None


class EndpointStepConfig(BaseModel):
    """Configuration for a custom-endpoint invocation step.

    Calls a registered custom endpoint (which typically fetches data from
    one or more deployed model services) and stores its output in the
    pipeline context so downstream agents can consume it.
    """

    step_type: Literal[StepType.ENDPOINT] = StepType.ENDPOINT
    name: str
    endpoint: str  # Endpoint name in the endpoint registry
    input: InputMapping = Field(default_factory=InputMapping)
    output: OutputMapping = Field(default_factory=OutputMapping)
    upstreams: list[str] | None = None
    condition: str | None = None
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    retry: RetryConfig | None = None
    timeout: float = Field(30.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    rollback: str | None = None


class IngestionStepConfig(BaseModel):
    """Configuration for an ingestor invocation step.

    Runs a registered ingestor (which packages the user's document-ingestion
    code) and stores its :class:`~agentomatic.ingestion.IngestionResult` in the
    pipeline context, so an ingest → index → answer flow can be expressed as a
    single pipeline.
    """

    step_type: Literal[StepType.INGESTION] = StepType.INGESTION
    name: str
    ingestor: str  # Ingestor name in the ingestion registry
    input: InputMapping = Field(default_factory=InputMapping)
    output: OutputMapping = Field(default_factory=OutputMapping)
    condition: str | None = None
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    retry: RetryConfig | None = None
    timeout: float = Field(300.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    rollback: str | None = None


class TransformStepConfig(BaseModel):
    """Configuration for a data-transformation step.

    ``code`` is a Python code block executed with ``ctx`` in scope.
    Must end with a ``return`` statement producing a dict.
    """

    step_type: Literal[StepType.TRANSFORM] = StepType.TRANSFORM
    name: str
    code: str
    condition: str | None = None
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    timeout: float = Field(10.0, gt=0)


class ParallelStepConfig(BaseModel):
    """Configuration for parallel (fan-out) execution."""

    step_type: Literal[StepType.PARALLEL] = StepType.PARALLEL
    name: str
    steps: list[AgentStepConfig]
    strategy: ParallelStrategy = ParallelStrategy.ALL
    max_concurrency: int = Field(5, ge=1, le=20)
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    timeout: float = Field(60.0, gt=0)


class MapStepConfig(BaseModel):
    """Configuration for dynamic map/fan-out over a runtime list of items.

    Unlike :class:`ParallelStepConfig`, which fans out a fixed set of
    heterogeneous sub-steps, ``MapStepConfig`` runs *one* agent (``agent``)
    against every element of a list resolved from the pipeline context
    (``items``, e.g. ``$.input.scopes``). Each element is exposed to the
    agent under ``item_key`` (default ``"item"``) and its index under
    ``index_key`` (default ``"index"``).

    Results are collected into a keyed fan-in structure so no keys are
    silently overwritten::

        {
            "items": [<result_0>, <result_1>, ...],
            "by_key": {"0": <result_0>, "1": <result_1>, ...},
        }
    """

    step_type: Literal[StepType.MAP] = StepType.MAP
    name: str
    agent: str = Field(..., description="Agent name to invoke per item.")
    items: str = Field(
        ...,
        description="Expression resolving to a list (e.g. ``$.input.scopes``).",
    )
    item_key: str = Field(
        default="item",
        description="Key under which each item is exposed to the agent.",
    )
    index_key: str = Field(
        default="index",
        description="Key under which each item index is exposed to the agent.",
    )
    input: InputMapping = Field(
        default_factory=InputMapping,
        description="Additional mappings merged into each per-item agent state.",
    )
    output: OutputMapping = Field(default_factory=OutputMapping)
    max_concurrency: int = Field(4, ge=1, le=64)
    strategy: ParallelStrategy = ParallelStrategy.ALL
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    retry: RetryConfig | None = Field(
        default=None,
        description="Optional per-item retry configuration.",
    )
    timeout: float = Field(120.0, gt=0)
    item_timeout: float = Field(60.0, gt=0)
    fallback_agent: str | None = None
    condition: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    rollback: str | None = None


class LoopStepConfig(BaseModel):
    """Configuration for iterative (loop) execution."""

    step_type: Literal[StepType.LOOP] = StepType.LOOP
    name: str
    step: AgentStepConfig
    max_iterations: int = Field(10, ge=1, le=100)
    until: str | None = None  # Python expression evaluated against ctx
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    timeout: float = Field(120.0, gt=0)


class SubPipelineStepConfig(BaseModel):
    """Configuration for nested pipeline execution."""

    step_type: Literal[StepType.SUB_PIPELINE] = StepType.SUB_PIPELINE
    name: str
    pipeline: str  # Pipeline name to invoke
    input: InputMapping = Field(default_factory=InputMapping)
    output: OutputMapping = Field(default_factory=OutputMapping)
    condition: str | None = None
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    timeout: float = Field(120.0, gt=0)


# Discriminated union of all step types
StepConfigUnion = (
    AgentStepConfig
    | PluginStepConfig
    | EndpointStepConfig
    | IngestionStepConfig
    | TransformStepConfig
    | ParallelStepConfig
    | MapStepConfig
    | LoopStepConfig
    | SubPipelineStepConfig
)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    name: str = Field(..., min_length=1, max_length=128)
    description: str = ""
    version: str = "1.0.0"
    steps: list[StepConfigUnion] = Field(..., min_length=1)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    on_error: Literal["fail_fast", "continue", "rollback"] = "fail_fast"
    strict_schema: bool = Field(
        default=False,
        description="When true, input/output-schema violations raise instead of "
        "logging a warning.",
    )
    timeout: float = Field(300.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def step_names(self) -> list[str]:
        """Return ordered list of step names."""
        return [s.name for s in self.steps]

    def get_step(self, name: str) -> StepConfigUnion | None:
        """Get step config by name."""
        for s in self.steps:
            if s.name == name:
                return s
        return None

    def get_agent_names(self) -> set[str]:
        """Return set of all referenced agent names."""
        agents: set[str] = set()
        for step in self.steps:
            if isinstance(step, AgentStepConfig):
                agents.add(step.agent)
                if step.fallback_agent:
                    agents.add(step.fallback_agent)
            elif isinstance(step, ParallelStepConfig):
                for sub in step.steps:
                    agents.add(sub.agent)
                    if sub.fallback_agent:
                        agents.add(sub.fallback_agent)
            elif isinstance(step, MapStepConfig):
                agents.add(step.agent)
                if step.fallback_agent:
                    agents.add(step.fallback_agent)
            elif isinstance(step, LoopStepConfig):
                agents.add(step.step.agent)
        return agents

    def get_plugin_names(self) -> set[str]:
        """Return the set of all referenced ML-plugin names."""
        return {s.plugin for s in self.steps if isinstance(s, PluginStepConfig)}

    def get_endpoint_names(self) -> set[str]:
        """Return the set of all referenced custom-endpoint names."""
        return {s.endpoint for s in self.steps if isinstance(s, EndpointStepConfig)}

    def get_ingestor_names(self) -> set[str]:
        """Return the set of all referenced ingestor names."""
        return {s.ingestor for s in self.steps if isinstance(s, IngestionStepConfig)}


# ---------------------------------------------------------------------------
# Execution results
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """Result of a single step execution."""

    name: str
    status: StepStatus = StepStatus.PENDING
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    agent_used: str | None = None
    retries: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # For parallel steps — individual sub-results
    sub_results: list[StepResult] | None = None

    # For loop steps — iteration results
    iterations: list[StepResult] | None = None


class PipelineResult(BaseModel):
    """Complete pipeline execution result."""

    pipeline_name: str
    status: PipelineStatus = PipelineStatus.PENDING
    output: dict[str, Any] = Field(default_factory=dict)
    steps: dict[str, StepResult] = Field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None
    started_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Whether the pipeline completed successfully."""
        return self.status == PipelineStatus.SUCCESS

    @property
    def step_count(self) -> int:
        """Number of steps executed."""
        return len(self.steps)
