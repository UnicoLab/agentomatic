"""Fluent builder API for agentomatic pipelines.

Provides the :class:`Pipeline` builder which lets users compose multi-step
agent pipelines with a concise, chainable method interface::

    from agentomatic.pipelines import Pipeline

    cfg = Pipeline("qa").step("researcher").step("writer").to_config()

The builder produces validated :class:`PipelineConfig` instances that can
be handed off to the pipeline engine for execution.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any, Literal

from .models import (
    AgentStepConfig,
    ErrorPolicy,
    InputMapping,
    LoopStepConfig,
    OutputMapping,
    ParallelStepConfig,
    ParallelStrategy,
    PipelineConfig,
    RetryConfig,
    StepConfigUnion,
    TransformStepConfig,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_input_mapping(raw: dict[str, str] | InputMapping | None) -> InputMapping:
    """Convert a plain dict to an :class:`InputMapping`.

    Args:
        raw: Either an ``InputMapping``, a plain ``dict``, or ``None``.

    Returns:
        An ``InputMapping`` instance (possibly empty).
    """
    if raw is None:
        return InputMapping()
    if isinstance(raw, InputMapping):
        return raw
    return InputMapping(mappings=raw)


def _to_output_mapping(raw: dict[str, str] | OutputMapping | None) -> OutputMapping:
    """Convert a plain dict to an :class:`OutputMapping`.

    Args:
        raw: Either an ``OutputMapping``, a plain ``dict``, or ``None``.

    Returns:
        An ``OutputMapping`` instance (possibly empty).
    """
    if raw is None:
        return OutputMapping()
    if isinstance(raw, OutputMapping):
        return raw
    return OutputMapping(mappings=raw)


def _to_retry_config(raw: dict[str, Any] | RetryConfig | None) -> RetryConfig | None:
    """Convert a plain dict to a :class:`RetryConfig`.

    Args:
        raw: Either a ``RetryConfig``, a plain ``dict``, or ``None``.

    Returns:
        A ``RetryConfig`` instance or ``None``.
    """
    if raw is None:
        return None
    if isinstance(raw, RetryConfig):
        return raw
    return RetryConfig(**raw)


def _to_error_policy(raw: str | ErrorPolicy | None) -> ErrorPolicy:
    """Convert a string to an :class:`ErrorPolicy`.

    Args:
        raw: A string matching an ``ErrorPolicy`` variant, an
            ``ErrorPolicy`` instance, or ``None`` (defaults to ``FAIL``).

    Returns:
        An ``ErrorPolicy`` enum member.
    """
    if raw is None:
        return ErrorPolicy.FAIL
    if isinstance(raw, ErrorPolicy):
        return raw
    return ErrorPolicy(raw)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


class Pipeline:
    """Fluent builder for constructing :class:`PipelineConfig` instances.

    The builder accumulates pipeline metadata and an ordered list of step
    configurations.  Call :meth:`to_config` to materialise a validated
    ``PipelineConfig``.

    Example:
        >>> cfg = Pipeline("qa").step("researcher").step("writer").to_config()
        >>> cfg.name
        'qa'
        >>> len(cfg.steps)
        2
    """

    def __init__(self, name: str) -> None:
        """Initialise a new pipeline builder.

        Args:
            name: A unique pipeline identifier (1-128 characters).
        """
        self._name: str = name
        self._description: str = ""
        self._version: str = "1.0.0"
        self._steps: list[StepConfigUnion] = []
        self._input_schema: dict[str, Any] | None = None
        self._output_schema: dict[str, Any] | None = None
        self._defaults: dict[str, Any] = {}
        self._on_error: Literal["fail_fast", "continue", "rollback"] = "fail_fast"
        self._timeout: float = 300.0
        self._metadata: dict[str, Any] = {}

    # -- metadata setters ---------------------------------------------------

    def description(self, text: str) -> Pipeline:
        """Set the pipeline description.

        Args:
            text: Human-readable description of the pipeline's purpose.

        Returns:
            ``self`` for method chaining.
        """
        self._description = text
        return self

    def version(self, ver: str) -> Pipeline:
        """Set the pipeline version.

        Args:
            ver: Semantic version string (e.g. ``"1.0.0"``).

        Returns:
            ``self`` for method chaining.
        """
        self._version = ver
        return self

    def input_schema(self, **fields: Any) -> Pipeline:
        """Declare expected input fields and their types / defaults.

        Each keyword argument defines a field.  Pass a bare type for a
        required field or a ``(type, default)`` tuple for an optional one::

            Pipeline("p").input_schema(query=str, max_depth=(int, 3))

        Args:
            **fields: Keyword arguments defining the input schema.

        Returns:
            ``self`` for method chaining.
        """
        schema: dict[str, Any] = {}
        for field_name, spec in fields.items():
            if isinstance(spec, tuple):
                type_hint, default = spec
                schema[field_name] = {
                    "type": type_hint.__name__ if isinstance(type_hint, type) else str(type_hint),
                    "default": default,
                }
            elif isinstance(spec, type):
                schema[field_name] = {"type": spec.__name__}
            else:
                schema[field_name] = {"type": str(spec)}
        self._input_schema = schema
        return self

    def output_schema(self, **fields: Any) -> Pipeline:
        """Declare expected output fields.

        Works identically to :meth:`input_schema`.

        Args:
            **fields: Keyword arguments defining the output schema.

        Returns:
            ``self`` for method chaining.
        """
        schema: dict[str, Any] = {}
        for field_name, spec in fields.items():
            if isinstance(spec, tuple):
                type_hint, default = spec
                schema[field_name] = {
                    "type": type_hint.__name__ if isinstance(type_hint, type) else str(type_hint),
                    "default": default,
                }
            elif isinstance(spec, type):
                schema[field_name] = {"type": spec.__name__}
            else:
                schema[field_name] = {"type": str(spec)}
        self._output_schema = schema
        return self

    def defaults(self, **kwargs: Any) -> Pipeline:
        """Set default values for the pipeline context.

        Args:
            **kwargs: Arbitrary key-value pairs to pre-populate context.

        Returns:
            ``self`` for method chaining.
        """
        self._defaults.update(kwargs)
        return self

    def on_error(
        self,
        policy: Literal["fail_fast", "continue", "rollback"],
    ) -> Pipeline:
        """Set the pipeline-level error policy.

        Args:
            policy: One of ``"fail_fast"``, ``"continue"``, or
                ``"rollback"``.

        Returns:
            ``self`` for method chaining.
        """
        self._on_error = policy
        return self

    def timeout(self, seconds: float) -> Pipeline:
        """Set the pipeline-level timeout in seconds.

        Args:
            seconds: Maximum execution time for the whole pipeline.

        Returns:
            ``self`` for method chaining.
        """
        self._timeout = seconds
        return self

    def meta(self, **kwargs: Any) -> Pipeline:
        """Attach arbitrary metadata to the pipeline.

        Args:
            **kwargs: Key-value pairs stored in ``PipelineConfig.metadata``.

        Returns:
            ``self`` for method chaining.
        """
        self._metadata.update(kwargs)
        return self

    # -- step builders ------------------------------------------------------

    def step(
        self,
        name: str,
        *,
        agent: str | None = None,
        input: dict[str, str] | InputMapping | None = None,  # noqa: A002
        output: dict[str, str] | OutputMapping | None = None,
        condition: str | None = None,
        on_error: str | ErrorPolicy | None = None,
        fallback_agent: str | None = None,
        retry: dict[str, Any] | RetryConfig | None = None,
        timeout: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> Pipeline:
        """Add an agent step to the pipeline.

        When *agent* is omitted it defaults to *name*, enabling the common
        pattern ``Pipeline("p").step("researcher")``.

        Args:
            name: Step identifier (unique within the pipeline).
            agent: Agent name in the registry. Defaults to *name*.
            input: Input mapping (dict or ``InputMapping``).
            output: Output mapping (dict or ``OutputMapping``).
            condition: Python expression; step runs only when truthy.
            on_error: Error policy string or ``ErrorPolicy``.
            fallback_agent: Agent to use when the primary one fails.
            retry: Retry configuration (dict or ``RetryConfig``).
            timeout: Per-step timeout in seconds.
            metadata: Arbitrary metadata attached to this step.

        Returns:
            ``self`` for method chaining.
        """
        step_cfg = AgentStepConfig(
            name=name,
            agent=agent or name,
            input=_to_input_mapping(input),
            output=_to_output_mapping(output),
            condition=condition,
            on_error=_to_error_policy(on_error),
            fallback_agent=fallback_agent,
            retry=_to_retry_config(retry),
            timeout=timeout,
            metadata=metadata or {},
        )
        self._steps.append(step_cfg)
        return self

    def transform(
        self,
        name: str,
        code: str,
        *,
        condition: str | None = None,
        on_error: str | ErrorPolicy | None = None,
        timeout: float = 10.0,
    ) -> Pipeline:
        """Add a data-transformation step.

        The *code* block is executed with ``ctx`` in scope and must end
        with a ``return`` statement producing a ``dict``.

        Args:
            name: Step identifier.
            code: Python code string.  Leading indentation is
                automatically stripped via :func:`textwrap.dedent`.
            condition: Python expression; step runs only when truthy.
            on_error: Error policy string or ``ErrorPolicy``.
            timeout: Per-step timeout in seconds.

        Returns:
            ``self`` for method chaining.
        """
        step_cfg = TransformStepConfig(
            name=name,
            code=textwrap.dedent(code).strip(),
            condition=condition,
            on_error=_to_error_policy(on_error),
            timeout=timeout,
        )
        self._steps.append(step_cfg)
        return self

    def parallel(
        self,
        name: str,
        steps: list[AgentStepConfig],
        *,
        strategy: str | ParallelStrategy = ParallelStrategy.ALL,
        max_concurrency: int = 5,
        on_error: str | ErrorPolicy | None = None,
        timeout: float = 60.0,
    ) -> Pipeline:
        """Add a parallel fan-out step.

        Use :meth:`agent` (static) to create the sub-step configs::

            Pipeline("p").parallel("research", [
                Pipeline.agent("web"),
                Pipeline.agent("kb", on_error="skip"),
            ])

        Args:
            name: Step identifier.
            steps: List of :class:`AgentStepConfig` to execute in
                parallel.
            strategy: Aggregation strategy (``"all"``, ``"first"``, or
                ``"majority"``).
            max_concurrency: Max concurrent sub-steps.
            on_error: Error policy string or ``ErrorPolicy``.
            timeout: Per-step timeout in seconds.

        Returns:
            ``self`` for method chaining.
        """
        if isinstance(strategy, str):
            strategy = ParallelStrategy(strategy)

        step_cfg = ParallelStepConfig(
            name=name,
            steps=steps,
            strategy=strategy,
            max_concurrency=max_concurrency,
            on_error=_to_error_policy(on_error),
            timeout=timeout,
        )
        self._steps.append(step_cfg)
        return self

    def loop(
        self,
        name: str,
        *,
        agent: str | None = None,
        max_iterations: int = 10,
        until: str | None = None,
        on_error: str | ErrorPolicy | None = None,
        timeout: float = 120.0,
        input: dict[str, str] | InputMapping | None = None,  # noqa: A002
        output: dict[str, str] | OutputMapping | None = None,
        retry: dict[str, Any] | RetryConfig | None = None,
        step_timeout: float = 30.0,
    ) -> Pipeline:
        """Add a loop step that iterates an agent until a condition is met.

        Args:
            name: Step identifier.
            agent: Agent name in the registry. Defaults to *name*.
            max_iterations: Maximum number of iterations.
            until: Python expression evaluated against ``ctx``; loop
                terminates when truthy.
            on_error: Error policy string or ``ErrorPolicy``.
            timeout: Overall timeout for the loop step.
            input: Input mapping for the inner agent step.
            output: Output mapping for the inner agent step.
            retry: Retry configuration for the inner agent step.
            step_timeout: Timeout for each individual agent call.

        Returns:
            ``self`` for method chaining.
        """
        inner_step = AgentStepConfig(
            name=f"{name}_agent",
            agent=agent or name,
            input=_to_input_mapping(input),
            output=_to_output_mapping(output),
            retry=_to_retry_config(retry),
            timeout=step_timeout,
        )
        step_cfg = LoopStepConfig(
            name=name,
            step=inner_step,
            max_iterations=max_iterations,
            until=until,
            on_error=_to_error_policy(on_error),
            timeout=timeout,
        )
        self._steps.append(step_cfg)
        return self

    # -- static helpers -----------------------------------------------------

    @staticmethod
    def agent(
        name: str,
        *,
        agent: str | None = None,
        input: dict[str, str] | InputMapping | None = None,  # noqa: A002
        output: dict[str, str] | OutputMapping | None = None,
        condition: str | None = None,
        on_error: str | ErrorPolicy | None = None,
        fallback_agent: str | None = None,
        retry: dict[str, Any] | RetryConfig | None = None,
        timeout: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentStepConfig:
        """Create a standalone :class:`AgentStepConfig`.

        Intended for use inside :meth:`parallel`::

            Pipeline.agent("web_researcher")
            Pipeline.agent("knowledge_base", on_error="skip")

        Args:
            name: Step / agent identifier.
            agent: Agent name in the registry. Defaults to *name*.
            input: Input mapping (dict or ``InputMapping``).
            output: Output mapping (dict or ``OutputMapping``).
            condition: Python expression; step runs only when truthy.
            on_error: Error policy string or ``ErrorPolicy``.
            fallback_agent: Fallback agent name.
            retry: Retry configuration (dict or ``RetryConfig``).
            timeout: Per-step timeout in seconds.
            metadata: Arbitrary metadata.

        Returns:
            A configured ``AgentStepConfig`` instance.
        """
        return AgentStepConfig(
            name=name,
            agent=agent or name,
            input=_to_input_mapping(input),
            output=_to_output_mapping(output),
            condition=condition,
            on_error=_to_error_policy(on_error),
            fallback_agent=fallback_agent,
            retry=_to_retry_config(retry),
            timeout=timeout,
            metadata=metadata or {},
        )

    # -- serialisation ------------------------------------------------------

    def to_config(self) -> PipelineConfig:
        """Build and validate the :class:`PipelineConfig`.

        Returns:
            A fully validated ``PipelineConfig`` ready for execution.

        Raises:
            pydantic.ValidationError: If the accumulated configuration
                is invalid (e.g. no steps, name too long).
        """
        return PipelineConfig(
            name=self._name,
            description=self._description,
            version=self._version,
            steps=self._steps,
            input_schema=self._input_schema,
            output_schema=self._output_schema,
            defaults=self._defaults,
            on_error=self._on_error,
            timeout=self._timeout,
            metadata=self._metadata,
        )

    def to_yaml(self) -> str:
        """Serialise the pipeline configuration to a YAML string.

        Requires *PyYAML* (``pip install pyyaml``).

        Returns:
            A YAML-formatted string representing the pipeline config.

        Raises:
            ImportError: If *PyYAML* is not installed.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML serialisation. Install it with: pip install pyyaml"
            ) from exc

        config = self.to_config()
        data = config.model_dump(mode="json", exclude_none=True)
        return yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    # -- dunder helpers -----------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        step_names = [s.name for s in self._steps]
        return f"Pipeline(name={self._name!r}, steps={step_names!r})"

    def __len__(self) -> int:
        """Return the number of steps added so far."""
        return len(self._steps)
