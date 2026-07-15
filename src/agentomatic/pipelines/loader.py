"""YAML pipeline loader and auto-discovery.

Loads pipeline configurations from YAML files, dictionaries, or raw
YAML strings and validates them against the Pydantic models defined in
:mod:`agentomatic.pipelines.models`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from agentomatic.pipelines.models import (
    AgentStepConfig,
    EndpointStepConfig,
    ErrorPolicy,
    IngestionStepConfig,
    InputMapping,
    LoopStepConfig,
    MapStepConfig,
    OutputMapping,
    ParallelStepConfig,
    ParallelStrategy,
    PipelineConfig,
    PluginStepConfig,
    RetryConfig,
    StepConfigUnion,
    SubPipelineStepConfig,
    TransformStepConfig,
)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# Keys used to discriminate the step type from a raw dict.
_STEP_TYPE_KEYS = (
    "agent",
    "plugin",
    "endpoint",
    "ingestion",
    "parallel",
    "map",
    "transform",
    "loop",
    "sub_pipeline",
)

# Recognised YAML file suffixes.
_YAML_SUFFIXES = {".yaml", ".yml"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_yaml() -> None:
    """Raise a clear error if PyYAML is not installed."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required for pipeline YAML loading. Install it with: pip install pyyaml"
        )


def _coerce_input_mapping(raw: Any) -> InputMapping:
    """Convert a raw value to an ``InputMapping``.

    Accepts:
    - ``None`` → empty mapping
    - ``dict``  → ``InputMapping(mappings=raw)``
    - ``InputMapping`` → pass-through

    Args:
        raw: The raw input mapping value from YAML.

    Returns:
        A validated ``InputMapping`` instance.

    Raises:
        TypeError: If *raw* is not a supported type.
    """
    if raw is None:
        return InputMapping()
    if isinstance(raw, InputMapping):
        return raw
    if isinstance(raw, dict):
        return InputMapping(mappings=raw)
    raise TypeError(f"Expected dict or InputMapping for input, got {type(raw).__name__}")


def _coerce_output_mapping(raw: Any) -> OutputMapping:
    """Convert a raw value to an ``OutputMapping``.

    Accepts:
    - ``None`` → empty mapping
    - ``dict``  → ``OutputMapping(mappings=raw)``
    - ``OutputMapping`` → pass-through

    Args:
        raw: The raw output mapping value from YAML.

    Returns:
        A validated ``OutputMapping`` instance.

    Raises:
        TypeError: If *raw* is not a supported type.
    """
    if raw is None:
        return OutputMapping()
    if isinstance(raw, OutputMapping):
        return raw
    if isinstance(raw, dict):
        return OutputMapping(mappings=raw)
    raise TypeError(f"Expected dict or OutputMapping for output, got {type(raw).__name__}")


def _parse_retry(raw: Any) -> RetryConfig | None:
    """Parse an optional retry configuration block.

    Args:
        raw: ``None``, a ``RetryConfig``, or a raw ``dict``.

    Returns:
        A ``RetryConfig`` or ``None``.
    """
    if raw is None:
        return None
    if isinstance(raw, RetryConfig):
        return raw
    if isinstance(raw, dict):
        return RetryConfig(**raw)
    raise TypeError(f"Expected dict or RetryConfig for retry, got {type(raw).__name__}")


def _parse_agent_step(data: dict[str, Any]) -> AgentStepConfig:
    """Build an ``AgentStepConfig`` from a raw dict.

    Handles the shorthand form ``{agent: "name"}`` where the step *name*
    defaults to the agent identifier.

    Args:
        data: Raw step dictionary containing at least the ``agent`` key.

    Returns:
        A validated ``AgentStepConfig``.
    """
    agent: str = data["agent"]
    name: str = data.get("name", agent)

    return AgentStepConfig(
        name=name,
        agent=agent,
        input=_coerce_input_mapping(data.get("input")),
        output=_coerce_output_mapping(data.get("output")),
        condition=data.get("condition"),
        on_error=ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL,
        fallback_agent=data.get("fallback_agent"),
        retry=_parse_retry(data.get("retry")),
        timeout=data.get("timeout", 30.0),
        metadata=data.get("metadata", {}),
        rollback=data.get("rollback"),
    )


def _parse_plugin_step(data: dict[str, Any]) -> PluginStepConfig:
    """Build a ``PluginStepConfig`` from a raw dict.

    Handles the shorthand form ``{plugin: "name"}`` where the step *name*
    defaults to the plugin identifier.

    Args:
        data: Raw step dictionary containing at least the ``plugin`` key.

    Returns:
        A validated ``PluginStepConfig``.
    """
    plugin: str = data["plugin"]
    name: str = data.get("name", plugin)

    return PluginStepConfig(
        name=name,
        plugin=plugin,
        input=_coerce_input_mapping(data.get("input")),
        output=_coerce_output_mapping(data.get("output")),
        condition=data.get("condition"),
        on_error=ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL,
        retry=_parse_retry(data.get("retry")),
        timeout=data.get("timeout", 30.0),
        metadata=data.get("metadata", {}),
        rollback=data.get("rollback"),
    )


def _parse_endpoint_step(data: dict[str, Any]) -> EndpointStepConfig:
    """Build an ``EndpointStepConfig`` from a raw dict.

    Handles the shorthand form ``{endpoint: "name"}`` where the step *name*
    defaults to the endpoint identifier.

    Args:
        data: Raw step dictionary containing at least the ``endpoint`` key.

    Returns:
        A validated ``EndpointStepConfig``.
    """
    endpoint: str = data["endpoint"]
    name: str = data.get("name", endpoint)

    return EndpointStepConfig(
        name=name,
        endpoint=endpoint,
        input=_coerce_input_mapping(data.get("input")),
        output=_coerce_output_mapping(data.get("output")),
        upstreams=data.get("upstreams"),
        condition=data.get("condition"),
        on_error=ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL,
        retry=_parse_retry(data.get("retry")),
        timeout=data.get("timeout", 30.0),
        metadata=data.get("metadata", {}),
        rollback=data.get("rollback"),
    )


def _parse_ingestion_step(data: dict[str, Any]) -> IngestionStepConfig:
    """Build an ``IngestionStepConfig`` from a raw dict.

    Handles the shorthand form ``{ingestion: "name"}`` where the step *name*
    defaults to the ingestor identifier.

    Args:
        data: Raw step dictionary containing at least the ``ingestion`` key.

    Returns:
        A validated ``IngestionStepConfig``.
    """
    ingestor: str = data["ingestion"]
    name: str = data.get("name", ingestor)

    return IngestionStepConfig(
        name=name,
        ingestor=ingestor,
        input=_coerce_input_mapping(data.get("input")),
        output=_coerce_output_mapping(data.get("output")),
        condition=data.get("condition"),
        on_error=ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL,
        retry=_parse_retry(data.get("retry")),
        timeout=data.get("timeout", 300.0),
        metadata=data.get("metadata", {}),
        rollback=data.get("rollback"),
    )


def _parse_transform_step(data: dict[str, Any]) -> TransformStepConfig:
    """Build a ``TransformStepConfig`` from a raw dict.

    Args:
        data: Raw step dictionary containing the ``transform`` key
              with the code string.

    Returns:
        A validated ``TransformStepConfig``.

    Raises:
        ValueError: If no ``name`` is provided.
    """
    if "name" not in data:
        raise ValueError("Transform steps require an explicit 'name'.")

    return TransformStepConfig(
        name=data["name"],
        code=data["transform"],
        condition=data.get("condition"),
        on_error=ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL,
        timeout=data.get("timeout", 10.0),
    )


def _parse_parallel_step(data: dict[str, Any]) -> ParallelStepConfig:
    """Build a ``ParallelStepConfig`` from a raw dict.

    The ``parallel`` key must be a dict with a ``steps`` list (each
    element parsed as an agent step) and optional ``strategy`` /
    ``max_concurrency`` overrides.

    Args:
        data: Raw step dictionary containing the ``parallel`` key.

    Returns:
        A validated ``ParallelStepConfig``.

    Raises:
        ValueError: If no ``name`` is provided.
    """
    if "name" not in data:
        raise ValueError("Parallel steps require an explicit 'name'.")

    par = data["parallel"]
    if not isinstance(par, dict):
        raise TypeError("The 'parallel' key must be a dict with at least a 'steps' list.")

    sub_steps = [_parse_agent_step(s) for s in par.get("steps", [])]
    strategy_raw = par.get("strategy", "all")
    strategy = ParallelStrategy(strategy_raw)

    return ParallelStepConfig(
        name=data["name"],
        steps=sub_steps,
        strategy=strategy,
        max_concurrency=par.get("max_concurrency", 5),
        on_error=(ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL),
        timeout=data.get("timeout", 60.0),
    )


def _parse_map_step(data: dict[str, Any]) -> MapStepConfig:
    """Build a ``MapStepConfig`` from a raw dict.

    The ``map`` key must be a dict containing at least ``agent`` (the agent
    to run per item) and ``items`` (an expression resolving to a list). All
    other fields (``item_key``, ``max_concurrency``, ``retry``, …) are
    optional.

    Args:
        data: Raw step dictionary containing the ``map`` key.

    Returns:
        A validated ``MapStepConfig``.

    Raises:
        ValueError: If no ``name`` is provided or the map body is invalid.
        TypeError: If ``map`` is not a dict.
    """
    if "name" not in data:
        raise ValueError("Map steps require an explicit 'name'.")

    body = data["map"]
    if not isinstance(body, dict):
        raise TypeError("The 'map' key must be a dict with at least 'agent' and 'items'.")
    if "agent" not in body:
        raise ValueError("Map step body must contain an 'agent' key.")
    if "items" not in body:
        raise ValueError("Map step body must contain an 'items' expression.")

    strategy_raw = body.get("strategy", "all")
    strategy = ParallelStrategy(strategy_raw)

    return MapStepConfig(
        name=data["name"],
        agent=body["agent"],
        items=body["items"],
        item_key=body.get("item_key", "item"),
        index_key=body.get("index_key", "index"),
        input=_coerce_input_mapping(body.get("input")),
        output=_coerce_output_mapping(body.get("output")),
        max_concurrency=body.get("max_concurrency", 4),
        strategy=strategy,
        on_error=(ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL),
        retry=_parse_retry(body.get("retry")),
        timeout=data.get("timeout", 120.0),
        item_timeout=body.get("item_timeout", 60.0),
        fallback_agent=body.get("fallback_agent"),
        condition=data.get("condition"),
        metadata=data.get("metadata", {}),
        rollback=data.get("rollback"),
    )


def _parse_loop_step(data: dict[str, Any]) -> LoopStepConfig:
    """Build a ``LoopStepConfig`` from a raw dict.

    Args:
        data: Raw step dictionary containing the ``loop`` key.

    Returns:
        A validated ``LoopStepConfig``.

    Raises:
        ValueError: If no ``name`` is provided or loop body is invalid.
    """
    if "name" not in data:
        raise ValueError("Loop steps require an explicit 'name'.")

    loop = data["loop"]
    if not isinstance(loop, dict):
        raise TypeError("The 'loop' key must be a dict with at least a 'step' entry.")

    inner_step = _parse_agent_step(loop["step"])

    return LoopStepConfig(
        name=data["name"],
        step=inner_step,
        max_iterations=loop.get("max_iterations", 10),
        until=loop.get("until"),
        on_error=(ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL),
        timeout=data.get("timeout", 120.0),
    )


def _parse_sub_pipeline_step(
    data: dict[str, Any],
) -> SubPipelineStepConfig:
    """Build a ``SubPipelineStepConfig`` from a raw dict.

    Args:
        data: Raw step dictionary containing the ``sub_pipeline`` key.

    Returns:
        A validated ``SubPipelineStepConfig``.
    """
    pipeline_name: str = data["sub_pipeline"]
    name: str = data.get("name", pipeline_name)

    return SubPipelineStepConfig(
        name=name,
        pipeline=pipeline_name,
        input=_coerce_input_mapping(data.get("input")),
        output=_coerce_output_mapping(data.get("output")),
        condition=data.get("condition"),
        on_error=(ErrorPolicy(data["on_error"]) if "on_error" in data else ErrorPolicy.FAIL),
        timeout=data.get("timeout", 120.0),
    )


def _parse_step(data: dict[str, Any]) -> StepConfigUnion:
    """Detect the step type and delegate to the appropriate parser.

    Detection is based on the first matching discriminator key found in
    *data*.  The priority order is: ``agent``, ``parallel``,
    ``transform``, ``loop``, ``sub_pipeline``.

    Args:
        data: A single raw step dictionary from the YAML ``steps`` list.

    Returns:
        One of the ``StepConfigUnion`` member types.

    Raises:
        ValueError: If no recognised step-type key is found.
    """
    if "agent" in data:
        return _parse_agent_step(data)
    if "plugin" in data:
        return _parse_plugin_step(data)
    if "endpoint" in data:
        return _parse_endpoint_step(data)
    if "ingestion" in data:
        return _parse_ingestion_step(data)
    if "parallel" in data:
        return _parse_parallel_step(data)
    if "map" in data:
        return _parse_map_step(data)
    if "transform" in data:
        return _parse_transform_step(data)
    if "loop" in data:
        return _parse_loop_step(data)
    if "sub_pipeline" in data:
        return _parse_sub_pipeline_step(data)

    raise ValueError(
        f"Cannot determine step type. Expected one of {_STEP_TYPE_KEYS} "
        f"in step keys: {set(data.keys())}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PipelineLoader:
    """Load and validate pipeline configurations from various sources.

    Supports YAML files, raw dictionaries, YAML strings, and automatic
    directory-based discovery.

    Examples:
        >>> config = PipelineLoader.from_yaml(Path("pipeline.yaml"))
        >>> config = PipelineLoader.from_dict(raw_dict)
        >>> pipelines = PipelineLoader.discover_pipelines(Path("./agents"))
    """

    @staticmethod
    def from_yaml(path: Path) -> PipelineConfig:
        """Load a pipeline from a YAML file.

        Args:
            path: Filesystem path to a ``.yaml`` / ``.yml`` file.

        Returns:
            A validated ``PipelineConfig``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ImportError: If PyYAML is not installed.
            ValueError: If the file content is not valid pipeline YAML.
        """
        _ensure_yaml()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        logger.debug("Loading pipeline from {}", path)
        raw_text = path.read_text(encoding="utf-8")

        data = yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a YAML mapping at top level in {path}, got {type(data).__name__}"
            )

        return PipelineLoader.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelineConfig:
        """Parse a pipeline from a raw dictionary.

        The dictionary must contain at least ``name`` and ``steps``.
        Each element of ``steps`` is inspected for a discriminator key
        (``agent``, ``parallel``, ``transform``, ``loop``, or
        ``sub_pipeline``) and converted to the appropriate model.

        Args:
            data: Raw pipeline dictionary (e.g. from ``yaml.safe_load``).

        Returns:
            A validated ``PipelineConfig``.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If step parsing or validation fails.
        """
        if "name" not in data:
            raise KeyError("Pipeline dict must contain a 'name' key.")
        if "steps" not in data:
            raise KeyError("Pipeline dict must contain a 'steps' key.")

        raw_steps: list[dict[str, Any]] = data["steps"]
        parsed_steps: list[StepConfigUnion] = [_parse_step(s) for s in raw_steps]

        return PipelineConfig(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            steps=parsed_steps,
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            defaults=data.get("defaults", {}),
            on_error=data.get("on_error", "fail_fast"),
            strict_schema=data.get("strict_schema", False),
            timeout=data.get("timeout", 300.0),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def from_yaml_string(content: str) -> PipelineConfig:
        """Parse a pipeline from a YAML string.

        Args:
            content: A YAML-formatted string.

        Returns:
            A validated ``PipelineConfig``.

        Raises:
            ImportError: If PyYAML is not installed.
            ValueError: If *content* is not valid pipeline YAML.
        """
        _ensure_yaml()

        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping at top level, got {type(data).__name__}")

        return PipelineLoader.from_dict(data)

    @staticmethod
    def discover_pipelines(
        directory: Path,
    ) -> dict[str, PipelineConfig]:
        """Auto-discover pipelines from a directory tree.

        Scans for:
        - ``pipeline.yaml`` or ``pipeline.yml`` files directly in
          *directory*.
        - ``*.yaml`` / ``*.yml`` files in a ``pipelines/`` subdirectory.
        - ``pipeline.yaml`` inside each agent folder
          (``agents/*/pipeline.yaml``).

        Args:
            directory: Root directory to scan.

        Returns:
            A mapping of ``pipeline_name → PipelineConfig`` for every
            successfully loaded pipeline.  Pipelines that fail to parse
            are logged as warnings and skipped.
        """
        _ensure_yaml()
        directory = Path(directory)
        configs: dict[str, PipelineConfig] = {}
        seen_paths: set[Path] = set()

        def _try_load(path: Path) -> None:
            """Attempt to load a pipeline file, skipping duplicates."""
            resolved = path.resolve()
            if resolved in seen_paths:
                return
            seen_paths.add(resolved)

            try:
                cfg = PipelineLoader.from_yaml(path)
                if cfg.name in configs:
                    logger.warning(
                        "Duplicate pipeline name '{}' – keeping first occurrence, skipping {}",
                        cfg.name,
                        path,
                    )
                    return
                configs[cfg.name] = cfg
                logger.debug("Discovered pipeline '{}' from {}", cfg.name, path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping {} – failed to load: {}", path, exc)

        # 1. pipeline.yaml / pipeline.yml in the root directory
        for suffix in ("yaml", "yml"):
            candidate = directory / f"pipeline.{suffix}"
            if candidate.is_file():
                _try_load(candidate)

        # 2. All YAML files under a `pipelines/` subdirectory
        pipelines_dir = directory / "pipelines"
        if pipelines_dir.is_dir():
            for child in sorted(pipelines_dir.iterdir()):
                if child.is_file() and child.suffix in _YAML_SUFFIXES:
                    _try_load(child)

        # 3. pipeline.yaml inside each `agents/*/` folder
        agents_dir = directory / "agents"
        if agents_dir.is_dir():
            for agent_folder in sorted(agents_dir.iterdir()):
                if not agent_folder.is_dir():
                    continue
                for suffix in ("yaml", "yml"):
                    candidate = agent_folder / f"pipeline.{suffix}"
                    if candidate.is_file():
                        _try_load(candidate)

        logger.info("Discovered {} pipeline(s) in {}", len(configs), directory)
        return configs
