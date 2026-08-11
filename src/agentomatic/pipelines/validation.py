"""Lightweight schema validation for pipeline input/output.

Pipelines may declare ``input_schema`` / ``output_schema`` dicts.  This module
validates data against them without requiring a full JSON-Schema engine, so the
contracts between chained steps can be enforced (advisory by default, strict on
demand).

Two schema shapes are supported per field:

* Shorthand — ``{"query": "str", "top_k": "int"}`` (field is required).
* Verbose — ``{"query": {"type": "str", "required": true}}``.

Recognised type names: ``str``, ``string``, ``int``, ``integer``, ``float``,
``number``, ``bool``, ``boolean``, ``list``, ``array``, ``dict``, ``object``,
``any`` (and ``None``/omitted → no type check).
"""

from __future__ import annotations

from typing import Any

from .models import (
    AgentStepConfig,
    EndpointStepConfig,
    ErrorPolicy,
    IngestionStepConfig,
    LoopStepConfig,
    MapStepConfig,
    ParallelStepConfig,
    PipelineConfig,
    PluginStepConfig,
    SubPipelineStepConfig,
    TransformStepConfig,
)

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "str": (str,),
    "string": (str,),
    "int": (int,),
    "integer": (int,),
    "float": (float, int),
    "number": (int, float),
    "bool": (bool,),
    "boolean": (bool,),
    "list": (list,),
    "array": (list,),
    "dict": (dict,),
    "object": (dict,),
}


def validate_against_schema(
    data: dict[str, Any] | None,
    schema: dict[str, Any] | None,
    *,
    label: str = "data",
) -> list[str]:
    """Validate ``data`` against a simple field ``schema``.

    Args:
        data: The payload to validate.
        schema: The declared schema (see module docstring for shapes).
        label: Prefix used in error messages (e.g. ``"input"``/``"output"``).

    Returns:
        A list of human-readable error messages (empty when valid).
    """
    if not schema:
        return []

    payload = data or {}
    errors: list[str] = []

    for field, spec in schema.items():
        required = True
        type_name: str | None = None

        if isinstance(spec, dict):
            required = bool(spec.get("required", True))
            raw_type = spec.get("type")
            type_name = str(raw_type).lower() if raw_type else None
        elif isinstance(spec, str):
            type_name = spec.lower()
        # Any other spec shape → presence-only check.

        if field not in payload:
            if required:
                errors.append(f"{label}: missing required field '{field}'")
            continue

        if not type_name or type_name in ("any", "none"):
            continue

        expected = _TYPE_MAP.get(type_name)
        if expected is None:
            # Unknown type name — skip rather than raise, to stay lenient.
            continue

        value = payload[field]
        # bool is a subclass of int; guard against silent acceptance.
        if type_name in ("int", "integer") and isinstance(value, bool):
            errors.append(f"{label}: field '{field}' expected {type_name}, got bool")
            continue
        if not isinstance(value, expected):
            got = type(value).__name__
            errors.append(f"{label}: field '{field}' expected {type_name}, got {got}")

    return errors


# ---------------------------------------------------------------------------
# Draft-level checks (for the visual builder's instant feedback)
# ---------------------------------------------------------------------------

# Roots a ``$`` expression may resolve against in the pipeline context.
_INPUT_ROOTS = ("input", "defaults", "context", "current", "steps")


def _check_python_syntax(code: str, label: str, errors: list[str], *, mode: str) -> None:
    """Append a syntax error to *errors* when *code* does not compile."""
    try:
        compile(code, "<pipeline>", mode)
    except SyntaxError as exc:
        errors.append(f"{label} has invalid syntax: {exc.msg} (line {exc.lineno})")


def _check_transform_code(code: str, label: str, errors: list[str]) -> None:
    """Compile transform code the same way the engine executes it.

    The engine wraps the block in ``async def _transform(ctx):`` before
    ``exec``, so a top-level ``return`` is valid there but not in a bare
    ``exec`` compile.  Mirror the wrapping to avoid false positives.
    """
    wrapped = "async def _transform(ctx):\n"
    for line in code.strip().splitlines():
        wrapped += f"    {line}\n"
    try:
        compile(wrapped, "<pipeline>", "exec")
    except SyntaxError as exc:
        errors.append(f"{label} has invalid syntax: {exc.msg} (line {exc.lineno})")


def _check_input_expr(
    expr: Any,
    label: str,
    current_index: int,
    step_index: dict[str, int],
    upstreams: set[str],
    input_schema: dict[str, Any] | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate one input-mapping expression (``$`` path into the context)."""
    if not isinstance(expr, str) or not expr.startswith("$"):
        return

    path = expr.lstrip("$").lstrip(".")
    parts = [p for p in path.split(".") if p]
    if not parts:
        warnings.append(
            f"{label}: expression '{expr}' resolves to nothing — use a path like '$.input.query'"
        )
        return

    root = parts[0]
    if root not in _INPUT_ROOTS:
        errors.append(
            f"{label}: unknown mapping root '{root}' in '{expr}' "
            f"(expected one of: {', '.join(_INPUT_ROOTS)})"
        )
        return

    if root == "steps" and len(parts) >= 2:
        ref = parts[1].split("[", 1)[0]
        if ref != "*":
            if ref not in step_index:
                errors.append(
                    f"{label}: '{expr}' references step '{ref}' which does not "
                    "exist in the pipeline"
                )
            elif step_index[ref] >= current_index and ref not in upstreams:
                errors.append(
                    f"{label}: '{expr}' references step '{ref}' which runs after "
                    "this step — move it earlier in the list or declare it in "
                    "upstreams"
                )
    elif root == "input" and input_schema and len(parts) >= 2:
        field = parts[1].split("[", 1)[0]
        if field not in input_schema:
            warnings.append(
                f"{label}: '{expr}' references input field '{field}' which is "
                "not declared in input_schema"
            )


def _check_output_expr(expr: Any, label: str, warnings: list[str]) -> None:
    """Validate one output-mapping expression.

    Output mappings resolve against the step's own output (not the pipeline
    context), so expressions like ``$.response`` are paths into that output.
    """
    if not isinstance(expr, str) or not expr.startswith("$"):
        return
    path = expr.lstrip("$").lstrip(".")
    parts = [p for p in path.split(".") if p]
    if not parts:
        return  # ``$`` alone = whole step output, which is valid
    if parts[0] in _INPUT_ROOTS:
        warnings.append(
            f"{label}: output mapping '{expr}' starts with '{parts[0]}' — output "
            "mappings resolve against this step's own output, not the pipeline "
            "context"
        )


def _warn_on_error_policy(step: Any, label: str, warnings: list[str]) -> None:
    """Add advisory warnings for risky error-policy combinations."""
    policy = getattr(step, "on_error", None)
    if policy == ErrorPolicy.SKIP:
        warnings.append(f"{label} uses on_error=skip — failures will be silently ignored")
    elif policy == ErrorPolicy.RETRY and getattr(step, "retry", None) is None:
        warnings.append(f"{label} uses on_error=retry but has no retry config")
    elif policy == ErrorPolicy.FALLBACK and getattr(step, "fallback_agent", None) is None:
        warnings.append(f"{label} uses on_error=fallback but has no fallback_agent")


def _check_step_mappings(
    step: Any,
    label: str,
    current_index: int,
    step_index: dict[str, int],
    upstreams: set[str],
    input_schema: dict[str, Any] | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check input/output mappings, condition syntax, and error policy."""
    input_mapping = getattr(step, "input", None)
    if input_mapping is not None:
        for key, expr in input_mapping.items():
            _check_input_expr(
                expr,
                f"{label} input mapping '{key}'",
                current_index,
                step_index,
                upstreams,
                input_schema,
                errors,
                warnings,
            )
    output_mapping = getattr(step, "output", None)
    if output_mapping is not None:
        for key, expr in output_mapping.items():
            _check_output_expr(expr, f"{label} output mapping '{key}'", warnings)

    condition = getattr(step, "condition", None)
    if condition:
        _check_python_syntax(condition, f"{label} condition", errors, mode="eval")

    _warn_on_error_policy(step, label, warnings)


def validate_pipeline_draft(config: PipelineConfig) -> tuple[list[str], list[str]]:
    """Run structural draft checks and return ``(errors, warnings)``.

    These checks complement :meth:`PipelineEngine.validate` (registry
    resolution) with wiring-level feedback so the visual builder can flag
    problems on every keystroke:

    - ``$`` mapping expressions must use a known context root
      (``input`` / ``defaults`` / ``context`` / ``current`` / ``steps``).
    - ``$.steps.X`` references must point at an earlier step (ordered flow).
    - Conditions, ``until`` expressions, and transform code must compile.
    - Advisory warnings: ``on_error=skip`` hiding failures, ``retry``
      policies without retry config, undeclared input fields, etc.

    Args:
        config: The parsed pipeline configuration.

    Returns:
        ``(errors, warnings)`` — human-readable messages, empty when clean.
    """
    errors: list[str] = []
    warnings: list[str] = []
    step_index = {name: i for i, name in enumerate(config.step_names)}
    input_schema = config.input_schema

    # Upstream dependency graph: unknown refs, self-refs, cycles.
    from .ordering import compute_execution_order

    try:
        compute_execution_order(config.steps)
    except ValueError as exc:
        errors.append(str(exc))

    for index, step in enumerate(config.steps):
        label = f"Step '{step.name}'"
        upstreams = set(getattr(step, "upstreams", None) or [])

        if isinstance(
            step,
            (
                AgentStepConfig,
                PluginStepConfig,
                EndpointStepConfig,
                IngestionStepConfig,
                SubPipelineStepConfig,
            ),
        ):
            _check_step_mappings(
                step, label, index, step_index, upstreams, input_schema, errors, warnings
            )

        elif isinstance(step, TransformStepConfig):
            _check_transform_code(step.code, f"{label} transform code", errors)
            if "return" not in step.code:
                warnings.append(
                    f"{label} transform code does not contain a 'return' — "
                    "the step will produce no output"
                )
            if step.condition:
                _check_python_syntax(step.condition, f"{label} condition", errors, mode="eval")
            _warn_on_error_policy(step, label, warnings)

        elif isinstance(step, ParallelStepConfig):
            for sub in step.steps:
                if getattr(sub, "upstreams", None):
                    warnings.append(
                        f"Step '{sub.name}' (inside parallel '{step.name}') declares "
                        "upstreams — nested steps are not independently scheduled; "
                        "declare the dependency on the container step instead"
                    )
                _check_step_mappings(
                    sub,
                    f"Step '{sub.name}' (inside parallel '{step.name}')",
                    index,
                    step_index,
                    upstreams,
                    input_schema,
                    errors,
                    warnings,
                )
            _warn_on_error_policy(step, label, warnings)

        elif isinstance(step, MapStepConfig):
            _check_input_expr(
                step.items,
                f"{label} items",
                index,
                step_index,
                upstreams,
                input_schema,
                errors,
                warnings,
            )
            _check_step_mappings(
                step, label, index, step_index, upstreams, input_schema, errors, warnings
            )

        elif isinstance(step, LoopStepConfig):
            if getattr(step.step, "upstreams", None):
                warnings.append(
                    f"Step '{step.step.name}' (inside loop '{step.name}') declares "
                    "upstreams — nested steps are not independently scheduled; "
                    "declare the dependency on the container step instead"
                )
            _check_step_mappings(
                step.step,
                f"Step '{step.step.name}' (inside loop '{step.name}')",
                index,
                step_index,
                upstreams,
                input_schema,
                errors,
                warnings,
            )
            if step.until:
                _check_python_syntax(step.until, f"{label} until expression", errors, mode="eval")
            _warn_on_error_policy(step, label, warnings)

    return errors, warnings
