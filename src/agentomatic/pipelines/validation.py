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
