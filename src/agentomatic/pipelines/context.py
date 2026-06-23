"""Pipeline context — shared state that flows between steps.

The context is the central data bus connecting pipeline steps.
It provides a ``$`` expression resolver for declarative data mapping.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from .models import StepResult


class PipelineContext:
    """Shared state flowing between pipeline steps.

    Attributes:
        input: Original pipeline input (read-only after creation).
        defaults: Pipeline-level default values (read-only).
        steps: Results from completed steps, keyed by step name.
        shared: Mutable shared context for cross-step communication.
        current: Most recent step's output (useful for loop conditions).
    """

    def __init__(
        self,
        input_data: dict[str, Any] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> None:
        self.input: dict[str, Any] = dict(input_data or {})
        self.defaults: dict[str, Any] = dict(defaults or {})
        self.steps: dict[str, StepResult] = {}
        self.shared: dict[str, Any] = {}
        self.current: dict[str, Any] = {}

    def set_step_result(self, name: str, result: StepResult) -> None:
        """Store a step's result and update ``current``."""
        self.steps[name] = result
        self.current = dict(result.output)

    def get_step_output(self, name: str) -> dict[str, Any]:
        """Get a step's output by name, or empty dict if not found."""
        result = self.steps.get(name)
        if result is None:
            return {}
        return dict(result.output)

    # ------------------------------------------------------------------
    # $ expression resolver
    # ------------------------------------------------------------------

    # Regex for array index access: name[0]
    _INDEX_RE = re.compile(r"^(\w+)\[(\d+)\]$")

    def resolve(self, expr: Any) -> Any:
        """Resolve a ``$`` expression against the context.

        Supported expressions:
            ``$.input.query``            → self.input["query"]
            ``$.input.*``                → self.input (entire dict)
            ``$.steps.plan.response``    → self.steps["plan"].output["response"]
            ``$.steps.plan.*``           → self.steps["plan"].output
            ``$.steps.research``         → [r.output for r in parallel results]
            ``$.steps.research[0].text`` → first parallel result's "text"
            ``$.defaults.language``      → self.defaults["language"]
            ``$.context.key``            → self.shared["key"]
            ``$.current.field``          → self.current["field"]

        Non-string values and strings not starting with ``$`` are
        returned as-is (literal pass-through).

        Args:
            expr: A ``$`` expression string or a literal value.

        Returns:
            The resolved value.
        """
        if not isinstance(expr, str) or not expr.startswith("$"):
            return expr

        # Strip leading "$." and split into parts
        path = expr.lstrip("$").lstrip(".")
        parts = path.split(".")

        if not parts:
            return expr

        root = parts[0]
        rest = parts[1:]

        try:
            return self._navigate(root, rest)
        except (KeyError, IndexError, TypeError) as exc:
            logger.debug(f"Could not resolve '{expr}': {exc}")
            return None

    def _navigate(self, root: str, rest: list[str]) -> Any:
        """Navigate the context tree from a root section."""
        if root == "input":
            return self._drill(self.input, rest)
        elif root == "defaults":
            return self._drill(self.defaults, rest)
        elif root == "context":
            return self._drill(self.shared, rest)
        elif root == "current":
            return self._drill(self.current, rest)
        elif root == "steps":
            return self._resolve_step(rest)
        else:
            raise KeyError(f"Unknown context root: {root}")

    def _resolve_step(self, parts: list[str]) -> Any:
        """Resolve a steps.X.field expression."""
        if not parts:
            # $.steps → dict of all step outputs
            return {name: dict(r.output) for name, r in self.steps.items()}

        step_name = parts[0]
        rest = parts[1:]

        # Check for array index: steps.research[0]
        match = self._INDEX_RE.match(step_name)
        if match:
            step_name = match.group(1)
            idx = int(match.group(2))
            result = self.steps.get(step_name)
            if result and result.sub_results:
                sub = result.sub_results[idx]
                return self._drill(sub.output, rest)
            return None

        result = self.steps.get(step_name)
        if result is None:
            return None

        # If no further path, return the output (or sub_results for parallel)
        if not rest:
            if result.sub_results is not None:
                return [dict(sr.output) for sr in result.sub_results]
            return dict(result.output)

        # Wildcard: steps.plan.* → entire output
        if rest == ["*"]:
            return dict(result.output)

        # Navigate into the output dict
        return self._drill(result.output, rest)

    def _drill(self, obj: Any, parts: list[str]) -> Any:
        """Drill into a nested dict/list by dotted path parts."""
        if not parts:
            return obj

        if parts == ["*"]:
            return obj

        current = obj
        for part in parts:
            if current is None:
                return None

            # Array index: field[0]
            match = self._INDEX_RE.match(part)
            if match:
                field = match.group(1)
                idx = int(match.group(2))
                if isinstance(current, dict):
                    current = current.get(field)
                else:
                    return None
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
                continue

            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                # If part is numeric string, use as index
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def resolve_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        """Resolve an entire input/output mapping dict.

        Args:
            mapping: Dict where values may be ``$`` expressions.

        Returns:
            Dict with all ``$`` expressions resolved.
        """
        resolved: dict[str, Any] = {}
        for key, expr in mapping.items():
            resolved[key] = self.resolve(expr)
        return resolved

    def to_eval_namespace(self) -> dict[str, Any]:
        """Create a namespace dict for evaluating condition expressions.

        The namespace exposes ``ctx`` (this context) and ``len``.
        """
        return {
            "ctx": self,
            "len": len,
            "any": any,
            "all": all,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "max": max,
            "min": min,
            "sum": sum,
            "sorted": sorted,
            "isinstance": isinstance,
        }
