"""Pydantic schema validation for agent I/O.

When an agent declares ``schemas.py`` with Request/Response models, the
platform can automatically validate inputs and outputs at the endpoint
layer.
"""

from __future__ import annotations

import importlib
from typing import Any

from loguru import logger
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Schema discovery conventions
# ---------------------------------------------------------------------------
#
# An agent may declare its input/output models in ``schemas.py`` under any of
# the following names (checked in priority order):
#
#   Input:  CustomInvokeRequest, {Title}Request, {Title}Input,
#           AgentInput, AgentInvokeRequest, Input
#   Output: CustomInvokeResponse, {Title}Response, {Title}Output,
#           AgentOutput, AgentInvokeResponse, Output
#
# ``{Title}`` is the agent folder name converted to CamelCase
# (e.g. ``weather_bot`` -> ``WeatherBot``).


def schema_model_candidates(
    agent_name: str,
) -> tuple[list[str], list[str]]:
    """Return the input/output class-name candidates for an agent.

    Args:
        agent_name: Agent folder name (e.g. ``"alpha"`` or ``"weather_bot"``).

    Returns:
        A ``(input_candidates, output_candidates)`` tuple of class names in
        priority order.
    """
    title = agent_name.title().replace("_", "")
    return (
        [
            "CustomInvokeRequest",
            f"{title}Request",
            f"{title}Input",
            "AgentInput",
            "AgentInvokeRequest",
            "Input",
        ],
        [
            "CustomInvokeResponse",
            f"{title}Response",
            f"{title}Output",
            "AgentOutput",
            "AgentInvokeResponse",
            "Output",
        ],
    )


def _is_pydantic_model(cls: Any) -> bool:
    """Whether *cls* looks like a Pydantic model (has JSON Schema output)."""
    return isinstance(cls, type) and hasattr(cls, "model_json_schema")


def _first_model(mod: Any, candidates: list[str]) -> type[BaseModel] | None:
    """Return the first candidate class that exists in *mod* and is a model."""
    for candidate in candidates:
        cls = getattr(mod, candidate, None)
        if _is_pydantic_model(cls):
            return cls
    return None


def load_schema_models(
    module_path: str,
    agent_name: str,
) -> tuple[type[BaseModel] | None, type[BaseModel] | None]:
    """Import an agent's ``schemas.py`` and resolve its input/output models.

    Args:
        module_path: Python dotted path of the agent package (without
            ``.schemas``).
        agent_name: Agent folder name used for CamelCase candidates.

    Returns:
        ``(input_model, output_model)`` — either may be ``None`` when the
        module is missing or no candidate class matches.
    """
    if not module_path:
        return None, None
    try:
        schemas_mod = importlib.import_module(f"{module_path}.schemas")
    except ImportError:
        return None, None

    input_candidates, output_candidates = schema_model_candidates(agent_name)
    return (
        _first_model(schemas_mod, input_candidates),
        _first_model(schemas_mod, output_candidates),
    )


class SchemaValidator:
    """Validates agent I/O against declared Pydantic schemas.

    Args:
        request_model: Pydantic model for input validation (or ``None``).
        response_model: Pydantic model for output validation (or ``None``).
    """

    def __init__(
        self,
        request_model: type[BaseModel] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> None:
        self.request_model = request_model
        self.response_model = response_model

    @property
    def has_request_schema(self) -> bool:
        """Whether a request schema is configured."""
        return self.request_model is not None

    @property
    def has_response_schema(self) -> bool:
        """Whether a response schema is configured."""
        return self.response_model is not None

    def validate_input(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce input data against the request model.

        Args:
            data: Raw input dictionary.

        Returns:
            Validated data as a dictionary.

        Raises:
            ValidationError: If the data does not match the schema.
        """
        if self.request_model is None:
            return data
        try:
            validated = self.request_model.model_validate(data)
            return validated.model_dump()
        except ValidationError:
            logger.warning(f"Input validation failed for {self.request_model.__name__}")
            raise

    def validate_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate output data against the response model.

        Validation failures for output are logged but not raised — the
        response is still returned to the caller.  This is intentional:
        a downstream schema change should not crash a running agent.

        Args:
            data: Raw output dictionary.

        Returns:
            The original data (validation is advisory).
        """
        if self.response_model is None:
            return data
        try:
            self.response_model.model_validate(data)
        except ValidationError as exc:
            logger.warning(f"Output validation warning for {self.response_model.__name__}: {exc}")
        return data

    def get_openapi_schemas(self) -> dict[str, Any]:
        """Return JSON-Schema representations for OpenAPI documentation.

        Returns:
            Dict with optional ``"request"`` and ``"response"`` keys.
        """
        schemas: dict[str, Any] = {}
        if self.request_model is not None:
            schemas["request"] = self.request_model.model_json_schema()
        if self.response_model is not None:
            schemas["response"] = self.response_model.model_json_schema()
        return schemas
