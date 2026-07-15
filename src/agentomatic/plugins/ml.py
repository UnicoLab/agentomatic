"""Base classes for ML model plugins."""

from __future__ import annotations

import typing
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class _EmptyPluginInput(BaseModel):
    """Fallback input schema when a plugin omits its generic parameters."""


class _EmptyPluginOutput(BaseModel):
    """Fallback output schema when a plugin omits its generic parameters."""

    result: dict[str, Any] = Field(default_factory=dict)


class BaseMLPlugin(Generic[InputT, OutputT]):
    """Base class for classical ML Model Plugins.

    Subclass this to expose custom classical ML models (e.g. Scikit-learn, PyTorch)
    via Agentomatic's auto-generated REST APIs.
    """

    plugin_name: str = "default_plugin"
    plugin_description: str = "A classical ML model plugin."
    plugin_version: str = "1.0.0"

    def __init__(self) -> None:
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return True if the model weights are loaded."""
        return self._is_loaded

    def get_input_schema(self) -> type[BaseModel]:
        """Extract the InputT Pydantic schema from the generic typing.

        Returns a concrete :class:`~pydantic.BaseModel` subclass only.
        Falls back to a safe empty model when generics are missing or
        unresolved (avoids OpenAPI schema generation failures).
        """
        schema = self._generic_arg(0)
        return schema or _EmptyPluginInput

    def get_output_schema(self) -> type[BaseModel]:
        """Extract the OutputT Pydantic schema from the generic typing.

        Returns a concrete :class:`~pydantic.BaseModel` subclass only.
        Falls back to a safe empty model when generics are missing or
        unresolved (avoids OpenAPI schema generation failures).
        """
        schema = self._generic_arg(1)
        return schema or _EmptyPluginOutput

    def _generic_arg(self, index: int) -> type[BaseModel] | None:
        """Return the ``index``-th generic type argument, if a BaseModel."""
        for base in getattr(self.__class__, "__orig_bases__", []):
            origin = typing.get_origin(base)
            if origin is BaseMLPlugin or origin is self.__class__:
                args = typing.get_args(base)
                if len(args) > index:
                    candidate = args[index]
                    if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                        return candidate
        return None

    async def load_model(self) -> None:
        """Load the ML model weights into memory.

        This is called automatically during the platform's lifespan startup.
        Override this method in subclasses to load your actual model files.
        """
        self._is_loaded = True

    async def predict(self, inputs: InputT) -> OutputT:
        """Run inference using the loaded model.

        Override this method in subclasses.
        """
        raise NotImplementedError("Plugins must implement predict()")

    def model_card(self) -> dict[str, Any]:
        """Return metadata about the model (params, metrics, architecture).

        Override this method to provide a custom model card.
        """
        return {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "version": self.plugin_version,
            "status": "loaded" if self._is_loaded else "unloaded",
        }
