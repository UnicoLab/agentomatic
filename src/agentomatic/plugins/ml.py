"""Base classes for ML model plugins."""

from __future__ import annotations

import asyncio
import typing
from datetime import UTC, datetime
from pathlib import Path
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
        self._loaded_at: str | None = None
        # Platform invocation paths and reloads share this lock.  A plugin can
        # contain mutable native-model state, so letting a prediction race a
        # model swap can expose a partially initialised model to production
        # traffic.
        self._operation_lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        """Return True if the model weights are loaded."""
        return self._is_loaded

    @property
    def loaded_at(self) -> str | None:
        """ISO-8601 timestamp of the last successful ``load_model`` / reload."""
        return self._loaded_at

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
        Subclasses that override must call ``await super().load_model()``
        (or set ``_is_loaded`` / ``_loaded_at`` themselves) so reload status
        stays accurate.
        """
        self._is_loaded = True
        self._loaded_at = datetime.now(UTC).isoformat()

    def mark_loaded(self) -> None:
        """Record that the model is loaded and ready to serve.

        Called by the platform after :meth:`load_model` returns, so a subclass
        that overrides ``load_model`` without calling ``super()`` still ends up
        in a usable state instead of answering 503 forever while the startup
        log claims success. Idempotent — an existing ``_loaded_at`` is kept.
        """
        self._is_loaded = True
        if self._loaded_at is None:
            self._loaded_at = datetime.now(UTC).isoformat()

    def artifact_dir(self) -> Path | None:
        """Return the active artifact bundle directory, or ``None``.

        Convenience for ``load_model`` implementations that read weights from
        :class:`~agentomatic.artifacts.ArtifactRegistry.current_dir`.
        """
        from agentomatic.artifacts import ArtifactRegistry

        return ArtifactRegistry().current_dir()

    async def reload_model(self) -> dict[str, Any]:
        """Reload model weights from the current artifact pointer safely.

        Reloading is serialized with platform predictions.  If loading raises,
        the plugin object's shallow state is restored, preserving the prior
        working model and readiness state.  Standard plugins assign a new
        model object in ``load_model``; plugins that mutate an existing model
        in place should construct a replacement object first and assign it
        only after it is ready.

        Returns:
            Status dict with name, version, loaded flag, loaded_at, model_card.
        """
        async with self._operation_lock:
            previous_state = self.__dict__.copy()
            try:
                # A subclass that omits ``super().load_model()`` still gets a
                # fresh timestamp through ``mark_loaded`` below.
                self._loaded_at = None
                await self.load_model()
                self.mark_loaded()
            except Exception:
                self.__dict__.clear()
                self.__dict__.update(previous_state)
                raise
            return self.info(include_model_card=True)

    async def invoke(self, inputs: InputT) -> OutputT:
        """Run a prediction serialized with model reloads.

        Platform routers, task dispatchers, and pipeline steps use this method
        instead of calling :meth:`predict` directly.  Direct user calls to an
        overridden ``predict`` remain supported for backwards compatibility.

        Readiness is intentionally enforced by the serving boundaries that
        require it (the REST router and task dispatcher).  Pipeline steps have
        historically supported lightweight, stateless plugins without a
        startup ``load_model`` implementation, so ``invoke`` must remain a
        synchronization primitive rather than changing that contract.
        """
        async with self._operation_lock:
            return await self.predict(inputs)

    def info(self, *, include_model_card: bool = False) -> dict[str, Any]:
        """Return a serialisable status snapshot for list/reload APIs.

        Args:
            include_model_card: When True, embed the full ``model_card()``.

        Returns:
            Dict with name, description, version, is_loaded, loaded_at.
        """
        payload: dict[str, Any] = {
            "name": self.plugin_name,
            "description": self.plugin_description,
            "version": self.plugin_version,
            "is_loaded": self.is_loaded,
            "loaded_at": self.loaded_at,
        }
        if include_model_card:
            payload["model_card"] = self.model_card()
        return payload

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
            "loaded_at": self._loaded_at,
        }
