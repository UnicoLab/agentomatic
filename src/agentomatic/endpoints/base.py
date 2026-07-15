"""Base class for custom endpoints.

A *custom endpoint* is a user-defined HTTP API exposed by the platform
(alongside agents, plugins and pipelines).  Endpoints typically call one
or more deployed model services via authenticated ``httpx`` requests and
aggregate their outputs — but they can implement any logic.

Subclass :class:`BaseEndpoint` and either:

* declare ``upstreams`` and rely on the default fan-out ``handle`` method, or
* override ``handle`` for fully custom behaviour.
"""

from __future__ import annotations

import time
import typing
from typing import Any, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel

from agentomatic.endpoints.client import MultiModelClient
from agentomatic.endpoints.models import (
    AggregationStrategy,
    EndpointCallRequest,
    EndpointCallResponse,
    UpstreamConfig,
)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseEndpoint(Generic[InputT, OutputT]):
    """Base class for a custom, auto-mounted HTTP endpoint.

    Class attributes configure identity and routing; ``upstreams`` lists the
    deployed model services this endpoint talks to.  The default
    :meth:`handle` implementation fans out the request payload to every
    upstream and aggregates the responses using :attr:`aggregation`.

    Example::

        class EnsembleEndpoint(BaseEndpoint[EndpointCallRequest, EndpointCallResponse]):
            endpoint_name = "ensemble"
            path = "/predict"
            aggregation = AggregationStrategy.MAJORITY
            upstreams = [
                UpstreamConfig(name="a", base_url="https://a.example.com", path="/v1/run"),
                UpstreamConfig(name="b", base_url="https://b.example.com", path="/v1/run"),
            ]
    """

    endpoint_name: str = "default_endpoint"
    endpoint_description: str = "A custom endpoint."
    endpoint_version: str = "1.0.0"

    #: Route path (relative to the endpoint's mount prefix).
    path: str = "/call"
    #: HTTP methods exposed for :attr:`path`.
    methods: list[str] = ["POST"]

    #: Upstream services this endpoint calls (may be empty).
    upstreams: list[UpstreamConfig] = []
    #: How to aggregate multiple upstream responses.
    aggregation: AggregationStrategy = AggregationStrategy.ALL
    #: Maximum concurrent upstream calls during fan-out.
    max_concurrency: int = 5

    def __init__(self) -> None:
        self._ready = False
        self._models: MultiModelClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return True once :meth:`startup` has completed."""
        return self._ready

    @property
    def models(self) -> MultiModelClient:
        """Return the fan-out client for the configured upstreams."""
        if self._models is None:
            self._models = MultiModelClient(
                self.get_upstreams(),
                strategy=self.aggregation,
                max_concurrency=self.max_concurrency,
            )
        return self._models

    def get_upstreams(self) -> list[UpstreamConfig]:
        """Return the upstream configs (override for dynamic upstreams)."""
        return list(self.upstreams)

    async def startup(self) -> None:
        """Initialise upstream clients — called during platform startup."""
        if self.get_upstreams():
            _ = self.models  # trigger lazy construction
        self._ready = True

    async def shutdown(self) -> None:
        """Release upstream clients — called during platform shutdown."""
        if self._models is not None:
            await self._models.aclose()
            self._models = None
        self._ready = False

    async def health_check(self) -> dict[str, Any]:
        """Report endpoint health for the platform status / control plane.

        The default implementation reports readiness and the number of
        configured upstreams. Override for deeper checks (e.g. pinging each
        upstream).

        Returns:
            A status mapping consumed by ``/status`` and the control plane.
        """
        return {
            "status": "healthy" if self._ready else "unloaded",
            "endpoint": self.endpoint_name,
            "version": self.endpoint_version,
            "upstreams": len(self.get_upstreams()),
        }

    # ------------------------------------------------------------------
    # Schema extraction (mirrors plugins.BaseMLPlugin)
    # ------------------------------------------------------------------

    def get_input_schema(self) -> type[BaseModel]:
        """Extract the ``InputT`` Pydantic schema from the generic typing."""
        schema = self._generic_arg(0)
        return schema or EndpointCallRequest

    def get_output_schema(self) -> type[BaseModel]:
        """Extract the ``OutputT`` Pydantic schema from the generic typing."""
        schema = self._generic_arg(1)
        return schema or EndpointCallResponse

    def _generic_arg(self, index: int) -> type[BaseModel] | None:
        """Return the ``index``-th generic type argument, if a BaseModel."""
        for base in getattr(self.__class__, "__orig_bases__", []):
            origin = typing.get_origin(base)
            if origin is BaseEndpoint or origin is self.__class__:
                args = typing.get_args(base)
                if len(args) > index:
                    candidate = args[index]
                    if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                        return candidate
        return None

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def handle(self, request: Any) -> Any:
        """Handle an incoming request.

        The default implementation fans out to every configured upstream and
        aggregates responses.  Override this for custom behaviour.

        Args:
            request: The parsed request model (``InputT``).

        Returns:
            The response model (``OutputT``).
        """
        t0 = time.perf_counter()
        payload = getattr(request, "payload", None)
        if payload is None and isinstance(request, BaseModel):
            payload = request.model_dump()
        subset = getattr(request, "upstreams", None)

        if not self.get_upstreams():
            return EndpointCallResponse(
                endpoint=self.endpoint_name,
                strategy=self.aggregation.value,
                ok=False,
                aggregated=None,
                results=[],
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        ok, aggregated, results = await self.models.fan_out(payload, upstreams=subset)
        return EndpointCallResponse(
            endpoint=self.endpoint_name,
            strategy=self.aggregation.value,
            ok=ok,
            aggregated=aggregated,
            results=results,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    async def call(self, payload: Any = None, **kwargs: Any) -> Any:
        """Convenience helper used by pipelines and other code.

        Wraps a raw ``payload`` in the default request model (unless the
        endpoint defines a custom input schema) and delegates to
        :meth:`handle`.

        Args:
            payload: JSON-serialisable payload for the upstream call.
            **kwargs: Extra fields (e.g. ``upstreams``) for the request model.

        Returns:
            The endpoint's response (an ``OutputT`` instance).
        """
        input_schema = self.get_input_schema()
        request: Any
        if input_schema is EndpointCallRequest:
            request = EndpointCallRequest(payload=payload or {}, **kwargs)
        else:
            data = payload if isinstance(payload, dict) else {"payload": payload}
            try:
                request = input_schema(**{**data, **kwargs})
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Falling back to raw payload for '{self.endpoint_name}': {exc}")
                request = payload
        return await self.handle(request)

    def info(self) -> dict[str, Any]:
        """Return metadata describing this endpoint."""
        return {
            "name": self.endpoint_name,
            "description": self.endpoint_description,
            "version": self.endpoint_version,
            "path": self.path,
            "methods": self.methods,
            "aggregation": self.aggregation.value,
            "upstreams": [u.name for u in self.get_upstreams()],
            "ready": self._ready,
        }
