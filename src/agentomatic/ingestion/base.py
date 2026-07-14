"""Base class for user-defined ingestors.

An *ingestor* packages your document-ingestion code — built with whatever
libraries you like (``docling``/``unstructured``/``pymupdf`` to read documents
to markdown, ``langchain-text-splitters`` to chunk, your embedding model, your
vector DB client) — as a first-class, deployable Agentomatic resource.

Agentomatic provides the ops around it: auto-discovery, REST endpoints,
task/queue execution with progress + cancellation, and status reporting. You
implement exactly one method: :meth:`ingest`.

Example::

    from agentomatic.ingestion import BaseIngestor, IngestionResult

    class PdfIngestor(BaseIngestor):
        ingestor_name = "pdf_docs"
        ingestor_description = "Parse PDFs to markdown and upsert to Qdrant."

        async def setup(self) -> None:
            from qdrant_client import AsyncQdrantClient
            self.client = AsyncQdrantClient(url="http://localhost:6333")

        async def ingest(self, request, ctx) -> IngestionResult:
            import pymupdf4llm  # user's choice of library
            from langchain_text_splitters import MarkdownTextSplitter

            md = pymupdf4llm.to_markdown(request.source)
            chunks = MarkdownTextSplitter().split_text(md)
            for i, chunk in enumerate(chunks):
                if ctx.cancelled:
                    break
                # ... embed + upsert with your own client ...
                await ctx.report(current=i + 1, total=len(chunks), message="upserting")
            return IngestionResult(chunks=len(chunks), upserted=len(chunks))
"""

from __future__ import annotations

import typing
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from .context import IngestionContext, NullIngestionContext
from .models import IngestionRequest, IngestionResult

InputT = TypeVar("InputT", bound=BaseModel)


class BaseIngestor(Generic[InputT]):
    """Base class for a user-defined, auto-mounted ingestion resource.

    Subclass this and implement :meth:`ingest`. Optionally declare a custom
    Pydantic input model via the generic parameter to get a strictly-typed
    REST body::

        class MyIngestor(BaseIngestor[MyRequest]):
            ...
    """

    ingestor_name: str = "default_ingestor"
    ingestor_description: str = "A document ingestion job."
    ingestor_version: str = "1.0.0"

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return ``True`` once :meth:`setup` has completed."""
        return self._ready

    async def setup(self) -> None:
        """Initialise resources (clients, models). Called on platform startup."""

    async def teardown(self) -> None:
        """Release resources. Called on platform shutdown."""

    async def startup(self) -> None:
        """Framework hook: run :meth:`setup` and mark the ingestor ready."""
        await self.setup()
        self._ready = True

    async def shutdown(self) -> None:
        """Framework hook: run :meth:`teardown` and mark the ingestor down."""
        await self.teardown()
        self._ready = False

    # ------------------------------------------------------------------
    # Core contract
    # ------------------------------------------------------------------

    async def ingest(self, request: InputT, ctx: IngestionContext) -> IngestionResult:
        """Run the ingestion. **Override this.**

        Args:
            request: The parsed request model (defaults to
                :class:`IngestionRequest`).
            ctx: Progress/cancellation handle. Call ``await ctx.report(...)`` to
                emit progress and check ``ctx.cancelled`` in long loops.

        Returns:
            An :class:`IngestionResult` (or a dict with the same fields).
        """
        raise NotImplementedError("Ingestors must implement ingest()")

    async def run(
        self, request: Any = None, ctx: IngestionContext | None = None
    ) -> IngestionResult:
        """Convenience wrapper: coerce raw input, run :meth:`ingest`, normalise.

        Used by the REST router and the task dispatcher. Accepts a dict, a model
        instance, or ``None`` and always returns an :class:`IngestionResult`.
        """
        import time

        ctx = ctx or NullIngestionContext()
        model = self._coerce_request(request)
        t0 = time.perf_counter()
        result = await self.ingest(model, ctx)
        if isinstance(result, dict):
            result = IngestionResult(**result)
        if not result.ingestor:
            result.ingestor = self.ingestor_name
        if not result.duration_ms:
            result.duration_ms = (time.perf_counter() - t0) * 1000
        return result

    def _coerce_request(self, request: Any) -> Any:
        """Coerce raw input into this ingestor's input model."""
        schema = self.get_input_schema()
        if isinstance(request, schema):
            return request
        data = request if isinstance(request, dict) else {}
        try:
            return schema(**data)
        except Exception:  # noqa: BLE001 - fall back to the raw payload
            return request

    # ------------------------------------------------------------------
    # Schema extraction (mirrors plugins / endpoints)
    # ------------------------------------------------------------------

    def get_input_schema(self) -> type[BaseModel]:
        """Return the ``InputT`` Pydantic model (defaults to IngestionRequest)."""
        for base in getattr(self.__class__, "__orig_bases__", []):
            origin = typing.get_origin(base)
            if origin is BaseIngestor or origin is self.__class__:
                args = typing.get_args(base)
                if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                    return args[0]
        return IngestionRequest

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Return metadata describing this ingestor."""
        return {
            "name": self.ingestor_name,
            "description": self.ingestor_description,
            "version": self.ingestor_version,
            "ready": self._ready,
            "input_schema": self.get_input_schema().__name__,
        }

    async def health_check(self) -> dict[str, Any]:
        """Return the ingestor's health status."""
        return {
            "status": "healthy" if self._ready else "not_ready",
            "ingestor": self.ingestor_name,
            "version": self.ingestor_version,
        }
