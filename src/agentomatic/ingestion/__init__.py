"""First-class ingestion / RAG ops layer.

Agentomatic packages *your* ingestion code — written with whatever libraries you
prefer — as a deployable, discoverable resource that runs synchronously, as a
background task with live progress, or on a queue, and shows up on the status
page. You implement :meth:`BaseIngestor.ingest`; Agentomatic handles the ops.

Example::

    from agentomatic.ingestion import BaseIngestor, IngestionResult

    class MyIngestor(BaseIngestor):
        ingestor_name = "my_docs"

        async def ingest(self, request, ctx) -> IngestionResult:
            # use any libraries you like (docling, unstructured, pymupdf, …)
            ...
            return IngestionResult(documents=1, chunks=42, upserted=42)
"""

from __future__ import annotations

from .base import BaseIngestor
from .context import IngestionContext, NullIngestionContext
from .formats import (
    MARKITDOWN_ACCEPT_ATTR,
    MARKITDOWN_EXTENSIONS,
    MARKITDOWN_MIME_SUFFIX,
    is_markitdown_extension,
)
from .models import IngestionRequest, IngestionResult
from .registry import IngestionRegistry
from .router import create_ingestion_router
from .text import (
    Chunk,
    IngestTextResult,
    Section,
    chunk_text,
    content_hash,
    convert_to_markdown,
    extract_sections,
    ingest_text,
    normalize_markdown,
    quality_score,
)

__all__ = [
    "MARKITDOWN_ACCEPT_ATTR",
    "MARKITDOWN_EXTENSIONS",
    "MARKITDOWN_MIME_SUFFIX",
    "BaseIngestor",
    "Chunk",
    "IngestTextResult",
    "IngestionContext",
    "IngestionRegistry",
    "IngestionRequest",
    "IngestionResult",
    "NullIngestionContext",
    "Section",
    "chunk_text",
    "content_hash",
    "convert_to_markdown",
    "create_ingestion_router",
    "extract_sections",
    "ingest_text",
    "is_markitdown_extension",
    "normalize_markdown",
    "quality_score",
]
