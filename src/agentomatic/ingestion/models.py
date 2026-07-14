"""Data contracts for the ingestion / RAG ops layer.

Agentomatic provides the *packaging and ops* around ingestion — discovery,
REST endpoints, task/queue execution, progress, and status — while the actual
work (parsing PDFs to markdown, chunking, embedding, upserting to a vector DB)
is implemented by the user with whatever libraries they prefer.

These models are intentionally generic: :class:`IngestionRequest` is a sensible
default input (override it per ingestor), and :class:`IngestionResult` is a
flexible, telemetry-friendly outcome you can return from any implementation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    """Default input model for an ingestor (override with your own schema).

    Nothing here is mandatory — pick the fields that make sense for your source
    (a folder path, an S3 URI, a list of inline documents, a collection name,
    and arbitrary ``options`` passed straight to your implementation).
    """

    source: str | None = Field(
        default=None,
        description="Where to read from (path, glob, URL, bucket URI, …).",
    )
    documents: list[Any] | None = Field(
        default=None,
        description="Optional inline documents to ingest instead of a source.",
    )
    collection: str | None = Field(
        default=None,
        description="Target collection / index / namespace name.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form options forwarded to the ingestor implementation.",
    )


class IngestionResult(BaseModel):
    """Flexible, telemetry-friendly result returned by an ingestor.

    Populate the counters that are relevant to your pipeline and use ``stats``
    / ``output`` for anything else you want to surface to callers and the
    status page.
    """

    ingestor: str = ""
    status: str = "succeeded"
    documents: int = Field(default=0, description="Number of source documents processed.")
    chunks: int = Field(default=0, description="Number of chunks produced.")
    upserted: int = Field(default=0, description="Number of records written to the store.")
    skipped: int = Field(default=0, description="Number of records skipped (dedup, filters).")
    duration_ms: float = 0.0
    collection: str | None = None
    errors: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
