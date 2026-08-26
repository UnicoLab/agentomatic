"""Minimal, typed ingestion resource used in deployment tests."""

from typing import Any

from pydantic import BaseModel

from agentomatic.ingestion import BaseIngestor, IngestionResult


class DocumentsInput(BaseModel):
    """Required deployment input used by the real task-contract probe."""

    source: str


class DocumentsIngestor(BaseIngestor[DocumentsInput]):
    ingestor_name = "documents"
    ingestor_description = "Deterministic fixture ingestor."

    async def ingest(self, request: DocumentsInput, _ctx: Any) -> IngestionResult:
        return IngestionResult(documents=1, chunks=1, upserted=1, collection="production")
