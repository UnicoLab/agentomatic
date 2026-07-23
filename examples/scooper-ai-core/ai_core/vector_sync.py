"""Helpers to sync artifact embeddings into the active vector provider.

When ``AI_VECTOR_PROVIDER=azure_cosmos``, historical updates and seeds also
upsert into Cosmos Mongo so the similarity plugin / RAG connections hit the
same backend used in production.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from loguru import logger

from ai_core.settings import get_settings


def uses_cosmos_vector() -> bool:
    """Return whether the active vector provider is Azure Cosmos Mongo."""
    return get_settings().vector_provider.strip().lower() == "azure_cosmos"


def cosmos_connection_config() -> Any:
    """Build a :class:`VectorConnectionConfig` from environment."""
    from agentomatic.connections import VectorConnectionConfig

    settings = get_settings()
    collection = os.getenv("AI_COSMOS_COLLECTION") or os.getenv("AI_COSMOS_CONTAINER", "cases")
    return VectorConnectionConfig(
        name="cases_sync",
        provider="azure_cosmos",
        url=os.getenv("AI_COSMOS_CONNECTION_STRING", ""),
        api_key="",
        collection=collection,
        dimension=settings.embed_dimension,
        distance="cosine",
        options={
            "database": os.getenv("AI_COSMOS_DATABASE", "scooper"),
            "collection": collection,
            "connection_string": os.getenv("AI_COSMOS_CONNECTION_STRING", ""),
        },
    )


async def sync_cases_to_active_vector_store(
    *,
    texts: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Upsert case embeddings into the active Agentomatic vector connection.

    No-ops when the provider is ``local_npz``. For ``azure_cosmos`` (Mongo
    vCore), builds a store via the registered provider and upserts.

    Returns:
        Status dict (``synced``, ``provider``, ``count`` / ``error``).
    """
    settings = get_settings()
    provider = settings.vector_provider.strip().lower()
    if provider != "azure_cosmos":
        return {"synced": False, "provider": provider, "reason": "local artifact store"}

    try:
        from ai_core.cosmos_vector import AzureCosmosVectorStore, build_cosmos_client
        from ai_core.vectorstore import register_vector_backends

        register_vector_backends()
        cfg = cosmos_connection_config()
        client = build_cosmos_client(cfg)
        store = AzureCosmosVectorStore(cfg, client)
        # Ensure numpy rows become plain float lists.
        emb_list = [list(map(float, row)) for row in embeddings]
        written = await store.upsert(
            texts=list(texts),
            metadatas=list(metadatas),
            ids=list(ids),
            embeddings=emb_list,
        )
        logger.info(
            "Synced {} case embeddings to Cosmos Mongo (db={}, collection={})",
            len(written),
            os.getenv("AI_COSMOS_DATABASE", "scooper"),
            os.getenv("AI_COSMOS_COLLECTION") or os.getenv("AI_COSMOS_CONTAINER", "cases"),
        )
        return {"synced": True, "provider": provider, "count": len(written)}
    except Exception as exc:  # noqa: BLE001 - never fail promote on sync
        logger.warning("Cosmos Mongo vector sync failed: {}", exc)
        return {"synced": False, "provider": provider, "error": str(exc)}
