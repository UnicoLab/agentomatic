"""Minimal document-store backend for MongoDB / Cosmos-Mongo-vCore clients.

Provides :class:`MinimalDocumentStore`, a partial :class:`BaseStore`
implementation that persists conversation *threads* and *messages* in
two collections of a Mongo-compatible database.  It is intentionally
lightweight (no HITL, checkpointer, or advanced feedback features) but
usable as a memory backend for agents that already declare a Cosmos DB
for MongoDB (vCore) connection — or any other client exposing the same
dict-like ``db[collection]`` interface.

The store accepts any object exposing::

    client[database_name][collection_name].insert_one(...)
    client[database_name][collection_name].find(...)
    client[database_name][collection_name].find_one(...)
    client[database_name][collection_name].delete_one(...)
    client[database_name][collection_name].update_one(...)

The methods may be sync or async; the store transparently awaits either.

Example::

    from agentomatic.storage.document import MinimalDocumentStore

    from pymongo import AsyncMongoClient  # or a Cosmos-vCore client

    client = AsyncMongoClient("mongodb+srv://...")
    store = MinimalDocumentStore(client, database="agentomatic")
    await store.initialize()
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any

from agentomatic.storage.base import BaseStore


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is a coroutine / awaitable; return it otherwise."""
    if inspect.isawaitable(value):
        return await value
    return value


class MinimalDocumentStore(BaseStore):
    """Partial document-DB backed :class:`BaseStore` for thread/message data.

    Only the *threads* + *messages* surface is persisted — feedback,
    HITL suspended states, and checkpoints fall back to the (in-memory)
    stubs from :class:`BaseStore`.  Use :class:`MinimalDocumentStore`
    where you want to reuse a Cosmos-Mongo-vCore (or any MongoDB) client
    as conversation memory without bringing SQLAlchemy along.

    Args:
        client: A dict-like async MongoDB client (indexable by database
            name, then collection name).
        database: Database name.  Defaults to ``"agentomatic"``.
        threads_collection: Collection name for threads.
        messages_collection: Collection name for messages.
    """

    def __init__(
        self,
        client: Any,
        *,
        database: str = "agentomatic",
        threads_collection: str = "threads",
        messages_collection: str = "messages",
    ) -> None:
        self._client = client
        self._database_name = database
        self._threads_name = threads_collection
        self._messages_name = messages_collection

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def _db(self) -> Any:
        return self._client[self._database_name]

    def _threads(self) -> Any:
        return self._db()[self._threads_name]

    def _messages(self) -> Any:
        return self._db()[self._messages_name]

    async def initialize(self) -> None:
        """No-op; the underlying client is expected to be ready."""

    async def close(self) -> None:
        """Close the underlying client if it exposes a close method."""
        for attr in ("aclose", "close", "disconnect"):
            fn = getattr(self._client, attr, None)
            if callable(fn):
                try:
                    await _maybe_await(fn())
                except Exception:  # noqa: BLE001
                    pass
                break

    async def health_check(self) -> dict[str, Any]:
        """Best-effort health check (client-level)."""
        return {"status": "healthy", "backend": self.__class__.__name__}

    # ------------------------------------------------------------------
    # Thread operations
    # ------------------------------------------------------------------

    async def create_thread(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert a new thread document."""
        now = datetime.now(UTC).isoformat()
        thread = {
            "_id": thread_id,
            "id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "title": title,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
        await _maybe_await(self._threads().insert_one(thread))
        return {k: v for k, v in thread.items() if k != "_id"}

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Fetch one thread by ID."""
        doc = await _maybe_await(self._threads().find_one({"_id": thread_id}))
        if doc is None:
            return None
        doc.pop("_id", None)
        return dict(doc)

    async def list_threads(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List threads with optional filtering."""
        query: dict[str, Any] = {}
        if agent_name:
            query["agent_name"] = agent_name
        if user_id:
            query["user_id"] = user_id
        cursor = self._threads().find(query)
        docs = await _collect_cursor(cursor, limit=limit, offset=offset)
        for doc in docs:
            doc.pop("_id", None)
        return docs

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and its messages."""
        result = await _maybe_await(self._threads().delete_one({"_id": thread_id}))
        await _maybe_await(self._messages().delete_many({"thread_id": thread_id}))
        deleted = getattr(result, "deleted_count", 0) or 0
        return int(deleted) > 0

    async def update_thread(
        self,
        thread_id: str,
        **updates: Any,
    ) -> dict[str, Any] | None:
        """Update thread fields; returns the updated doc or None."""
        updates["updated_at"] = datetime.now(UTC).isoformat()
        await _maybe_await(self._threads().update_one({"_id": thread_id}, {"$set": dict(updates)}))
        return await self.get_thread(thread_id)

    # ------------------------------------------------------------------
    # Message operations
    # ------------------------------------------------------------------

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert a new message and bump the thread's counter."""
        now = datetime.now(UTC).isoformat()
        existing = await self.get_messages(thread_id, limit=10_000)
        msg = {
            "id": len(existing) + 1,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": now,
        }
        await _maybe_await(self._messages().insert_one(dict(msg)))
        await _maybe_await(
            self._threads().update_one(
                {"_id": thread_id},
                {"$inc": {"message_count": 1}, "$set": {"updated_at": now}},
            )
        )
        return msg

    async def get_messages(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return messages for a thread in insertion order."""
        cursor = self._messages().find({"thread_id": thread_id})
        docs = await _collect_cursor(cursor, limit=limit, offset=offset)
        for doc in docs:
            doc.pop("_id", None)
        return docs

    async def get_stats(self) -> dict[str, Any]:
        """Return best-effort counts (no queries when the client lacks support)."""
        return {
            "backend": self.__class__.__name__,
            "database": self._database_name,
        }


async def _collect_cursor(cursor: Any, *, limit: int, offset: int) -> list[dict[str, Any]]:
    """Materialise a Mongo-style cursor into a list of dicts.

    Handles both async iterators (pymongo async / motor) and sync cursors
    that support ``skip`` / ``limit`` chaining, so the same store works
    with several client flavours.
    """
    if hasattr(cursor, "skip") and hasattr(cursor, "limit"):
        try:
            cursor = cursor.skip(int(offset)).limit(int(limit))
        except Exception:  # noqa: BLE001
            pass

    if hasattr(cursor, "to_list"):
        try:
            docs = await _maybe_await(cursor.to_list(length=limit))
            return [dict(d) for d in docs]
        except TypeError:
            docs = await _maybe_await(cursor.to_list())
            return [dict(d) for d in docs]

    if hasattr(cursor, "__aiter__"):
        collected: list[dict[str, Any]] = []
        async for doc in cursor:
            collected.append(dict(doc))
        return collected[offset : offset + limit] if not hasattr(cursor, "skip") else collected

    try:
        docs = list(cursor)
    except TypeError:
        docs = []
    return [dict(d) for d in docs[offset : offset + limit]]
