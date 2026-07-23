# Custom Database & Vector Store Connections

Agentomatic is **provider-agnostic**: it ships the *ops* (config, lifecycle,
`${ENV}` interpolation, the async `VectorStore` protocol, scoping, health
checks) and you bring the *client*. Any Python client package — an official
vendor SDK, an in-house wrapper, or a thin `httpx` shim — can be wired in as a
first-class connection with correct startup/shutdown lifecycle.

!!! info "No first-party vendor connectors"
    The core never ships vendor-specific connectors (no Cosmos, no proprietary
    backends). You register your client against a stable, documented surface —
    so you stay in control of versions, auth, and dependencies.

## Built-in: local `.npz` (zero-infra RAG)

For demos, tests, and artifact bundles that need an on-disk store without
Qdrant/Chroma, use the built-in ``local_npz`` provider
(requires ``pip install numpy`` or ``agentomatic[vector]``):

```python
from agentomatic.connections import (
    ConnectionPurpose,
    TextEncoder,
    VectorConnectionConfig,
    get_connections,
    initialize_connections,
)

await initialize_connections(
    "rag_agent",
    [
        VectorConnectionConfig(
            name="kb",
            provider="local_npz",
            url=".local/vectors/kb",  # directory for embeddings.npz + metadata.jsonl
            purpose=ConnectionPurpose.VECTOR,
        )
    ],
)
store = await get_connections("rag_agent").vector("kb").as_store()
await store.upsert(texts=["Agentomatic is batteries-included"], ids=["1"])
hits = await store.query(text="batteries", k=3)
```

:class:`~agentomatic.connections.TextEncoder` overlays stack embedding
settings when the provider is real; otherwise it falls back to a
deterministic hash embedder so offline tests keep working.

## The 3-layer model

Every custom integration follows the same three layers. Keep them separate and
the rest of the platform (RAG nodes, memory, health checks) just works.

```
┌─ 1. Connection (config + lifecycle) ─────────────────────────────┐
│   VectorConnectionConfig / CustomConnectionConfig                 │
│   builds ONE client per process, lazily, and closes it on         │
│   shutdown. Credentials come from ${ENV}.                         │
└───────────────────────────────────────────────────────────────────┘
┌─ 2. Domain adapter (provider-agnostic surface) ──────────────────┐
│   A VectorStore adapter (upsert/query/delete) OR a plain client   │
│   you call directly. Maps the agnostic API onto your SDK.         │
└───────────────────────────────────────────────────────────────────┘
┌─ 3. Agent node usage ────────────────────────────────────────────┐
│   store = await get_connections(self.agent_name).vector("kb")     │
│                 .as_store()                                       │
│   hits = await store.query(text=..., k=3)                         │
└───────────────────────────────────────────────────────────────────┘
```

| Layer | Built-in registration hook | You provide |
| --- | --- | --- |
| Vector client | `register_vector_provider(name, builder)` | a builder `cfg -> client` |
| Vector adapter | `register_vector_store_adapter(name, adapter)` | `(cfg, client) -> VectorStore` |
| Any backend | `CustomConnectionConfig(factory=...)` | a factory callable |
| Any conn. class | `register_connection_type(cfg_cls, builder)` | a lifecycle object |

!!! danger "Build the client in the builder/factory — never in `__init__`"
    Do **not** create the client in `BaseGraphAgent.__init__`. Agents are
    instantiated eagerly (discovery, optimization, tests) and a client created
    there leaks sockets, isn't closed on shutdown, and breaks per-worker
    scaling. Always build it in the provider builder / factory so there is
    **one client per process, built lazily, closed on shutdown**.

---

## Example A — a custom **async** vector database

Register a provider builder that returns your async client, plus an adapter that
maps `upsert`/`query`/`delete` onto its API.

=== "`providers.py` (register once, on import)"

    ```python
    from __future__ import annotations

    from typing import Any

    from agentomatic.connections import (
        VectorConnectionConfig,
        register_vector_provider,
        register_vector_store_adapter,
    )
    from agentomatic.endpoints.auth import resolve_env


    def build_my_db(cfg: VectorConnectionConfig) -> Any:
        """Return your native async client (one per process)."""
        from my_vector_sdk import AsyncClient  # your package

        return AsyncClient(
            url=resolve_env(cfg.url),
            api_key=resolve_env(cfg.api_key) or None,
            **cfg.options,  # extra vendor knobs
        )


    class MyDbStore:
        """Map the provider-agnostic VectorStore surface onto the SDK."""

        def __init__(self, cfg: VectorConnectionConfig, client: Any) -> None:
            self._cfg = cfg
            self._client = client

        async def upsert(self, texts, metadatas=None, ids=None, *, embeddings=None):
            ids = list(ids or [str(i) for i in range(len(texts))])
            await self._client.add(
                collection=self._cfg.collection,
                ids=ids,
                documents=list(texts),
                metadatas=list(metadatas or [{} for _ in texts]),
            )
            return ids

        async def query(self, text=None, *, embedding=None, k=5, filter=None):
            res = await self._client.search(
                collection=self._cfg.collection, query=text, top_k=k, where=filter
            )
            return [
                {"id": r.id, "score": r.score, "text": r.document, "metadata": r.metadata}
                for r in res
            ]

        async def delete(self, ids):
            await self._client.remove(collection=self._cfg.collection, ids=list(ids))


    register_vector_provider("my_db", build_my_db)
    register_vector_store_adapter("my_db", MyDbStore)
    ```

=== "`connections.py` (declare per agent)"

    ```python
    from __future__ import annotations

    import my_agent.providers  # noqa: F401  — registers "my_db" on import

    from agentomatic.connections import ConnectionPurpose, VectorConnectionConfig

    CONNECTIONS = [
        VectorConnectionConfig(
            name="kb",
            provider="my_db",              # any registered name — not just built-ins
            url="${MY_DB_URL}",
            api_key="${MY_DB_API_KEY}",
            collection="knowledge_base",
            purpose=ConnectionPurpose.RAG,
            options={"tenant": "${MY_DB_TENANT}"},
        )
    ]
    ```

=== "`agent.py` (use in a RAG node)"

    ```python
    from __future__ import annotations

    from agentomatic import BaseGraphAgent, get_connections


    class RagAgent(BaseGraphAgent):
        agent_name = "rag_agent"

        async def retrieve(self, state):
            store = await get_connections(self.agent_name).vector("kb").as_store()
            hits = await store.query(text=state["question"], k=3)
            state["context"] = [h["text"] for h in hits]
            return state
    ```

The connection is **initialised on startup** and **closed on shutdown**
automatically when served by `AgentPlatform` / `agentomatic run` (the platform
discovers `connections.py`). `close()` accepts any client: it calls the first of
`aclose` / `close` / `disconnect` that exists, awaiting it if needed.

---

## Example B — a **sync-only** SDK (don't block the event loop)

Many database SDKs are synchronous. Wrap each blocking call in
`asyncio.to_thread` **at the adapter boundary** so the event loop stays free.

```python
from __future__ import annotations

import asyncio
from typing import Any

from agentomatic.connections import (
    VectorConnectionConfig,
    register_vector_provider,
    register_vector_store_adapter,
)
from agentomatic.endpoints.auth import resolve_env


def build_sync_db(cfg: VectorConnectionConfig) -> Any:
    from legacy_sdk import Client  # sync-only

    return Client(dsn=resolve_env(cfg.url))


class SyncDbStore:
    def __init__(self, cfg: VectorConnectionConfig, client: Any) -> None:
        self._cfg = cfg
        self._client = client

    async def upsert(self, texts, metadatas=None, ids=None, *, embeddings=None):
        ids = list(ids or [str(i) for i in range(len(texts))])
        # offload the blocking call to a worker thread
        await asyncio.to_thread(
            self._client.insert, self._cfg.collection, ids, list(texts)
        )
        return ids

    async def query(self, text=None, *, embedding=None, k=5, filter=None):
        rows = await asyncio.to_thread(
            self._client.search, self._cfg.collection, text, k
        )
        return [{"id": r["id"], "score": r["score"], "text": r["doc"], "metadata": {}} for r in rows]

    async def delete(self, ids):
        await asyncio.to_thread(self._client.delete, self._cfg.collection, list(ids))


register_vector_provider("sync_db", build_sync_db)
register_vector_store_adapter("sync_db", SyncDbStore)
```

!!! tip "Rule of thumb"
    If a client method does I/O and is **not** `async`, wrap it in
    `asyncio.to_thread(...)`. Never call a blocking method directly inside an
    `async def` node.

---

## Example C — a non-vector specialised DB via a factory

For graph DBs, time-series stores, feature stores, caches — anything that
isn't a vector store — use `CustomConnectionConfig(factory=...)`. No class
required; the factory returns a client and Agentomatic manages its lifecycle.

=== "`connections.py`"

    ```python
    from __future__ import annotations

    from agentomatic.connections import ConnectionPurpose, CustomConnectionConfig

    CONNECTIONS = [
        CustomConnectionConfig(
            name="graph",
            factory="neo4j:AsyncGraphDatabase.driver",  # dotted path or a callable
            args=["${NEO4J_URI}"],
            kwargs={"auth": ("${NEO4J_USER}", "${NEO4J_PASSWORD}")},
            purpose=ConnectionPurpose.GENERAL,
            # close_method="close",   # auto-detected: aclose/close/disconnect
        )
    ]
    ```

=== "`agent.py`"

    ```python
    from agentomatic import get_connections


    async def expand(self, state):
        driver = await get_connections(self.agent_name).client("graph")
        async with driver.session() as session:
            result = await session.run("MATCH (n)-[:REL]->(m) RETURN m LIMIT 10")
            state["neighbors"] = [r["m"] async for r in result]
        return state
    ```

Need a full connection class (custom health checks, pooling)? Register the type
instead:

```python
from agentomatic.connections import register_connection_type

register_connection_type(MyTimeseriesConfig, MyTimeseriesConnection)
# builder(config) -> object with async initialize()/health_check()/close() + .name
```

---

## Secrets & environment (`${ENV}`)

Every string field (and `CustomConnectionConfig` `args`/`kwargs`, recursively)
supports `${VAR}` interpolation, resolved at connect time — secrets never live
in code.

```bash
# .env.example
MY_DB_URL=https://vectors.example.com
MY_DB_API_KEY=change-me
MY_DB_TENANT=acme
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change-me
```

In containers, pass these as environment variables (see
[Production Deployment](deployment.md)); the same `connections.py` works in dev
and in production without edits.

## Scoping: per-agent vs shared

- **Per-agent** (recommended): put `CONNECTIONS` in the agent's
  `connections.py`. The scope is the agent name; retrieve with
  `get_connections("agent_name")`.
- **Platform-wide** (shared): pass `connections=[...]` to `AgentPlatform(...)`;
  the scope is `PLATFORM_SCOPE`, retrieved with `get_connections()`.

Find backends by intent regardless of kind:

```python
from agentomatic.connections import ConnectionPurpose, get_connections

conns = get_connections("rag_agent")
for name, conn in conns.by_purpose(ConnectionPurpose.RAG).items():
    ...  # every RAG backend (vector + database + http)
```

## Lifecycle

- **One client per process**, built **lazily** on first use.
- **Closed on shutdown** — `close()` tries `aclose` → `close` → `disconnect`.
- **Idempotent init** — building twice is a no-op.
- **Never** construct clients in `BaseGraphAgent.__init__` (see the warning
  above).

## Standalone runs (outside `AgentPlatform`)

`AgentPlatform` / `agentomatic run` register **and** initialise connections for
you. A bare `get_graph()` under `langgraph dev`, a notebook, or a script has no
platform lifecycle, so wire connections yourself with the one-call helper:

```python
from agentomatic.connections import initialize_connections
from my_agent.connections import CONNECTIONS

# in an app factory / lifespan hook / before first use
await initialize_connections("rag_agent", CONNECTIONS)

# now agent code can use them:
store = await get_connections("rag_agent").vector("kb").as_store()
```

`initialize_connections(scope, configs)` registers the configs and awaits their
initialisation, returning the `ConnectionManager`. It is idempotent per client.
Close them on teardown with `await get_connections("rag_agent").close()`.

## Testing & mocking

No network needed — register a fake client under a novel provider name and
drive the agnostic surface:

```python
from __future__ import annotations

import pytest

from agentomatic.connections import (
    VectorConnectionConfig,
    get_connections,
    initialize_connections,
    register_vector_provider,
    register_vector_store_adapter,
)
from agentomatic.connections.manager import reset_connections


@pytest.fixture(autouse=True)
def _clean():
    reset_connections()
    yield
    reset_connections()


async def test_custom_store_roundtrip():
    class FakeClient:
        def __init__(self): self.docs = {}
        async def aclose(self): ...

    class FakeStore:
        def __init__(self, cfg, client): self.c = client
        async def upsert(self, texts, metadatas=None, ids=None, *, embeddings=None):
            for i, t in zip(ids, texts): self.c.docs[i] = t
            return list(ids)
        async def query(self, text=None, *, embedding=None, k=5, filter=None):
            return [{"id": i, "text": t, "score": 1.0, "metadata": {}}
                    for i, t in self.c.docs.items() if text in t][:k]
        async def delete(self, ids):
            for i in ids: self.c.docs.pop(i, None)

    register_vector_provider("fake", lambda cfg: FakeClient())
    register_vector_store_adapter("fake", FakeStore)

    await initialize_connections(
        "agent", [VectorConnectionConfig(name="kb", provider="fake")]
    )
    store = await get_connections("agent").vector("kb").as_store()
    await store.upsert(texts=["hello"], ids=["1"])
    assert (await store.query(text="hello"))[0]["id"] == "1"
```

See `tests/test_custom_client_integration.py` in the repo for full async, sync,
factory, `register_connection_type`, and shutdown-close examples.

## See also

- [Per-Agent Connections](connections.md) — the full connections reference.
- [Ingestion & RAG](ingestion.md) — building retrieval pipelines.
- [Production Deployment](deployment.md) — passing `${ENV}` into containers.
