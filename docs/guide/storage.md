# Storage Backends

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Pluggable Persistence Stack</h3>
</div>

---

Agentomatic decouples chat persistence and session state from your agent logic. It uses a pluggable storage system where all backends inherit from the `BaseStore` abstract class. Build with in-memory storage locally, then swap to PostgreSQL in production by changing a single line.

---

## 🏗️ Storage Architecture

```mermaid
flowchart TB
    subgraph Platform["AgentPlatform"]
        direction TB
        Router["Router Layer<br/>/invoke, /chat, /threads"]
        Checkpointer["AgentomaticCheckpointer<br/>(LangGraph integration)"]
        Feedback["FeedbackCollector"]
    end

    subgraph Store["BaseStore Interface"]
        direction TB
        Threads["create/get/list/delete_thread"]
        Messages["add/get_messages"]
        FB["add/get_feedback"]
        HITL["save/get/list_suspended_state"]
        CP["save/get/list_checkpoints"]
        Fork["fork_thread / get_lineage"]
    end

    subgraph Backends["Storage Backends"]
        Memory["MemoryStore<br/>(development)"]
        SQL["SQLAlchemyStore<br/>(production)"]
        Custom["Custom Store<br/>(Redis, Mongo, etc.)"]
    end

    Router --> Store
    Checkpointer --> Store
    Feedback --> Store
    Store --> Memory
    Store --> SQL
    Store --> Custom
```

### Data Model

Agentomatic's data model consists of **five** core entities:

| Entity | Table | Description |
|--------|-------|-------------|
| **Thread** | `threads` | A conversation session — tracks owner, agent, timestamps, metadata, and fork lineage |
| **Message** | `messages` | Chronological log of user/assistant/system/tool messages within a thread |
| **Feedback** | `feedback` | User ratings (1–5), comments, corrections linked to threads |
| **SuspendedState** | `suspended_states` | Execution state saved for human-in-the-loop approval (7-day TTL) |
| **Checkpoint** | `checkpoints` | LangGraph execution checkpoints for graph state persistence |

---

## 📦 Backend Comparison

| Feature | `MemoryStore` | `SQLAlchemyStore` |
|---------|:------------:|:-----------------:|
| **Setup** | Zero-config | Requires database + driver |
| **Persistence** | ❌ Lost on restart | ✅ Durable |
| **Performance** | ⚡ Ultra-fast | 🚀 Fast (pooled) |
| **Concurrency** | Single-process | Multi-process safe |
| **ACID compliance** | ❌ | ✅ |
| **Thread forking** | ✅ | ✅ |
| **HITL suspension** | ✅ (in-memory) | ✅ (persisted) |
| **Checkpointing** | ✅ (in-memory) | ✅ (persisted) |
| **Connection pooling** | N/A | ✅ Configurable |
| **PostgreSQL** | — | ✅ `asyncpg` |
| **SQLite** | — | ✅ `aiosqlite` |
| **MySQL** | — | ✅ `asyncmy` |
| **Best for** | Development, testing | Production |

---

## 🧪 MemoryStore (Development)

Zero-configuration in-memory storage. All data is stored in Python dictionaries and lost when the process exits.

```python
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore

store = MemoryStore()

platform = AgentPlatform.from_folder(
    "agents/",
    store=store,
)
app = platform.build()
```

!!! warning "Volatility"
    `MemoryStore` is designed for development and testing only. All threads, messages, and feedback are **wiped** when the application restarts.

### Internal Data Structures

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_threads` | `dict[str, dict]` | Thread records keyed by thread ID |
| `_messages` | `dict[str, list[dict]]` | Message lists keyed by thread ID |
| `_feedback` | `list[dict]` | All feedback records |
| `_suspended_states` | `dict[str, dict]` | HITL states keyed by approval ID |
| `_checkpoints` | `dict[tuple, dict]` | Checkpoints keyed by (thread_id, ns, cp_id) |

---

## 🐘 SQLAlchemyStore (Production)

Production-ready async storage supporting any SQLAlchemy-compatible database. Features connection pooling, automatic table creation, and full ACID compliance.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"sqlite+aiosqlite:///data/platform.db"` | Database connection URL |
| `pool_size` | `int` | `10` | Number of persistent connections in the pool |
| `max_overflow` | `int` | `20` | Maximum temporary connections above `pool_size` |
| `pool_recycle` | `int` | `3600` | Recycle connections after N seconds |
| `pool_pre_ping` | `bool` | `True` | Test connections before use (prevents stale connections) |
| `echo` | `bool` | `False` | Log all SQL statements (debug mode) |

!!! info "SQLite Pool Settings"
    When using SQLite, `pool_size`, `max_overflow`, `pool_recycle`, and `pool_pre_ping` are automatically ignored since SQLite doesn't support connection pooling.

### Setup Examples

=== "PostgreSQL"

    ```python
    from agentomatic import AgentPlatform
    from agentomatic.storage import SQLAlchemyStore

    store = SQLAlchemyStore(
        "postgresql+asyncpg://postgres:secret@localhost:5432/agent_db",
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
        pool_pre_ping=True,
    )

    platform = AgentPlatform.from_folder("agents/", store=store)
    app = platform.build()
    ```

    **Required package:** `pip install asyncpg`

=== "SQLite (File)"

    ```python
    from agentomatic.storage import SQLAlchemyStore

    store = SQLAlchemyStore("sqlite+aiosqlite:///./data/agentomatic.db")
    ```

    **Required package:** `pip install aiosqlite`

=== "MySQL"

    ```python
    from agentomatic.storage import SQLAlchemyStore

    store = SQLAlchemyStore(
        "mysql+asyncmy://user:password@localhost:3306/agent_db",
        pool_size=10,
    )
    ```

    **Required package:** `pip install asyncmy`

### Database Table Schema

Tables are created automatically during `platform.build()` via the lifespan `initialize()` call:

```mermaid
erDiagram
    threads ||--o{ messages : contains
    threads ||--o{ feedback : receives
    threads ||--o{ suspended_states : suspends
    threads ||--o{ threads : forks

    threads {
        string id PK
        string user_id
        string agent_name
        string title
        json metadata_json
        datetime created_at
        datetime updated_at
        int message_count
        string parent_thread_id FK
        int fork_message_index
    }

    messages {
        int id PK
        string thread_id FK
        string role
        text content
        json metadata_json
        datetime created_at
    }

    feedback {
        int id PK
        string thread_id FK
        int message_id
        string user_id
        string agent_name
        int rating
        text comment
        string feedback_type
        datetime created_at
    }

    suspended_states {
        string id PK
        string thread_id FK
        string agent_name
        string node_name
        json state_json
        datetime created_at
        datetime expires_at
    }

    checkpoints {
        string thread_id PK
        string checkpoint_ns PK
        string checkpoint_id PK
        string parent_checkpoint_id
        json checkpoint_json
        json metadata_json
        datetime created_at
    }
```

---

## 🔌 Integrating the Store

Pass your store instance to the `AgentPlatform`. Agentomatic handles initialization at startup and cleanup at shutdown automatically:

```python
from agentomatic import AgentPlatform
from agentomatic.storage import SQLAlchemyStore

db_store = SQLAlchemyStore(
    "postgresql+asyncpg://postgres:secret@localhost:5432/agent_db"
)

platform = AgentPlatform.from_folder(
    "agents/",
    store=db_store,
)

app = platform.build()
# On startup:  await store.initialize() — creates tables, opens pool
# On shutdown: await store.close()      — disposes engine, releases connections
```

---

## 🗂️ Thread & Conversation Management

### Creating and Managing Threads

```python
# Threads are typically managed via the auto-generated API endpoints
# POST /api/v1/{agent_name}/threads      → Create thread
# GET  /api/v1/{agent_name}/threads      → List threads
# GET  /api/v1/{agent_name}/threads/{id} → Get thread
# DELETE /api/v1/{agent_name}/threads/{id} → Delete thread

# Programmatic usage:
thread = await store.create_thread(
    thread_id="thread_abc123",
    user_id="user_42",
    agent_name="qa_agent",
    title="Tax Questions",
    metadata={"department": "finance"},
)
```

### Thread Forking

Fork a conversation at any message index to create a branch:

```python
forked = await store.fork_thread(
    parent_thread_id="thread_abc123",
    message_index=5,          # Fork after message #5
    new_thread_id="thread_fork_001",
    title="Fork: Different approach",
)
# Copies the thread and messages 0–5 into a new thread
```

### Thread Lineage

Retrieve the full ancestry/descendant tree:

```python
lineage = await store.get_thread_lineage("thread_fork_001")
# {
#     "thread_id": "thread_fork_001",
#     "ancestors": [{"id": "thread_abc123", ...}],
#     "descendants": []
# }
```

---

## ✅ LangGraph Checkpointer Integration

Agentomatic provides `AgentomaticCheckpointer`, a LangGraph `BaseCheckpointSaver` that delegates all checkpoint storage to your configured `BaseStore` backend.

### How It Works

```mermaid
flowchart LR
    LG["LangGraph<br/>CompiledGraph"] -->|"save/load<br/>checkpoint"| CP["AgentomaticCheckpointer"]
    CP -->|"save_checkpoint()<br/>get_checkpoint()"| Store["BaseStore<br/>(Memory/SQL)"]
```

### Usage

```python
from agentomatic.storage import SQLAlchemyStore
from agentomatic.storage.checkpointer import AgentomaticCheckpointer

# 1. Create your store
store = SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/db")

# 2. Wrap it in a checkpointer
checkpointer = AgentomaticCheckpointer(store)

# 3. Compile your graph with the checkpointer
from langgraph.graph import StateGraph, END
from agentomatic import BaseAgentState

def build_graph():
    g = StateGraph(BaseAgentState)
    g.add_node("process", process_node)
    g.set_entry_point("process")
    g.add_edge("process", END)
    return g

graph = build_graph().compile(checkpointer=checkpointer)

# 4. Invoke with thread_id for state persistence
result = await graph.ainvoke(
    {"current_query": "Hello!"},
    config={"configurable": {"thread_id": "thread_123"}},
)
```

### Checkpointer API

The checkpointer implements the full LangGraph `BaseCheckpointSaver` interface:

| Method | Description |
|--------|-------------|
| `get_tuple(config)` | Retrieve a checkpoint (sync wrapper) |
| `aget_tuple(config)` | Retrieve a checkpoint (async) |
| `put(config, checkpoint, metadata, new_versions)` | Save a checkpoint (sync wrapper) |
| `aput(config, checkpoint, metadata, new_versions)` | Save a checkpoint (async) |
| `list(config, ...)` | List checkpoints (sync wrapper, returns `Iterator`) |
| `alist(config, ...)` | List checkpoints (async, returns `AsyncIterator`) |

!!! info "JSON Serialization"
    The checkpointer automatically ensures all checkpoint and metadata values are JSON-serializable using a `default=str` fallback. Custom objects, datetimes, and bytes are converted to their string representation.

---

## 🛠️ Implementing a Custom Backend

Subclass `BaseStore` and implement the abstract methods to create a custom backend:

### Required Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_thread` | `(thread_id, user_id, agent_name, *, title, metadata) → dict` | Create a conversation session |
| `get_thread` | `(thread_id) → dict \| None` | Retrieve a thread by ID |
| `list_threads` | `(*, agent_name, user_id, limit, offset) → list[dict]` | List/filter threads |
| `add_message` | `(thread_id, role, content, *, metadata) → dict` | Add a message to a thread |
| `get_messages` | `(thread_id, *, limit, offset) → list[dict]` | Get messages chronologically |

### Optional Methods

| Method | Description | Default |
|--------|-------------|---------|
| `initialize()` | Async setup (open connections) | No-op |
| `close()` | Graceful shutdown | No-op |
| `health_check()` | Liveness/readiness info | `{"status": "healthy"}` |
| `update_thread()` | Update thread metadata/title | Returns `None` |
| `delete_thread()` | Cascade delete thread + messages | Returns `False` |
| `add_feedback()` | Record user feedback | Returns stub dict |
| `get_feedback()` | Retrieve feedback records | Returns `[]` |
| `get_stats()` | Storage statistics | Returns backend name |
| `save_suspended_state()` | HITL state persistence | Returns `{}` |
| `get_suspended_state()` | HITL state retrieval | Returns `None` |
| `list_suspended_states()` | List pending HITL states | Returns `[]` |
| `delete_suspended_state()` | Delete completed HITL state | Returns `False` |
| `cleanup_expired_states()` | Remove expired HITL states | Returns `0` |
| `fork_thread()` | Fork a conversation | Returns `None` |
| `get_thread_lineage()` | Ancestry/descendant tree | Returns empty tree |
| `save_checkpoint()` | LangGraph checkpoint save | No-op |
| `get_checkpoint()` | LangGraph checkpoint load | Returns `None` |
| `list_checkpoints()` | LangGraph checkpoint listing | Returns `[]` |

### Example: Redis Backend

```python
import json
import redis.asyncio as aioredis
from typing import Any
from agentomatic.storage import BaseStore

class RedisStore(BaseStore):
    """Redis-backed storage for high-throughput scenarios."""

    def __init__(self, url: str = "redis://localhost:6379") -> None:
        self.url = url
        self._redis: aioredis.Redis | None = None

    async def initialize(self) -> None:
        """Open the Redis connection pool."""
        self._redis = aioredis.from_url(
            self.url, decode_responses=True
        )

    async def close(self) -> None:
        """Close all Redis connections."""
        if self._redis:
            await self._redis.close()

    async def health_check(self) -> dict[str, Any]:
        """Verify Redis connectivity."""
        try:
            await self._redis.ping()
            return {"status": "healthy", "backend": "RedisStore"}
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}

    async def create_thread(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from datetime import UTC, datetime

        thread = {
            "id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "title": title or "New Conversation",
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.now(UTC).isoformat(),
        }
        await self._redis.hset(f"thread:{thread_id}", mapping=thread)
        # Index for listing
        await self._redis.sadd(f"threads:{agent_name}", thread_id)
        await self._redis.sadd(f"user_threads:{user_id}", thread_id)
        return thread

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        data = await self._redis.hgetall(f"thread:{thread_id}")
        if not data:
            return None
        data["metadata"] = json.loads(data.get("metadata", "{}"))
        return data

    async def list_threads(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if agent_name:
            thread_ids = await self._redis.smembers(f"threads:{agent_name}")
        elif user_id:
            thread_ids = await self._redis.smembers(f"user_threads:{user_id}")
        else:
            thread_ids = await self._redis.keys("thread:*")
            thread_ids = [k.replace("thread:", "") for k in thread_ids]

        threads = []
        for tid in list(thread_ids)[offset:offset + limit]:
            thread = await self.get_thread(tid)
            if thread:
                threads.append(thread)
        return threads

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from datetime import UTC, datetime

        message = {
            "role": role,
            "content": content,
            "metadata": json.dumps(metadata or {}),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._redis.rpush(
            f"messages:{thread_id}", json.dumps(message)
        )
        return message

    async def get_messages(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        raw = await self._redis.lrange(
            f"messages:{thread_id}", offset, offset + limit - 1
        )
        return [json.loads(m) for m in raw]
```

### Using Your Custom Backend

```python
from agentomatic import AgentPlatform

store = RedisStore("redis://localhost:6379")

platform = AgentPlatform.from_folder("agents/", store=store)
app = platform.build()
```

---

## 📊 Storage Statistics

All backends implement a `get_stats()` method, exposed at `/api/v1/storage/stats`:

```http
GET /api/v1/storage/stats
```

```json
{
  "backend": "SQLAlchemyStore",
  "threads": 142,
  "messages": 8463,
  "feedback": 37
}
```

---

## 🔗 Further Reading

> 🚦 For details on how checkpointing, thread forking, and human-in-the-loop suspension integrate at the platform level, see the [Advanced Platform Features Guide](platform-features.md).

---

## 📚 Related Documentation

| Topic | Link |
|-------|------|
| Platform configuration | [Configuration](configuration.md) |
| Thread management endpoints | [API Reference](../architecture/api-reference.md) |
| HITL & suspended states | [Platform Features](platform-features.md) |
| Conversation memory | [Platform Features](platform-features.md) |
