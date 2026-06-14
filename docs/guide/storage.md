# Storage Backends

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Pluggable Persistence Stack</h3>
</div>

---

Agentomatic decouples chat persistence and session state from your agent logic. It uses a pluggable storage system where all storage backends inherit from the `BaseStore` abstract base class (ABC). This means you can build with an in-memory database locally and swap to an enterprise-grade PostgreSQL or custom Redis database in production by changing a single line of code.

---

## 🏗️ Storage Architecture & Models

Agentomatic’s data model consists of three core tables/entities:

1. **Threads**: A conversation session belonging to a unique user and agent. Tracks the session state, creation/update timestamps, and custom dictionary metadata.
2. **Messages**: Chronological logs of user questions, assistant responses, system instructions, and tool output traces within a specific thread.
3. **Feedback**: Ratings (1 to 5 stars), commentary, corrections, and query/response records submitted by users or API clients.

---

## 📦 Built-in Backends

### 1. `MemoryStore` (Development)

The default store when launching Agentomatic. It stores all threads, messages, and feedback in dictionary structures in RAM. 

- **Pros**: Zero-config, extremely fast, no external dependencies.
- **Cons**: Volatile (wiped on application restart), unsuitable for production.

```python
from agentomatic.storage import MemoryStore

store = MemoryStore()
```

### 2. `SQLAlchemyStore` (Production)

The production-ready store supporting any SQL database supported by SQLAlchemy's async driver (e.g., PostgreSQL, MySQL, SQLite, or CockroachDB).

- **Pros**: Relational ACID storage, indexed thread lookups, connection pooling.
- **Cons**: Requires setup and driver installations (e.g., `asyncpg` or `aiosqlite`).

#### PostgreSQL Configuration
```python
from agentomatic.storage import SQLAlchemyStore

store = SQLAlchemyStore(
    "postgresql+asyncpg://postgres:secret@localhost:5432/agent_db",
    pool_size=10,         # Minimum persistent connections
    max_overflow=20,      # Maximum temporary overflow connections
    pool_timeout=30.0,    # Seconds to wait for a connection from the pool
)
```

#### SQLite Configuration (Local file persistence)
```python
store = SQLAlchemyStore("sqlite+aiosqlite:///./data/agentomatic.db")
```

---

## 🛠️ Implementing a Custom Store (e.g., Redis)

To use a custom backend like Redis, MongoDB, or DynamoDB, subclass `BaseStore` and implement the abstract database operations:

```python
import json
import redis.asyncio as aioredis
from typing import Any
from agentomatic.storage import BaseStore

class RedisStore(BaseStore):
    def __init__(self, url: str) -> None:
        self.url = url
        self._redis: aioredis.Redis | None = None

    async def initialize(self) -> None:
        """Called automatically during platform startup."""
        self._redis = aioredis.from_url(self.url, decode_responses=True)

    async def close(self) -> None:
        """Called automatically during platform shutdown."""
        if self._redis:
            await self._redis.close()

    async def create_thread(
        self,
        thread_id: str,
        user_id: str,
        agent_name: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        thread = {
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_name": agent_name,
            "title": title or "New Conversation",
            "metadata": json.dumps(metadata or {}),
        }
        await self._redis.hset(f"thread:{thread_id}", mapping=thread)
        return thread

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        data = await self._redis.hgetall(f"thread:{thread_id}")
        if not data:
            return None
        data["metadata"] = json.loads(data["metadata"])
        return data

    async def list_threads(
        self,
        *,
        agent_name: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        # Scan and return matching thread hashes...
        return []

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message = {
            "role": role,
            "content": content,
            "metadata": json.dumps(metadata or {}),
        }
        await self._redis.rpush(f"messages:{thread_id}", json.dumps(message))
        return message

    async def get_messages(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        raw_msgs = await self._redis.lrange(f"messages:{thread_id}", offset, offset + limit - 1)
        return [json.loads(m) for m in raw_msgs]
```

---

## 🔌 Integrating the Store with your Platform

Pass your store instance directly to the `AgentPlatform` initialization. Agentomatic takes care of opening connection pools at startup, closing them gracefully at shutdown, and handling session context across all HTTP routes:

```python
from agentomatic import AgentPlatform
from agentomatic.storage import SQLAlchemyStore

# 1. Initialize the persistent backend
db_store = SQLAlchemyStore("postgresql+asyncpg://postgres:secret@localhost:5432/agent_db")

# 2. Bind the store to the Platform
platform = AgentPlatform.from_folder(
    folder_path="agents/",
    store=db_store,
)

app = platform.build()
```

---

## 🗂️ The `BaseStore` API Interface Reference

Every storage adapter implements the following async interface methods:

| Method Signature | Core / Option | Description |
|---|---|---|
| `async def initialize(self) -> None` | Optional | Async database pool setup |
| `async def close(self) -> None` | Optional | Graceful resource cleanup |
| `async def health_check(self) -> dict` | Optional | Used for liveness/readiness probe reports |
| `async def create_thread(...) -> dict` | **Required** | Register a new conversation session |
| `async def get_thread(id) -> dict \| None` | **Required** | Fetch a single thread |
| `async def list_threads(...) -> list[dict]` | **Required** | Search/paginate session listings |
| `async def update_thread(id, **kw) -> dict` | Optional | Update metadata or title |
| `async def delete_thread(id) -> bool` | Optional | Cascade delete thread and messages |
| `async def add_message(...) -> dict` | **Required** | Save a new query/response message |
| `async def get_messages(id) -> list[dict]` | **Required** | Fetch chronological chat histories |
| `async def add_feedback(...) -> dict` | Optional | Save user scores and comment metrics |
| `async def get_feedback(...) -> list[dict]` | Optional | Search feedback records |
| `async def get_stats() -> dict` | Optional | Return storage statistics (threads, messages, feedback counts) |
| `async def save_suspended_state(...) -> dict` | Optional | Save execution state for human approval (auto-sets 7-day TTL) |
| `async def get_suspended_state(id) -> dict \| None` | Optional | Retrieve a suspended state |
| `async def list_suspended_states(...) -> list[dict]` | Optional | List all pending suspended states |
| `async def delete_suspended_state(id) -> bool` | Optional | Delete suspended state on completion |
| `async def cleanup_expired_states() -> int` | Optional | Delete expired HITL states, returns count removed |
| `async def fork_thread(...) -> dict \| None` | Optional | Fork a conversation history up to index |
| `async def get_thread_lineage(id) -> dict` | Optional | Ancestry/descendant tree for a thread |
| `async def save_checkpoint(...) -> None` | Optional | Save a LangGraph checkpoint state |
| `async def get_checkpoint(...) -> dict \| None` | Optional | Retrieve a LangGraph checkpoint state |
| `async def list_checkpoints(...) -> list[dict]` | Optional | List checkpoints for a thread namespace |

> 🚦 *For details on how checkpointer, thread forking, and human-in-the-loop suspension integrate with these storage methods, see the [Advanced Platform Features Guide](platform-features.md).*

