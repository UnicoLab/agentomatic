# Storage Backends

Agentomatic uses a pluggable storage system. All backends implement the `BaseStore` ABC.

## Built-in Backends

### MemoryStore (Development)

```python
from agentomatic.storage import MemoryStore

store = MemoryStore()
```

### SQLAlchemyStore (Production)

```python
from agentomatic.storage import SQLAlchemyStore

# PostgreSQL
store = SQLAlchemyStore(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=10,
    max_overflow=20,
)

# SQLite
store = SQLAlchemyStore("sqlite+aiosqlite:///data/app.db")
```

## Custom Backend

```python
from agentomatic.storage import BaseStore

class RedisStore(BaseStore):
    async def initialize(self):
        self.redis = aioredis.from_url("redis://localhost")

    async def create_thread(self, thread_id, user_id, agent_name, **kw):
        await self.redis.hset(f"thread:{thread_id}", mapping={...})
        return {...}

    async def get_thread(self, thread_id):
        data = await self.redis.hgetall(f"thread:{thread_id}")
        return dict(data) if data else None

    # ... implement remaining abstract methods
```

## Storage Protocol

| Method | Required | Description |
|---|---|---|
| `create_thread()` | ✅ | Create conversation thread |
| `get_thread()` | ✅ | Get thread by ID |
| `list_threads()` | ✅ | List threads with filters |
| `add_message()` | ✅ | Add message to thread |
| `get_messages()` | ✅ | Get thread messages |
| `delete_thread()` | Optional | Delete thread + messages |
| `add_feedback()` | Optional | Store user feedback |
| `get_feedback()` | Optional | Retrieve feedback |
| `health_check()` | Optional | Backend health info |
| `get_stats()` | Optional | Backend statistics |
