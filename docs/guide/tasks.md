# Tasks & Execution Modes

Agentomatic ships a **unified task subsystem** that lets you run *any* platform
resource — an agent, an ML plugin, a pipeline, or a custom endpoint — in
**synchronous**, **asynchronous (background)**, **batch**, or **streaming**
modes, all through one consistent API.

Every submission returns the same [`TaskRecord`](#taskrecord) shape, so your
frontend can submit work, poll status, subscribe to live progress, fetch the
result, or cancel — regardless of what runs underneath.

!!! tip "Why one task API?"
    Instead of sprinkling bespoke async endpoints on every resource, the task
    board gives you a single, uniform contract for *submit → track → result →
    cancel*. It is the backbone for long-running work such as document
    ingestion, batch scoring, and multi-step pipelines.

## Capabilities

| Capability | Supported |
|---|---|
| Synchronous (blocking) execution | ✅ |
| Asynchronous background execution + task id | ✅ |
| Status polling (`queued`/`running`/`succeeded`/`failed`/`cancelled`) | ✅ |
| Live progress (percent / message / stage) over SSE | ✅ |
| Bounded in-process queue (concurrency limit) | ✅ |
| Batch fan-out with per-item progress | ✅ |
| Cancellation | ✅ |
| Completion webhooks (`callback_url`) | ✅ |
| A2A task lifecycle (real, pollable) | ✅ |
| Pluggable persistence (`TaskStore`) | ✅ |

## Per-resource execution modes

You don't have to use the task board directly. Every resource also gets
**ergonomic `/async` and `/batch` companions** next to its synchronous route,
all backed by the same task system. Submit to these and you get back a `202`
with a task id and `links` to poll, stream, or cancel via the task board.

| Resource | Sync | Async | Batch |
|----------|------|-------|-------|
| Agent | `POST /api/v1/{agent}/invoke` | `.../invoke/async` | `.../invoke/batch` |
| Plugin | `POST /api/v1/plugins/{name}/predict` | `.../predict/async` | `.../predict/batch` |
| Pipeline | `POST /api/v1/pipelines/{name}/run` | `.../run/async` | `.../run/batch` |
| Ingestor | `POST /api/v1/ingestion/{name}/run` | `.../run/async` | `.../run/batch` |
| Endpoint | `POST /api/v1/endpoints/{name}{path}` | `.../{path}/async` | `.../{path}/batch` |

The **async** body is identical to the sync body. The **batch** body wraps a
list of those inputs:

```bash
# Fan out one agent over many inputs as a single batch task
curl -X POST http://localhost:8000/api/v1/researcher/invoke/batch \
  -H 'content-type: application/json' \
  -d '{"inputs": [{"query": "a"}, {"query": "b"}, {"query": "c"}],
       "callback_url": "https://example.com/webhook"}'
# -> 202 { "id": "task_...", "mode": "batch", "links": { ... } }
```

Poll the returned task id on the task board just like any other task; a batch
task's `result` is the ordered list of per-item results.

## The task board API

The task board is mounted at `{api_prefix}/tasks` (default `/api/v1/tasks`) and
is enabled by default (`enable_tasks=True`).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks` | Submit work (`202` async, `200` when `wait=true`) |
| `GET` | `/tasks` | List / filter tasks |
| `GET` | `/tasks/{id}` | Poll status + progress |
| `GET` | `/tasks/{id}/result` | Fetch the result (`409` if still pending) |
| `GET` | `/tasks/{id}/events` | Live SSE progress stream |
| `POST` | `/tasks/{id}/cancel` | Request cancellation |
| `DELETE` | `/tasks/{id}` | Delete a terminal record |

### Submitting a task

=== "Async (default)"

    ```bash
    curl -X POST http://localhost:8000/api/v1/tasks \
      -H 'content-type: application/json' \
      -d '{
        "target_type": "agent",
        "target": "researcher",
        "input": {"query": "What is Agentomatic?"}
      }'
    ```

    Responds `202` immediately:

    ```json
    {
      "id": "task_a1b2c3d4e5f6a7b8",
      "target_type": "agent",
      "target": "researcher",
      "status": "queued",
      "progress": {"percent": null, "message": "", "current": 0, "total": null},
      "links": {
        "self": "tasks/task_a1b2c3d4e5f6a7b8",
        "result": "tasks/task_a1b2c3d4e5f6a7b8/result",
        "events": "tasks/task_a1b2c3d4e5f6a7b8/events",
        "cancel": "tasks/task_a1b2c3d4e5f6a7b8/cancel"
      }
    }
    ```

=== "Synchronous"

    Set `wait: true` (or `mode: "sync"`) to block until the task is terminal.
    The endpoint returns `200` with the full record including `result`.

    ```bash
    curl -X POST http://localhost:8000/api/v1/tasks \
      -H 'content-type: application/json' \
      -d '{
        "target_type": "plugin",
        "target": "sentiment_analyzer",
        "input": {"text": "I love this"},
        "wait": true,
        "timeout": 30
      }'
    ```

=== "Batch"

    Provide `batch` (a list of inputs). Items run with bounded concurrency and
    the task reports `done/total` progress. The result is a list aligned to the
    inputs (failed items become `{"error": "..."}`).

    ```bash
    curl -X POST http://localhost:8000/api/v1/tasks \
      -H 'content-type: application/json' \
      -d '{
        "target_type": "agent",
        "target": "classifier",
        "batch": [{"query": "a"}, {"query": "b"}, {"query": "c"}]
      }'
    ```

=== "With webhook"

    Provide `callback_url` to have the final record POSTed to your service on
    completion — no polling required.

    ```bash
    curl -X POST http://localhost:8000/api/v1/tasks \
      -H 'content-type: application/json' \
      -d '{
        "target_type": "pipeline",
        "target": "ingest_docs",
        "input": {"source": "s3://bucket/docs/"},
        "callback_url": "https://myapp.example.com/hooks/task-done"
      }'
    ```

### Polling status

```bash
curl http://localhost:8000/api/v1/tasks/task_a1b2c3d4e5f6a7b8
```

```json
{
  "id": "task_a1b2c3d4e5f6a7b8",
  "status": "running",
  "progress": {"percent": 66.6, "message": "Completed 2/3 batch items", "current": 2, "total": 3},
  "duration_ms": 812.4
}
```

### Streaming progress (SSE)

Subscribe to `/tasks/{id}/events` for live updates. Each frame is a
`TaskEvent`; the stream ends with `data: [DONE]`.

```javascript
const es = new EventSource("/api/v1/tasks/task_a1b2c3d4e5f6a7b8/events");
es.onmessage = (e) => {
  if (e.data === "[DONE]") { es.close(); return; }
  const evt = JSON.parse(e.data);
  console.log(evt.status, evt.progress?.percent, evt.progress?.message);
};
```

### Fetching the result

```bash
curl http://localhost:8000/api/v1/tasks/task_a1b2c3d4e5f6a7b8/result
```

* `200` with `{"task_id": ..., "result": ...}` when succeeded.
* `409` while the task is still pending.
* `422` if the task failed or was cancelled (the error is in `detail`).

### Cancelling

```bash
curl -X POST http://localhost:8000/api/v1/tasks/task_a1b2c3d4e5f6a7b8/cancel
```

Running work is cancelled cooperatively; the record transitions to `cancelled`.

## A2A task lifecycle

When the task subsystem is enabled, each agent's A2A endpoints run through the
task manager and expose a **real, pollable lifecycle** (previously a stub):

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/{agent}/a2a/tasks` | Submit an A2A task (async) |
| `GET` | `/api/v1/{agent}/a2a/tasks/{id}` | A2A task status |
| `POST` | `/api/v1/{agent}/a2a/tasks/{id}/cancel` | Cancel the task |

Statuses are mapped to canonical A2A states: `submitted`, `working`,
`completed`, `failed`, `canceled`.

## Programmatic use

```python
from agentomatic import TaskManager, TargetType
from agentomatic.tasks.dispatchers import make_agent_dispatcher

manager = TaskManager(max_concurrency=8)
manager.register_dispatcher(TargetType.AGENT, make_agent_dispatcher(registry))

record = await manager.submit(TargetType.AGENT, "researcher", input={"query": "hi"})
status = await manager.get(record.id)

# Or block until done:
done = await manager.submit_and_wait(TargetType.AGENT, "researcher", input={"query": "hi"})
print(done.status, done.result)
```

### Reporting progress from your own code

Every dispatcher (and each ingestor's `ingest`) receives a **`TaskContext`** —
use it to publish live progress (surfaced via `/tasks/{id}` and the SSE
`/events` stream) and to honour cancellation:

```python
from agentomatic.tasks import TaskContext

async def dispatcher(target: str, payload, ctx: TaskContext):
    total = len(payload["items"])
    for i, item in enumerate(payload["items"], start=1):
        if ctx.cancelled:            # cooperative cancellation
            break
        await ctx.report(percent=100 * i / total, message=f"item {i}/{total}",
                         current=i, total=total)
        ...
    return {"processed": total}
```

Nested graph nodes and pipeline step functions often **do not** receive the
``TaskContext`` argument. The platform installs a ContextVar bridge so you can
report from anywhere under a task run:

```python
from agentomatic.tasks import report_stage, report_stage_sync

async def my_node(state):
    await report_stage("embed", percent=40.0, message="Embedding documents")
    ...

def sync_node(state):
    report_stage_sync("encode", percent=20.0)  # schedules on the running loop
    ...
```

When no task is bound, both helpers are no-ops.

## Configuration

```python
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder(
    "agents/",
    enable_tasks=True,          # default
    task_max_concurrency=8,     # bounded in-process queue size
    # task_store=SQLAlchemyTaskStore(...),  # optional durable backend (see below)
)
```

### TaskRecord

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique task id (`task_...`) |
| `target_type` | `agent`/`plugin`/`pipeline`/`endpoint` | Resource kind |
| `target` | `str` | Resource name |
| `mode` | `str` | `sync`/`async`/`batch`/`stream` |
| `status` | `queued`/`running`/`succeeded`/`failed`/`cancelled` | Lifecycle |
| `progress` | `TaskProgress` | `percent`, `message`, `current`, `total`, `stage` |
| `input` / `batch` | `Any` | Submitted payload(s) |
| `result` / `error` | `Any` / `str` | Terminal outcome |
| `created_at` / `started_at` / `finished_at` | `float` | Unix timestamps |
| `duration_ms` | `float \| None` | Wall-clock duration |
| `callback_url` | `str \| None` | Completion webhook |

## Persistence & durability

The default [`InMemoryTaskStore`][store] keeps records in a bounded, TTL-aware
dict — perfect for single-process deployments and tests. For multi-worker or
restart-durable deployments, use the built-in **`SQLAlchemyTaskStore`** (or
implement the `TaskStore` interface against any shared backend) and pass it via
`task_store=`.

### Durable SQLAlchemy store

`SQLAlchemyTaskStore` is a drop-in, durable `TaskStore` backed by any
SQLAlchemy async driver. Records survive process restarts and are shared across
workers/replicas through a common database.

```python
from agentomatic import AgentPlatform
from agentomatic.tasks import SQLAlchemyTaskStore

# SQLite for development…
store = SQLAlchemyTaskStore("sqlite+aiosqlite:///data/tasks.db")

# …PostgreSQL for production.
store = SQLAlchemyTaskStore("postgresql+asyncpg://user:pass@localhost/db")

platform = AgentPlatform.from_folder("agents/", task_store=store)
```

Install the optional dependency:

=== "SQLite"

    ```bash
    pip install "agentomatic[db]"
    ```

=== "PostgreSQL"

    ```bash
    pip install "agentomatic[db-postgres]"
    ```

The platform initialises and disposes the store automatically as part of its
lifespan — no manual `initialize()`/`close()` needed.

**Configuration** (all optional, safe defaults):

| Argument | Default | Description |
|----------|---------|-------------|
| `url` | `sqlite+aiosqlite:///data/tasks.db` | Any SQLAlchemy async URL |
| `table_name` | `agentomatic_tasks` | Table to store records in |
| `pool_size` / `max_overflow` / `pool_recycle` / `pool_pre_ping` | `10` / `20` / `3600` / `True` | Connection pool (ignored for SQLite) |
| `echo` | `False` | Log all SQL (debug) |
| `ttl_seconds` | `604800` (7 days) | Terminal records older than this are evicted; `None` disables |
| `max_records` | `100000` | Soft cap; oldest terminal records evicted first; `None` disables |
| `eviction_interval` | `200` | Run eviction at most once every N saves |
| `engine` | `None` | Reuse an existing async engine (e.g. an agent's `DatabaseConnection`); not disposed on `close()` |

!!! tip "Forward-compatible schema"
    Records are stored as an indexed JSON payload plus a handful of indexed
    scalar columns (`status`, `target`, `created_at`, …), so adding fields to
    `TaskRecord` never requires a migration. Eviction is best-effort and never
    blocks or fails a `save`.

!!! note "Importing is always safe"
    `agentomatic.tasks` never imports SQLAlchemy eagerly. The dependency is only
    required when you actually construct `SQLAlchemyTaskStore`; a clear install
    hint is raised otherwise.

!!! warning "Multi-worker deployments"
    The in-memory store and in-process queue are per-process. If you run with
    `workers > 1`, provide a shared `TaskStore` (the `SQLAlchemyTaskStore`
    works out of the box) so status and results are consistent across workers.

[store]: #persistence-durability
