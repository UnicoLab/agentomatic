# Agentomatic Studio 🎨

Agentomatic provides a built-in visual development environment specifically designed to help you debug, inspect, and trace the execution of your agents in real-time. It works with **any agent framework** — LangGraph, LangChain, CrewAI, AutoGen, or raw Python — via a universal adapter system.

## Quick Start

The studio is bundled directly into the `agentomatic` pip package. No separate setup required.

```bash
pip install "agentomatic[studio]"
agentomatic run --studio
```

The unified platform starts serving your API endpoints at `http://localhost:8000` and the Studio UI at `http://localhost:8000/studio/ui/`.

---

## Framework Support

Agentomatic Studio uses a **universal adapter system** to provide the best possible debugging experience for every agent:

| Capability | LangGraph | LangChain / Custom | With Decorators |
|---|:---:|:---:|:---:|
| Graph Topology | ✅ Real graph | ✅ Synthetic linear | ✅ Custom graph |
| SSE Node Streaming | ✅ `astream_events` | ✅ Trace-based | ✅ Custom stream |
| Time-Travel History | ✅ Checkpointer | ✅ In-memory traces | ✅ In-memory traces |
| State Inspection | ✅ Checkpointer | ✅ Last I/O capture | ✅ Custom provider |
| State Mutation | ✅ `aupdate_state` | ⚠️ In-memory only | ⚠️ In-memory only |
| Breakpoints | ✅ `interrupt_before` | ❌ | ❌ |
| HITL Support | ✅ Native | ❌ | ❌ |

---

## Key Features

### 1. Live Node Streaming

When you execute an agent query, the **Graph View** maps directly to your agent's topology. As the execution progresses, nodes pulse and light up in real-time.

- **LangGraph agents**: Server-Sent Events stream node transitions directly from `astream_events`.
- **Other agents**: The generic adapter wraps execution with trace events that capture timing, input/output payloads, and exceptions.

### 2. Time-Travel Debugging

Agentomatic records every execution step for historical replay.

- **History View**: The **Time Travel** tab lists all past checkpoints (LangGraph) or execution traces (other frameworks).
- **Replay**: Click **"Replay from here"** on any snapshot to branch your thread and resume from that state.

### 3. Conditional Breakpoints

Freeze execution before a critical node (LangGraph only).

- **Setting Breakpoints**: Right-click any node in the Graph View → **"Add Breakpoint"**.
- **Execution**: The graph pauses before the target node. The node pulses, and the thread is suspended.
- **Resuming**: Resume execution or edit the state before continuing.

### 4. Live State Editing

During a breakpoint pause or HITL interrupt, you can mutate the graph state.

- **State View**: Navigate to the **State** tab in the Debug Console.
- **Editing**: Click **"Edit State"**, modify the JSON, and click **"Save"**.
- **LangGraph**: Changes are persisted via `graph.aupdate_state()`.
- **Other frameworks**: Changes are stored in the in-memory trace store.

---

## Studio Decorators

For non-LangGraph agents, you can incrementally opt-in to richer Studio capabilities using decorators. These let you provide custom graph topologies, state providers, and stream functions.

### `@studio_graph`

Register a custom graph topology for your agent:

```python
from agentomatic.studio import studio_graph

@studio_graph
def my_topology():
    return {
        "nodes": [
            {"id": "__start__", "name": "Start", "type": "start"},
            {"id": "fetch_data", "name": "Fetch Data", "type": "tool"},
            {"id": "process", "name": "Process", "type": "agent"},
            {"id": "validate", "name": "Validate", "type": "condition"},
            {"id": "__end__", "name": "End", "type": "end"},
        ],
        "edges": [
            {"source": "__start__", "target": "fetch_data"},
            {"source": "fetch_data", "target": "process"},
            {"source": "process", "target": "validate"},
            {"source": "validate", "target": "__end__", "condition": "valid"},
            {"source": "validate", "target": "process", "condition": "retry"},
        ]
    }
```

### `@studio_state`

Register a custom state provider:

```python
from agentomatic.studio import studio_state

@studio_state
async def get_my_state(thread_id: str) -> dict:
    """Return the current state for a thread."""
    return await my_database.get_thread_state(thread_id)
```

### `@studio_stream`

Register a custom SSE event stream:

```python
from agentomatic.studio import studio_stream
from agentomatic.studio.models import StudioRunEvent

@studio_stream
async def my_streamer(state, config, breakpoints):
    yield StudioRunEvent(event="node_start", run_id="", timestamp="...", node="my_node")
    result = await my_agent.process(state)
    yield StudioRunEvent(event="node_end", run_id="", timestamp="...", node="my_node", data={"output": result})
```

---

## Architecture

The Studio uses a layered adapter architecture:

```
┌──────────────────────────────────────────────────────┐
│                    Studio Router                      │
│            (FastAPI endpoints at /studio/)             │
└──────────────────┬───────────────────────────────────┘
                   │ resolve_adapter(agent)
    ┌──────────────┼──────────────────┐
    │              │                  │
    ▼              ▼                  ▼
┌────────┐   ┌──────────┐   ┌──────────────┐
│LangGraph│   │ Generic  │   │   Custom     │
│Adapter  │   │ Adapter  │   │   Adapter    │
│ (full)  │   │ (traces) │   │(user-defined)│
└────────┘   └──────────┘   └──────────────┘
```

- **Studio Router**: Framework-agnostic FastAPI endpoints.
- **Adapter Factory**: Automatically selects the best adapter based on the agent's configuration.
- **LangGraphAdapter**: Full-featured — uses `CompiledGraph` APIs natively.
- **GenericAdapter**: Trace-based — wraps `node_fn()` with timing and I/O capture.
- **Custom Adapter**: User-registered via decorators or direct assignment.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/studio/info` | GET | Platform metadata and capabilities |
| `/studio/agents` | GET | List agents with capabilities |
| `/studio/agents/{name}/graph` | GET | Graph topology (real or synthetic) |
| `/studio/agents/{name}/schemas` | GET | Input/output JSON schemas |
| `/studio/agents/{name}/runs/stream` | POST | SSE-streamed execution |
| `/studio/agents/{name}/threads/{tid}/state` | GET | Thread state snapshot |
| `/studio/agents/{name}/threads/{tid}/state` | POST | Update thread state |
| `/studio/agents/{name}/threads/{tid}/history` | GET | Checkpoint/trace history |
