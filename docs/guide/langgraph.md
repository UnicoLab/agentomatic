# LangGraph Integration Guide

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Build Production-Grade Agent Workflows with LangGraph + Agentomatic</h3>
</div>

---

## Why LangGraph + Agentomatic?

[LangGraph](https://langchain-ai.github.io/langgraph/) gives you **stateful, cyclical agent graphs** — the building blocks for complex AI workflows. Agentomatic takes those graphs and adds everything needed for production:

| LangGraph Provides | Agentomatic Adds |
|---|---|
| `StateGraph` → `CompiledGraph` | Auto-generated REST API per agent |
| `add_messages` reducer | Persistent conversation memory |
| `interrupt_before` / `interrupt_after` | Full HITL approval workflow with UI |
| In-memory checkpointer | Pluggable storage (SQLAlchemy, Memory) |
| `astream_events` | Studio SSE real-time visualization |
| Graph topology via `get_graph()` | Interactive graph explorer in Studio |

!!! tip "Zero Configuration Required"
    Drop a LangGraph agent into your `agents/` folder with a `graph_fn` and Agentomatic auto-generates 20+ production endpoints, Studio debugging, and persistent checkpointing.

---

## Architecture

```mermaid
graph TB
    subgraph "Your Code"
        SG["StateGraph\n(nodes, edges, conditions)"]
        CG["CompiledGraph\n(graph_fn returns this)"]
    end

    subgraph "Agentomatic Platform"
        RF["RouterFactory\nauto-generates REST endpoints"]
        LGA["LangGraphAdapter\ngraph topology · SSE · checkpoints"]
        SR["Studio Router\n/studio/* debug API"]
        CP["AgentomaticCheckpointer\nBridges LangGraph ↔ BaseStore"]
    end

    subgraph "Client Layer"
        REST["REST API\n/invoke · /chat · /stream"]
        STUDIO["Agentomatic Studio\nGraph viz · State · Time-travel"]
        A2A["A2A Protocol\nAgent-to-agent discovery"]
    end

    subgraph "Storage"
        MEM["MemoryStore\nDev / Testing"]
        SQL["SQLAlchemyStore\nPostgreSQL · SQLite"]
    end

    SG -->|".compile()"| CG
    CG -->|"graph_fn"| RF
    CG -->|"graph_fn"| LGA
    RF --> REST
    LGA --> SR --> STUDIO
    RF --> A2A
    CP --> MEM
    CP --> SQL
    LGA -.->|"checkpoint ops"| CP

    style SG fill:#4a9eff,color:#fff
    style CG fill:#4a9eff,color:#fff
    style LGA fill:#ff6b6b,color:#fff
    style CP fill:#ff6b6b,color:#fff
    style STUDIO fill:#51cf66,color:#fff
```

---

## Quick Start

### Minimal LangGraph Agent

Create a folder `agents/my_agent/` with these files:

=== "`__init__.py`"

    ```python
    """My first LangGraph agent with Agentomatic."""

    from langgraph.graph import StateGraph, END
    from agentomatic.core.state import BaseAgentState
    from agentomatic.core.manifest import AgentManifest

    # 1. Define your graph
    async def process_query(state: BaseAgentState) -> dict:
        query = state.get("current_query", "")
        return {
            "response": f"Processed: {query}",
            "steps_taken": ["process_query"],
        }

    # 2. Build the StateGraph
    builder = StateGraph(BaseAgentState)
    builder.add_node("process", process_query)
    builder.set_entry_point("process")
    builder.add_edge("process", END)

    # 3. Export as graph_fn (required by Agentomatic)
    def graph_fn():
        return builder.compile()

    # 4. Declare the manifest
    manifest = AgentManifest(
        name="my_agent",
        slug="my-agent",
        description="A minimal LangGraph agent",
        version="1.0.0",
        framework="langgraph",
    )
    ```

=== "`prompts.json`"

    ```json
    {
      "v1": {
        "system": "You are a helpful assistant."
      }
    }
    ```

### Launch the Platform

```python
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore

platform = AgentPlatform.from_folder(
    "agents/",
    store=MemoryStore(),
)

# Run with: uvicorn main:platform.app --reload
```

Your agent now has a full REST API:

```bash
# Invoke synchronously
curl -X POST http://localhost:8000/api/v1/my_agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, agent!"}'

# Stream via SSE
curl -X POST http://localhost:8000/api/v1/my_agent/invoke/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, agent!"}'

# Session-aware chat
curl -X POST http://localhost:8000/api/v1/my_agent/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello!", "thread_id": "thread_001"}'
```

---

## State Management

### `BaseAgentState` — The Default State

Agentomatic provides `BaseAgentState`, a `TypedDict` with **annotated reducers** that handle parallel writes safely:

```python
from agentomatic.core.state import BaseAgentState
```

| Field | Type | Reducer | Description |
|---|---|---|---|
| `messages` | `Annotated[list, add_messages]` | LangGraph `add_messages` | Chat message history |
| `thread_id` | `str` | — | Conversation thread ID |
| `user_id` | `str` | — | User identifier |
| `current_query` | `str` | — | Current user input |
| `response` | `Annotated[str, _last_value]` | Last-writer-wins | Final agent response |
| `agent_type` | `Annotated[str, _last_value]` | Last-writer-wins | Agent identifier |
| `suggestions` | `Annotated[list[str], operator.add]` | Concatenation | Follow-up suggestions |
| `citations` | `Annotated[list[dict], operator.add]` | Concatenation | Source citations |
| `routing_decision` | `Annotated[str, _last_value]` | Last-writer-wins | Router decision label |
| `context` | `Annotated[dict, _merge_dicts]` | Dict merge | Arbitrary caller data |
| `prompt_version` | `Annotated[str, _last_value]` | Last-writer-wins | Active prompt version |
| `steps_taken` | `Annotated[list[str], operator.add]` | Concatenation | Processing step log |
| `metadata` | `Annotated[dict, _merge_dicts]` | Dict merge | Extra metadata |
| `error` | `Annotated[str \| None, _last_value]` | Last-writer-wins | Error message |

!!! info "All Fields are Optional"
    `BaseAgentState` uses `total=False`, so every field is optional. Your nodes only need to return the fields they modify.

### Custom State with TypedDict

For domain-specific agents, extend `BaseAgentState` or define your own:

=== "Extending BaseAgentState"

    ```python
    from typing import Annotated
    from agentomatic.core.state import BaseAgentState, _last_value

    class RAGAgentState(BaseAgentState, total=False):
        """RAG agent with retrieval-specific fields."""
        retrieved_documents: Annotated[list[dict], operator.add]
        similarity_scores: Annotated[list[float], operator.add]
        rerank_applied: Annotated[bool, _last_value]
    ```

=== "Standalone TypedDict"

    ```python
    from typing import Annotated, TypedDict
    from langgraph.graph import add_messages

    class MinimalState(TypedDict, total=False):
        messages: Annotated[list, add_messages]
        current_query: str
        response: str
    ```

### State Channels & Reducers

Reducers resolve conflicts when multiple graph branches write to the same key:

```python
import operator
from typing import Annotated

# Concatenation — parallel branches append to the same list
steps_taken: Annotated[list[str], operator.add]

# Last-writer-wins — last branch to write wins
response: Annotated[str, _last_value]

# Dict merge — keys from both branches are merged
metadata: Annotated[dict, _merge_dicts]
```

!!! warning "Without Reducers"
    If two parallel nodes write to the same key without a reducer, LangGraph raises `InvalidUpdateError`. Always annotate shared fields.

### State Inspection via Studio

The Studio adapter extracts state from the LangGraph checkpointer automatically:

```bash
# Get current thread state
curl http://localhost:8000/studio/agents/my_agent/threads/thread_001/state

# Update state manually
curl -X POST http://localhost:8000/studio/agents/my_agent/threads/thread_001/state \
  -H "Content-Type: application/json" \
  -d '{"updates": {"response": "Overridden response"}}'
```

---

## Graph Patterns

### Linear: A → B → C

The simplest pattern — nodes execute in sequence:

```python
from langgraph.graph import StateGraph, END
from agentomatic.core.state import BaseAgentState

async def retrieve(state: BaseAgentState) -> dict:
    return {"steps_taken": ["retrieve"], "context": {"docs": ["doc1", "doc2"]}}

async def generate(state: BaseAgentState) -> dict:
    docs = state.get("context", {}).get("docs", [])
    return {"steps_taken": ["generate"], "response": f"Based on {len(docs)} docs..."}

async def validate(state: BaseAgentState) -> dict:
    return {"steps_taken": ["validate"], "metadata": {"validated": True}}

builder = StateGraph(BaseAgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("validate", validate)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "validate")
builder.add_edge("validate", END)

def graph_fn():
    return builder.compile()
```

```mermaid
graph LR
    S(("__start__")) --> R["retrieve"]
    R --> G["generate"]
    G --> V["validate"]
    V --> E(("__end__"))
```

### Conditional: Router with Multiple Paths

Use `add_conditional_edges` to branch based on state:

```python
from langgraph.graph import StateGraph, END
from agentomatic.core.state import BaseAgentState

async def router_node(state: BaseAgentState) -> dict:
    query = state.get("current_query", "").lower()
    if "code" in query:
        return {"routing_decision": "code_agent"}
    elif "search" in query:
        return {"routing_decision": "search_agent"}
    return {"routing_decision": "general_agent"}

def route_decision(state: BaseAgentState) -> str:
    return state.get("routing_decision", "general_agent")

async def code_agent(state: BaseAgentState) -> dict:
    return {"response": "Here's the code...", "agent_type": "code"}

async def search_agent(state: BaseAgentState) -> dict:
    return {"response": "Search results...", "agent_type": "search"}

async def general_agent(state: BaseAgentState) -> dict:
    return {"response": "General response...", "agent_type": "general"}

builder = StateGraph(BaseAgentState)
builder.add_node("router", router_node)
builder.add_node("code_agent", code_agent)
builder.add_node("search_agent", search_agent)
builder.add_node("general_agent", general_agent)

builder.set_entry_point("router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "code_agent": "code_agent",
        "search_agent": "search_agent",
        "general_agent": "general_agent",
    },
)
builder.add_edge("code_agent", END)
builder.add_edge("search_agent", END)
builder.add_edge("general_agent", END)

def graph_fn():
    return builder.compile()
```

```mermaid
graph TD
    S(("__start__")) --> R["router"]
    R -->|"code"| C["code_agent"]
    R -->|"search"| SE["search_agent"]
    R -->|"general"| G["general_agent"]
    C --> E(("__end__"))
    SE --> E
    G --> E
```

### Cyclic: Agent Loop with Tools

The ReAct pattern — agent decides whether to call tools or finish:

```python
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from agentomatic.core.state import BaseAgentState

# Define tools
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

tools = [search_web, calculate]

async def agent_node(state: BaseAgentState) -> dict:
    """LLM decides: call a tool or respond directly."""
    from agentomatic.providers.llm import get_llm
    llm = get_llm(provider="openai", model="gpt-4o").bind_tools(tools)
    messages = state.get("messages", [])
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

def should_continue(state: BaseAgentState) -> str:
    """Check if the last message has tool calls."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    return "end"

builder = StateGraph(BaseAgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
builder.add_edge("tools", "agent")  # Loop back after tool execution

def graph_fn():
    return builder.compile()
```

```mermaid
graph TD
    S(("__start__")) --> A["agent"]
    A -->|"tool_calls"| T["tools"]
    A -->|"no tools"| E(("__end__"))
    T --> A
```

### Multi-Agent: Supervisor + Workers

A supervisor orchestrates specialized worker agents:

```python
from langgraph.graph import StateGraph, END
from agentomatic.core.state import BaseAgentState

async def supervisor(state: BaseAgentState) -> dict:
    """Route to the best worker based on query analysis."""
    query = state.get("current_query", "").lower()
    if "data" in query or "chart" in query:
        return {"routing_decision": "analyst"}
    elif "write" in query or "draft" in query:
        return {"routing_decision": "writer"}
    return {"routing_decision": "researcher"}

def route_to_worker(state: BaseAgentState) -> str:
    decision = state.get("routing_decision", "researcher")
    if decision == "done":
        return "end"
    return decision

async def researcher(state: BaseAgentState) -> dict:
    return {
        "response": "Research findings...",
        "routing_decision": "done",
        "steps_taken": ["researcher"],
    }

async def analyst(state: BaseAgentState) -> dict:
    return {
        "response": "Analysis complete...",
        "routing_decision": "done",
        "steps_taken": ["analyst"],
    }

async def writer(state: BaseAgentState) -> dict:
    return {
        "response": "Draft written...",
        "routing_decision": "done",
        "steps_taken": ["writer"],
    }

builder = StateGraph(BaseAgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("analyst", analyst)
builder.add_node("writer", writer)

builder.set_entry_point("supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "end": END,
    },
)
# Workers route back through supervisor or finish
for worker in ["researcher", "analyst", "writer"]:
    builder.add_conditional_edges(
        worker,
        route_to_worker,
        {"done": END, "end": END, "researcher": "researcher",
         "analyst": "analyst", "writer": "writer"},
    )

def graph_fn():
    return builder.compile()
```

```mermaid
graph TD
    S(("__start__")) --> SUP["supervisor"]
    SUP -->|"data/chart"| ANA["analyst"]
    SUP -->|"write/draft"| WR["writer"]
    SUP -->|"default"| RES["researcher"]
    ANA -->|"done"| E(("__end__"))
    WR -->|"done"| E
    RES -->|"done"| E
```

---

## Checkpointing & Time-Travel

### `AgentomaticCheckpointer`

The `AgentomaticCheckpointer` bridges LangGraph's checkpoint interface with Agentomatic's storage backends. It implements `BaseCheckpointSaver` and delegates all persistence to your configured `BaseStore`.

```mermaid
sequenceDiagram
    participant LG as LangGraph Runtime
    participant CP as AgentomaticCheckpointer
    participant ST as BaseStore (Memory/SQL)

    LG->>CP: aput(config, checkpoint, metadata)
    CP->>CP: _ensure_json_serializable(checkpoint)
    CP->>ST: save_checkpoint(thread_id, ns, id, data)
    ST-->>CP: saved
    CP-->>LG: RunnableConfig

    LG->>CP: aget_tuple(config)
    CP->>ST: get_checkpoint(thread_id, ns, id)
    ST-->>CP: checkpoint_data
    CP-->>LG: CheckpointTuple
```

### Memory Checkpointer (Development)

For rapid development and testing:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

builder = StateGraph(MyState)
# ... add nodes and edges ...

# In-memory — data lost on restart
graph = builder.compile(checkpointer=MemorySaver())
```

### SQLAlchemy Checkpointer (Production)

For production with persistent checkpoints:

```python
from agentomatic.storage import SQLAlchemyStore
from agentomatic.storage.checkpointer import AgentomaticCheckpointer
from langgraph.graph import StateGraph, END

# 1. Create the persistent store
store = SQLAlchemyStore(
    "postgresql+asyncpg://user:pass@localhost:5432/agents"
)

# 2. Wrap it as a LangGraph checkpointer
checkpointer = AgentomaticCheckpointer(store)

# 3. Compile with the checkpointer
builder = StateGraph(MyState)
# ... add nodes and edges ...
graph = builder.compile(checkpointer=checkpointer)
```

!!! tip "Safe Serialization"
    The checkpointer automatically handles non-JSON-serializable objects (datetimes, bytes, custom classes) via `_ensure_json_serializable()`. No extra configuration needed.

### Replaying from Checkpoints

Time-travel lets you rewind execution to any checkpoint and replay from there:

=== "Via Studio API"

    ```bash
    # 1. List checkpoint history
    curl http://localhost:8000/studio/agents/my_agent/threads/thread_001/history

    # 2. Replay from a specific checkpoint
    curl -X POST http://localhost:8000/studio/agents/my_agent/runs/stream \
      -H "Content-Type: application/json" \
      -d '{
        "query": "Re-run with different input",
        "thread_id": "thread_001",
        "checkpoint_id": "cp_abc123"
      }'
    ```

=== "Via Python"

    ```python
    # Replay from a checkpoint
    config = {
        "configurable": {
            "thread_id": "thread_001",
            "checkpoint_id": "cp_abc123",
        }
    }
    result = await graph.ainvoke(
        {"current_query": "Re-run with new input"},
        config=config,
    )
    ```

---

## Breakpoints & Human-in-the-Loop

### `interrupt_before` / `interrupt_after`

LangGraph supports pausing execution at specific nodes. The Agentomatic adapter maps these to Studio events:

```python
from langgraph.graph import StateGraph, END
from agentomatic.core.state import BaseAgentState

async def analyze(state: BaseAgentState) -> dict:
    return {"steps_taken": ["analyze"], "response": "Analysis ready for review"}

async def execute_action(state: BaseAgentState) -> dict:
    """This node requires human approval before running."""
    return {"steps_taken": ["execute_action"], "response": "Action executed!"}

builder = StateGraph(BaseAgentState)
builder.add_node("analyze", analyze)
builder.add_node("execute_action", execute_action)

builder.set_entry_point("analyze")
builder.add_edge("analyze", "execute_action")
builder.add_edge("execute_action", END)

def graph_fn():
    return builder.compile(
        interrupt_before=["execute_action"],  # ← pause before this node
    )
```

When the graph reaches `execute_action`, it pauses and the adapter emits:

```json
{
  "event": "breakpoint_hit",
  "node": "execute_action",
  "data": {
    "reason": "interrupt_before",
    "resumable": true,
    "interrupt_type": "NodeInterrupt"
  }
}
```

### Studio Approval Workflow

```mermaid
sequenceDiagram
    participant UI as Studio UI
    participant SR as Studio Router
    participant LG as LangGraph

    UI->>SR: POST /studio/agents/{name}/runs/stream
    SR->>LG: astream_events(state)
    LG-->>SR: node_start: analyze
    SR-->>UI: SSE: node_start
    LG-->>SR: node_end: analyze
    SR-->>UI: SSE: node_end
    LG-->>SR: GraphInterrupt (execute_action)
    SR-->>UI: SSE: breakpoint_hit

    Note over UI: Human reviews state in Studio

    UI->>SR: POST /studio/agents/{name}/threads/{id}/resume
    Note over SR: body: {"value": {"approved": true}, "action": "approve"}
    SR->>LG: astream_events(Command(resume=value))
    LG-->>SR: node_start: execute_action
    SR-->>UI: SSE: node_start
    LG-->>SR: node_end: execute_action
    SR-->>UI: SSE: node_end
    LG-->>SR: run_complete
    SR-->>UI: SSE: done
```

### Resume Endpoint

After a breakpoint is hit, resume execution via the Studio API:

```bash
# Approve and resume
curl -X POST \
  http://localhost:8000/studio/agents/my_agent/threads/thread_001/resume \
  -H "Content-Type: application/json" \
  -d '{"value": {"approved": true, "notes": "Looks good"}, "action": "approve"}'

# Reject
curl -X POST \
  http://localhost:8000/studio/agents/my_agent/threads/thread_001/resume \
  -H "Content-Type: application/json" \
  -d '{"value": null, "action": "reject"}'
```

### `AgentSuspendedException` — Programmatic HITL

For more control, raise `AgentSuspendedException` directly in your node:

```python
from agentomatic.core.router_factory import AgentSuspendedException

async def transfer_funds(state: BaseAgentState) -> dict:
    metadata = state.get("metadata", {})

    if not metadata.get("hitl_approved"):
        raise AgentSuspendedException(
            approval_id=f"tx_{state.get('thread_id', 'unknown')}",
            node_name="transfer_funds",
            state_snapshot=dict(state),
            message=f"Transfer ${state.get('context', {}).get('amount', 0)} requires approval",
        )

    # Approved — proceed
    return {"response": "Transfer completed!", "steps_taken": ["transfer_funds"]}
```

Then approve/reject via the per-agent REST endpoints:

```bash
# List pending
GET /api/v1/my_agent/threads/{thread_id}/pending

# Approve
POST /api/v1/my_agent/threads/{thread_id}/approve
{"approval_id": "tx_thread_001", "context": {"approved_limit": 1000}}

# Reject
POST /api/v1/my_agent/threads/{thread_id}/reject
{"approval_id": "tx_thread_001", "reason": "Amount exceeds policy"}
```

---

## Streaming

### `astream_events` Event Types

LangGraph's `astream_events(version="v2")` emits detailed events. The `LangGraphAdapter` maps them to Studio-compatible `StudioRunEvent` types:

| LangGraph Event | Studio Event | Description |
|---|---|---|
| `on_chain_start` (node) | `node_start` | Node begins execution |
| `on_chain_end` (node) | `node_end` | Node completes with output |
| `on_chain_start` (subgraph) | `subagent_start` | Subgraph/subagent begins |
| `on_chain_end` (subgraph) | `subagent_end` | Subgraph/subagent completes |
| `on_chat_model_stream` | `message_chunk` | LLM token streamed |
| `on_tool_start` | `node_start` (tool:name) | Tool invocation begins |
| `on_tool_end` | `node_end` (tool:name) | Tool returns result |
| `on_tool_start` (write_todos) | `task_update` | Planning tool detected |
| `GraphInterrupt` / `NodeInterrupt` | `breakpoint_hit` | Execution paused |

### Studio SSE Mapping

When streaming through Studio, events arrive as SSE frames:

```
data: {"event":"node_start","run_id":"run_abc","timestamp":"2026-06-18T20:00:00","node":"retrieve","data":{"tags":["seq:step:1"]}}

data: {"event":"node_end","run_id":"run_abc","timestamp":"2026-06-18T20:00:01","node":"retrieve","data":{"output":{"docs":["..."]}}}

data: {"event":"message_chunk","run_id":"run_abc","timestamp":"2026-06-18T20:00:02","node":"gpt-4o","data":{"content":"The "}}

data: {"event":"message_chunk","run_id":"run_abc","timestamp":"2026-06-18T20:00:02","node":"gpt-4o","data":{"content":"answer "}}

data: {"event":"message_chunk","run_id":"run_abc","timestamp":"2026-06-18T20:00:02","node":"gpt-4o","data":{"content":"is..."}}

data: {"event":"done"}
```

### Custom Stream Modes

Agentomatic supports two streaming endpoints:

=== "Agent Streaming (`/invoke/stream`)"

    Uses `graph.astream(state)` — yields full state diffs per node:

    ```bash
    curl -X POST http://localhost:8000/api/v1/my_agent/invoke/stream \
      -H "Content-Type: application/json" \
      -d '{"query": "Analyze this data"}'
    ```

    ```
    data: {"response": "", "steps_taken": ["retrieve"]}
    data: {"response": "Analysis complete", "steps_taken": ["retrieve", "analyze"]}
    data: [DONE]
    ```

=== "Studio Streaming (`/studio/agents/{name}/runs/stream`)"

    Uses `graph.astream_events(version="v2")` via `LangGraphAdapter` — yields fine-grained `StudioRunEvent` objects:

    ```bash
    curl -X POST http://localhost:8000/studio/agents/my_agent/runs/stream \
      -H "Content-Type: application/json" \
      -d '{"query": "Analyze this data"}'
    ```

    ```
    data: {"event":"run_start","run_id":"run_xyz",...}
    data: {"event":"node_start","node":"retrieve",...}
    data: {"event":"node_end","node":"retrieve","data":{"output":{...}}}
    data: {"event":"message_chunk","data":{"content":"The "}}
    data: {"event":"run_complete",...}
    ```

---

## Studio Integration

### Graph Visualization

The `LangGraphAdapter` extracts real graph topology from `graph.get_graph()` and classifies each node:

```bash
curl http://localhost:8000/studio/agents/my_agent/graph
```

```json
{
  "agent_name": "my_agent",
  "nodes": [
    {"id": "__start__", "name": "__start__", "type": "start", "metadata": {}},
    {"id": "retrieve", "name": "retrieve", "type": "agent", "metadata": {}},
    {"id": "tools", "name": "tools", "type": "tool", "metadata": {}},
    {"id": "human_review", "name": "human_review", "type": "human", "metadata": {}},
    {"id": "__end__", "name": "__end__", "type": "end", "metadata": {}}
  ],
  "edges": [
    {"id": "edge_0", "source": "__start__", "target": "retrieve", "condition": null},
    {"id": "edge_1", "source": "retrieve", "target": "tools", "condition": "conditional"},
    {"id": "edge_2", "source": "tools", "target": "retrieve", "condition": null}
  ],
  "entry_point": "__start__",
  "end_points": ["__end__"]
}
```

### Node Classification Rules

The adapter auto-classifies nodes based on naming conventions:

| Node Name Contains | Classified As | Studio Icon |
|---|---|---|
| `__start__` | `start` | ● (green) |
| `__end__` | `end` | ● (red) |
| `tool` | `tool` | 🔧 |
| `condition`, `router`, `route`, `branch` | `condition` | ◇ |
| `human`, `approval`, `review` | `human` | 👤 |
| `task`, `delegate`, `subagent`, `spawn` | `subagent` | 🤖 |
| `plan`, `todo`, `write_todo` | `planning` | 📋 |
| Everything else | `agent` | ⬡ |

!!! tip "Naming Convention"
    Name your nodes descriptively (e.g., `human_review`, `search_tool`, `route_query`) to get automatic icon classification in Studio.

### Real-Time Execution Tracking

Studio tracks every run with full event history:

```bash
# Create a run and get results
curl -X POST http://localhost:8000/studio/agents/my_agent/runs \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello"}'

# List recent runs
curl http://localhost:8000/studio/agents/my_agent/runs?limit=10

# Get a specific run with all events
curl http://localhost:8000/studio/agents/my_agent/runs/run_abc123
```

### Adapter Capabilities

The adapter dynamically reports its capabilities based on the graph's configuration:

```python
# These capabilities are auto-detected:
adapter.capabilities
# → ["graph", "streaming", "checkpoints", "state", "breakpoints", "hitl"]

adapter.supports_graph          # True — has graph_fn
adapter.supports_checkpoints    # True — graph has checkpointer
adapter.supports_state_mutation # True — checkpointer present
adapter.supports_breakpoints    # True — checkpointer present
```

---

## Advanced Patterns

### Subgraphs

Nest graphs inside other graphs for modular architectures:

```python
from langgraph.graph import StateGraph, END
from agentomatic.core.state import BaseAgentState

# Inner graph (research module)
async def search(state: BaseAgentState) -> dict:
    return {"steps_taken": ["search"], "context": {"results": ["r1", "r2"]}}

async def summarize(state: BaseAgentState) -> dict:
    return {"steps_taken": ["summarize"], "response": "Summary of findings"}

research_builder = StateGraph(BaseAgentState)
research_builder.add_node("search", search)
research_builder.add_node("summarize", summarize)
research_builder.set_entry_point("search")
research_builder.add_edge("search", "summarize")
research_builder.add_edge("summarize", END)
research_graph = research_builder.compile()

# Outer graph
async def intake(state: BaseAgentState) -> dict:
    return {"steps_taken": ["intake"]}

async def format_output(state: BaseAgentState) -> dict:
    return {"steps_taken": ["format"]}

outer_builder = StateGraph(BaseAgentState)
outer_builder.add_node("intake", intake)
outer_builder.add_node("research", research_graph)  # ← subgraph as node
outer_builder.add_node("format", format_output)

outer_builder.set_entry_point("intake")
outer_builder.add_edge("intake", "research")
outer_builder.add_edge("research", "format")
outer_builder.add_edge("format", END)

def graph_fn():
    return outer_builder.compile()
```

The Studio adapter detects subgraph events via the `langgraph_checkpoint_ns` metadata and emits `subagent_start` / `subagent_end` events with the namespace path.

### Dynamic Graphs

Build graphs dynamically based on configuration:

```python
from langgraph.graph import StateGraph, END
from agentomatic.core.state import BaseAgentState

def build_pipeline(steps: list[str]):
    """Build a graph dynamically from a list of step names."""
    builder = StateGraph(BaseAgentState)

    for step_name in steps:
        async def node_fn(state, name=step_name):
            return {"steps_taken": [name]}
        builder.add_node(step_name, node_fn)

    builder.set_entry_point(steps[0])
    for i in range(len(steps) - 1):
        builder.add_edge(steps[i], steps[i + 1])
    builder.add_edge(steps[-1], END)

    return builder.compile()

# Configure via manifest or environment
PIPELINE_STEPS = ["ingest", "validate", "transform", "load"]

def graph_fn():
    return build_pipeline(PIPELINE_STEPS)
```

### Tool Nodes

Use LangGraph's `ToolNode` for structured tool execution:

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight between two cities."""
    return f"Flight booked: {origin} → {destination} on {date}"

# ToolNode handles tool dispatch automatically
tool_node = ToolNode([get_weather, book_flight])

builder = StateGraph(BaseAgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)  # ← ToolNode manages multiple tools

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")
```

The Studio adapter detects `on_tool_start` / `on_tool_end` events and shows tool inputs/outputs with the `tool:{name}` node label.

### Memory & Persistence

Combine checkpointing with Agentomatic's thread store for full persistence:

```python
from agentomatic import AgentPlatform
from agentomatic.storage import SQLAlchemyStore
from agentomatic.storage.checkpointer import AgentomaticCheckpointer

# Shared store for threads AND checkpoints
store = SQLAlchemyStore(
    "postgresql+asyncpg://user:pass@localhost:5432/agents"
)

# In your agent's __init__.py:
checkpointer = AgentomaticCheckpointer(store)

builder = StateGraph(BaseAgentState)
# ... build graph ...
def graph_fn():
    return builder.compile(checkpointer=checkpointer)

# Platform uses the same store
platform = AgentPlatform.from_folder(
    "agents/",
    store=store,  # threads + messages + checkpoints all in one DB
)
```

This gives you:

- **Thread persistence** — conversation history survives restarts
- **Checkpoint persistence** — graph state survives restarts
- **Time-travel** — replay from any checkpoint
- **State inspection** — view/edit state via Studio

---

## Complete Example: RAG Agent with HITL

Here's a production-ready RAG agent that combines retrieval, generation, and human approval:

```python
"""agents/rag_agent/__init__.py — RAG agent with human review."""

import operator
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agentomatic.core.manifest import AgentManifest
from agentomatic.core.state import BaseAgentState, _last_value
from agentomatic.storage.checkpointer import AgentomaticCheckpointer


# Extended state for RAG
class RAGState(BaseAgentState, total=False):
    retrieved_docs: Annotated[list[dict[str, Any]], operator.add]
    confidence: Annotated[float, _last_value]


# Tools
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base."""
    return f"Found 3 relevant documents for: {query}"


tools = [search_knowledge_base]


# Nodes
async def retrieve(state: RAGState) -> dict:
    query = state.get("current_query", "")
    return {
        "retrieved_docs": [{"content": f"Doc about {query}", "score": 0.95}],
        "steps_taken": ["retrieve"],
    }


async def generate(state: RAGState) -> dict:
    docs = state.get("retrieved_docs", [])
    return {
        "response": f"Based on {len(docs)} documents: ...",
        "confidence": 0.87,
        "steps_taken": ["generate"],
    }


async def quality_check(state: RAGState) -> dict:
    confidence = state.get("confidence", 0.0)
    if confidence < 0.8:
        return {"routing_decision": "human_review"}
    return {"routing_decision": "publish"}


def route_after_check(state: RAGState) -> str:
    return state.get("routing_decision", "publish")


async def human_review(state: RAGState) -> dict:
    """This node is paused by interrupt_before for human review."""
    return {"steps_taken": ["human_review"], "metadata": {"reviewed": True}}


async def publish(state: RAGState) -> dict:
    return {"steps_taken": ["publish"]}


# Build graph
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("quality_check", quality_check)
builder.add_node("human_review", human_review)
builder.add_node("publish", publish)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "quality_check")
builder.add_conditional_edges(
    "quality_check",
    route_after_check,
    {"human_review": "human_review", "publish": "publish"},
)
builder.add_edge("human_review", "publish")
builder.add_edge("publish", END)


def graph_fn():
    return builder.compile(
        interrupt_before=["human_review"],  # Pause for HITL
    )


manifest = AgentManifest(
    name="rag_agent",
    slug="rag-agent",
    description="RAG agent with quality-gated human review",
    version="1.0.0",
    framework="langgraph",
)
```

```mermaid
graph TD
    S(("__start__")) --> R["retrieve"]
    R --> G["generate"]
    G --> QC["quality_check"]
    QC -->|"confidence < 0.8"| HR["👤 human_review\n(interrupt_before)"]
    QC -->|"confidence ≥ 0.8"| P["publish"]
    HR --> P
    P --> E(("__end__"))

    style HR fill:#ff9800,color:#fff
```

---

## Reference

### Key Imports

```python
# State
from agentomatic.core.state import BaseAgentState

# Manifest
from agentomatic.core.manifest import AgentManifest

# Checkpointing
from agentomatic.storage.checkpointer import AgentomaticCheckpointer

# HITL
from agentomatic.core.router_factory import AgentSuspendedException

# Storage
from agentomatic.storage import MemoryStore, SQLAlchemyStore

# Platform
from agentomatic import AgentPlatform
```

### Related Documentation

- [Architecture Overview](../architecture/overview.md) — System design and component relationships
- [API Reference](../architecture/api-reference.md) — Complete endpoint documentation
- [Studio Guide](studio.md) — Full Studio debugging UI guide
- [Platform Features](platform-features.md) — HITL, memory, checkpointing details
- [Storage Guide](storage.md) — Storage backend configuration
