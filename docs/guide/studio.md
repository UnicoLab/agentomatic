# Agentomatic Studio 🎨

Agentomatic provides a built-in visual development environment specifically designed to help you debug, inspect, and trace the execution of your agents in real-time. It works with **any agent framework** — LangGraph, LangChain, CrewAI, AutoGen, or raw Python — via a universal adapter system.

!!! info "Studio vs Chat Interface"
    | | **Agentomatic Studio** (this page) | [Chat Interface (Chainlit)](debug-ui.md) |
    |---|---|---|
    | **Purpose** | Visual debugging & state inspection | Conversational testing |
    | **Launch** | `agentomatic run --studio` | `agentomatic run --with-ui` |
    | **URL** | `/studio/ui/` | `/chat` |
    | **Best for** | Graph visualization, SSE tracing, breakpoints, time-travel | Testing responses, prompt A/B testing, feedback |

## Quick Start

The studio is bundled directly into the `agentomatic` pip package. No separate setup required.

=== "With Your Agents"

    ```bash
    pip install "agentomatic[studio]"
    agentomatic run --studio
    ```

=== "Quick Demo (No Setup)"

    ```bash
    agentomatic demo
    ```

    Launches a built-in demo agent with Studio enabled — perfect for first-time exploration or CI smoke tests. See the [Demo Command](demo.md) page for details.

The unified platform starts serving your API endpoints at `http://localhost:8000` and the Studio UI at `http://localhost:8000/studio/ui/`.

---

## Framework Support

Agentomatic Studio uses a **universal adapter system** to provide the best possible debugging experience for every agent:

| Capability | LangGraph | LangChain | Custom / Raw Python | With Decorators |
|---|:---:|:---:|:---:|:---:|
| Graph Topology | ✅ Real graph | ✅ LCEL extraction or synthetic chain | ✅ Synthetic linear | ✅ Custom graph |
| SSE Node Streaming | ✅ `astream_events` | ✅ `astream_events` (v2) | ✅ Trace-based | ✅ Custom stream |
| Time-Travel History | ✅ Checkpointer | ✅ In-memory traces | ✅ In-memory traces | ✅ In-memory traces |
| State Inspection | ✅ Checkpointer | ✅ Message + I/O capture | ✅ Last I/O capture | ✅ Custom provider |
| State Mutation | ✅ `aupdate_state` | ⚠️ In-memory only | ⚠️ In-memory only | ⚠️ In-memory only |
| Breakpoints | ✅ `interrupt_before` | ❌ | ❌ | ❌ |
| HITL Support | ✅ Native | ❌ | ❌ | ❌ |

---

## LangChain Integration

Agentomatic Studio provides first-class support for LangChain-based agents, chatbots, and LCEL chains. When your agent's manifest declares `framework='langchain'`, the Studio automatically uses the dedicated `LangChainAdapter`.

### Automatic Features

The LangChain adapter automatically provides:

- **LCEL graph extraction** — If your chain/runnable exposes `.get_graph()`, Studio extracts the real topology.
- **Synthetic chain graph** — If no `.get_graph()` is found, Studio renders a typical chain layout: `Input → Prompt → LLM → Output Parser → Output`.
- **Rich SSE streaming** — If `astream_events` is available on the runnable, the Studio streams `on_chain_start`, `on_chain_end`, `on_chat_model_stream`, `on_tool_start`, `on_tool_end`, and `on_llm_start/end` events in real-time.
- **Automatic message tracking** — Captures conversation messages per thread for the State tab.

### Example: LangChain Chatbot

```python
# agents/chatbot/__init__.py
from agentomatic.core.manifest import AgentManifest

manifest = AgentManifest(
    name="chatbot",
    slug="my-langchain-chatbot",
    description="A conversational chatbot using LangChain",
    framework="langchain",  # ← This triggers the LangChain adapter
)

async def node_fn(state: dict) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{query}"),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm

    result = await chain.ainvoke({"query": state["current_query"]})
    return {"response": result.content}
```

That's it! Drop this agent into your `agents/` folder and launch with `agentomatic run --studio`. The Studio will automatically:

1. Show a chain-style graph in the Graph View
2. Stream LLM tokens in real-time via SSE
3. Track conversation state per thread
4. Record execution history for the History tab

### Advanced: Exposing LCEL Graphs

For richer graph visualization, export your runnable as a module-level variable named `chain`, `runnable`, or `agent`. The `LangChainAdapter` will discover it and extract the real LCEL graph:

```python
# agents/rag_bot/__init__.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([...])
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Export as module-level — Studio will discover this automatically
chain = prompt | llm | parser
```

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
