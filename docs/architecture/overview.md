# Architecture Overview

Agentomatic follows a **convention-over-configuration** design where agents are discovered, registered, and served with minimal boilerplate. This page describes the system architecture, component relationships, request lifecycle, and deployment patterns.

---

## System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CL["REST Client\n(curl · SDK · frontend)"]
        ST["Agentomatic Studio\n(React debug UI)"]
        CH["Chainlit\n(Chat UI)"]
        A2AC["A2A Client\n(Agent-to-Agent)"]
    end

    subgraph "Middleware Pipeline"
        AUTH["Auth Middleware\nAPI key · JWT · Custom"]
        RL["Rate Limiter\nPer-user · Per-agent"]
        CORS["CORS\nCross-origin config"]
        OTEL["OpenTelemetry\nTraces · Spans"]
        PROM["Prometheus\nHistograms · Counters"]
        LOG["Loguru\nStructured logging"]
    end

    subgraph "Agentomatic Platform Core"
        AP["AgentPlatform\nCentral orchestrator"]
        REG["AgentRegistry\nIn-memory agent store"]
        RF["RouterFactory\nAuto-generated REST\n(26 endpoints per agent)"]
        SR["StudioRouter\nDebug API (/studio/*)"]
        MM["MemoryManager\nConversation history\nSummarization · Windowing"]
        FB["FeedbackCollector\nRatings · Corrections"]
    end

    subgraph "Agent Layer"
        A1["Agent A\n(LangGraph)\ngraph_fn → CompiledGraph"]
        A2["Agent B\n(LangChain)\nnode_fn → LCEL chain"]
        A3["Agent C\n(Raw Python)\nnode_fn → async callable"]
        A4["Agent D\n(Deep Agent)\ngraph_fn → Subagent graph"]
        A5["Agent E\n(Class Agent)\nBaseGraphAgent → AgentGraph"]
    end

    subgraph "Studio Adapter Layer"
        LGA["LangGraphAdapter\n✅ Real graph topology\n✅ Checkpoints + time-travel\n✅ SSE via astream_events\n✅ HITL breakpoints"]
        LCA["LangChainAdapter\n✅ LCEL graph extraction\n✅ SSE via astream_events\n○ Synthetic checkpoints"]
        GA["GenericAdapter\n✅ Synthetic graph\n✅ Trace-based events\n○ Basic state tracking"]
    end

    subgraph "Storage Layer"
        MEM["MemoryStore\nIn-process dict\n(development)"]
        SQL["SQLAlchemyStore\nPostgreSQL · SQLite\n(production)"]
        CP["AgentomaticCheckpointer\nLangGraph ↔ BaseStore\nbridge"]
    end

    subgraph "Observability"
        OT["OpenTelemetry Collector"]
        PM["Prometheus /metrics"]
        LK["Loguru → stdout/file"]
    end

    %% Client → Middleware
    CL --> AUTH
    ST --> AUTH
    CH --> AUTH
    A2AC --> AUTH

    %% Middleware chain
    AUTH --> RL --> CORS --> AP

    %% Platform routing
    AP --> RF
    AP --> SR
    AP --> REG

    %% Router → Registry → Agent
    RF --> REG
    SR --> REG
    REG --> A1
    REG --> A2
    REG --> A3
    REG --> A4
    REG --> A5

    %% Agent → Adapter
    A1 --> LGA
    A2 --> LCA
    A3 --> GA
    A4 --> LGA
    A5 --> GA

    %% Storage
    RF --> MM --> SQL
    RF --> FB
    LGA --> CP --> SQL
    LGA --> CP --> MEM

    %% Observability
    AP --> OTEL --> OT
    AP --> PROM --> PM
    AP --> LOG --> LK

    style AP fill:#4a9eff,color:#fff
    style REG fill:#4a9eff,color:#fff
    style RF fill:#4a9eff,color:#fff
    style LGA fill:#ff6b6b,color:#fff
    style CP fill:#ff6b6b,color:#fff
    style SQL fill:#51cf66,color:#fff
```

---

## Component Descriptions

### AgentPlatform

The central orchestrator. Created via `AgentPlatform.from_folder()`:

1. **Scans** the agents directory for valid agent folders
2. **Imports** each agent's `__init__.py` to extract `manifest` and `node_fn`/`graph_fn`
3. **Registers** agents into the `AgentRegistry`
4. **Generates** REST endpoints per agent via `RouterFactory`
5. **Mounts** middleware, Studio router, and Chainlit UI as requested

```python
from agentomatic import AgentPlatform
from agentomatic.storage import SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    store=SQLAlchemyStore("postgresql+asyncpg://..."),
    enable_studio=True,
    enable_chainlit=True,
)
app = platform.app  # FastAPI application
```

### AgentRegistry

An in-memory registry holding `RegisteredAgent` instances. Each agent carries:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Machine name (folder name) |
| `slug` | `str` | URL-safe identifier |
| `manifest` | `AgentManifest` | Name, description, framework, version |
| `node_fn` | `Callable` | Async callable that processes state |
| `graph_fn` | `Callable \| None` | LangGraph `StateGraph` factory |
| `config` | `BaseModel \| None` | Pydantic configuration |
| `prompt_manager` | `PromptManager \| None` | Template versioning from `prompts.json` |
| `module_path` | `str \| None` | Python module path for schema discovery |

### RouterFactory

Auto-generates a full FastAPI router per agent with 26 endpoints:

| Category | Endpoints | Description |
|---|---|---|
| **Execution** | `POST /invoke`, `POST /invoke/stream`, `POST /chat` | Sync, streaming, session-aware |
| **A2A Protocol** | `GET /card`, `POST /a2a/tasks`, `GET /a2a/tasks/{id}` | Agent-to-agent interop |
| **Threads** | `POST /threads`, `GET /threads`, `GET /threads/{id}`, `PATCH /threads/{id}`, `DELETE /threads/{id}` | Full CRUD |
| **Messages** | `GET /threads/{id}/messages`, `DELETE /threads/{id}/messages`, `GET /threads/{id}/summary` | History + summarization |
| **HITL** | `GET /threads/{id}/pending`, `POST /threads/{id}/approve`, `POST /threads/{id}/reject` | Human-in-the-loop approval |
| **Forking** | `POST /threads/{id}/fork`, `GET /threads/{id}/lineage` | Thread branching + ancestry |
| **Feedback** | `POST /feedback`, `GET /feedback`, `GET /feedback/export` | User ratings + JSONL export |
| **Inspection** | `GET /health`, `GET /config`, `GET /prompts` | Agent health + config |
| **Optimization** | `POST /optimize/invoke` | Full-context pipeline for DeepEval |

### Studio Adapters

The Studio uses an **adapter pattern** to provide debugging capabilities across different agent frameworks:

```mermaid
graph LR
    subgraph "Resolution Chain"
        R["resolve_adapter()"]
    end

    subgraph "Adapters"
        R -->|"graph_fn present?"| LG["LangGraphAdapter"]
        R -->|"framework=graph_agent?"| GA["GraphAgentAdapter"]
        R -->|"framework=langchain?"| LC["LangChainAdapter"]
        R -->|"_studio_adapter set?"| CU["Custom Adapter"]
        R -->|"fallback"| GN["GenericAdapter"]
    end

    subgraph "Capabilities"
        LG --> G1["✅ Real Graph Topology"]
        LG --> G2["✅ Checkpoints + Time-travel"]
        LG --> G3["✅ HITL Breakpoints"]
        LG --> G4["✅ SSE via astream_events"]
        GA --> A1["✅ Graph Topology"]
        GA --> A2["✅ SSE Streaming"]
        GA --> A3["✅ Execution Traces"]
        LC --> L1["✅ LCEL Graph"]
        LC --> L2["✅ astream_events"]
        GN --> N1["✅ Synthetic Graph"]
        GN --> N2["✅ Trace Events"]
    end
```

Each adapter implements the `StudioAdapter` ABC:

- **`get_graph()`** — Returns `StudioGraphTopology` with nodes and edges
- **`stream_execution()`** — Yields `StudioRunEvent` via SSE
- **`get_state()`** — Returns `StudioStateSnapshot` for a thread
- **`update_state()`** — Applies partial state updates
- **`get_history()`** — Returns checkpoint history

### Storage Layer

Abstract `BaseStore` with swappable implementations:

| Store | Backend | Use Case |
|---|---|---|
| `MemoryStore` | In-process `dict` | Development, testing, CI |
| `SQLAlchemyStore` | PostgreSQL / SQLite | Production deployment |

The `AgentomaticCheckpointer` bridges LangGraph's `BaseCheckpointSaver` to `BaseStore`, enabling checkpoint persistence through the same storage backend used for threads and messages.

---

## Request Lifecycle

### Invoke Flow

```mermaid
sequenceDiagram
    participant Client
    participant Middleware
    participant RouterFactory
    participant Registry
    participant MemoryManager
    participant Agent
    participant Storage

    Client->>+Middleware: POST /api/v1/{agent}/invoke
    Middleware->>Middleware: Auth · Rate Limit · Metrics
    Middleware->>+RouterFactory: Authenticated Request

    RouterFactory->>+Registry: get(agent_name)
    Registry-->>-RouterFactory: RegisteredAgent

    RouterFactory->>RouterFactory: _build_initial_state(request)

    alt Thread store configured
        RouterFactory->>+MemoryManager: get_or_create_thread()
        MemoryManager->>+Storage: get/create thread
        Storage-->>-MemoryManager: thread_id
        MemoryManager-->>-RouterFactory: thread_id

        RouterFactory->>+MemoryManager: load_history(thread_id, query)
        MemoryManager->>+Storage: get_messages()
        Storage-->>-MemoryManager: messages
        MemoryManager-->>-RouterFactory: LangChain messages
    end

    RouterFactory->>RouterFactory: Execute before_node hooks

    alt Has graph_fn
        RouterFactory->>+Agent: graph.ainvoke(state)
    else Has node_fn
        RouterFactory->>+Agent: node_fn(state)
    end
    Agent-->>-RouterFactory: result dict

    RouterFactory->>RouterFactory: Execute after_node hooks

    alt Thread store configured
        RouterFactory->>+MemoryManager: save_turn(query, response)
        MemoryManager->>+Storage: add_message(user) + add_message(assistant)
        Storage-->>-MemoryManager: persisted
        MemoryManager-->>-RouterFactory: done
    end

    RouterFactory->>RouterFactory: _extract_response(result)
    RouterFactory-->>-Middleware: AgentInvokeResponse
    Middleware-->>-Client: JSON Response
```

### HITL Suspend/Resume Flow

```mermaid
sequenceDiagram
    participant Client
    participant Router
    participant Agent
    participant Storage

    Client->>+Router: POST /invoke
    Router->>+Agent: graph.ainvoke(state)
    Agent-->>-Router: raises AgentSuspendedException

    Router->>+Storage: save_suspended_state()
    Storage-->>-Router: saved

    Router-->>-Client: 202 Accepted + approval_id

    Note over Client: Human reviews and approves

    Client->>+Router: POST /threads/{id}/approve
    Router->>+Storage: get_suspended_state(approval_id)
    Storage-->>-Router: state_snapshot

    Router->>+Storage: delete_suspended_state()
    Storage-->>-Router: deleted

    Router->>Router: Merge human context into state
    Router->>Router: Set metadata.hitl_approved = true

    Router->>+Agent: graph.ainvoke(resumed_state)
    Agent-->>-Router: result

    Router-->>-Client: AgentInvokeResponse
```

---

## Agent Registration Flow

```mermaid
graph TD
    SCAN["AgentPlatform.from_folder('agents/')"] --> DISC["Discovery: scan for __init__.py"]
    DISC --> IMP["Import each agent module"]

    IMP --> EXTRACT["Extract:\n• manifest (required)\n• graph_fn or node_fn\n• config (optional)\n• prompts.json (optional)"]

    EXTRACT --> VAL["Validate manifest"]

    VAL --> REG["Register in AgentRegistry"]

    REG --> SCHEMA["Discover custom schemas\n(schemas.py module)"]

    SCHEMA --> ROUTER["RouterFactory creates\n26 endpoints per agent"]

    ROUTER --> STUDIO["Studio adapter resolved\n(LangGraph/LangChain/Generic)"]

    ROUTER --> MOUNT["Mount on FastAPI app\nat /api/v1/{agent_name}/"]

    style SCAN fill:#4a9eff,color:#fff
    style ROUTER fill:#ff6b6b,color:#fff
    style MOUNT fill:#51cf66,color:#fff
```

### Agent Folder Structure

```
agents/
├── my_agent/
│   ├── __init__.py       # Required: manifest + graph_fn/node_fn
│   ├── prompts.json      # Optional: prompt versions
│   ├── schemas.py        # Optional: custom request/response models
│   ├── config.py         # Optional: Pydantic config class
│   └── api.py            # Optional: custom router (replaces auto-generated)
└── another_agent/
    └── __init__.py
```

---

## Deployment Patterns

### Single Process (Development)

```mermaid
graph LR
    UV["uvicorn main:app"] --> AP["AgentPlatform"]
    AP --> MEM["MemoryStore"]
```

```python
# main.py
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore

platform = AgentPlatform.from_folder("agents/", store=MemoryStore())
app = platform.app

# uvicorn main:app --reload --port 8000
```

### Multi-Worker (Staging / Production)

```mermaid
graph TB
    LB["Load Balancer\n(nginx / ALB)"]
    LB --> W1["Gunicorn Worker 1\n(uvicorn)"]
    LB --> W2["Gunicorn Worker 2\n(uvicorn)"]
    LB --> W3["Gunicorn Worker 3\n(uvicorn)"]

    W1 --> DB["PostgreSQL\n(SQLAlchemyStore)"]
    W2 --> DB
    W3 --> DB

    W1 --> OTEL["OpenTelemetry\nCollector"]
    W2 --> OTEL
    W3 --> OTEL
```

```bash
gunicorn main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000
```

!!! warning "MemoryStore is Not Multi-Worker Safe"
    Use `SQLAlchemyStore` when running with multiple workers. `MemoryStore` is per-process and does not share state.

### Containerized (Kubernetes)

```mermaid
graph TB
    ING["Ingress Controller"] --> SVC["K8s Service"]

    subgraph "Deployment (replicas: 3)"
        SVC --> P1["Pod 1\nAgentomatic"]
        SVC --> P2["Pod 2\nAgentomatic"]
        SVC --> P3["Pod 3\nAgentomatic"]
    end

    subgraph "Stateful Services"
        P1 --> PG["PostgreSQL\n(StatefulSet / RDS)"]
        P2 --> PG
        P3 --> PG

        P1 --> REDIS["Redis\n(Optional: rate limiting)"]
        P2 --> REDIS
        P3 --> REDIS
    end

    subgraph "Observability Stack"
        P1 --> PROM["Prometheus"]
        P1 --> JAEGER["Jaeger / Tempo\n(traces)"]
    end
```

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[all]"
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# k8s deployment excerpt
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: agentomatic
          image: myregistry/agentomatic:latest
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
          readinessProbe:
            httpGet:
              path: /readiness
              port: 8000
```

---

## Class-Agent Architecture (v0.7)

The **Class-Owned Graph Agent** system introduces an alternative agent paradigm: define agents as Python classes with an ML-inspired lifecycle, using a built-in graph runtime that requires no LangGraph dependency.

### Component Overview

```mermaid
graph TB
    subgraph "Platform Integration"
        AP["AgentPlatform"]
        REG["AgentRegistry"]
        RA["RegisteredAgent"]
    end

    subgraph "Class-Agent System"
        BGA["BaseGraphAgent\n(user subclass)"]
        BG["build_graph()\nnew_graph()"]
        GB["GraphBuilder\nLangGraph-compatible API"]
        AG["AgentGraph\nlightweight runtime"]
    end

    subgraph "Mixins"
        GEM["GraphExecutionMixin\ncompile · transform"]
        DSM["DatasetMixin\nload · split · iterate"]
        EVM["EvaluationMixin\nevaluate · report"]
        OPM["OptimizationMixin\nfit · optimize"]
        SRM["SerializationMixin\nsave · load_compiled"]
        TRM["TracingMixin\nTraceEvent logging"]
    end

    subgraph "ML Lifecycle"
        DS["AgentDataset\nAgentExample"]
        MET["Metrics\nExactKeyMatch\nContainsTerms\nCallable"]
        OPT["Optimizers\nNoOp · GridSearch\nPromptFitterBridge"]
    end

    subgraph "Registration"
        RCA["register_class_agent()"]
        DISC["Auto-discovery\n(agent.py)"]
    end

    %% Class agent construction
    BGA --> AN
    AN --> GB
    GB --> AG

    %% Mixins compose into BaseGraphAgent
    GEM --> BGA
    DSM --> BGA
    EVM --> BGA
    OPM --> BGA
    SRM --> BGA
    TRM --> BGA

    %% ML lifecycle
    DS --> EVM
    DS --> OPM
    MET --> EVM
    OPT --> OPM

    %% Registration into platform
    BGA --> RCA
    DISC --> RCA
    RCA --> RA
    RA --> REG
    REG --> AP

    style BGA fill:#51cf66,color:#fff
    style AG fill:#51cf66,color:#fff
    style AP fill:#4a9eff,color:#fff
    style REG fill:#4a9eff,color:#fff
```

### ML Lifecycle

Class agents follow a training-inspired workflow:

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Agent as BaseGraphAgent
    participant Graph as AgentGraph
    participant DS as AgentDataset
    participant Opt as Optimizer

    Dev->>Agent: compile(dataset, metrics, optimizer)
    Agent->>Graph: Build graph from build_graph()
    Agent->>Agent: Store metrics + optimizer refs

    Dev->>Agent: fit(dataset)
    Agent->>Opt: optimize(agent, dataset)
    Opt-->>Agent: Optimized parameters

    Dev->>Agent: evaluate(test_split, metrics)
    loop Each example
        Agent->>Graph: Execute graph
        Graph-->>Agent: Output
        Agent->>Agent: Score with metrics
    end
    Agent-->>Dev: EvaluationReport

    Dev->>Agent: transform(input_data)
    Agent->>Graph: Execute compiled graph
    Graph-->>Agent: State
    Agent-->>Dev: Output dict

    Dev->>Agent: save("path/")
    Agent-->>Dev: Serialized state + config
```

### Key Classes

| Class | Module | Purpose |
|---|---|---|
| `BaseGraphAgent[S]` | `agentomatic.agents` | Abstract base — subclass to define your agent |
| `build_graph()` | `BaseGraphAgent` | Primary override — wire nodes with `new_graph()` |
| `GraphBuilder` | `agentomatic.agents.builder` | LangGraph-compatible API for graph construction |
| `AgentGraph` | `agentomatic.agents.graph` | Lightweight graph runtime (sync + async) |
| `AgentDataset` | `agentomatic.agents.types` | JSONL-backed dataset with train/test splits |
| `AgentExample` | `agentomatic.agents.types` | Single input/expected-output pair |
| `TraceEvent` | `agentomatic.agents.types` | Per-node execution trace event |
| `register_class_agent()` | `AgentRegistry` | Register a class agent into the platform |

### Integration with Platform

Class agents integrate with the platform through `register_class_agent()`, which wraps the agent's `transform()` method as a standard `node_fn` and registers it as a `RegisteredAgent`:

```python
from __future__ import annotations

from agentomatic.agents import BaseGraphAgent


class MyAgent(BaseGraphAgent[MyState]):
    agent_name = "my_agent"

    def build_graph(self):
        g = self.new_graph()
        g.add_node("process", self.process)
        g.set_entry_point("process")
        g.set_finish_point("process")
        return g.compile()

    # ... methods ...

# Register into the platform
agent = MyAgent(llm=my_llm)
agent.compile(dataset, metrics)
registry.register_class_agent(agent)
```

Alternatively, the platform auto-discovers class agents via `agent.py` files using the `agentomatic init --template class` scaffold.

---

## Extension Points

Agentomatic is designed for extensibility at every layer:

| Extension Point | Mechanism | Example |
|---|---|---|
| **Custom agents** | Drop folder in `agents/` | Any Python async function |
| **Custom routers** | `api.py` in agent folder | Replace auto-generated endpoints |
| **Custom schemas** | `schemas.py` in agent folder | Domain-specific request/response models |
| **Custom adapters** | `_studio_adapter` attribute | Framework-specific Studio integration |
| **Before/after hooks** | `register_before_node_hook()` | Audit logging, security scanning |
| **Storage backends** | Subclass `BaseStore` | Custom databases, cloud storage |
| **Middleware** | FastAPI middleware | Auth, rate limiting, custom headers |
| **Prompt versions** | `prompts.json` | A/B testing, version management |
| **LLM providers** | `get_llm(provider=...)` | OpenAI, Azure, Ollama, Anthropic |
| **Checkpointers** | `AgentomaticCheckpointer` | Bridge any store to LangGraph |

---

## Key Design Decisions

1. **Convention over configuration** — Drop a folder, get a full API
2. **Everything is optional** — Only `__init__.py` with a manifest is required
3. **Override anything** — Custom `api.py` routers replace auto-generated ones
4. **Async-first** — All I/O uses `async`/`await`
5. **ABC-based storage** — Swap backends without code changes
6. **Universal Studio** — Adapter pattern degrades gracefully across frameworks
7. **Middleware pipeline** — Composable, ordered middleware with per-request context
8. **Schema discovery** — Custom Pydantic models auto-integrate into OpenAPI docs
9. **Checkpoint bridge** — Single storage backend for threads, messages, and LangGraph checkpoints
10. **HITL as first-class** — Suspend/resume built into the router factory, not bolted on
11. **Class agents** — ML lifecycle (`compile`/`fit`/`evaluate`/`transform`) with zero framework deps
12. **Composable pipelines** — YAML, Builder, and decorator interfaces for multi-agent orchestration
