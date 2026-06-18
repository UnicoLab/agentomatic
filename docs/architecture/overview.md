# Architecture Overview

Agentomatic follows a **convention-over-configuration** design where agents are discovered, registered, and served with minimal boilerplate.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CL["REST Client<br/>(curl, SDK, frontend)"]
        ST["Agentomatic Studio<br/>(React debug UI)"]
        CH["Chainlit<br/>(Chat UI)"]
    end

    subgraph "Agentomatic Platform"
        MW["Middleware Stack<br/>Auth · Rate Limit · Metrics · Logging"]
        RF["RouterFactory<br/>Auto-generated endpoints per agent"]
        SR["Studio Router<br/>Debug API (/studio/)"]
        REG["Agent Registry"]
        SM["Storage Manager"]
    end

    subgraph "Agent Layer"
        A1["Agent A<br/>(LangGraph)"]
        A2["Agent B<br/>(LangChain)"]
        A3["Agent C<br/>(Raw Python)"]
    end

    subgraph "Studio Adapter Layer"
        LGA["LangGraphAdapter<br/>Full graph + checkpoints"]
        LCA["LangChainAdapter<br/>LCEL + astream_events"]
        GA["GenericAdapter<br/>Trace-based fallback"]
    end

    subgraph "Infrastructure"
        DB["Storage Backend<br/>Memory · SQLite · PostgreSQL"]
        OT["OpenTelemetry<br/>Traces · Metrics"]
        PM["Prometheus<br/>Histograms · Counters"]
    end

    CL --> MW
    ST --> SR
    CH --> RF
    MW --> RF
    MW --> SR
    RF --> REG
    SR --> REG
    REG --> A1
    REG --> A2
    REG --> A3
    A1 --> LGA
    A2 --> LCA
    A3 --> GA
    RF --> SM
    SM --> DB
    RF --> OT
    MW --> PM
```

---

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant Middleware
    participant Router
    participant Registry
    participant Agent
    participant Storage

    Client->>+Middleware: HTTP Request
    Middleware->>Middleware: Auth · Rate Limit · Metrics
    Middleware->>+Router: Authenticated Request
    Router->>+Registry: get_agent(name)
    Registry-->>-Router: RegisteredAgent
    Router->>+Agent: invoke / stream / chat
    Agent->>Agent: graph_fn or node_fn execution
    Agent-->>-Router: Result
    Router->>+Storage: Save thread + messages
    Storage-->>-Router: Persisted
    Router-->>-Middleware: Response
    Middleware-->>-Client: JSON / SSE Stream
```

---

## Studio Architecture

The Studio uses an **adapter pattern** to provide debugging capabilities across different agent frameworks:

```mermaid
graph LR
    subgraph "Resolution Chain"
        R["resolve_adapter()"]
    end

    subgraph "Adapters"
        R -->|"graph_fn?"| LG["LangGraphAdapter"]
        R -->|"framework=langchain?"| LC["LangChainAdapter"]
        R -->|"fallback"| GN["GenericAdapter"]
        R -->|"_studio_adapter?"| CU["Custom Adapter"]
    end

    subgraph "Capabilities"
        LG --> G1["✅ Real Graph"]
        LG --> G2["✅ Checkpoints"]
        LG --> G3["✅ HITL"]
        LC --> L1["✅ LCEL Graph"]
        LC --> L2["✅ astream_events"]
        GN --> N1["✅ Synthetic Graph"]
        GN --> N2["✅ Trace Events"]
    end
```

---

## Core Components

### AgentPlatform

The central orchestrator. Created via `AgentPlatform.from_folder()` which:

1. **Scans** the agents directory for valid agent folders
2. **Imports** each agent's `__init__.py` to extract `manifest` and `node_fn`/`graph_fn`
3. **Registers** agents into the `AgentRegistry`
4. **Generates** REST endpoints per agent via `RouterFactory`
5. **Mounts** middleware, Studio router, and Chainlit UI as requested

### Agent Registry

An in-memory registry that holds `RegisteredAgent` instances. Each agent carries:

- **manifest** — `AgentManifest` with name, description, framework, schemas
- **node_fn** — The async callable that processes state
- **graph_fn** — Optional LangGraph `StateGraph` factory
- **config** — Optional Pydantic configuration class
- **prompt_manager** — Template versioning from `prompts.json`

### RouterFactory

Auto-generates 20+ FastAPI endpoints per agent:

| Category | Endpoints |
|---|---|
| **Execution** | `POST /invoke`, `POST /invoke/stream`, `POST /chat` |
| **A2A Protocol** | `GET /card`, `POST /a2a/tasks` |
| **Threads** | `GET/POST /threads`, `GET /threads/{id}/messages` |
| **HITL** | `POST /threads/{id}/approve`, `POST /threads/{id}/reject` |
| **Inspection** | `GET /health`, `GET /config`, `GET /prompts` |
| **Forking** | `POST /threads/{id}/fork`, `GET /threads/{id}/lineage` |

### Storage Layer

Abstract `BaseStore` with implementations:

- **MemoryStore** — In-process dict, perfect for development
- **SQLAlchemyStore** — PostgreSQL / SQLite via async SQLAlchemy

---

## Key Design Decisions

1. **Convention over configuration** — Drop a folder, get a full API
2. **Everything is optional** — Only `__init__.py` is required
3. **Override anything** — Custom `api.py` routers replace auto-generated ones
4. **Async-first** — All I/O uses async/await
5. **ABC-based storage** — Swap backends without code changes
6. **Universal Studio** — Adapter pattern degrades gracefully across frameworks
7. **Middleware pipeline** — Composable, ordered middleware with per-request context
