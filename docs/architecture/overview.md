# Architecture Overview

## Core Components

```mermaid
graph TB
    subgraph "AgentPlatform"
        P[Platform] --> R[AgentRegistry]
        P --> S[Storage Backend]
        P --> MW[Middleware Stack]
        P --> RF[RouterFactory]
    end

    subgraph "Per Agent"
        R --> A[RegisteredAgent]
        A --> M[AgentManifest]
        A --> G["graph_fn / node_fn"]
        A --> C[Config]
        A --> PM[PromptManager]
    end

    subgraph "Auto-Generated"
        RF --> E1["POST /invoke"]
        RF --> E2["POST /invoke/stream"]
        RF --> E3["POST /chat"]
        RF --> E4["POST /a2a/tasks"]
        RF --> E5["GET /health"]
        RF --> E6["GET /card"]
    end
```

## Request Flow

```mermaid
sequenceDiagram
    Client->>+Middleware: HTTP Request
    Middleware->>+RouterFactory: Authenticated
    RouterFactory->>+Registry: get(agent_name)
    Registry-->>-RouterFactory: RegisteredAgent
    RouterFactory->>+Agent: graph.ainvoke(state)
    Agent-->>-RouterFactory: result
    RouterFactory->>+Storage: save thread/messages
    Storage-->>-RouterFactory: ok
    RouterFactory-->>-Middleware: Response
    Middleware-->>-Client: JSON / SSE
```

## Key Design Decisions

1. **Convention over configuration** — drop a folder, get an API
2. **Everything is optional** — only `__init__.py` required
3. **Override anything** — custom routers replace auto-generated ones
4. **Async-first** — all I/O is async
5. **ABC-based storage** — swap backends without code changes
