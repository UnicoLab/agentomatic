---
hide:
  - navigation
---

# Agentomatic

<div align="center">
  <p align="center">
    <img src="assets/logo.png" width="300" alt="agentomatic logo">
  </p>

  <h3>Drop agents, not code. ⚡</h3>
  <p><b>The zero-code multi-agent API platform framework.</b><br>
  Turn any Python function, LangGraph workflow, LangChain pipeline, or Deep Agent into a production-ready microservice — with auto-discovery, SSE streaming, thread persistence, visual debugging, and prompt optimization. Every agent, plugin, pipeline, endpoint, and ingestor is automatically callable <b>sync, async, batch, streaming, or as a tracked background task</b>, with a unified task board and a whole-platform status dashboard.</p>

  <p>
    <a href="https://pypi.org/project/agentomatic/"><img src="https://img.shields.io/pypi/v/agentomatic.svg" alt="PyPI version"></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/tests-1185%20passing-brightgreen.svg" alt="Tests passing">
    <img src="https://img.shields.io/badge/framework-LangGraph%20%7C%20LangChain%20%7C%20Deep%20Agent%20%7C%20Python-purple.svg" alt="Frameworks">
  </p>
</div>

---

## :octicons-zap-24: What is Agentomatic?

**Agentomatic** is a production-ready application server for AI agents. Drop a Python folder containing a manifest and an execution function into your `agents/` directory — Agentomatic auto-discovers the code and mounts a complete FastAPI application with REST endpoints, SSE streaming, database persistence, middleware, telemetry, and a visual debugging studio.

It works with **any agent framework** — LangGraph, LangChain, Deep Agent, or raw Python — and requires **zero boilerplate configuration**.

!!! success "Why teams choose Agentomatic"
    - **Ship in minutes** — go from a Python function to a documented, streaming REST API with one command
    - **Debug visually** — Studio provides graph visualization, time-travel debugging, and live state editing
    - **Scale with confidence** — built-in auth, rate limiting, metrics, telemetry, and PostgreSQL persistence
    - **Stay framework-agnostic** — switch between LangGraph, LangChain, Deep Agent, or plain Python without changing your infrastructure

---

## :material-view-grid-plus: Features at a Glance

### :material-server-network: Platform

<div class="grid cards" markdown>

- :material-flash:{ .lg .middle } **Auto-Discovery & REST API**

    ---

    Drop a folder with `agent.py` (class-based) or `__init__.py` + `manifest` (functional) into `agents/`. Agentomatic generates **26 REST endpoints** per agent automatically — invoke, stream, chat, health, config, threads, feedback, HITL, A2A, and more.

    [:octicons-arrow-right-24: Agent Structure](guide/agent-structure.md)

- :material-waves:{ .lg .middle } **SSE Streaming**

    ---

    Every agent gets synchronous `/invoke` and asynchronous `/invoke/stream` endpoints. Stream intermediate thoughts, tool calls, and final answers to clients in real-time via Server-Sent Events.

    [:octicons-arrow-right-24: Platform Features](guide/platform-features.md)

- :material-database:{ .lg .middle } **Multi-Turn Threads**

    ---

    The `/chat` endpoint manages conversation history automatically. Pass a `thread_id` and Agentomatic handles context. Swap between **MemoryStore**, **SQLite**, or **PostgreSQL** backends.

    [:octicons-arrow-right-24: Storage Backends](guide/storage.md)

- :material-swap-horizontal:{ .lg .middle } **Universal Frameworks**

    ---

    First-class support for **LangGraph**, **LangChain**, **Deep Agent**, and raw Python. The adapter pattern ensures every framework gets the best debugging experience possible.

    [:octicons-arrow-right-24: Deep Agent Integration](guide/deep-agents.md)

- :material-timer-sand:{ .lg .middle } **Universal Execution Modes**

    ---

    Every agent, plugin, pipeline, endpoint, and ingestor runs **sync**, **async**, **batch**, **streaming**, or as a **background task** — automatically. Poll status/progress, stream events, cancel, and get completion webhooks from a unified `/api/v1/tasks` board.

    [:octicons-arrow-right-24: Tasks & Execution Modes](guide/tasks.md)

- :material-heart-pulse:{ .lg .middle } **Unified Status Dashboard**

    ---

    One `/status` HTML page (and `/api/v1/status` JSON) rolls up the health of every agent, plugin, pipeline, endpoint, ingestor, the storage backend, and the task engine into a single control-plane view.

    [:octicons-arrow-right-24: Status Dashboard](guide/status.md)

- :material-file-import:{ .lg .middle } **Ingestion & RAG Packaging**

    ---

    Bring any library (PDF→markdown, loaders, splitters, embedders, vector stores); Agentomatic *packages* it as a discoverable ingestor callable sync/async/as-a-task and usable as a pipeline step. Ops, not implementation.

    [:octicons-arrow-right-24: Ingestion & RAG](guide/ingestion.md)

- :material-vector-polyline:{ .lg .middle } **Composable Pipelines**

    ---

    Chain agents, plugins, endpoints, ingestors, transforms, loops, and sub-pipelines with typed data-passing, conditionals, retries, timeouts, rollback/compensation, and optional schema enforcement.

    [:octicons-arrow-right-24: Pipelines](guide/pipelines.md)

</div>

### :material-bug-play: Debugging & Development

<div class="grid cards" markdown>

- :material-palette:{ .lg .middle } **Agentomatic Studio**

    ---

    Visual debugging environment with graph visualization, SSE node streaming, time-travel debugging, state inspection, and live editing. Works with every framework via universal adapters.

    [:octicons-arrow-right-24: Studio Guide](guide/studio.md)

- :material-message-text:{ .lg .middle } **Chainlit Chat Interface**

    ---

    Built-in conversational testing UI at `/chat`. Token-by-token streaming, conversation history, file uploads, and built-in feedback collection — no frontend code needed.

    [:octicons-arrow-right-24: Chat Interface](guide/debug-ui.md)

- :material-console:{ .lg .middle } **Powerful CLI**

    ---

    Scaffold agents, run the platform, test interactively, inspect configurations, diagnose environments, optimize prompts, and launch debug interfaces — all from the terminal.

    [:octicons-arrow-right-24: CLI Reference](cli/commands.md)

- :material-tune-vertical:{ .lg .middle } **Prompt Optimization & Local Training**

    ---

    DSPy-inspired prompt fitting with **5 optimizer strategies** (GEPA, MIPRO, rewrite, few-shot bootstrap, param search). Train against a local LLM — no cloud keys, no HTTP server — using the `compile → fit → evaluate` ML lifecycle.

    [:octicons-arrow-right-24: Optimization Guide](guide/optimization.md)

</div>

### :material-shield-lock: Production & Operations

<div class="grid cards" markdown>

- :material-shield-key:{ .lg .middle } **Enterprise Middleware**

    ---

    API key auth, token-bucket rate limiting, Prometheus metrics at `/metrics`, structured logging with Loguru, and user feedback capture — toggle globally or per-agent.

    [:octicons-arrow-right-24: Middleware Guide](guide/middleware.md)

- :material-chart-line:{ .lg .middle } **OpenTelemetry Observability**

    ---

    Distributed tracing, spans, and metrics export via the OpenTelemetry SDK. Instrument every request, LLM call, and tool invocation with zero code changes.

    [:octicons-arrow-right-24: Telemetry Guide](guide/telemetry.md)

- :material-file-document-edit:{ .lg .middle } **Prompt Versioning**

    ---

    Track prompt templates in `prompts.json`. Hot-reload changes, A/B test versions, inspect history through the API, and run optimization experiments — all without redeploying.

    [:octicons-arrow-right-24: Prompt Management](guide/prompts.md)

- :material-docker:{ .lg .middle } **Container-Ready**

    ---

    Production `Dockerfile` and `docker-compose.yml` included. Distroless variant available for minimal attack surface. CI/CD-ready with GitHub Actions.

    [:octicons-arrow-right-24: Deployment Guide](guide/deployment.md)

- :material-database-cog:{ .lg .middle } **Per-Agent Connections**

    ---

    Give every agent its **own** authenticated databases, vector stores (RAG), HTTP services, and any custom backend (redis, mongo, …) — declared in `connections.py` with `${ENV}` secrets and purpose tagging.

    [:octicons-arrow-right-24: Connections Guide](guide/connections.md)

- :material-transit-connection-variant:{ .lg .middle } **Custom Endpoints**

    ---

    Expose custom APIs that fan out to multiple deployed models over authenticated upstreams (OAuth2), aggregate results, and feed them into pipelines — no router code.

    [:octicons-arrow-right-24: Endpoints Guide](guide/endpoints.md)

- :material-view-dashboard-edit:{ .lg .middle } **Production Control Plane**

    ---

    Inspect agents, endpoints, and connection health at runtime; drain or re-enable individual agents; toggle maintenance mode — via REST or the Studio Control view.

    [:octicons-arrow-right-24: Control Plane](guide/control-plane.md)

</div>

---

## :material-rocket-launch: The 3-Line Deploy

```python title="main.py"
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder("agents/")  # (1)!
app = platform.build()  # (2)!
# Run: uvicorn main:app --reload  (3)
```

1. Scans the `agents/` directory and auto-discovers all agent packages containing a `manifest`
2. Builds a complete FastAPI application with routes, middleware, storage, and Studio
3. Visit `http://localhost:8000/docs` for your auto-generated OpenAPI specification

That's it. Every folder inside `agents/` becomes a full REST API with streaming, persistence, health checks, and documentation — **no router code, no endpoint wiring, no boilerplate**.

---

## :material-sitemap: Platform Architecture

```mermaid
graph TB
    subgraph Clients["Client Layer"]
        CLI["CLI / Terminal"]
        REST["REST / curl / SDK"]
        STUDIO["Studio UI<br/>(React)"]
        CHAT["Chat UI<br/>(Chainlit)"]
    end

    subgraph Platform["AgentPlatform"]
        direction TB
        MW["Middleware Stack<br/>(Auth · Rate Limit · Metrics · Logging)"]
        RF["Router Factory<br/>(auto-generates 26 routes per agent)"]
        REG["Agent Registry<br/>(auto-discovery from agents/ folder)"]
        PM["Prompt Manager<br/>(hot-reload · versioning · optimization)"]
        STORE["Storage Backend<br/>(Memory · SQLite · PostgreSQL)"]
        TEL["Telemetry & Feedback<br/>(OpenTelemetry · Prometheus)"]
        TASK["Task Engine<br/>(async · batch · progress · webhooks)"]
        RES["Resources<br/>(Plugins · Pipelines · Endpoints · Ingestors)"]
    end

    subgraph Agents["Agent Layer"]
        A1["Agent A<br/>(LangGraph)"]
        A2["Agent B<br/>(LangChain LCEL)"]
        A3["Agent C<br/>(Deep Agent)"]
        A4["Agent D<br/>(Raw Python)"]
    end

    subgraph StudioBackend["Studio Backend"]
        GI["GraphInspector<br/>(graph topology, nodes, edges)"]
        RT["RunTracker<br/>(time-travel, state history)"]
    end

    CLI --> MW
    REST --> MW
    STUDIO --> MW
    CHAT --> MW
    MW --> RF
    RF --> REG
    REG --> A1
    REG --> A2
    REG --> A3
    REG --> A4
    RF --> STORE
    RF --> TEL
    RF --> TASK
    RF --> RES
    TASK --> RES
    REG --> PM
    STUDIO --> StudioBackend
    StudioBackend --> REG
```

!!! info "How it works"
    1. **Agent folders** are dropped into the `agents/` directory — each containing a `manifest` and either a `graph_fn` (for LangGraph/Deep Agent) or a `node_fn` (for LangChain/custom agents)
    2. **`AgentRegistry`** auto-discovers and validates every agent at startup
    3. **`RouterFactory`** generates REST endpoints for each agent — invoke, stream, chat, health, config, prompts, and more
    4. **Middleware** wraps every request with auth, rate limiting, metrics, logging, and telemetry
    5. **Studio** connects to the `GraphInspector` and `RunTracker` for visual debugging

---

## :material-swap-horizontal: How Does It Compare?

| Feature | **Agentomatic** | LangServe | AgentOps | Raw FastAPI |
| :--- | :---: | :---: | :---: | :---: |
| **Auto-generated REST API** | :material-check-bold:{ .success } 26 routes/agent | :material-check:{ .success } Limited | :material-close:{ .error } — | :material-close:{ .error } Manual |
| **SSE streaming** | :material-check-bold:{ .success } Native | :material-check:{ .success } Built-in | :material-close:{ .error } — | :material-close:{ .error } Custom |
| **Multi-framework support** | :material-check-bold:{ .success } LG · LC · DA · Py | :material-close:{ .error } LC only | :material-check:{ .success } Any | :material-close:{ .error } Manual |
| **Visual debugging (Studio)** | :material-check-bold:{ .success } Built-in | :material-close:{ .error } — | :material-check:{ .success } Dashboard | :material-close:{ .error } — |
| **Chat interface** | :material-check-bold:{ .success } Chainlit | :material-close:{ .error } — | :material-close:{ .error } — | :material-close:{ .error } — |
| **Thread persistence** | :material-check-bold:{ .success } Memory · SQL · PG | :material-close:{ .error } — | :material-close:{ .error } — | :material-close:{ .error } Manual |
| **Prompt versioning & optimization** | :material-check-bold:{ .success } DSPy-inspired | :material-close:{ .error } — | :material-close:{ .error } — | :material-close:{ .error } — |
| **Auth, rate limiting, metrics** | :material-check-bold:{ .success } Toggle-based | :material-close:{ .error } — | :material-check:{ .success } Partial | :material-close:{ .error } Manual |
| **OpenTelemetry tracing** | :material-check-bold:{ .success } Auto-instrumented | :material-close:{ .error } — | :material-check:{ .success } Custom | :material-close:{ .error } Manual |
| **CLI scaffolding & management** | :material-check-bold:{ .success } `init` · `run` · `demo` | :material-close:{ .error } — | :material-close:{ .error } — | :material-close:{ .error } — |
| **Zero config required** | :material-check-bold:{ .success } Drop folder & go | :material-close:{ .error } — | :material-close:{ .error } — | :material-close:{ .error } — |

---

## :material-download: Quick Installation

=== "pip (Recommended)"

    ```bash
    pip install agentomatic[all]
    ```

=== "uv"

    ```bash
    uv add agentomatic --extra all
    ```

=== "poetry"

    ```bash
    poetry add agentomatic -E all
    ```

=== "Minimal"

    ```bash
    pip install agentomatic
    ```

!!! tip "Install Extras"
    Agentomatic uses optional extras to keep the base package light:

    | Extra | Includes | Use Case |
    |-------|----------|----------|
    | `all` | Everything below | Full development & production |
    | `langgraph` | LangGraph + LangChain Core | LangGraph-based agents |
    | `ollama` | LangChain-Ollama bindings | Local LLM development |
    | `openai` | LangChain-OpenAI bindings | OpenAI API agents |
    | `ui` | Chainlit chat interface | Conversational testing at `/chat` |
    | `studio` | React-based visual debugger | Graph debugging at `/studio/ui/` |
    | `db` | SQLAlchemy + database drivers | SQLite / PostgreSQL persistence |
    | `optimize` | DeepEval, DSPy metrics | Automatic prompt tuning |
    | `metrics` | Prometheus client | `/metrics` endpoint for monitoring |
    | `cli` | Rich terminal output + questionary | Enhanced CLI experience |
    | `telemetry` | OpenTelemetry SDK | Distributed tracing |

---

## :material-play: Create Your First Agent in 60 Seconds

```bash
# 1. Scaffold a chatbot from the built-in template
agentomatic init my_chatbot --template basic

# 2. Launch the platform with Studio
agentomatic run --studio --reload
```

Your platform is now running:

| Service | URL |
|---------|-----|
| :material-api: **API Docs** (Swagger) | `http://localhost:8000/docs` |
| :material-palette: **Agentomatic Studio** | `http://localhost:8000/studio/ui/` |
| :material-lightning-bolt: **Agent Endpoint** | `http://localhost:8000/api/v1/my_chatbot/invoke` |

=== "curl"

    ```bash
    # 3. Query your agent
    curl -X POST http://localhost:8000/api/v1/my_chatbot/invoke \
      -H "Content-Type: application/json" \
      -d '{"query": "Hello! What can you do?"}'
    ```

=== "Python"

    ```python title="test_agent.py"
    import httpx

    response = httpx.post(
        "http://localhost:8000/api/v1/my_chatbot/invoke",
        json={"query": "Hello! What can you do?"},
    )
    print(response.json())
    ```

=== "Interactive CLI"

    ```bash
    agentomatic test my_chatbot
    ```

**Expected Response:**

```json
{
  "response": "Hello! I'm your chatbot assistant. How can I help you today?",
  "agent_type": "agent-my_chatbot",
  "thread_id": "auto-generated-uuid",
  "suggestions": [],
  "citations": [],
  "steps_taken": ["greeting_node"],
  "metadata": {},
  "duration_ms": 114.2
}
```

---

## :material-key-chain: Core Public API

The main exports you'll use when building with Agentomatic:

| Export | Module | Purpose |
|--------|--------|---------|
| `AgentPlatform` | `agentomatic` | Main entry point — scans folders, builds FastAPI app |
| `AgentManifest` | `agentomatic` | Dataclass defining agent identity (name, slug, framework) |
| `BaseAgentState` | `agentomatic` | Default LangGraph `TypedDict` state with reducers |
| `AgentRegistry` | `agentomatic` | Manages registered agents, lookups, and health checks |
| `PromptManager` | `agentomatic` | Hot-reload prompt templates from `prompts.json` |
| `GraphInspector` | `agentomatic` | Extracts graph topology (nodes, edges) for Studio |
| `RunTracker` | `agentomatic` | Records execution history for time-travel debugging |
| `MemoryStore` | `agentomatic.storage` | In-memory thread storage (development) |
| `SQLAlchemyStore` | `agentomatic.storage` | SQL-based thread storage (production) |
| `BaseGraphAgent` | `agentomatic.agents` | Base class for class-based agents with graph wiring |
| `History` / `EarlyStopping` / `Loss` | `agentomatic` | Keras-style training lifecycle primitives for `fit()` |
| `SQLAlchemyTaskStore` | `agentomatic.tasks` | Durable, multi-worker task persistence backend |
| `BaseIngestor` | `agentomatic.ingestion` | Base class to package any ingestion/RAG job as a resource |

```python title="main.py"
from agentomatic import AgentPlatform, AgentManifest, BaseAgentState
from agentomatic.storage import SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    store=SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/agents"),
    enable_auth=True,
    auth_api_key="my-secret-key",
    enable_metrics=True,
    enable_studio=True,
)
app = platform.build()
```

---

## :material-compass: Where to Go Next

<div class="grid cards" markdown>

- :material-play-circle:{ .lg .middle } **[Quick Start](getting-started/quickstart.md)**

    ---

    Install, scaffold, and run your first agent in under 60 seconds with tabbed examples for every framework.

- :material-pencil:{ .lg .middle } **[Your First Agent](getting-started/first-agent.md)**

    ---

    Step-by-step tutorial building an agent from scratch with annotated code and debugging.

- :material-folder-cog:{ .lg .middle } **[Agent Structure](guide/agent-structure.md)**

    ---

    Understand the convention-over-configuration folder layout — manifest, graph_fn, node_fn, and more.

- :material-palette:{ .lg .middle } **[Agentomatic Studio](guide/studio.md)**

    ---

    Visual debugging with graph view, state inspection, time-travel, and live editing.

- :material-robot:{ .lg .middle } **[Deep Agent Integration](guide/deep-agents.md)**

    ---

    Register and debug Deep Agent workflows with full Studio support.

- :material-tune-vertical:{ .lg .middle } **[Prompt Optimization](guide/optimization.md)**

    ---

    Auto-tune your prompts with DSPy-inspired optimization loops and evaluation metrics.

- :material-console:{ .lg .middle } **[CLI Reference](cli/commands.md)**

    ---

    Every command, flag, and workflow documented — `init`, `run`, `test`, `demo`, `optimize`, and more.

- :material-cogs:{ .lg .middle } **[Architecture](architecture/overview.md)**

    ---

    Deep dive into platform internals, request flow, adapters, and design decisions.

- :material-code-braces:{ .lg .middle } **[Class-Based Agents](guide/class-agents.md)**

    ---

    Build agents as Python classes with typed state, graph wiring, and ML lifecycle.

- :material-book-open-page-variant:{ .lg .middle } **[Cookbook & Recipes](guide/cookbook.md)**

    ---

    10 copy-paste patterns: RAG, routing, HITL, schemas, prompts, and ML plugins.

</div>

---

<div align="center" markdown>

[Get Started :material-arrow-right:](getting-started/quickstart.md){ .md-button .md-button--primary }
[Your First Agent :material-pencil:](getting-started/first-agent.md){ .md-button }
[CLI Reference :material-console:](cli/commands.md){ .md-button }
[View on GitHub :material-github:](https://github.com/UnicoLab/agentomatic){ .md-button }

</div>
