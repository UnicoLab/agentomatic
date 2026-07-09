---
name: agentomatic
description: |
  Design, build, and debug multi-agent AI systems using the Agentomatic platform.
  Activate this skill when working with agentomatic agent creation, configuration,
  Studio debugging, LangGraph integration, prompt optimization, or CLI commands.
---

# Agentomatic Development Skill

Use this skill when creating, modifying, debugging, or extending agents built on the Agentomatic platform.

## Package Overview

**Agentomatic** is a zero-code multi-agent API platform framework built on FastAPI.
It auto-generates REST API endpoints from agent folders and provides a built-in
Studio debug UI, prompt optimization, storage backends, and observability.

For production deployments it also provides (see dedicated sections below):

- **Custom Endpoints** — user-defined APIs that call deployed ML model services
  via authenticated `httpx` requests and aggregate their outputs (usable as
  pipeline steps to feed context into agents)
- **Per-Agent Connections** — scoped, authenticated databases, vector stores
  (RAG / vector search), HTTP services, and *any* backend via a factory, with
  purpose tagging (memory / rag / vector / cache …) and caching
- **Production Control Plane** — an admin API to inspect/drain agents and toggle
  maintenance mode
- **Security** — API-key auth, JWT/OAuth2, and per-agent zero-trust policies
- **Observability** — Prometheus metrics + OpenTelemetry tracing with a
  ready-to-run Grafana/Prometheus/OTel stack in `deploy/observability/`

```
pip install agentomatic                    # Core
pip install "agentomatic[langgraph]"       # + LangGraph support
pip install "agentomatic[all]"             # Everything
```

## Architecture

```
src/agentomatic/
├── __init__.py              # Public API: AgentPlatform, AgentManifest, BaseAgentState, BaseGraphAgent, AgentGraph, GraphBuilder
├── core/
│   ├── platform.py          # AgentPlatform — central orchestrator
│   ├── registry.py          # AgentRegistry — agent registration and lookup
│   ├── manifest.py          # AgentManifest model + RegisteredAgent dataclass
│   ├── router_factory.py    # Auto-generates 26 REST endpoints per agent
│   ├── state.py             # BaseAgentState — typed agent state
│   ├── memory_manager.py    # ConversationMemoryManager
│   └── lifespan.py          # FastAPI lifespan events
├── studio/
│   ├── adapters/
│   │   ├── __init__.py      # resolve_adapter() factory
│   │   ├── langgraph.py     # LangGraphAdapter — full graph/streaming/HITL
│   │   ├── langchain.py     # LangChainAdapter — LCEL streaming
│   │   └── generic.py       # GenericAdapter — fallback for any framework
│   ├── router.py            # Studio API endpoints (graph, stream, state, resume)
│   ├── models.py            # Pydantic models (StudioRunEvent, StudioGraphTopology, etc.)
│   ├── decorators.py        # @studio_graph, @studio_state, @studio_stream
│   ├── graph_inspector.py   # Runtime graph introspection
│   ├── run_tracker.py       # Execution run tracking
│   └── serve.py             # Static file serving for React UI
├── cli/
│   ├── commands.py          # CLI: init, run, demo, list, inspect, doctor, optimize
│   └── templates.py         # Scaffolding: basic, full, rag, chatbot, deepagent, custom, legacy_dict, plugin, endpoint, connection
├── storage/
│   ├── base.py              # BaseStore ABC
│   ├── memory.py            # MemoryStore — in-memory dict
│   ├── sqlalchemy.py        # SQLAlchemyStore — PostgreSQL/SQLite
│   ├── checkpointer.py      # LangGraph-compatible checkpoint serde
│   └── models.py            # SQLAlchemy ORM models
├── middleware/
│   ├── auth.py              # API key authentication
│   ├── rate_limit.py        # Token bucket rate limiting
│   ├── metrics.py           # Prometheus metrics
│   ├── feedback.py          # Feedback collection
│   ├── logging.py           # Structured logging
│   ├── zero_trust.py        # Per-agent zero-trust policy enforcement
│   └── connections.py       # Attaches request.state.connections per request
├── endpoints/               # Custom endpoints: httpx calls to deployed ML models
│   ├── models.py            # AuthType, UpstreamConfig/Auth, AggregationStrategy
│   ├── auth.py              # ${ENV} + OAuth2 client-credentials (token cache)
│   ├── client.py            # UpstreamClient + MultiModelClient (fan-out/aggregate)
│   ├── base.py              # BaseEndpoint ABC (auto /call, /health, /info)
│   ├── registry.py          # EndpointRegistry — auto-discovery
│   └── router.py            # Dynamic APIRouter per endpoint
├── connections/             # Per-agent connections (any DB, minimal code)
│   ├── models.py            # ConnectionKind/Purpose + *ConnectionConfig models
│   ├── database.py          # DatabaseConnection (SQLAlchemy async) + create_store()
│   ├── vector.py            # VectorConnection + pluggable provider registry
│   ├── http.py              # HttpConnection (authenticated, reuses UpstreamClient)
│   ├── custom.py            # CustomConnection — any backend via a factory
│   └── manager.py           # ConnectionManager, get_connections(scope), registries
├── control/                 # Production control plane
│   ├── state.py             # ControlPlaneState (maintenance, drained agents)
│   ├── middleware.py        # MaintenanceMiddleware (503 gating)
│   ├── models.py            # Control API response models
│   └── router.py            # /api/v1/control admin API
├── security/                # JWT auth + zero-trust policies (AgentSecurityPolicy)
├── optimize/                # DSPy-style prompt optimization (18 modules)
├── observability/           # OpenTelemetry tracing + Prometheus metrics
├── prompts/                 # PromptManager for prompt versioning
├── providers/               # LLM providers (Ollama, OpenAI, Azure, Vertex)
├── config/                  # Settings management
├── ui/                      # Chainlit chat interface
└── demo/                    # Built-in demo agent

deploy/observability/        # Docker Compose: Prometheus + OTel Collector + Grafana
                             # (pre-provisioned "Agentomatic Overview" dashboard)
```

## Creating Agents

### Agent Structure (Class-based Pattern - Recommended)

The primary and recommended way to build an Agentomatic agent is by subclassing `BaseGraphAgent`. This approach provides better encapsulation, allows easy injection of dependencies (like LLMs or databases) in `__init__`, and minimizes boilerplate.

Every agent is a Python package or module in the `agents/` directory:

```
agents/
└── my_agent/
    ├── __init__.py          # Optional: Python package init
    ├── agent.py             # REQUIRED: Contains your BaseGraphAgent subclass
    ├── config.py            # Optional: agent-specific config
    └── prompts.json         # Optional: versioned prompt templates
```

### Example: Class-based Agent (`agent.py`)

```python
# agents/my_agent/agent.py
from dataclasses import dataclass, field
from typing import Any
from agentomatic.agents import BaseGraphAgent

@dataclass
class MyAgentState:
    request: str = ""
    output: dict[str, Any] = field(default_factory=dict)

class MyAgentAgent(BaseGraphAgent[MyAgentState]):
    # These properties automatically populate the AgentManifest
    agent_name = "my_agent"
    agent_description = "A helpful class-based agent"
    agent_framework = "graph_agent"

    def __init__(self, *, llm: Any = None):
        super().__init__()
        self.llm = llm  # easily inject dependencies!

    def build_graph(self):
        # Build the execution graph
        g = self.new_graph()  # returns GraphBuilder (agentomatic's own builder)
        g.add_node("process", self.process)
        g.set_entry_point("process")
        g.set_finish_point("process")
        return g.compile()

    def process(self, state: MyAgentState) -> MyAgentState:
        state.output = {"response": f"Hello! You said: {state.request}"}
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> MyAgentState:
        return MyAgentState(request=input_data.get("current_query", ""))

    def state_to_output(self, state: MyAgentState) -> dict[str, Any]:
        return state.output
```

Agentomatic's registry will automatically discover the `MyAgentAgent` class, instantiate it, and expose it via REST endpoints and the Studio.

### Legacy Functional Pattern (LangGraph Dict)

You can still use the older dict-based pattern where you export `manifest` and `node_fn` or `graph_fn` in `__init__.py`. This is generated if you use `agentomatic init --template legacy_dict`.

### Deep Agent Pattern

```python
# agents/my_agent/agent.py
from functools import lru_cache
from deepagents import create_deep_agent

@lru_cache(maxsize=1)
def create_agent():
    return create_deep_agent(
        model="google_genai:gemini-3.5-flash",
        system_prompt="You are an expert assistant.",
        tools=[my_tool],
    )
```

## Creating ML Plugins

Agentomatic supports classical ML models (PyTorch, TensorFlow, Scikit-learn) via `BaseMLPlugin`. These plugins are auto-discovered from a `plugins/` directory and mapped to strict Pydantic REST endpoints.

```python
# plugins/sentiment/plugin.py
from pydantic import BaseModel, Field
from agentomatic.plugins import BaseMLPlugin

class SentimentInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float

class SentimentPlugin(BaseMLPlugin[SentimentInput, SentimentOutput]):
    plugin_name = "sentiment_analyzer"
    plugin_description = "A classical ML model plugin."
    plugin_version = "1.0.0"

    async def load_model(self) -> None:
        # Load weights into self.model
        self.model = ...
        await super().load_model()

    async def predict(self, inputs: SentimentInput) -> SentimentOutput:
        # Run inference
        return SentimentOutput(sentiment="positive", confidence=0.99)
```

The platform auto-generates:
- `POST /api/v1/plugins/sentiment_analyzer/predict` (strictly typed by Pydantic)
- `GET /api/v1/plugins/sentiment_analyzer/model_card`
- `GET /api/v1/plugins/sentiment_analyzer/health`

## BaseAgentState

The standard state dictionary used across all agents:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `Annotated[list, add_messages]` | Conversation message history (parallel-safe) |
| `thread_id` | `str` | Conversation thread ID |
| `user_id` | `str` | User identifier |
| `current_query` | `str` | The user's input query |
| `response` | `Annotated[str, _last_value]` | Agent's final response (last-writer-wins) |
| `agent_type` | `Annotated[str, _last_value]` | Which agent handled this (last-writer-wins) |
| `suggestions` | `Annotated[list[str], operator.add]` | Follow-up suggestions (merged across branches) |
| `citations` | `Annotated[list[dict], operator.add]` | Source citations (merged across branches) |
| `routing_decision` | `Annotated[str, _last_value]` | Orchestrator routing target |
| `context` | `Annotated[dict, _merge_dicts]` | Additional context (dict-merged) |
| `prompt_version` | `Annotated[str, _last_value]` | Active prompt version |
| `steps_taken` | `Annotated[list[str], operator.add]` | Execution trace (merged across branches) |
| `metadata` | `Annotated[dict, _merge_dicts]` | Request metadata (dict-merged) |
| `error` | `Annotated[str \| None, _last_value]` | Error message if any |

## Platform Configuration

```python
from agentomatic import AgentPlatform
from agentomatic.storage import SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    store=SQLAlchemyStore("postgresql+asyncpg://..."),
    enable_metrics=True,              # Prometheus metrics
    enable_auth=True,                 # API key auth
    auth_api_key="secret",            # Auth key
    enable_rate_limit=True,           # Rate limiting
    rate_limit_requests=100,          # Requests per window
    # --- Security ---
    enable_jwt_auth=True,             # JWT / OAuth2 bearer validation
    enable_zero_trust=True,           # Per-agent role/scope/auth enforcement
    # --- Production control plane ---
    enable_control_plane=True,        # Mount /api/v1/control admin API
    control_token="ops-secret",       # Protects mutating control ops
    # --- Connections (platform-wide; agents also declare their own) ---
    connections=[...],                # DatabaseConnectionConfig, VectorConnectionConfig, ...
    enable_connections_context=True,  # request.state.connections per request (default)
)
app = platform.build()
```

## Custom Endpoints (API calls to deployed ML models)

Custom endpoints call deployed model services with authenticated `httpx`
requests, aggregate their outputs, and can feed context into agents via
pipelines. Auto-discovered from an `endpoints/` directory; scaffold with
`agentomatic init NAME --template endpoint`.

```python
# endpoints/ensemble/endpoint.py
from agentomatic.endpoints import (
    AggregationStrategy, AuthType, BaseEndpoint, UpstreamAuthConfig, UpstreamConfig,
)

class EnsembleEndpoint(BaseEndpoint):
    endpoint_name = "ensemble"
    endpoint_description = "Fan out to several model services and aggregate."
    aggregation = AggregationStrategy.MAJORITY   # ALL | FIRST_SUCCESS | MAJORITY

    upstreams = [
        UpstreamConfig(
            name="model_a",
            base_url="${MODEL_A_URL}",
            auth=UpstreamAuthConfig(
                type=AuthType.OAUTH2_CLIENT_CREDENTIALS,
                token_url="${MODEL_A_TOKEN_URL}",
                client_id="${MODEL_A_CLIENT_ID}",
                client_secret="${MODEL_A_CLIENT_SECRET}",
            ),
        ),
        UpstreamConfig(name="model_b", base_url="${MODEL_B_URL}",
                       auth=UpstreamAuthConfig(type=AuthType.BEARER, api_key="${MODEL_B_KEY}")),
    ]
```

Auto-generates `POST /api/v1/endpoints/ensemble/call`, `GET .../health`,
`GET .../info`. Use in a pipeline step to inject model context into an agent:

```yaml
# pipeline.yaml
steps:
  - endpoint: ensemble        # calls the endpoint, stores result in context
    input: {payload: {text: "$input.query"}}
  - agent: summarizer
    input: {current_query: "$steps.ensemble.output"}
```

Auth types: `NONE`, `API_KEY`, `BEARER`, `BASIC`, `OAUTH2_CLIENT_CREDENTIALS`
(token cached + auto-refreshed). All string fields support `${ENV}`.

## Per-Agent Connections (databases, vector search, memory, any backend)

Declare a `connections.py` (`CONNECTIONS = [...]`) in an agent package; it is
auto-discovered and registered under the agent's **scope**. Scaffold with
`agentomatic init NAME --template connection`. Every connection has a **kind**
(how it connects) and a **purpose** (`memory` / `rag` / `vector` / `cache` /
`analytics` / `documents` / `general`).

```python
# agents/rag_agent/connections.py
from agentomatic.connections import (
    ConnectionPurpose, CustomConnectionConfig, DatabaseConnectionConfig,
    HttpConnectionConfig, VectorConnectionConfig,
)
from agentomatic.endpoints import AuthType, UpstreamAuthConfig

CONNECTIONS = [
    # Any SQL database — just a URL (Postgres/MySQL/SQLite/…)
    DatabaseConnectionConfig(name="main", url="${DB_URL}", purpose=ConnectionPurpose.GENERAL),
    # Conversation memory backed by the agent's own database
    DatabaseConnectionConfig(name="memory", url="${MEM_DB_URL}", purpose=ConnectionPurpose.MEMORY),
    # Vector store for RAG / vector search (qdrant|chroma|weaviate|pinecone|milvus)
    VectorConnectionConfig(name="kb", provider="qdrant", url="${QDRANT_URL}",
                           api_key="${QDRANT_API_KEY}", collection="kb",
                           purpose=ConnectionPurpose.RAG),
    # Authenticated HTTP service
    HttpConnectionConfig(name="scoring", base_url="${SCORING_URL}",
                         auth=UpstreamAuthConfig(type=AuthType.OAUTH2_CLIENT_CREDENTIALS,
                                                 token_url="${T}", client_id="${I}",
                                                 client_secret="${S}")),
    # ANY other backend (redis, mongo, elasticsearch…) with zero new classes
    CustomConnectionConfig(name="cache", factory="redis.asyncio.from_url",
                           args=["${REDIS_URL}"], purpose=ConnectionPurpose.CACHE),
]
```

Access live, initialised connections anywhere with `get_connections(scope)`:

```python
from agentomatic.connections import ConnectionPurpose, get_connections

conns = get_connections("rag_agent")
async with conns.database("main").session() as session: ...   # SQLAlchemy
kb = conns.vector("kb").client                                 # native vector client
redis = await conns.client("cache")                            # any factory client (lazy)
result = await conns.http("scoring").post("/score", payload={"x": 1})
for name, c in conns.by_purpose(ConnectionPurpose.RAG).items(): ...   # lookup by intent

memory_store = await conns.database("memory").create_store()   # back memory with a DB
```

Extensibility (minimal code):

- **`VectorConnectionConfig` + `register_vector_provider(name, builder)`** — add
  or override any vector backend.
- **`CustomConnectionConfig`** — wrap *any* client by pointing `factory` at a
  callable or dotted path (`"pkg.mod:func"`); `${ENV}` resolved deeply, sync/async
  factories supported, lifecycle auto-detected (`aclose`/`close`/`disconnect`).
- **`register_connection_type(config_cls, builder)`** — register a full custom
  wrapper class as a first-class connection.

Connections are lifecycle-managed (init on startup, closed on shutdown), emit
`agentomatic_connection_calls_total`, appear in the control plane, and are
exposed per request at `request.state.connections`.

## Production Control Plane

Enable with `enable_control_plane=True`. Admin API under `/api/v1/control`
(mutating ops require the `X-Control-Token` header when `control_token` is set):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/control` | Platform overview snapshot |
| `GET` | `/api/v1/control/agents` | List agents + health + policy |
| `GET` | `/api/v1/control/endpoints` | List custom endpoints |
| `GET` | `/api/v1/control/connections` | Connection health by scope |
| `GET` | `/api/v1/control/health` | Aggregate health |
| `GET` | `/api/v1/control/metrics/summary` | Coarse counters |
| `POST` | `/api/v1/control/agents/{name}/disable` | Drain an agent (503) |
| `POST` | `/api/v1/control/agents/{name}/enable` | Re-enable an agent |
| `POST` | `/api/v1/control/maintenance` | Toggle maintenance mode |

## Observability & Monitoring

- Prometheus metrics (agent, endpoint, upstream model, and connection calls);
  OpenTelemetry tracing auto-configured when `enable_telemetry=True`.
- `deploy/observability/` ships a Docker Compose stack (Prometheus + OTel
  Collector + Grafana) with a pre-provisioned **Agentomatic Overview** dashboard:
  `cd deploy/observability && docker compose up -d`.

## CLI Commands

```bash
agentomatic init NAME [--template basic|full|rag|chatbot|deepagent|custom|legacy_dict|plugin|endpoint|connection]
agentomatic run [--studio] [--with-ui] [--port 8000] [--agents-dir agents/]
agentomatic demo                    # Scaffold + run demo
agentomatic list [--agents-dir]     # List registered agents
agentomatic inspect NAME            # Inspect agent details
agentomatic doctor                  # Verify installation
agentomatic optimize NAME           # Run prompt optimization
agentomatic test NAME               # Interactive agent testing
agentomatic ui                      # Launch Chainlit chat UI
agentomatic stack init              # Initialize stack configuration
agentomatic stack list              # List available stacks
```

## Studio Integration

Studio provides visual debugging via a React UI at `/studio/ui/`.

### Adapter System

The Studio uses a universal adapter system:

| Framework | Adapter | Capabilities |
|-----------|---------|-------------|
| LangGraph | `LangGraphAdapter` | graph, streaming, checkpoints, state, breakpoints, hitl |
| Class Agent | `GraphAgentAdapter` | graph, streaming, traces |
| Deep Agent | `LangGraphAdapter` | + deep_agent, subagents, planning |
| LangChain | `LangChainAdapter` | streaming, traces, graph (LCEL) |
| Custom | `GenericAdapter` | streaming, traces |

### Custom Studio Integration

Use decorators to customize Studio behavior for any framework:

```python
from agentomatic.studio.decorators import studio_graph, studio_state, studio_stream

@studio_graph
def my_graph_provider(agent):
    return StudioGraphTopology(...)

@studio_state
async def my_state_provider(agent, thread_id):
    return StudioStateSnapshot(...)

@studio_stream
async def my_stream_provider(agent, state, config):
    yield StudioRunEvent(...)
```

### Studio SSE Event Types

| Event | Description |
|-------|-------------|
| `run_start` | Execution begins |
| `node_start` | Graph node starts |
| `node_end` | Graph node completes |
| `message_chunk` | LLM token streaming |
| `state_update` | State change |
| `subagent_start` | Deep Agent subagent delegation |
| `subagent_end` | Subagent returns |
| `task_update` | Planning (write_todos) |
| `breakpoint_hit` | HITL interrupt |
| `run_complete` | Execution finished |
| `run_error` | Error occurred |

## Auto-Generated Endpoints

For each registered agent, these endpoints are created:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/{name}/invoke` | Synchronous invocation |
| `POST` | `/api/v1/{name}/invoke/stream` | SSE streaming invocation |
| `POST` | `/api/v1/{name}/chat` | Session-aware conversation with memory |
| `GET` | `/api/v1/{name}/health` | Agent health check |
| `GET` | `/api/v1/{name}/config` | Agent configuration |
| `GET` | `/api/v1/{name}/prompts` | Available prompt versions |
| `GET` | `/api/v1/{name}/card` | A2A agent card |
| `POST` | `/api/v1/{name}/a2a/tasks` | A2A task submission |
| `GET` | `/api/v1/{name}/a2a/tasks/{id}` | A2A task status |
| `POST` | `/api/v1/{name}/threads` | Create thread |
| `GET` | `/api/v1/{name}/threads` | List threads |
| `GET` | `/api/v1/{name}/threads/{tid}` | Get thread |
| `PATCH` | `/api/v1/{name}/threads/{tid}` | Update thread |
| `DELETE` | `/api/v1/{name}/threads/{tid}` | Delete thread |
| `GET` | `/api/v1/{name}/threads/{tid}/messages` | Get messages |
| `DELETE` | `/api/v1/{name}/threads/{tid}/messages` | Clear messages |
| `GET` | `/api/v1/{name}/threads/{tid}/summary` | Conversation summary |
| `POST` | `/api/v1/{name}/optimize/invoke` | Optimization invocation |
| `POST` | `/api/v1/{name}/feedback` | Submit feedback |
| `GET` | `/api/v1/{name}/feedback` | List feedback |
| `GET` | `/api/v1/{name}/feedback/export` | Export feedback as JSONL |
| `GET` | `/api/v1/{name}/threads/{tid}/pending` | Pending HITL approvals |
| `POST` | `/api/v1/{name}/threads/{tid}/approve` | HITL approval |
| `POST` | `/api/v1/{name}/threads/{tid}/reject` | HITL rejection |
| `POST` | `/api/v1/{name}/threads/{tid}/fork` | Fork thread |
| `GET` | `/api/v1/{name}/threads/{tid}/lineage` | Thread lineage tree |

## Testing Conventions

- Tests use `pytest` with `pytest-asyncio` (auto mode)
- Test files: `tests/test_*.py`
- Use `MagicMock`/`AsyncMock` for agent mocks
- For node data in LangGraph tests, use `SimpleNamespace` instead of `MagicMock` (MagicMock's `name` attribute is special)
- Run: `uv run pytest tests/ --override-ini='addopts='`
- Lint: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`

## Code Style

- `from __future__ import annotations` in ALL files
- Google-style docstrings
- Line length: 99 characters
- Ruff for linting + formatting (rules: E, F, I, W, UP)
- Conventional commits for git messages
- Type hints on all public functions

## Common Patterns

### Adding a new middleware

```python
# src/agentomatic/middleware/my_middleware.py
from starlette.middleware.base import BaseHTTPMiddleware

class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Pre-processing
        response = await call_next(request)
        # Post-processing
        return response
```

### Adding a new storage backend

```python
# Implement BaseStore ABC
from agentomatic.storage.base import BaseStore

class MyStore(BaseStore):
    async def save_thread(self, thread_id, agent_name, state): ...
    async def get_thread(self, thread_id): ...
    async def list_threads(self, agent_name, limit): ...
    # ... see base.py for full interface
```

### Adding a new Studio adapter

```python
# Implement StudioAdapter ABC
from agentomatic.studio.adapter import StudioAdapter

class MyAdapter(StudioAdapter):
    @property
    def capabilities(self) -> list[str]: ...
    async def get_graph(self) -> StudioGraphTopology: ...
    async def stream_execution(self, state, config): ...
    async def get_state(self, thread_id): ...
    async def update_state(self, thread_id, updates): ...
    async def get_history(self, thread_id): ...
```
