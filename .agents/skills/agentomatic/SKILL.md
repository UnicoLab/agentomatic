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

```
pip install agentomatic                    # Core
pip install "agentomatic[langgraph]"       # + LangGraph support
pip install "agentomatic[all]"             # Everything
```

## Architecture

```
src/agentomatic/
├── __init__.py              # Public API: AgentPlatform, AgentManifest, BaseAgentState
├── core/
│   ├── platform.py          # AgentPlatform — central orchestrator
│   ├── registry.py          # AgentRegistry — agent registration and lookup
│   ├── manifest.py          # AgentManifest model + RegisteredAgent dataclass
│   ├── router_factory.py    # Auto-generates 20+ REST endpoints per agent
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
│   └── templates.py         # Scaffolding: basic, full, rag, chatbot, deepagent, custom
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
│   └── logging.py           # Structured logging
├── optimize/                # DSPy-style prompt optimization (18 modules)
├── observability/           # OpenTelemetry tracing + Prometheus metrics
├── prompts/                 # PromptManager for prompt versioning
├── providers/               # LLM providers (Ollama, OpenAI, Azure, Vertex)
├── config/                  # Settings management
├── ui/                      # Chainlit chat interface
└── demo/                    # Built-in demo agent
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
        # Build the LangGraph execution graph
        g = self.new_graph()
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
| `current_query` | `str` | The user's input query |
| `user_id` | `str` | User identifier |
| `thread_id` | `str` | Conversation thread ID |
| `messages` | `list` | Conversation message history |
| `context` | `dict` | Additional context |
| `metadata` | `dict` | Request metadata |
| `steps_taken` | `list[str]` | Execution trace |
| `response` | `str` | Agent's final response |
| `suggestions` | `list[str]` | Follow-up suggestions |
| `citations` | `list[dict]` | Source citations |
| `prompt_version` | `str` | Active prompt version |
| `agent_type` | `str` | Which agent handled this |

## Platform Configuration

```python
from agentomatic import AgentPlatform
from agentomatic.storage import SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    store=SQLAlchemyStore("postgresql+asyncpg://..."),
    enable_metrics=True,       # Prometheus metrics
    enable_auth=True,          # API key auth
    auth_api_key="secret",     # Auth key
    enable_cors=True,          # CORS headers
    enable_rate_limit=True,    # Rate limiting
    rate_limit=100,            # Requests per minute
)
app = platform.build()
```

## CLI Commands

```bash
agentomatic init NAME [--template basic|full|rag|chatbot|deepagent|custom]
agentomatic run [--studio] [--with-ui] [--port 8000] [--agents-dir agents/]
agentomatic demo                    # Scaffold + run demo
agentomatic list [--agents-dir]     # List registered agents
agentomatic inspect NAME            # Inspect agent details
agentomatic doctor                  # Verify installation
agentomatic optimize NAME           # Run prompt optimization
```

## Studio Integration

Studio provides visual debugging via a React UI at `/studio/ui/`.

### Adapter System

The Studio uses a universal adapter system:

| Framework | Adapter | Capabilities |
|-----------|---------|-------------|
| LangGraph | `LangGraphAdapter` | graph, streaming, checkpoints, state, breakpoints, hitl |
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
| `POST` | `/api/v1/{name}/invoke` | Invoke agent |
| `POST` | `/api/v1/{name}/stream` | Stream SSE response |
| `POST` | `/api/v1/{name}/chat` | Chat with conversation memory |
| `GET` | `/api/v1/{name}/threads` | List threads |
| `GET` | `/api/v1/{name}/threads/{tid}` | Get thread |
| `DELETE` | `/api/v1/{name}/threads/{tid}` | Delete thread |
| `POST` | `/api/v1/{name}/threads/{tid}/fork` | Fork thread |
| `POST` | `/threads/{tid}/approve` | HITL approval |
| `POST` | `/api/v1/{name}/feedback` | Submit feedback |
| `GET` | `/api/v1/{name}/suggestions` | Get suggestions |

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
