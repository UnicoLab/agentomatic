<p align="center">
  <img src="docs/assets/logo.png" width="200" alt="Agentomatic Logo">
  <h1 align="center">🤖 Agentomatic</h1>
  <p align="center"><em>Drop agents, not code.</em></p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#api-reference">API Reference</a>
  </p>
</p>

---

**Agentomatic** is a zero-code multi-agent API platform framework for Python. Create an agent folder, drop in your logic, and get 12+ production-ready REST endpoints automatically — including streaming, A2A protocol, health checks, and thread management.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Auto-Discovery** | Agents are discovered from folder structure — no registration code |
| 🔌 **12+ Endpoints Per Agent** | invoke, stream, chat, health, config, prompts, A2A, threads |
| 🧩 **Framework-Agnostic** | LangGraph, LangChain, or plain async functions |
| ⚡ **Production-Ready** | Circuit breakers, metrics, rate limiting, auth, concurrency control |
| 📊 **Observability** | Prometheus metrics, structured logging, request tracing |
| 🔄 **A2A Protocol** | Agent-to-Agent communication with auto-generated agent cards |
| 🗄️ **Pluggable Storage** | In-memory (dev) or SQLAlchemy (production) |
| 🎛️ **Feature Flags** | Toggle streaming, auth, metrics, A2A, DB — all via env vars |
| 📝 **Prompt Versioning** | JSON-based prompt management with hot-reload |
| 🖥️ **CLI** | `agentomatic init`, `agentomatic run`, `agentomatic list` |

## Installation

```bash
# Core (minimal)
pip install agentomatic

# With LangGraph support (recommended)
pip install agentomatic[langgraph]

# With Ollama LLM
pip install agentomatic[langgraph,ollama]

# With Prometheus metrics
pip install agentomatic[langgraph,metrics]

# Everything
pip install agentomatic[all]

# For development
pip install agentomatic[all,dev]
```

### Optional Extras

| Extra | Packages Added |
|-------|----------------|
| `langgraph` | langgraph, langchain-core |
| `langchain` | langchain, langchain-core, langchain-community |
| `ollama` | langchain-ollama, ollama |
| `openai` | langchain-openai |
| `azure` | langchain-openai |
| `vertex` | langchain-google-vertexai |
| `metrics` | prometheus-client |
| `db` | sqlalchemy[asyncio], aiosqlite |
| `db-postgres` | sqlalchemy[asyncio], asyncpg, psycopg |
| `all` | langgraph + ollama + metrics + db |

## Quick Start

### 1. Create your project

```bash
mkdir my-platform && cd my-platform
agentomatic init hello --dir agents
```

This creates:
```
my-platform/
└── agents/
    └── hello/
        ├── __init__.py   # AgentManifest + node_fn
        ├── graph.py       # LangGraph StateGraph
        └── nodes.py       # Node functions
```

### 2. Write `main.py` — 3 lines!

```python
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder("agents/", package_prefix="agents")
app = platform.build()

# Run: uvicorn main:app --reload
```

### 3. Start the server

```bash
uvicorn main:app --reload
```

### 4. Test it!

```bash
# Health check
curl http://localhost:8000/health

# Invoke the agent
curl -X POST http://localhost:8000/api/v1/hello/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello world!"}'

# Chat with the agent
curl -X POST http://localhost:8000/api/v1/hello/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "Tell me a joke"}'

# Stream response
curl -X POST http://localhost:8000/api/v1/hello/invoke/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Stream this!"}'

# Get agent card (A2A)
curl http://localhost:8000/api/v1/hello/card

# List all agents
curl http://localhost:8000/api/v1/agents
```

## Architecture

### Agent Folder Structure

Every agent is a Python package in the agents directory:

```
agents/
├── holidays/                 # Agent folder name = agent name
│   ├── __init__.py          # REQUIRED: manifest + node_fn
│   ├── graph.py             # Optional: LangGraph StateGraph
│   ├── nodes.py             # Optional: Node functions
│   ├── config.py            # Optional: Agent-specific config
│   ├── schemas.py           # Optional: Pydantic I/O models
│   ├── prompts.json         # Optional: Versioned prompts
│   ├── tools.py             # Optional: LangChain tools
│   └── api.py               # Optional: Custom FastAPI router
```

### Minimal Agent (`__init__.py`)

```python
from agentomatic import AgentManifest

manifest = AgentManifest(
    name="holidays",
    slug="my-platform-holidays",
    description="Manages holiday requests",
    intent_keywords=["holiday", "vacation", "leave"],
)

async def node_fn(state):
    from .graph import get_graph
    return await get_graph().ainvoke(state)
```

### Auto-Generated Endpoints

For every discovered agent, Agentomatic generates:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/{agent}/invoke` | Synchronous invocation |
| `POST` | `/api/v1/{agent}/invoke/stream` | SSE streaming |
| `POST` | `/api/v1/{agent}/chat` | Session-aware chat |
| `GET` | `/api/v1/{agent}/health` | Per-agent health check |
| `GET` | `/api/v1/{agent}/config` | Agent configuration |
| `GET` | `/api/v1/{agent}/prompts` | Prompt versions |
| `GET` | `/api/v1/{agent}/card` | A2A agent card |
| `POST` | `/api/v1/{agent}/a2a/tasks` | A2A task submission |
| `GET` | `/api/v1/{agent}/a2a/tasks/{id}` | A2A task status |
| `GET` | `/api/v1/{agent}/threads` | List threads |
| `GET` | `/api/v1/{agent}/threads/{id}` | Get thread |
| `GET` | `/api/v1/{agent}/threads/{id}/messages` | Get messages |

### Platform Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Platform info |
| `GET` | `/health` | Overall health |
| `GET` | `/readiness` | Readiness probe |
| `GET` | `/api/v1/agents` | List all agents |
| `GET` | `/.well-known/agent.json` | A2A discovery |

## Configuration

### Environment Variables

All settings use double-underscore nesting:

```bash
# LLM
LLM__PROVIDER=ollama
LLM__MODEL=mistral:7b
LLM__TEMPERATURE=0.1

# Features
FEATURES__ENABLE_STREAMING=true
FEATURES__ENABLE_METRICS=true
FEATURES__ENABLE_A2A=true
FEATURES__MAX_CONCURRENT_AGENTS=10

# Database
DB__URL=postgresql+asyncpg://user:pass@localhost:5432/mydb
```

See [`.env.example`](.env.example) for all options.

### Programmatic Configuration

```python
from agentomatic import AgentPlatform
from agentomatic.config.settings import PlatformSettings

settings = PlatformSettings(
    llm={"provider": "openai", "model": "gpt-4"},
    features={"enable_metrics": True},
)

platform = AgentPlatform.from_folder(
    "agents/",
    title="My Platform",
    settings=settings,
)
```

## Advanced Usage

### Custom Endpoints (Override)

Create `api.py` in your agent folder:

```python
from fastapi import APIRouter, UploadFile
router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile):
    return {"filename": file.filename, "status": "processed"}
```

### Programmatic Agent Registration

```python
from agentomatic import AgentPlatform, AgentManifest

platform = AgentPlatform()

async def my_agent(state):
    return {"response": f"Got: {state['current_query']}"}

platform.register_agent(
    manifest=AgentManifest(name="dynamic", slug="dynamic-agent"),
    node_fn=my_agent,
)

app = platform.build()
```

### Lifecycle Hooks

```python
platform = AgentPlatform.from_folder("agents/")

@platform.on_startup
async def init_database():
    print("Database connected!")

@platform.on_shutdown
async def cleanup():
    print("Cleaning up...")

app = platform.build()
```

### LangGraph Studio Integration

Create `langgraph.json` per agent:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agents/holidays/graph.py:get_graph"
  }
}
```

## CLI Reference

```bash
# Scaffold a new agent
agentomatic init my_agent --dir agents

# Run the platform
agentomatic run --agents-dir agents --port 8000 --reload

# List discovered agents
agentomatic list --agents-dir agents
```

## Project Structure

```
src/agentomatic/
├── __init__.py          # Public API
├── _version.py          # Version
├── core/                # Platform, Registry, Manifest
├── config/              # Settings system
├── providers/           # LLM/Embedding factories
├── middleware/           # Logging, CORS, Auth
├── observability/       # Metrics, Concurrency
├── storage/             # Memory, SQLAlchemy
├── prompts/             # Prompt versioning
├── protocols/           # A2A, Decorators
├── cli/                 # CLI commands
└── templates/           # Scaffolding
```

## License

MIT © Piotr Laczkowski