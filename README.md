<div align="center">

# ⚡ Agentomatic

### Drop agents, not code

[![CI](https://github.com/UnicoLab/agentomatic/actions/workflows/ci.yml/badge.svg)](https://github.com/UnicoLab/agentomatic/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agentomatic.svg)](https://pypi.org/project/agentomatic/)
[![Python](https://img.shields.io/pypi/pyversions/agentomatic.svg)](https://pypi.org/project/agentomatic/)
[![License](https://img.shields.io/github/license/UnicoLab/agentomatic.svg)](https://github.com/UnicoLab/agentomatic/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://unicolab.github.io/agentomatic)

**Zero-code multi-agent API platform framework.**
Create production-ready AI agent APIs with auto-discovery, auto-routing, streaming, A2A protocol, and full observability — in 3 lines of code.

[Documentation](https://unicolab.github.io/agentomatic) · [Quick Start](#-quick-start) · [CLI Reference](#-cli) · [Templates](#-templates) · [Contributing](CONTRIBUTING.md)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Auto-Discovery** | Drop an agent folder → endpoints appear automatically |
| 🚀 **12+ Endpoints Per Agent** | invoke, stream, chat, A2A, health, config, threads |
| 🗄️ **Pluggable Storage** | MemoryStore, SQLAlchemy, or bring your own |
| 🔐 **Middleware Pipeline** | Auth, rate limiting, Prometheus metrics — all toggleable |
| 🎨 **Built-in Debug UI** | ChatGPT-like interface via Chainlit |
| 📦 **5 Scaffolding Templates** | basic, full, rag, chatbot, custom |
| 🤖 **A2A Protocol** | Agent-to-agent communication out of the box |
| 🔌 **Framework Agnostic** | LangGraph, LangChain, or raw Python |
| 🩺 **Rich CLI** | Beautiful terminal experience with doctor, inspect, test |
| ⚡ **Prompt Optimizer** | DSPy-inspired prompt optimization (7 strategies) |
| 🧪 **Data Synthesizer** | Auto-generate & augment eval datasets via LLM |
| 📊 **HTML Reports** | SVG charts, prompt diffs, experiment tracking |

## 🚀 Quick Start

### Install

```bash
pip install agentomatic[all]
```

### Create an Agent

```bash
agentomatic init my_agent --template basic
```

### Build & Run

```python
# main.py
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder("agents/")
app = platform.build()
```

```bash
uvicorn main:app --reload
```

### Test

```bash
# CLI
agentomatic test my_agent

# curl
curl -X POST http://localhost:8000/api/v1/my_agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!"}'
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentPlatform                            │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Registry │  │ Middleware   │  │ Storage               │ │
│  │          │  │ ├─ Auth      │  │ ├─ MemoryStore        │ │
│  │ agent_a  │  │ ├─ RateLimit │  │ ├─ SQLAlchemyStore   │ │
│  │ agent_b  │  │ ├─ Metrics   │  │ └─ YourStore(ABC)    │ │
│  │ agent_c  │  │ └─ Logging   │  │                       │ │
│  └──────────┘  └──────────────┘  └───────────────────────┘ │
│                                                             │
│  Per Agent: POST /invoke, /stream, /chat, /a2a/tasks ...   │
└─────────────────────────────────────────────────────────────┘
```

## 📂 Agent Structure

Only `__init__.py` is required. Everything else is optional overrides:

```
agents/my_agent/
├── __init__.py      ← REQUIRED: manifest + node_fn
├── graph.py         ← Optional: LangGraph StateGraph
├── nodes.py         ← Optional: node functions
├── config.py        ← Optional: Pydantic config
├── schemas.py       ← Optional: custom request/response models
├── tools.py         ← Optional: LangChain tools
├── api.py           ← Optional: custom router (REPLACES auto-gen)
├── prompts.json     ← Optional: versioned prompt templates
├── langgraph.json   ← Optional: LangGraph Studio config
├── .env.example     ← Optional: environment variables
└── README.md        ← Optional: agent documentation
```

## 📦 Templates

```bash
agentomatic init my_agent --template <template>
```

| Template | Files | Description |
|----------|-------|-------------|
| `basic` | 7 | Minimal agent — quick start |
| `full` | 11 | All override files — config, schemas, api, tools |
| `rag` | 9 | Retrieve → Generate pipeline |
| `chatbot` | 8 | Conversational with memory |
| `custom` | 4 | Framework-agnostic — no LangGraph |

## 🖥️ CLI

```
⚡ Agentomatic — Drop agents, not code

  init <name>      Scaffold a new agent from template
  run              Start the platform server
  list             List discovered agents (Rich table)
  test <name>      Interactive terminal testing
  inspect <name>   Show agent structure + config
  doctor           Environment health check
  ui               Launch Chainlit debug UI
```

## ⚙️ Configuration

```python
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore  # or SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    # Storage
    store=MemoryStore(),
    # Auth
    enable_auth=True,
    auth_api_key="your-secret-key",
    # Rate limiting
    enable_rate_limit=True,
    rate_limit_requests=100,
    rate_limit_window=60,
    # Prometheus metrics
    enable_metrics=True,
    # Custom middleware
    middleware=[(MyMiddleware, {"arg": "value"})],
)
app = platform.build()
```

## 🗄️ Storage Backends

```python
# Development
from agentomatic.storage import MemoryStore
store = MemoryStore()

# Production (PostgreSQL)
from agentomatic.storage import SQLAlchemyStore
store = SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/db")

# Custom
from agentomatic.storage import BaseStore
class RedisStore(BaseStore):
    async def create_thread(self, ...): ...
    async def get_thread(self, ...): ...
```

## 🎨 Debug UI

Built-in ChatGPT-like interface powered by Chainlit:

```bash
pip install agentomatic[ui]
agentomatic run --with-ui
# → http://localhost:8000/chat
```

Features: agent selector, streaming, tool call visualization, chain-of-thought, feedback collection.

## 📊 Auto-Generated Endpoints

Every agent gets 12+ endpoints automatically:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/{agent}/invoke` | Synchronous invocation |
| `POST` | `/api/v1/{agent}/invoke/stream` | SSE streaming |
| `POST` | `/api/v1/{agent}/chat` | Session-aware chat |
| `GET` | `/api/v1/{agent}/health` | Per-agent health |
| `GET` | `/api/v1/{agent}/card` | A2A agent card |
| `POST` | `/api/v1/{agent}/a2a/tasks` | A2A task submission |
| `GET` | `/api/v1/{agent}/threads` | List threads |
| ... | ... | + config, prompts, thread messages |

## 🛠️ Development

```bash
# Install
git clone https://github.com/UnicoLab/agentomatic.git
cd agentomatic
make dev  # Installs all deps + pre-commit hooks

# Quality
make lint          # Ruff linter
make format        # Auto-format
make typecheck     # Mypy
make test          # All tests
make test-cov      # With coverage
make check-all     # lint + typecheck + test

# Docs
make docs-serve    # Local docs server
make docs-build    # Build static site

# Build
make build         # Package
make publish       # PyPI
```

## 📜 License

MIT — see [LICENSE](LICENSE).

## 👥 Authors

**[UnicoLab](https://github.com/UnicoLab)** — Building the future of AI agent platforms.

---

<div align="center">

**[⭐ Star us on GitHub](https://github.com/UnicoLab/agentomatic)** — it helps!

Made with ❤️ by UnicoLab

</div>