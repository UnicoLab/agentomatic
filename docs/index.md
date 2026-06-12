---
hide:
  - navigation
---

# Agentomatic

## Drop agents, not code ⚡

**Agentomatic** is a zero-code multi-agent API platform framework. Create production-ready AI agent APIs with auto-discovery, auto-routing, streaming, A2A protocol, and full observability — in 3 lines of code.

```python
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder("agents/")
app = platform.build()  # Full FastAPI app — ready to deploy
```

---

## Why Agentomatic?

| Feature | What You Get |
|---|---|
| 🔍 **Auto-Discovery** | Drop an agent folder → endpoints appear automatically |
| 🚀 **12+ Endpoints Per Agent** | invoke, stream, chat, A2A, health, config, threads |
| 🗄️ **Pluggable Storage** | MemoryStore, SQLAlchemy, or bring your own |
| 🔒 **Middleware Pipeline** | Auth, rate limiting, metrics — all toggleable |
| 🎨 **Built-in Debug UI** | ChatGPT-like interface via Chainlit |
| 📦 **5 Scaffolding Templates** | basic, full, rag, chatbot, custom |
| 🤖 **A2A Protocol** | Agent-to-agent communication out of the box |
| 🔌 **Framework Agnostic** | LangGraph, LangChain, or raw Python |

---

## Quick Install

```bash
pip install agentomatic

# With all batteries
pip install agentomatic[all]
```

## Create Your First Agent

```bash
agentomatic init my_agent --template basic
agentomatic run
# → http://localhost:8000/docs
```

---

## Architecture

```mermaid
graph TB
    subgraph "Your Code"
        A["agent folder"] --> B["__init__.py\nmanifest + node_fn"]
        A --> C["graph.py\nnodes.py"]
    end

    subgraph "Agentomatic Platform"
        D[AgentPlatform] -->|auto-discover| E[AgentRegistry]
        E -->|per agent| F[RouterFactory]
        F --> G["12+ endpoints"]
        D --> H[Middleware Stack]
        D --> I[Storage Backend]
        D --> J[Debug UI]
    end

    B --> D
```

---

[Get Started :material-arrow-right:](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub :material-github:](https://github.com/UnicoLab/agentomatic){ .md-button }
