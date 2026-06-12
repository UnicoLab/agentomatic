---
hide:
  - navigation
---

# Agentomatic

[![PyPI version](https://img.shields.io/pypi/v/agentomatic.svg)](https://pypi.org/project/agentomatic/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-161%20passing-brightgreen.svg)](#)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg)](https://unicolab.github.io/agentomatic)

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
| 🚀 **12+ Endpoints Per Agent** | invoke, stream, chat, A2A, health, config, threads, feedback |
| 🗄️ **Pluggable Storage** | MemoryStore, SQLAlchemy, or bring your own |
| 🔒 **Middleware Pipeline** | Auth, rate limiting, metrics — all toggleable |
| 🎯 **Prompt Optimization** | 7 DSPy-inspired strategies, 8+ metrics, DataSynthesizer |
| 🎨 **Built-in Debug UI** | ChatGPT-like interface via Chainlit |
| 📦 **5 Scaffolding Templates** | basic, full, rag, chatbot, custom |
| 🤖 **A2A Protocol** | Agent-to-agent communication out of the box |
| 📊 **Observability** | OpenTelemetry tracing + Prometheus metrics |
| 🔌 **Framework Agnostic** | LangGraph, LangChain, or raw Python |
| 🛡️ **Red Team Testing** | Adversarial evaluation for safety & security |
| ⚡ **CLI Tooling** | init, run, list, test, inspect, doctor, ui |

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
        D --> K[Telemetry]
    end

    subgraph "Optimization"
        L[PromptOptimizer] --> M["7 Strategies"]
        L --> N["8+ Metrics"]
        L --> O[DataSynthesizer]
        L --> P[HTML Reports]
    end

    B --> D
```

---

## Feature Highlights

### 🎯 Prompt Optimization (DSPy-inspired)

```python
from agentomatic.optimize import PromptOptimizer, Dataset

optimizer = PromptOptimizer(
    agent="hr_bot",
    metrics=["answer_relevancy", "faithfulness"],
    strategy="iterative_rewrite",
)
result = await optimizer.optimize(
    dataset=Dataset.from_jsonl("eval.jsonl"),
    max_iterations=10,
)
result.apply()  # Save optimized prompt
```

### 📊 Full Observability

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_metrics=True,     # Prometheus at /metrics
    enable_telemetry=True,   # OpenTelemetry tracing
    enable_feedback=True,    # User feedback collection
)
```

### ⚡ Click-Based CLI

```bash
agentomatic init hr_bot --template rag
agentomatic run --reload --with-ui
agentomatic test hr_bot
agentomatic doctor
```

---

[Get Started :material-arrow-right:](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub :material-github:](https://github.com/UnicoLab/agentomatic){ .md-button }
