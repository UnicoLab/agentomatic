<div align="center">

<p align="center">
  <img src="assets/logo.png" width="300" alt="agentomatic">
</p>

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
| 🚀 **25+ Endpoints Per Agent** | invoke, stream, chat, A2A, health, config, threads, memory, feedback |
| 🗄️ **Pluggable Storage** | MemoryStore, SQLAlchemy, or bring your own |
| 🔐 **Middleware Pipeline** | Auth, rate limiting, Prometheus metrics — all toggleable |
| 🎨 **Built-in Debug UI** | ChatGPT-like interface via Chainlit |
| 🎯 **Agentomatic Studio** | Visual agent debugger with graph view, SSE streaming, state inspection, time-travel |
| 📦 **5 Scaffolding Templates** | basic, full, rag, chatbot, custom |
| 🤖 **A2A Protocol** | Agent-to-agent communication out of the box |
| 🔌 **Framework Agnostic** | LangGraph, LangChain, or raw Python |
| 🩺 **Rich CLI** | Beautiful terminal experience with doctor, inspect, test |
| ⚡ **Prompt Optimizer** | ML-like prompt/config fitting with 5 optimizers + deployment recommendations |
| 🧪 **Data Synthesizer** | Auto-generate & augment eval datasets via LLM |
| 📊 **HTML Reports** | SVG charts, prompt diffs, experiment tracking |
| 🚦 **Human-in-the-Loop** | Suspend/resume execution with human approval gates |
| 🌳 **Thread Lineage** | First-class parent/child tracking with ancestry traversal |
| ⏰ **HITL TTL Expiry** | Auto-cleanup of stale suspended states (7-day default) |
| 🛡️ **LLM Failover** | Multi-provider fallback chains with telemetry |
| 🧬 **Thread Forking** | Clone conversations at any message index |
| 🔀 **A/B Prompt Routing** | Weight-based prompt version selection |
| 🪝 **State Hooks** | Before/after node interceptors for audit & telemetry |
| 🧠 **Conversation Memory** | Automatic session + long-term memory with windowing |
| 📝 **Auto-Summarization** | LLM-powered compression of long conversations |
| 📋 **Thread CRUD** | Full thread lifecycle management (create/update/delete/clear) |
| 💬 **Message Persistence** | Every turn auto-saved — history survives restarts |

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
  run --studio     Start the platform server AND the visual Agentomatic Studio 🎨
  list             List discovered agents (Rich table)
  test <name>      Interactive terminal testing
  inspect <name>   Show agent structure + config
  doctor           Environment health check
  ui               Launch Chainlit debug UI
```

## 🎨 Agentomatic Studio

Agentomatic ships with a built-in React-based visual studio designed for time-travel debugging, real-time node streaming, and state inspection for all underlying LangGraph agents.

To use the studio, install the optional package dependencies and run with the `--studio` flag:

```bash
pip install "agentomatic[studio]"
agentomatic run --studio
```

The unified server will bind to `http://localhost:8000` and mount the studio at `http://localhost:8000/studio/ui/`. 

**Key Studio Features**:
- **Live Node Streaming**: Watch Server-Sent Events (SSE) transition node activity dynamically.
- **Conditional Breakpoints**: Right-click graph nodes to intercept flow before execution triggers.
- **Time-Travel History**: Rewind to any state checkpoint and replay from historical forks.
- **Live State Editing**: Mutate graph state payloads on the fly during a breakpoint pause.

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

## 🎯 Agentomatic Studio

Visual debugging interface for any agent — graph visualization, real-time execution tracing, state inspection, and time-travel debugging. Works with LangGraph, LangChain, or custom agents.

```bash
# Option 1: Bundled (zero-config)
agentomatic run --studio
# → Studio at http://localhost:8000/studio/ui/

# Option 2: Standalone Docker
docker run -p 3000:80 -e REACT_APP_API_URL=http://your-server:8000 agentomatic-studio
```

**Studio API Endpoints** (mounted at `/studio/`):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/studio/info` | Server info + capabilities |
| `GET` | `/studio/agents` | List agents with debugging capabilities |
| `GET` | `/studio/agents/{name}/graph` | Graph topology (nodes, edges) |
| `GET` | `/studio/agents/{name}/schemas` | Input/output JSON schemas |
| `POST` | `/studio/agents/{name}/runs/stream` | Execute with SSE event streaming |
| `GET` | `/studio/agents/{name}/threads/{tid}/state` | Thread state snapshot |
| `GET` | `/studio/agents/{name}/threads/{tid}/history` | Checkpoint history |

**Capabilities degrade gracefully:**
- **LangGraph agents** → Full graph visualization, checkpoints, time-travel, state editing
- **Custom/simple agents** → Chat, invoke, thread management, response inspection

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
| `POST` | `/api/v1/{agent}/threads/{id}/approve` | HITL: approve suspended state |
| `POST` | `/api/v1/{agent}/threads/{id}/reject` | HITL: reject suspended state |
| `GET` | `/api/v1/{agent}/threads/{id}/pending` | HITL: list pending approvals |
| `POST` | `/api/v1/{agent}/threads/{id}/fork` | Fork thread at message index |
| `GET` | `/api/v1/{agent}/threads/{id}/lineage` | Thread ancestry/descendant tree |
| ... | ... | + config, prompts, thread messages |

### 🔧 Prompt Fitting (ML-like API)

Agentomatic Optimize treats your deployed agent configuration as a **parameter surface**
to fit against real evaluation data. The output is never a compiled program — it's a
**better deployment configuration**: an improved prompt, tuned model parameters,
optimized RAG settings, and a rollout recommendation you can ship with confidence.

> **Philosophy:** Your agent is already deployed. Optimization produces a *better version*
> of that deployment, not a new artifact. Every result includes a `DeploymentRecommendation`
> with canary weights and confidence scores so you can roll out safely.

#### EvalContract — Structural Quality Gate

Define what a valid agent response looks like *before* you optimize:

```python
from agentomatic.optimize import EvalContract

contract = EvalContract(
    name="scoping_response",
    input_fields=["query", "context"],
    output_format="json",
    required_output_fields=["answer", "confidence", "risks", "next_questions"],
    constraints=["confidence must be between 0.0 and 1.0"],
)

score = contract.validate(response_text)       # 0.0 – 1.0
metric = contract.as_metric(weight=0.10)       # use inside CompositeMetric
criteria = contract.as_judge_criteria()         # feed to LLM judge
```

#### CompositeMetric — Multi-Dimensional Scoring

Combine quality judges with **negative-weight** cost/latency penalties so the optimizer
balances accuracy against operational cost:

```python
from agentomatic.optimize import (
    CompositeMetric, WeightedMetric,
    LocalJudgeMetric, LatencyMetric, CostMetric,
)

metric = CompositeMetric(metrics=[
    WeightedMetric("completeness",   LocalJudgeMetric("completeness"),      weight=0.30),
    WeightedMetric("relevance",      LocalJudgeMetric("business_relevance"),weight=0.25),
    WeightedMetric("risk_detection", LocalJudgeMetric("risk_detection"),    weight=0.20),
    WeightedMetric("format",         contract.as_metric(),                  weight=0.10),
    WeightedMetric("latency",        LatencyMetric(),                       weight=-0.10),
    WeightedMetric("cost",           CostMetric(),                          weight=-0.05),
])
```

> Negative weights penalize candidates that are slower or more expensive, steering the
> fitter toward cost-effective configurations.

#### PromptSearchSpace — Full Configuration Surface

Tell the fitter *what* it's allowed to change:

```python
from agentomatic.optimize import PromptSearchSpace

space = PromptSearchSpace(
    optimize_system_prompt=True,
    optimize_few_shot=True,
    optimize_model_params=True,
    optimize_model_choice=True,
    model_choices=["ollama/qwen2.5:7b", "openai/gpt-4.1"],
    fallback_models=["openai/gpt-4.1-mini"],
    model_param_space={
        "temperature": [0.0, 0.1, 0.2, 0.4, 0.7],
        "top_p": [0.7, 0.9, 1.0],
    },
    rag_param_space={"top_k": [3, 5, 8, 12], "rerank": [True, False]},
    optimize_rag_params=True,
)
```

#### PromptFitter — The scikit-learn-like API

```python
from agentomatic.optimize import PromptFitter

fitter = PromptFitter(
    agent="scope_agent",
    task_model="ollama/qwen2.5:7b",
    rewrite_model="openai/gpt-4.1",
    optimizer="gepa_like",
    search_space=space,
    max_trials=30,
    min_absolute_improvement=0.05,
    concurrency=5,
)
result = await fitter.fit(trainset, valset, metric, testset=testset)
```

Access the full result surface:

```python
result.best_prompt              # optimized system prompt
result.best_params              # {"temperature": 0.2, "top_p": 0.9}
result.best_few_shot_examples   # selected few-shot examples
result.metric_deltas            # per-dimension improvement
result.suggestions              # actionable recommendations
result.deployment_recommendation # canary rollout config
result.summary()                # human-readable summary
result.apply(version="v2_optimized")
```

**Five optimisation strategies:**

| Strategy | What it does |
|---|---|
| `rewrite` | LLM-driven prompt rewrite based on failure analysis |
| `few_shot_bootstrap` | Score²-weighted example selection with diversity scoring |
| `mipro_like` | Multi-perspective instruction generation + cross-product search |
| `gepa_like` | Feedback-guided targeted prompt mutations |
| `param_search` | Grid search over model/RAG/tool parameters |

#### DeploymentRecommendation — Ship With Confidence

Every `PromptFitResult` includes a deployment recommendation based on the observed
improvement magnitude and variance:

```python
rec = result.deployment_recommendation
print(rec.confidence)              # "high" / "medium" / "low"
print(rec.rollout.strategy)        # "canary"
print(rec.rollout.initial_weight)  # 0.40
print(rec.summary())               # human-readable deployment plan
```

#### Failure Clusters — Targeted Diagnostics

The fitter groups validation failures into actionable clusters, each with the parameters
most likely to resolve the issue and the expected metric gain:

```
Failure cluster 1:
  Agent answered without using retrieval context.
  → Suggested fix: force context-first behavior.
  → Affected params: rag.top_k, tool_policy.force_retrieval
  → Expected metric gain: faithfulness +0.18

Failure cluster 2:
  Agent produced unstructured answers.
  → Suggested fix: stronger output format block.
  → Affected params: prompt.output_contract
  → Expected metric gain: format_compliance +0.12
```

#### Ideal CLI Flow

```bash
# 1. Run your agents
agentomatic run

# 2. Generate a synthetic evaluation dataset from your docs
agentomatic dataset synth scope_agent --from-docs docs/scoping.md --n 100

# 3. Evaluate the current version
agentomatic eval scope_agent --dataset scope_eval.jsonl --metrics scoping_quality

# 4. Fit a better configuration
agentomatic optimize scope_agent --optimize prompt,params,rag,tools

# 5. Canary release — send 20 % traffic to the new version
agentomatic route scope_agent --version v2_optimized --weight 20

# 6. Promote when satisfied
agentomatic promote scope_agent --version v2_optimized
```

#### Vocabulary

| ❌ Avoid | ✅ Use instead |
|----------|----------------|
| Program | Agent endpoint |
| Compile | Fit / optimize / tune |
| Signature | EvalContract |
| Module | Deployment component |
| Predictor | Agent version |
| Compiled artifact | Optimized config version |

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
