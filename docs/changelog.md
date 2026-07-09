# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- **Custom Endpoints**
  - `BaseEndpoint` — user-defined HTTP APIs that call deployed model services
    via authenticated `httpx` requests and aggregate their outputs
  - Auto-discovery from the `endpoints/` directory; routes mounted under
    `/api/v1/endpoints/{name}` with auto-generated `/health` and `/info`
  - Upstream auth: API key, bearer, basic, and OAuth2 client-credentials
    (with token caching); `${ENV}` interpolation keeps secrets out of code
  - `MultiModelClient` fan-out with `ALL` / `FIRST_SUCCESS` / `MAJORITY`
    aggregation, retries, timeouts, and Prometheus metrics
  - Usable as a pipeline step (`endpoint:`) to feed model outputs into agents
  - `agentomatic init <name> --template endpoint` scaffolding
- **Per-Agent Connections**
  - `DatabaseConnectionConfig` / `HttpConnectionConfig` for scoped, authenticated
    databases (SQLAlchemy async) and HTTP services per agent
  - `get_connections(scope)` runtime accessor; auto-discovery from an agent's
    `connections.py` (`CONNECTIONS = [...]`)
  - Lifecycle-managed (initialised on startup, closed on shutdown) with health
    checks; `agentomatic init <name> --template connection` scaffolding
  - **Vector stores** (`VectorConnectionConfig`) for RAG / vector search with
    lazy, provider-agnostic clients (Qdrant, Chroma, Weaviate, Pinecone,
    Milvus) and `register_vector_provider()` for custom backends
  - **Purpose tagging** (`ConnectionPurpose`: memory/rag/vector/cache/analytics…)
    with `by_purpose()` / `for_purpose()` / `first_for_purpose()` lookups
  - **Any backend, zero classes** — `CustomConnectionConfig` wraps any factory
    callable / dotted path (redis, mongo, elasticsearch, neo4j…) with lazy
    build, deep `${ENV}` resolution, sync/async factories and auto-detected
    lifecycle; fetch the native client via `await conns.client(name)`
  - **Pluggable type registry** (`register_connection_type()`) so any backend
    becomes a first-class connection with a full custom wrapper
  - Conversation **memory** can be backed by a connection's own engine via
    `DatabaseConnection.create_store()` (shared pool, no double-dispose)
  - **`ConnectionsMiddleware`** exposes the routed agent's manager on
    `request.state.connections` (enabled by `enable_connections_context`)
- **Production Control Plane** (`enable_control_plane=True`)
  - Admin API under `/api/v1/control` to inspect agents/endpoints/connections,
    read health/metrics/config, drain/re-enable agents, and toggle maintenance
    mode; mutating ops protected by an optional `X-Control-Token`
- **Observability & Monitoring**
  - New metrics for endpoints, upstream model calls, and connections
  - Ready-to-run stack in `deploy/observability/` (Prometheus + OpenTelemetry
    Collector + Grafana) with a pre-provisioned **Agentomatic Overview** dashboard
- **Zero-Trust enforcement** activated via middleware so per-agent security
  policies (roles/scopes/auth) are enforced on the request path
- **Swagger/OpenAPI fixes**: structured tag metadata, cleaner operation IDs,
  de-duplicated pipeline tags, and Studio UI routes excluded from the schema

- **Class-Owned Graph Agents (v0.7)**
  - `BaseGraphAgent` — define agents as Python classes with ML lifecycle
  - `build_graph()` + `new_graph()` — LangGraph-style graph wiring (primary API)
  - `GraphBuilder` with LangGraph-compatible aliases: `add_node()`, `add_edge()`,
    `set_entry_point()`, `set_finish_point()`, `add_conditional_edge()`, `compile()`
  - `@agent_node` decorator — optional fallback for simple linear chains
  - `AgentGraph` — lightweight internal graph runtime (no LangGraph dependency)
  - `AgentDataset` / `AgentExample` — rich evaluation datasets with JSONL I/O
  - Evaluation metrics: `ExactKeyMatchMetric`, `ContainsTermsMetric`, `CallableMetric`
  - Optimizers: `NoOpOptimizer`, `GridSearchOptimizer`, `PromptFitterBridge`
  - ML lifecycle: `compile()` → `fit()` → `evaluate()` → `transform()`
  - Per-node observability with `TraceEvent` tracing
  - Serialization: `save()`/`load_compiled()` for compiled agent state
  - Auto-discovery: class agents found via `agent.py` or `__init__.py` bridge
  - Registry integration: `register_class_agent()` + `_discover_class_agent()`
  - `agentomatic init --template class` generates full package with
    `__init__.py`, `agent.py`, `llm.py`, `prompts.json`, `train.py`
  - 200 tests covering graph, dataset, metrics, lifecycle, and API aliases

- **Agent Pipelines / Composition DSL**
  - YAML, Builder, and Flow (decorator) interfaces
  - Parallel, sequential, conditional, loop, and transform steps
  - Auto-schema detection and delegation
  - 99 tests with full platform integration

- **Custom LLM Injection — Pluggable Models Everywhere**
  - `set_llm()` — inject a custom LLM as the global singleton
  - `get_llm(instance=...)`, `get_named_llm(instance=...)`,
    `get_structured_llm(instance=...)` — bypass factory with pre-built LLMs
  - `LLMSpec = str | LLMCallable` type in the `optimize` module —
    all optimizers, metrics, synthesizers, and fitters accept custom callables
  - `call_llm()` / `call_llm_json()` — unified dispatch with graceful
    error handling (string, async/sync callable, LangChain model)
  - `PromptFitterBridge` accepts `LLMSpec` for `task_model` / `rewrite_model`
  - Eliminated 5 raw `httpx` Ollama calls — all LLM traffic goes through
    `LLMCaller` or `call_llm()`

### Changed

- **Studio enabled by default** — `agentomatic run` now enables Studio
  at `/studio/ui/` by default. Use `--no-studio` to disable.
- **Studio resilience** — branded error pages (503) when Studio is
  disabled or assets are missing, instead of raw 404s.

### Fixed

- Fixed literal `\\n` characters appearing in generated `__init__.py` files
  from `agentomatic init` templates.
- Added `from __future__ import annotations` to `plugins/__init__.py` and
  `demo/__init__.py` per project rules.
- Fixed `protocols/__init__.py` missing re-exports and `__all__`.
- Fixed `Makefile` `check-ci` target using auto-fix `format` instead of
  read-only `format-check`.

---

## [0.3.0] - 2026-06-18

### Added

- **Agentomatic Studio** — Built-in visual debugging environment
  - Real-time graph visualization for any agent framework
  - Universal adapter system (LangGraph, LangChain, Deep Agent, Custom)
  - SSE streaming with node-level event mapping
  - Time-travel debugging via checkpoints
  - Breakpoints and state editing (LangGraph)
  - `@studio_graph`, `@studio_state`, `@studio_stream` decorators for custom integration
  - Studio UI bundled into the PyPI package

- **Deep Agent Integration**
  - Full support for LangChain's `deepagents` harness
  - Subagent event tracking (`subagent_start`/`subagent_end`)
  - Task planning visualization (`task_update`)
  - HITL interrupt handling with resume endpoint
  - `agentomatic init --template deepagent` scaffold

- **LangChain Adapter**
  - LCEL chain graph extraction
  - `astream_events` v2 streaming
  - Message history tracking
  - Automatic framework detection

- **Enhanced LangGraph Adapter**
  - Retriever event mapping (`on_retriever_start`/`on_retriever_end`)
  - LLM event mapping (`on_llm_start`/`on_llm_end`)
  - Deep Agent node classification (subagent, planning, filesystem, execute)
  - Interrupt/breakpoint event detection
  - Capability auto-detection from graph topology

- **Demo Command**
  - `agentomatic demo` scaffolds a temporary agent and launches Studio
  - Instant hands-on experience without project setup

- **Agent Skills for AI Assistants**
  - `.agents/skills/agentomatic/SKILL.md` — comprehensive package skill
  - `.agents/AGENTS.md` — project rules and conventions

### Changed

- CI/CD pipeline now includes docs build verification and import smoke tests
- PR checks use `uv` consistently (replaced pip-based pre-commit install)
- Makefile uses `uv sync` instead of `uv pip install` for consistency
- Build verification step in CI now checks wheel contents
- Coverage threshold raised from 40% to 55%

---

## [0.2.0] - 2026-06-14

### Added

- **Universal Studio Adapter Architecture**
  - Pluggable adapter system for any agent framework
  - `GenericAdapter` as fallback for custom/unknown frameworks
  - Adapter resolution chain with automatic framework detection

- **Comprehensive Documentation**
  - MkDocs Material site with 22+ documentation pages
  - CLI reference, user guide, architecture overview
  - Mike versioning for release documentation

- **Enhanced CLI**
  - `agentomatic inspect` — show agent structure and configuration
  - `agentomatic doctor` — environment health check
  - `agentomatic optimize` — prompt optimization runner

### Changed

- Test suite expanded to 393+ tests with multi-Python matrix CI
- Improved error messages and validation

---

## [0.1.0] - 2026-06-12

### Added

- **Core Framework**
  - `AgentPlatform` class for zero-code multi-agent API creation
  - Auto-discovery: drop agent folder → 12+ REST endpoints generated
  - Agent manifest system with `MANIFEST` dict or `@agent` decorator
  - `BaseAgentState` for typed agent state management
  - Pluggable storage: `MemoryStore`, `SQLAlchemyStore`, custom via `BaseStore`

- **CLI** (click-based)
  - `agentomatic init <name>` — scaffold from templates (basic, full, rag, chatbot, custom)
  - `agentomatic run` — start platform server with uvicorn
  - `agentomatic list` — rich table of discovered agents
  - `agentomatic test <name>` — interactive agent testing
  - `agentomatic ui` — launch Chainlit debug interface

- **Prompt Optimization** (DSPy-inspired)
  - 7 optimization strategies: IterativeRewrite, FewShotBootstrap, ChainOfThought, MIPRO, BootstrapRandomSearch, EnsembleOptimizer
  - 8 metric types: DeepEval, LLMJudge, GEval, Contains, ExactMatch, Custom, RedTeam
  - `DataSynthesizer` for generating/augmenting eval datasets
  - HTML report generation with interactive dashboards
  - Red team adversarial testing (PII, bias, prompt injection)

- **Middleware Pipeline**
  - API key authentication with configurable header
  - Token bucket rate limiting (per-IP or global)
  - Prometheus metrics (request counts, latencies, error rates)
  - Structured request/response logging
  - Feedback collection endpoint

- **A2A Protocol**
  - Auto-generated agent cards (model cards)
  - Task submission and status endpoints
  - Inter-agent communication

- **Observability**
  - OpenTelemetry tracing integration
  - Prometheus metrics exporter
  - `CircuitBreaker` for external service protection
  - `AgentSemaphore` for concurrency control

- **Providers**
  - Ollama, OpenAI, Azure, Vertex AI LLM support
  - Configurable embedding providers

- **Developer Experience**
  - Comprehensive test suite (161 tests)
  - CI/CD with GitHub Actions
  - Semantic release for automated versioning
  - Ruff linting + formatting
  - `py.typed` marker for mypy/pyright support

---

[0.3.0]: https://github.com/UnicoLab/agentomatic/releases/tag/v0.3.0
[0.2.0]: https://github.com/UnicoLab/agentomatic/releases/tag/v0.2.0
[0.1.0]: https://github.com/UnicoLab/agentomatic/releases/tag/v0.1.0
