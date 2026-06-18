# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

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
