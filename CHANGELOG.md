# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-12

### Added

- **Core Framework**
  - `AgentPlatform` class for zero-code multi-agent API creation
  - Auto-discovery: drop agent folder → 12+ REST endpoints generated
  - Agent manifest system with `MANIFEST` dict or `@agent` decorator
  - `BaseAgentState` for typed agent state management
  - Pluggable storage: `MemoryStore`, `SQLAlchemyStore`, custom via `BaseStore`

- **CLI** (click-based)
  - `agentomatic init <name>` — scaffold from 5 templates (basic, full, rag, chatbot, custom)
  - `agentomatic run` — start platform server with uvicorn
  - `agentomatic list` — rich table of discovered agents
  - `agentomatic test <name>` — interactive agent testing
  - `agentomatic inspect <name>` — show agent structure & config
  - `agentomatic doctor` — environment health check
  - `agentomatic ui` — launch Chainlit debug interface

- **Prompt Optimization** (DSPy-inspired)
  - 7 optimization strategies: IterativeRewrite, FewShotBootstrap, ChainOfThought, MIPRO, BootstrapRandomSearch, EnsembleOptimizer
  - 8 metric types: DeepEval, LLMJudge, GEval, Contains, ExactMatch, Custom, RedTeam
  - `DataSynthesizer` for generating/augmenting eval datasets
  - HTML report generation with interactive dashboards (HolySheet or SVG fallback)
  - Red team adversarial testing (PII, bias, prompt injection)
  - Experiment tracking via `.optimize/{agent}/experiments.json`

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
  - `track_agent_invocation` context manager

- **Providers**
  - Ollama, OpenAI, Azure, Vertex AI LLM support
  - Configurable embedding providers
  - LiteLLM integration for 100+ models

- **Developer Experience**
  - MkDocs Material documentation site with mike versioning
  - Comprehensive test suite (161 tests)
  - CI/CD with GitHub Actions (lint, test matrix, typecheck)
  - Semantic release for automated versioning
  - Ruff linting + formatting
  - `py.typed` marker for mypy/pyright support
  - Pre-commit hooks support

[0.1.0]: https://github.com/UnicoLab/agentomatic/releases/tag/v0.1.0
