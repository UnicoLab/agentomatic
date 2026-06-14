# CHANGELOG


## v0.3.0 (2026-06-14)

### Features

- Implement memory management, platform features, and router updates
  ([`5cf8aa8`](https://github.com/UnicoLab/agentomatic/commit/5cf8aa88ef750e2fa443948981aad2f1f9f1f9c4))


## v0.2.0 (2026-06-14)

### Documentation

- Add telemetry and feedback correlation notes for A/B testing
  ([`2c27a86`](https://github.com/UnicoLab/agentomatic/commit/2c27a86ac83b44a022e5d002d88316953f01ab35))

### Features

- Implement advanced platform features (HITL, checkpointer, structured output, fork, A/B routing,
  fallbacks, hooks) and update documentation
  ([`c1a55c3`](https://github.com/UnicoLab/agentomatic/commit/c1a55c39b1efd24084f1453946df27a830d3d7ec))

- Increase core test coverage over 85% and enhance homepage aesthetics
  ([`98f28f5`](https://github.com/UnicoLab/agentomatic/commit/98f28f56308b7540d642149aa2a7edc20b2eeafb))

### Testing

- Enhance unit test coverage for sqlalchemy store, sync checkpointers, and structured fallbacks
  ([`91ab042`](https://github.com/UnicoLab/agentomatic/commit/91ab04254a623fb4fbd108303265ad8d4b3aeba5))


## v0.1.1 (2026-06-13)

### Bug Fixes

- Resolve import block sorting and unused typing imports
  ([`99c83ba`](https://github.com/UnicoLab/agentomatic/commit/99c83ba9c0170a29584db4dca316999f764a4c13))

- **ci**: Force Node.js 24 runtime and disable PyPI attestations to remove warnings
  ([`275a233`](https://github.com/UnicoLab/agentomatic/commit/275a2336cbfe8ca6860e95545172d02ff118234e))

### Documentation

- Add detailed input and output schemas guide and improve custom schema extraction
  ([`4c5e079`](https://github.com/UnicoLab/agentomatic/commit/4c5e0790aeaf68cfcbe6f0f0c52963c9ce3bdffe))

- Expand agent structure, customization overrides, and platform settings references
  ([`5aa9956`](https://github.com/UnicoLab/agentomatic/commit/5aa9956f25f2a56abf27d364f56f086d4a105616))

- Overhaul documentation with detailed guides, visual diagrams, and logo accents
  ([`ef3de5a`](https://github.com/UnicoLab/agentomatic/commit/ef3de5adb9056184b4a66e7c7130c673c84088e6))

- **agnostic**: Replace all remaining hr_bot references with project-agnostic placeholders
  ([`4f868d6`](https://github.com/UnicoLab/agentomatic/commit/4f868d623362d6fa0e1e961cccf52440df4d4aa4))

- **prompts**: Document versioned prompts and PromptManager usage
  ([`f0fe872`](https://github.com/UnicoLab/agentomatic/commit/f0fe8726d73cafd935afa67724b427f452c6d33c))


## v0.1.0 (2026-06-13)

### Bug Fixes

- Add parallel-safe Annotated reducers to BaseAgentState
  ([`04ed6b8`](https://github.com/UnicoLab/agentomatic/commit/04ed6b886dd1bf28329a7165010919e7d14d71d0))

All state fields now use Annotated reducers for LangGraph parallel fan-out compatibility: -
  metadata: dict merge (last-writer-wins per key) - response/agent_type/routing_decision/error:
  last-writer-wins - suggestions/citations/steps_taken: list concatenation (operator.add) -
  messages: add_messages (existing)

This fixes 'Can receive only one value per step' errors when parallel graph branches write to the
  same state field.

- **ci**: Fix codecov param on v7, add pr-title check filter, and import Callable
  ([`490583c`](https://github.com/UnicoLab/agentomatic/commit/490583ce5c793cb397fff8a61c48db5a89e42083))

- **ci**: Fix semantic release token and upgrade deprecated Node.js 20 actions
  ([`7dfea0d`](https://github.com/UnicoLab/agentomatic/commit/7dfea0d3e2dd5fff2f4c005bba9f797d0b246d53))

- **release**: Disable semantic-release build command and pypi upload
  ([`2b1c427`](https://github.com/UnicoLab/agentomatic/commit/2b1c4276e9ac20f75f22f30815e91a08ff4d9165))

### Chores

- Fix workflows, docker, setup-telemetry, and resolve all type/mypy errors for first release
  ([`54cd42d`](https://github.com/UnicoLab/agentomatic/commit/54cd42d8a85760150238ed9d56a884d7c52d677e))

- Run pre-commit quality gate autofix, add importlib.util, resolve mypy and formatting
  ([`1140c7b`](https://github.com/UnicoLab/agentomatic/commit/1140c7bcf0f6a3fb192130d7d91457c7670a0482))

### Documentation

- Comprehensive v0.1.0 release documentation
  ([`06c34b1`](https://github.com/UnicoLab/agentomatic/commit/06c34b104607d0cb892ebe242eab168d82d182eb))

- Create root CHANGELOG.md with detailed release notes - Update docs/changelog.md with full feature
  breakdown - Update docs/index.md with badges, expanded features, architecture diagram - Update
  docs/getting-started/installation.md with extras table, pip/uv/poetry tabs - Update
  docs/getting-started/quickstart.md with expected outputs, next steps - Update docs/cli/commands.md
  with click-style help for all 7 commands - Update docs/guide/optimization.md with all 7
  strategies, GEval, per-agent pattern - Fix noqa WPS433 → F811 in optimize/report.py - Format
  cli/commands.py - mkdocs build --strict passes with 0 errors

- Mkdocs Material + CI/CD + semantic release + pre-commit
  ([`2f672e8`](https://github.com/UnicoLab/agentomatic/commit/2f672e8632bbeb919f64204273299697a9495e44))

Documentation (MkDocs Material): - 16 documentation pages (getting-started, user guide, CLI,
  architecture) - mike versioning (dev/latest aliases) - Dark/light theme with deep purple accent -
  Mermaid diagrams, tabbed content, code copy

CI/CD (GitHub Actions): - ci.yml: lint (ruff), test (Python 3.11/3.12/3.13 matrix), typecheck (mypy)
  - release.yml: python-semantic-release + PyPI publish + GitHub Release - docs.yml: auto-deploy
  docs via mike on push/tag

Pre-commit: - trailing whitespace, end-of-file-fixer, check-yaml/toml/json - ruff lint + format -
  mypy type checking - conventional-pre-commit (commit message enforcement)

Author: UnicoLab - Updated pyproject.toml, LICENSE, README, all URLs - Added semantic-release
  config, coverage thresholds - Added docs and cli optional deps - Cleaned up 12 stale test files,
  removed conflicting pytest.ini - Enhanced Makefile with docs-serve, docs-build, docs-deploy,
  check-all

88/88 tests passing, docs build clean in strict mode

### Features

- Add Pixar logo, fix 130 lint errors, format all files
  ([`902072d`](https://github.com/UnicoLab/agentomatic/commit/902072d55a60791cf4c8c5e7ee439cb4566c4d9d))

- Add assets/logo.png (Pixar-style 3D robot mascot) - Update README.md with centered logo - Fix 130
  ruff lint errors (import sorting, unused imports, etc.) - Reformat 51 files with ruff format - All
  161 tests pass, all linting clean - Verify CI workflows reference UnicoLab/agentomatic

- Add PromptOptimizationLoop — local-first prompt optimizer
  ([`3b4143f`](https://github.com/UnicoLab/agentomatic/commit/3b4143f1b1ad641e4a4cf3c01a964ad5ed2552f3))

Generic, framework-agnostic iterative prompt optimization engine: - PromptOptimizationLoop: evaluate
  → analyse failures → LLM rewrite → repeat - 4 rewrite strategies: iterative, adversarial,
  structured, minimal - Built-in scorers: keyword_overlap, contains_score - Pluggable scoring: sync
  or async (LLM-as-a-judge) - Pluggable rewrite LLM: any LangChain model or raw callable - Early
  stopping with patience - Rich HTML reports with SVG evolution charts - JSON experiment tracking
  for reproducibility - Zero project-specific deps — works with any agent

- Complete framework with SQLAlchemy storage, full example, and integration tests
  ([`c358ed1`](https://github.com/UnicoLab/agentomatic/commit/c358ed137213edaa87e9f7c4323c0130d92dbe0a))

New modules: - storage/models.py: SQLAlchemy ORM (ThreadModel, MessageModel, FeedbackModel) -
  storage/sqlalchemy.py: Async SQLAlchemyStore with connection pooling - examples/full_agent/: Full
  weather agent demonstrating ALL overwrite options (manifest, graph, nodes, config, schemas, tools,
  api, prompts, langgraph.json)

Fixes: - platform.py: Mount routers for programmatic agents at build-time (previously only mounted
  during lifespan, breaking TestClient)

Tests: - test_integration.py: 20 integration tests (platform + agent endpoints) - 41/41 tests
  passing (unit + integration)

- Deep DeepEval + HolySheet integration
  ([`fc2dc44`](https://github.com/UnicoLab/agentomatic/commit/fc2dc4443f42081fc23ead06e5db67d080651474))

Metrics (metrics.py): - GEvalMetric — DeepEval GEval with custom criteria + eval_steps
  (chain-of-thought LLM-as-judge, Ollama fallback) - DeepEvalMetric — wrap ANY deepeval metric
  instance as BaseMetric - RedTeamMetric — adversarial safety scoring (bias+toxicity) -
  resolve_metrics() now supports 'geval:criteria' shorthand syntax - All DeepEval imports gracefully
  degrade via try/except

Reports (report.py): - HolySheet integration for interactive React dashboards - KPI cards (baseline,
  best, improvement, duration, iteration) - LineChart (score vs iteration per metric) - DataTable
  (full iteration history) - Markdown (prompt diffs via difflib) - Falls back to inline SVG/HTML if
  HolySheet not installed

Synthesizer (synthesizer.py): - generate_from_docs() — DeepEval
  Synthesizer.generate_goldens_from_docs() - red_team() — DeepEval RedTeamer with 40+ vulnerability
  scans - to_deepeval_dataset() / from_deepeval_dataset() — bidirectional bridge - Convenience:
  generate_from_docs(), red_team() at module level

Dependencies: - optimize extra: deepeval>=2.0, holysheet>=0.1

Tests: 139/139 passing | Docs: builds clean

- Make skip_paths customizable in AuthMiddleware
  ([`acc6de2`](https://github.com/UnicoLab/agentomatic/commit/acc6de28526865b4d4b4fbcfa7004ffb144f2382))

- Optimization endpoints, feedback collection, OpenTelemetry
  ([`3dcfc8c`](https://github.com/UnicoLab/agentomatic/commit/3dcfc8cad1f3349c0a4efffecde077bc14b7bbfb))

API Endpoints: - POST /{agent}/optimize/invoke — full pipeline context (retrieval_context,
  tool_calls, reasoning, citations) for DeepEval metrics - POST /{agent}/feedback — async user
  feedback (thumbs, rating, correction) - GET /{agent}/feedback — list feedback entries - GET
  /{agent}/feedback/export — JSONL export for optimization datasets

Feedback System (middleware/feedback.py): - FeedbackCollector — async collector with in-memory
  buffer + BaseStore backend - @collect_feedback decorator — auto-record agent I/O for dataset
  building - Module-level singleton via get_collector()/set_collector() - Corrections feed back as
  expected_answers for optimization

OpenTelemetry (observability/telemetry.py): - setup_telemetry(app) — auto-configures from env vars -
  Auto-instruments FastAPI + httpx - @traced decorator — creates spans for any sync/async function -
  OTLP exporter (gRPC → HTTP fallback) or ConsoleSpanExporter - Graceful no-op when opentelemetry
  not installed

Runner Context (optimize/runner.py): - RunResult now carries retrieval_context, tool_calls,
  reasoning, steps_taken - Auto-tries /optimize/invoke first, falls back to /invoke -
  submit_feedback() method for feeding optimization results back

Platform (core/platform.py): - enable_feedback=True (default) — auto-attaches feedback endpoints -
  enable_telemetry=True (default) — auto-configures OTEL on build()

Dependencies: - telemetry = [opentelemetry-api, opentelemetry-sdk, fastapi+httpx instrumentors] -
  all extra now includes telemetry

Tests: 161/161 passing | Docs: builds clean

- Production-ready pluggable architecture
  ([`f096a9d`](https://github.com/UnicoLab/agentomatic/commit/f096a9d5285210ddeb7de24233b25df533a22cad))

Storage: - BaseStore ABC defining the universal storage protocol - MemoryStore and SQLAlchemyStore
  both inherit BaseStore - Users can implement RedisStore, MongoStore, etc. by subclassing -
  Platform accepts store= param with auto-init/close lifecycle

Middleware pipeline: - AuthMiddleware: API key auth via header/query param - RateLimitMiddleware:
  sliding-window per-IP rate limiter - MetricsMiddleware: Prometheus counters + histograms - All
  toggleable via platform constructor params - Custom middleware via middleware=[] param

Platform integration: - Storage wired into router_factory (threads, messages, feedback) - Health
  endpoint includes storage health check - /api/v1/storage/stats and /api/v1/feedback endpoints -
  Lifecycle hooks properly init/close storage

Tests: - 63/63 passing (21 unit + 42 integration) - Coverage: invoke, chat, stream, A2A, threads,
  lifecycle, middleware, storage, feedback, programmatic reg, custom routers

- Prompt optimization engine with DSPy-inspired strategies
  ([`c12f00e`](https://github.com/UnicoLab/agentomatic/commit/c12f00e3616bc79b34b51688c418fb6cc57e8548))

Optimization Module (agentomatic.optimize): - PromptOptimizer — like model.fit() for prompts -
  Separate rewrite_llm and eval_llm for full model control - 7 optimization strategies
  (iterative_rewrite, few_shot, chain_of_thought, mipro, bootstrap_random_search, ensemble) - Rich
  progress display with per-iteration metrics table - Auto HTML reports with SVG charts + prompt
  diffs - Experiment tracking (JSON logs) - Prompt versioning (branching in prompts.json) - Early
  stopping with patience + target score - Cross-prompt A/B comparison

Data Synthesis (agentomatic.optimize.synthesizer): - DataSynthesizer — generate eval datasets from
  descriptions - 5 augmentation strategies (paraphrase, perturbation, expansion, adversarial,
  formality_shift) - generate_dataset() + augment_dataset() convenience functions - Generate from
  system prompts or agent descriptions

Metrics (agentomatic.optimize.metrics): - Built-in: ExactMatchMetric, ContainsMetric (no LLM needed)
  - DeepEval: answer_relevancy, faithfulness, hallucination, etc. - LLMJudgeMetric — custom criteria
  with LLM-as-judge - CustomMetric — wrap any sync/async callable

CLI: agentomatic optimize <agent> --dataset qa.jsonl

Docs: guide/optimization.md with full usage reference

Tests: 51 new tests (139/139 total passing)

Install: pip install agentomatic[optimize]

- Rich CLI/TUI + Chainlit debug UI + 5 scaffolding templates
  ([`642cdf7`](https://github.com/UnicoLab/agentomatic/commit/642cdf7727ec1392dbd71edbd7098561c26fdb0a))

CLI (agentomatic command): - init: Interactive agent scaffolding with 5 templates (basic, full, rag,
  chatbot, custom) - run: Start platform with --with-ui flag - list: Rich table of discovered agents
  - test: Interactive terminal testing against running agents - inspect: Show agent structure,
  manifest, config - doctor: Environment health check with dep status - ui: Launch Chainlit debug UI
  standalone

Templates generate ALL agent files: - __init__.py, graph.py, nodes.py (always) - config.py,
  schemas.py, tools.py, api.py (full template) - prompts.json, langgraph.json, .env.example,
  README.md

Chainlit Debug UI (pip install agentomatic[ui]): - Mounts at /chat inside the same FastAPI app -
  Agent selector, real-time streaming, tool call visualization - Chain-of-thought display, feedback
  collection - Dark theme with agentomatic branding

Dependencies: - cli extra: rich>=13.0, questionary>=2.0 - ui extra: chainlit>=2.0

Tests: 88/88 passing (21 unit + 42 integration + 25 CLI)

- 🚀 agentomatic v0.1.0 — zero-code multi-agent API framework
  ([`ba9d6ae`](https://github.com/UnicoLab/agentomatic/commit/ba9d6ae8f17c3e01ed122be630755a3c80126393))

Complete rewrite of lm_agents_api into a reusable Python package:

Core Framework: - AgentPlatform: one-liner app factory with auto-discovery - AgentRegistry:
  auto-discovers agents from folder structure - AgentManifest: frozen dataclass identity card per
  agent - BaseAgentState: LangGraph-compatible default state - RouterFactory: auto-generates 12+
  REST endpoints per agent

Features: - Auto-generated endpoints: invoke, stream, chat, health, config, prompts, A2A -
  Framework-agnostic: LangGraph, LangChain, or plain async functions - Circuit breakers &
  concurrency limiting - Prometheus metrics with graceful fallback - Pluggable storage
  (memory/SQLAlchemy) - Prompt versioning with JSON-based management - A2A protocol with
  auto-generated agent cards - Feature flags via env vars (streaming, auth, metrics, etc.) - CLI:
  init, run, list commands - LoggingMiddleware with X-Request-ID and timing

Package: - hatchling build system - Optional extras: langgraph, ollama, openai, azure, vertex,
  metrics, db - PEP 561 py.typed marker - Comprehensive test suite (21 tests, all passing) - MIT
  license

Example: - hello_agent: 3-line main.py demonstrating the framework

- **prompts**: Export PromptManager from root and support prompts_file parameter
  ([`bb3cbf2`](https://github.com/UnicoLab/agentomatic/commit/bb3cbf2397f211ee0e7222bca83175d7c7ecaf42))

### Refactoring

- Cleaning the implementation
  ([`36f035b`](https://github.com/UnicoLab/agentomatic/commit/36f035b290bd3101eab77935ebbdeb273f9826c3))

- Improving and testing
  ([`af37c0a`](https://github.com/UnicoLab/agentomatic/commit/af37c0a6a2512be656446fb5a65593ae42797b51))

- Improving dockers
  ([`5f65ae3`](https://github.com/UnicoLab/agentomatic/commit/5f65ae35663147f1900fd00c9cc6c2527faebc1b))

- Migrate CLI from argparse to click, replace print with loguru
  ([`6b00b9e`](https://github.com/UnicoLab/agentomatic/commit/6b00b9e6c5144d1200e2159b33049e7a721cb4a2))

- Replace argparse with click decorators for all 8 commands - All print() → loguru
  logger.info/success/warning/error - Add click>=8.1.0 to dependencies - Update entry point to cli
  click group - 161/161 tests pass

- Saving state
  ([`0e73eaf`](https://github.com/UnicoLab/agentomatic/commit/0e73eafea194cc9629262d0100f4a97ba52d76eb))

- Simplification
  ([`32e5987`](https://github.com/UnicoLab/agentomatic/commit/32e5987ae3449849d84eea73ff78a8b2abfb3f5e))

- Simplifying
  ([`fabdc64`](https://github.com/UnicoLab/agentomatic/commit/fabdc64bfcf29f07a0476c13f2310fccf90d51b3))

### Testing

- Adding some first tests
  ([`ce21191`](https://github.com/UnicoLab/agentomatic/commit/ce21191e9f9a5b4cd584056e66911f2a9fb21f70))
