# Changelog

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

Targeting **v1.8.0** (PyPI may still show 1.7.0 until the release job
publishes). Install from git / an editable checkout until then.

### Added

- **`EvalConfig` / `evaluate_and_report`** (1.8.6): Thin eval scripts
  mirroring `train_and_report` — stack load, structured + LLM-judge
  metrics, split selection, optional augmented dataset reuse, and
  HolySheet `generate_eval_report`. Scaffold `agents/*/eval.py` updated.
  `OptimizeMetricAdapter` now prefers `question` over meta-`query` and
  forwards snapshot `context` + rich expected references to judges.
- **Per-agent invocation log history + optional LLM analysis**: Opt-in
  flags `logs_history` / `AGENTOMATIC_LOGS_HISTORY` and
  `allow_logsllm_analysis` / `AGENTOMATIC_ALLOW_LOGSLLM_ANALYSIS` persist
  full invoke/chat/stream payloads into the platform store
  (`AgentInvocationLog`). REST endpoints: `GET /logs`, `GET /logs/{id}`,
  `POST /logs/analyze`, `GET /logs/analysis`. Analyser samples/truncates
  logs for API budget safety and falls back to a heuristic when no LLM
  is configured. Related: `OptimizationRunStore` + `optimize/fit_store`
  for auditable retrain artefacts.
- **Ordered LLM model fallbacks** (ships in **1.8.0**): Configure
  `fallbacks` / `fallback_on` on stack LLM profiles or via
  `get_llm(..., fallbacks=..., fallback_on=...)`. On timeout, connection
  error, rate limit, or empty response the next model is tried;
  `record_failover` and a success log identify which model answered.
  Single-model stacks are unchanged when `fallbacks` is omitted. See
  [LLM Providers](guide/llm-providers.md) and [Stacks](guide/stacks.md).
- **Multi-pass optimize (SLM + LLM)**: `optimize/briefing.py` builds a full
  optimization briefing (runtime config, search space, dataset samples,
  eval I/O, metrics, history) for rewrite / GEPA / MIPRO. Auto multi-pass:
  **3** (draft→critique→revise) for SLMs / local providers, **2**
  (draft→self-check) for frontier LLMs. Prompt style and briefing size
  adapt per model class. Knobs: `rewrite_passes`, `multipass`,
  `slm_multipass`, `llm_multipass`, `slm_default_passes`,
  `llm_default_passes`. See [Optimization](guide/optimization.md).
- **`gemini/` optimize provider**: `LLMCaller` routes
  `gemini/gemini-…` via the Generative Language API (`GEMINI_API_KEY`).
  Live suite: `tests/test_live_gemini_optimize.py`.
- **Hardened `openai/` optimize routing**: cloud `gpt-*` / `o1`/`o3` models
  are not hijacked by a local `OPENAI_BASE_URL`; clear `OPENAI_API_KEY`
  errors; `max_completion_tokens` for reasoning models; live suite
  `tests/test_live_openai_optimize.py`.
- **Thinking / reasoning LLMs**: `message_text`, `message_thinking`,
  `strip_thinking_for_json`, `astream_with_thinking`, and
  `invoke_with_retry(strip_thinking=…)` normalize Qwen3 / tagged `<think>` /
  `reasoning_content` responses. Stack `extra:` forwards `enable_thinking`,
  `chat_template_kwargs`, `response_format`, and `extra_body` for oMLX /
  OpenAI-compatible servers. See [LLM Providers](guide/llm-providers.md).
- **Plugin reload API**: `POST /api/v1/plugins/reload` (all) and
  `POST /api/v1/plugins/{name}/reload` re-call `load_model()` on the live
  registry instance and return status + `loaded_at` + `model_card`. List
  plugins now includes `loaded_at`. See [ML Plugins](guide/ml-plugins.md).

### Fixed

- **Train / optimize mechanics**: Fitted prompts and overrides apply during
  evaluate/reevaluate; GridSearch/PromptFitterBridge persist
  `system_prompt`; local in-process runner for `fit()`; honest LLM-as-judge
  failures; class-agent `/invoke/stream` per-node frames.
  `GEvalMetric` falls through on any deepeval failure; `MultiJudgePanel`
  sets `evaluation_failed` when all judges soft-fail.
- **optimize / oMLX**: `omlx/` provider routing; disable thinking on local
  OpenAI-compatible servers; strip residual CoT from optimize LLM calls;
  skip DeepEval for oMLX/local specs; MIPRO accepts `DataPoint` samples;
  `_wrap_local_agent` injects top-level `system_prompt_override`.
- **Invoke context passthrough**: `AgentInvokeRequest` / chat / Studio /
  async task dispatch use `extra="allow"` and
  `build_invoke_state()` so the **entire** client payload (rich `context` +
  unknown top-level keys) reaches class-agent `input_to_state`
- **OpenAPI full schema**: `StudioResumeRequest` moved to module scope so
  `/openapi.json` no longer falls back to the ~13-path stub catalog
- **Class-agent async tasks**: default agent dispatcher uses
  `invoke_registered_agent` (same as sync REST) so `input_to_state` runs
- **`agentomatic run` + `main.py`**: prefers `uvicorn main:app` when present;
  maps `AGENTOMATIC_ENABLE_METRICS` on the `from_folder` fallback path
- **Invoke `context`**: flattened into the transform payload before
  `input_to_state` for class agents
- **Docs**: invoke paths documented as `/api/v1/{name}/invoke` (not
  `/api/v1/agents/{name}/invoke`)

---

## [1.3.0] — 2026-07-16

Minor release: deployment ergonomics and coding-agent enablement. Builds on the
1.2.x production-readiness fixes (which ship together in this release). No
breaking API changes.

### Added

- **Fully-featured, env-driven scaffolded `main.py`**: the project `main.py`
  builds a module-level `app` that is feature-identical to `agentomatic run`
  (Studio, docs, health, metrics on by default; all component dirs discovered),
  so a deployed container running `uvicorn main:app` drops no functionality.
  Behaviour is driven by `AGENTOMATIC_*` env vars (`ENABLE_STUDIO`,
  `ENABLE_METRICS`, `ENABLE_AUTH`, `ENABLE_JWT`, `REQUIRE_AUTH`,
  `ENABLE_CONTROL_PLANE`, `ENABLE_RATE_LIMIT`, `TITLE`, `LOG_LEVEL`) so the same
  file works in dev and in the container
- **Deploy profiles**: `agentomatic deploy --profile full|minimal` (and the
  `--minimal` shorthand). `full` (default) runs everything; `minimal` is a
  production-lean image that disables the Studio debug UI and quiets logging
  via baked-in `AGENTOMATIC_*` env vars while keeping the core REST API,
  health/readiness, metrics, and auth. **Swagger/OpenAPI (`/docs`, `/redoc`,
  `/openapi.json`) is always available in both profiles.** Both profiles share
  one env-driven `main.py` code path (no separate image), and `minimal` still
  installs `agentomatic[all]` so no required functionality is dropped
- **`agentomatic agents-guide` command**: prints an Agentomatic primer for
  bootstrapping any coding agent, or writes it into a project with
  `--write AGENTS.md|CLAUDE.md|.cursor/skills/agentomatic/SKILL.md` (refuses to
  overwrite without `--force`). Content comes from a single source of truth
  (`agentomatic.cli.agent_guide`) so the CLI, docs, and in-repo agent knowledge
  files stay in sync
- **Custom DB / vector-store connections for any Python client**: the connection
  abstraction robustly accepts arbitrary custom clients (async or sync SDKs,
  graph/time-series DBs, in-house packages) with correct lifecycle. Provider
  names are no longer limited to the built-ins — any name registered via
  `register_vector_provider` resolves (unknown names raise a clear, actionable
  error); `VectorConnection.close()` now also handles clients exposing only
  `disconnect`; and a new `initialize_connections(scope, configs)` helper
  registers **and** initialises connections in one call for standalone runs
  (`get_graph()`, scripts) that have no platform lifecycle. New dedicated guide:
  [Custom DB & Vector Store Connections](guide/custom-connections.md)

---

## [1.2.1] — 2026-07-16

Patch release: production/wiring fixes that landed after the 1.2.0 cut. No
breaking API changes.

### Fixed

- **Class-agent `langgraph.json`**: points at `./agent.py:get_graph` (with a
  module-level `get_graph()` export) instead of broken `./graph.py:get_graph`
- **Plugin train/optimize stubs**: exit with `SystemExit(1)` instead of
  logging fake success, so `python -m` lifecycle commands fail loudly
- **Plugin `eval`**: computes real accuracy from labelled examples (and fails
  with a clear message when metrics cannot be derived) instead of reporting a
  fabricated score
- **`agentomatic optimize --llm`**: defaults from `AGENTOMATIC_TASK_MODEL` /
  `LLM__MODEL` (then `ollama/mistral:7b`) instead of a hardwired default
- **OpenAPI fallback**: minimal path list from `app.routes` when schema
  generation fails (no empty `"paths": {}`, keeps `/docs` usable)
- **CLI list/inspect**: `has_graph` is true when `agent.py` or `graph.py`
  exists
- **Studio auth**: client sends both `X-Api-Key` and `Authorization: Bearer`
  on REST + SSE requests so either credential scheme authenticates
- **Type-check hardening**: resolved mypy errors in the task store / manager,
  registry LLM injection, pipeline steps, and status/health probes

### Fixed (production-readiness audit)

- **Class agents on every server path (P0)**: REST `invoke` / `chat` /
  `invoke/stream` / `optimize/invoke` / A2A / approve and the Studio streaming
  adapter now route class agents through `invoke_registered_agent` /
  `input_to_state` instead of `graph.ainvoke(dict)`, so dataclass-state agents
  no longer raise `AttributeError` → HTTP 500 / Studio `run_error`
- **`agentomatic deploy` Dockerfile (P0)**: generates a project-appropriate
  image that installs `agentomatic[all]==<version>` from PyPI, copies only the
  project dirs that exist (`main.py`, `agents/`, …), and launches
  `uvicorn main:app`; compose build context/volumes point back to the project
  root. `init --project` now emits a pinned `requirements.txt`
- **`agentomatic run --reload` / `workers>1` (P1)**: run via a module-level
  factory import string (`agentomatic._runtime:create_app`) instead of passing
  an app instance (which made uvicorn `exit(1)`); programmatic platforms
  degrade to a single instance with a clear message instead of crashing
- **`--require-auth-globally` JWT bypass (P1)**: refuses to start when no JWKS
  (or API-key auth) is configured instead of silently accepting forged/unsigned
  tokens; expiry (`exp`) is now always verified, even in dev mode
- **`AGENTOMATIC_AGENTS` allow-list (P1)**: agent discovery now honours the env
  var (comma-separated names) so `deploy --with-agent-stubs` actually scopes
  each replica to a single agent
- **`PromptFitterBridge.optimize()` (P2)**: records a structured
  `agent._last_optimize_status` (`"ok"` / `"skipped: <reason>"`) so callers can
  tell whether optimization ran instead of silently no-op'ing
- **CORS hardening (P2)**: wildcard origins (`cors_origins=["*"]`) no longer
  send `Access-Control-Allow-Credentials: true` — credentials are auto-disabled
  (with a one-time warning) unless explicit origins are configured

---

## [1.2.0] — 2026-07-15

Release notes for the develop / optimize / deploy platform wave:
scaffolding, stacks, connections, deploy, Keras-style training polish,
and the wiring fixes below.

### Fixed

- **Scaffolded `main.py`**: removed invalid `pipelines_dir=` kwarg that caused
  `TypeError` on boot (pipelines auto-discovered from sibling `pipelines/`)
- **Store auto-derive timing**: routers now use a lazy store proxy so MEMORY
  connections derived in lifespan wire conversation threads correctly
- **Class-agent LLM / stack**: `apply_stack_defaults` runs before discovery;
  registry passes `stack_manager` into `get_llm_for_agent` for role-aware LLMs
- **Pipelines + class agents**: invoke via `atransform` / `input_to_state`
  instead of raw `graph.ainvoke(dict)` (dataclass states no longer break)
- **`--require-auth-globally`**: auto-enables JWT so the flag cannot lock out
  all traffic without a credential path
- **PromptFitterBridge**: strips bridge-only kwargs (`metric`, optimize
  toggles) before constructing `PromptFitter` (no more silent TypeError skip);
  runs fitter on a worker thread when called inside an existing event loop
- **`evaluate()`**: defaults to metrics from `compile()`; clear error when none
- **`save()` / `load()`**: persists and restores Keras-style `History` and
  `evaluation_history`; `load()` is an alias of `load_compiled()`
- **Docker / compose healthcheck**: probes `/health` (not non-existent
  `/api/v1/health`); compose mounts plugins/endpoints/ingestion/pipelines
- **Studio class agents**: `resolve_adapter` prefers `GraphAgentAdapter` for
  `framework=graph_agent` / `class_instance` *before* treating `graph_fn` as
  LangGraph (unblocks Studio for scaffolded class agents)
- **Studio auth / SSE**: `/studio` and `/status` skipped by JWT/API-key
  middleware (prefix match); SSE calls `onDone` when the stream ends without
  `[DONE]` (clears stuck “thinking…”)
- **Plugin templates**: relative imports (`from .plugin import …`); project
  scaffold includes `agents/__init__.py` + `plugins/__init__.py` for
  `python -m` train/eval
- **CLI**: `--template class` alias; `agentomatic run` accepts
  `--endpoints-dir` / `--ingestion-dir` / `--stacks-dir`

- Top-level exports: `WeightedMetric`, `PromptFitterBridge`, optimizers,
  `VectorStore`, `register_store_provider`, `register_embedding_provider`
- Warn when `fit(..., search_space=...)` knobs are ignored by a non-bridge
  optimizer; warn on `compile()` without metrics
- Scaffold `main.py` reuses one platform instance for `app` + `run()`
- **Platform hardening (Swagger, Studio, scaffolding, generic RAG ops, deploy)**
  - Resilient OpenAPI: plugin/`response_model` BaseModel guards +
    `custom_openapi()` fallback so one bad schema no longer blanks `/docs`
  - Studio: resolve agents by **name or slug**; SSE errors surface in chat/
    debug (no more stuck "thinking…"); stream chunk accumulation + node-prefix
    stripping for correct answers/graph highlight
  - Project scaffold: `agentomatic init --project` / `agentomatic new` writes
    `main.py`, stacks, `.env.example`, and component dirs
  - Agent templates emit `AgentManifest` cards + stack-driven `llm.py`;
    registry injects LLM + PromptManager into class agents
  - Safe merge init (no overwrite without `--force`); `agentomatic add
    connection|ingestion`; ingestion/plugin/endpoint templates land in the
    correct top-level discovery dirs
  - Provider-agnostic `VectorStore` Protocol + `register_vector_provider` /
    `register_embedding_provider` / `register_store_provider` (users own
    vendor SDKs such as Cosmos — no first-party vendor connectors in-core)
  - Any-DB memory via connection→store factory + `MinimalDocumentStore`;
    auto-derive platform store from `MEMORY` connections
  - Stack-aware `get_llm`/`set_llm` with custom `base_url` /
    `openai_compatible`; apply stack defaults at startup
  - Pipeline `map` step (parallel scope fan-out), markdown ingestor,
    task retry/checkpoints, `extraction` template
  - `agentomatic deploy` (distroless/rootless), `stack export --env`,
    HTTPS cert flags, `require_auth_globally`
  - Optimize CLI + `BaseGraphAgent.fit(..., search_space=...,
    optimize_mode=..., optimize_prompt/params=...)` for per-call control
    over what to tune; weighted multi-criteria train/eval templates
- **Durable `SQLAlchemyTaskStore`**
  - Drop-in, production-ready `TaskStore` backed by any SQLAlchemy async driver
    (SQLite/PostgreSQL/MySQL) — task status, progress, and results survive
    process restarts and are shared across workers/replicas
  - Wire it in one line: `AgentPlatform.from_folder(..., task_store=SQLAlchemyTaskStore(url))`;
    the platform lifespan initialises and disposes it automatically
  - Fully configurable and safe by default: connection pooling, `table_name`,
    TTL/`max_records` eviction (best-effort, never blocks a `save`), reusable
    external `engine`, and a forward-compatible indexed-JSON schema
  - Lazy import — `agentomatic.tasks` never requires the optional `db` extra;
    a clear install hint is raised only when the store is constructed
  - Exposed as `agentomatic.tasks.SQLAlchemyTaskStore`; install via
    `agentomatic[db]` (SQLite) or `agentomatic[db-postgres]` (PostgreSQL)
- **Keras-style agent training lifecycle**
  - `BaseGraphAgent.fit()` is now epoch-aware and returns a real **`History`**
    object (`.history` log-key → per-epoch values, plus `.epoch`, `.params`,
    `final()`, `best()`, `to_dict()`, `summary()`); also stored on
    `agent.history`
  - `fit(dataset, *, epochs, verbose, callbacks, validation_data)` — per-epoch
    optimizer step + train/validation evaluation, Keras-like verbose log lines,
    and `val_*` metrics when `validation_data` is supplied
  - **Callbacks**: `Callback` base class (`on_train_begin`/`on_epoch_begin`/
    `on_epoch_end(epoch, logs)`/`on_train_end`) and a built-in `EarlyStopping`
    that halts training via `agent.stop_training`
  - **Loss abstraction**: `compile(..., loss=...)` accepts a `Loss`, any
    metric-like object (converted to `1 - score` via `MetricLoss`), or a
    callable (`CallableLoss`); `resolve_loss()` coerces any of these
  - **Optimize-engine wiring**: `PromptFitterBridge` now actually runs the
    async `optimize.PromptFitter` from `fit()`, applies the best prompt config
    back onto the agent, and stores the full `PromptFitResult` on
    `agent._last_fit_result` (gracefully degrades to a baseline pass when the
    `optimize` extra is missing or called inside a running event loop)
  - `compile()` arguments are now all optional (dataset/metrics can be provided
    at `fit()` time); `History`, `Callback`, `EarlyStopping`, and `Loss` are
    exported from `agentomatic` and `agentomatic.agents`
- **Pipeline data-passing hardening**
  - New **`plugin:` step type** — call a registered ML plugin's `predict()`
    mid-pipeline; the resolved input mapping is coerced into the plugin's
    declared input schema before inference (with input/output mapping,
    condition, retry, timeout, and `on_error` like every other step)
  - **Rollback / compensation**: under `on_error: rollback`, completed steps
    are compensated in reverse order via an optional per-step `rollback` code
    block (with `ctx` and `output` in scope); compensated steps are reported in
    `result.metadata["rolled_back_steps"]`
  - **Optional input/output schema enforcement**: `input_schema` /
    `output_schema` are now validated (advisory by default, failing when
    `strict_schema: true`) via a lightweight type checker
  - Plugin registry is threaded through the pipeline engine, router, task
    dispatcher, and sub-pipelines so plugin steps work in every run mode
    (sync, `/run/async`, `/run/batch`, and nested pipelines)
- **Per-resource execution-mode sugar**
  - Every resource now exposes consistent `/<sync>/async` and `/<sync>/batch`
    companion routes backed by the unified task system: agents
    (`/invoke/async`, `/invoke/batch`), plugins (`/predict/async`,
    `/predict/batch`), pipelines (`/run/async`, `/run/batch`), ingestors, and
    custom endpoints
  - All return `202` with a task id and hypermedia `links` (status / events /
    result / cancel); batch bodies wrap a list of the normal input and fan out
    as one task with per-item progress
  - Shared `attach_execution_modes` helper + `BatchSubmitRequest` in
    `agentomatic.tasks` keep behaviour and links uniform across resource types
- **Unified Status Dashboard**
  - Self-contained, auto-refreshing HTML dashboard at `/status` showing the
    health of every resource — agents, plugins, custom endpoints, ingestors,
    and pipelines — plus the task executor and storage backend, with no
    external assets
  - Structured `GET /api/v1/status` endpoint exposing the same snapshot as JSON
    (platform uptime/version, per-resource healthy/total summaries, task stats
    by status + concurrency, and storage health) for custom tooling
  - Extended `GET /health` to aggregate plugins, endpoints, ingestors, and
    pipelines (previously agents + storage only); root index now advertises
    resource counts and the `/status` link
  - `TaskManager.stats()` snapshot (totals, per-status breakdown, running vs.
    max concurrency, supported targets)
- **First-class Ingestion / RAG ops layer**
  - `BaseIngestor` packages *your* document-ingestion code — built with any
    libraries you like (`docling`/`unstructured`/`pymupdf4llm` to parse,
    `langchain-text-splitters` to chunk, your own vector DB client) — as a
    deployable resource. Agentomatic provides the ops, not the implementation
  - Auto-discovery from an `ingestion/` directory; routes mounted under
    `/api/v1/ingestion/{name}` with `/run` (sync), `/run/async` (background
    task), `/info`, and `/health`
  - Async runs use the unified task system for live progress, SSE streaming,
    cancellation, and webhooks; the `ingest(request, ctx)` context exposes
    `await ctx.report(...)` and `ctx.cancelled`
  - Flexible `IngestionResult` telemetry (documents/chunks/upserted/skipped +
    free-form `stats`/`output`) and a default `IngestionRequest` input model
  - Hardened embeddings factory: cached per `(provider, kwargs)`, new
    dependency-free `HashEmbedder`, and `openai` provider support
  - Fully connected to the rest of the platform: new **`ingestion:` pipeline
    step type** (with input/output mapping, condition, retry, timeout, and
    sub-pipeline support), inclusion in the unified `/health` status (alongside
    plugins, endpoints, and pipelines), and resource counts on the root index
  - `agentomatic init <name> --template ingestion` scaffolding
  - Public API: `BaseIngestor`, `IngestionRegistry`, `IngestionRequest`,
    `IngestionResult`
- **Unified Task / Execution Subsystem**
  - Run any resource — agent, ML plugin, pipeline, or custom endpoint — in
    **sync**, **async (background)**, **batch**, or **streaming** modes through a
    single, uniform `TaskRecord` contract
  - Task board API at `/api/v1/tasks`: submit (`202`/`200`), list/filter, poll
    status, fetch result, cancel, delete, and a live SSE progress stream
    (`/tasks/{id}/events`)
  - `TaskManager` with a bounded in-process queue (`task_max_concurrency`),
    cooperative cancellation, batch fan-out with per-item progress, and
    completion webhooks (`callback_url`)
  - Pluggable persistence via `TaskStore` (default bounded, TTL-aware
    `InMemoryTaskStore`); enable/disable with `enable_tasks`
  - **A2A task lifecycle is now real and pollable** — `POST /{agent}/a2a/tasks`
    runs asynchronously and returns a trackable id; `GET .../a2a/tasks/{id}`
    reports live status mapped to canonical A2A states, plus a new
    `POST .../a2a/tasks/{id}/cancel` (replaces the previous synchronous stub)
  - Public API: `TaskManager`, `TaskRecord`, `TaskStatus`, `TargetType`
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
