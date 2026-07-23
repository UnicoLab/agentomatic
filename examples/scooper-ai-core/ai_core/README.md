# `ai_core` — Scooper domain library

Shared Python package for the AI platform. It holds **what Agentomatic does not
own**: estimation math, artifact versioning, embeddings/vector adapters,
schemas, language helpers, audit/telemetry, and JSON repair for small LLMs.

LLM clients, prompt managers, Studio, and the REST surface come from
**Agentomatic** (`stacks/*.yaml`, per-agent `llm.py` + `prompts.json`). Do not
add a parallel LLM layer here.

| Need | Use | Avoid |
| --- | --- | --- |
| Invoke an LLM | Agentomatic `invoke_with_retry` on `self.llm` | Custom OpenAI clients in `ai_core` |
| Domain settings (artifacts, plugins, embed) | `ai_core.settings` | Duplicating stack LLM URL/model here |
| Combine effort plugins | `ai_core.estimation.combine_cascade` | Re-implementing cascade in an agent |
| Shared Postgres URL | `ai_core.db_connections` | Hardcoding DSN in agents |

## Module map

### Runtime & config

| Module | Role | Typical consumers |
| --- | --- | --- |
| `settings` | Non-LLM knobs (`AI_*` env) | Agents, plugins, ingestion, vectors |
| `db_connections` | Shared Postgres connection config | Agent `connections.py`, `main` |
| `util` | `now_iso`, clamps, `message_text` | All graph agents |
| `requestutil` | Alias `query` → `question` | All graph agents |
| `language` | Detect language + output directive | All graph agents |
| `jsonutil` | Extract/repair JSON from LLM text | Agents with structured output |
| `audit` | Structured JSONL audit sink | `main.py` |
| `telemetry` | Prometheus wrappers + mode middleware | `main.py`, assistant |
| `task_progress` | Bind TaskContext → `report_stage` | Pipelines (`historical_update`) |

### Domain models & estimation

| Module | Role | Typical consumers |
| --- | --- | --- |
| `schemas` | Pydantic contracts (features, cases, effort) | Agents, plugins, pipelines |
| `scope` | Fingerprint, normalize, heuristic scope | `scope_analysis`, `effort_*` |
| `estimation` | Plugin runner + cascade combiner | `effort_estimation` |
| `estimator_core/` | Deterministic math (multipliers, PERT, confidence) | Plugins, `estimation` |
| `featureutil` | Feature vectors for Bayesian / case rows | PyMC plugin, `historical_update` |

### Ingestion & retrieval

| Module | Role | Typical consumers |
| --- | --- | --- |
| `ingestion` | Markdown normalize / sections / chunks / quality | MarkItDown ingestor, `project_context` |
| `markitdown_formats` | Canonical extensions + MIME map (keep FE in sync) | Ingestor, frontend accept list |
| `cardtext` | Short retrieval text for embeddings | Seed / historical / similarity |
| `embeddings` | OpenAI-compatible encoder + hash fallback | Vector stores, similarity plugin |
| `similarity` | Cosine search + rerank helpers | Similarity plugin, Cosmos fallback |
| `vectorstore` | `.npz` store + provider registration | `connections/`, seed, plugins |
| `cosmos_vector` | Azure Cosmos Mongo vCore adapter | When `AI_VECTOR_PROVIDER=azure_cosmos` |
| `vector_sync` | Push cases into the active vector backend | Seed, historical update |
| `artifacts` | Versioned artifact bundles (blue/green) | Plugins, pipelines |

## Layout

```text
ai_core/
├── README.md                 # this file
├── __init__.py
├── settings.py
├── schemas.py
├── estimation.py
├── estimator_core/           # pure math (no I/O)
├── scope.py / featureutil.py / cardtext.py
├── ingestion.py / markitdown_formats.py
├── embeddings.py / similarity.py / vectorstore.py
├── cosmos_vector.py / vector_sync.py / artifacts.py
├── language.py / jsonutil.py / requestutil.py / util.py
├── db_connections.py
├── audit.py / telemetry.py / task_progress.py
```

## Design rules

1. **No Agentomatic duplicates** — stacks own LLM config; agents own prompts.
2. **No frontend contracts** — callers wrap schemas for the UI.
3. **Prefer pure helpers** — I/O (Cosmos, disk artifacts) stays in dedicated modules.
4. **Keep the surface small** — add a module only when two+ consumers need it.

## Docs

- Component handbook: [`../docs/ai-core/index.md`](../docs/ai-core/index.md)
- Product page (monorepo): [`../../docs/ai-platform/ai-core.md`](../../docs/ai-platform/ai-core.md)
- Tests: `tests/test_ai_core_*.py`, plus plugin/pipeline suites that import these helpers
