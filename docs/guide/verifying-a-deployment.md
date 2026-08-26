# Verifying a Deployment

`scripts/e2e_verify.py` drives every surface the platform publishes against a
**running server** and reports pass/fail per group. It is deployment-agnostic:
point it at `agentomatic run`, at `uvicorn main:app`, or at a container built
by `agentomatic deploy`, and it adapts to what that deployment actually has
switched on.

Use it to answer the question a test suite cannot: *does this container, with
this configuration, behave correctly right now?*

```bash
uv run python scripts/e2e_verify.py \
  --base-url http://localhost:8000 \
  --agent my_agent --plugin my_plugin --pipeline my_pipeline \
  --endpoint my_endpoint --ingestor my_ingestor \
  --builder-smoke-name deployment_builder_check \
  --api-key "$AGENTOMATIC_API_KEY" \
  --control-token "$AGENTOMATIC_CONTROL_TOKEN" \
  --expect-auth \
  --json report.json
```

The exit code is `0` only when every check passes, so it drops straight into
CI or a post-deploy gate.

`--agent` is optional: when omitted, the verifier selects the first agent
published by `/api/v1/agents`. Pass it explicitly when a particular agent's
schema or runtime contract is the deployment gate.

## Full local production fixture (Docker + oMLX + Postgres)

The repository's default Compose fixture is intentionally more than a health
check. It mounts five discovered resource types (`agents`, `plugins`,
`endpoints`, `ingestion`, and `pipelines`), enables API-key auth and the
control plane, connects its `writer` agent to the host's OpenAI-compatible
oMLX service, and persists threads and invocation history in Postgres. Its
`root_echo_agent` additionally publishes `RootModel[str]`, so native scalar
agent input and the Studio `agent_input` envelope can be verified together.

```bash
env \
  DATABASE_URL='postgresql+asyncpg://agentomatic:agentomatic@db:5432/agentomatic' \
  AGENTOMATIC_LOGS_HISTORY=1 \
  AGENTOMATIC_AUDIT_LOG=/app/logs/audit.jsonl \
  OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  AGENTOMATIC_ENABLE_AUTH=1 \
  AGENTOMATIC_API_KEY=production-key \
  AGENTOMATIC_CONTROL_TOKEN=control-key \
  AGENTOMATIC_ENABLE_RATE_LIMIT=1 \
  AGENTOMATIC_RATE_LIMIT_REQUESTS=200 \
  docker compose --profile db up --build -d

uv run python scripts/e2e_verify.py \
  --base-url http://localhost:8010 \
  --agent researcher --plugin length_scorer --endpoint enricher \
  --read-endpoint lookup \
  --ingestor documents --pipeline production_flow \
  --builder-smoke-name e2e_builder_audit \
  --api-key production-key --control-token control-key --expect-auth
```

Run the root-schema proof against the same live container after the broad
matrix (the commands deliberately use different public entry points):

```bash
# Native generated agent route: the entire request body is a JSON string.
curl -sS -H 'X-Api-Key: production-key' -H 'Content-Type: application/json' \
  --data '"root deployment input"' \
  http://localhost:8010/api/v1/root_echo_agent/invoke

# The Studio Task Board uses this generic route; its input is also native JSON.
curl -sS -H 'X-Api-Key: production-key' -H 'Content-Type: application/json' \
  --data '{"target_type":"agent","target":"root_echo_agent","input":"root task input","wait":true}' \
  http://localhost:8010/api/v1/tasks

# Studio uses an object run envelope and forwards this value as state.__root__.
curl -sS -H 'X-Api-Key: production-key' -H 'Content-Type: application/json' \
  --data '{"agent_input":"root Studio input"}' \
  http://localhost:8010/studio/agents/root_echo_agent/runs
```

The deterministic resources prove routing, delegation, pipeline composition,
and control actions. `omlx_echo` is the separate live-model proof: invoke it
with a fixed prompt and verify that its response comes from your host oMLX
model. Restart the `platform` service before the final thread/log check to
prove the configured Postgres store, rather than the container layer, holds
the data.

For a deployment that retains audit files, mount the directory containing
`AGENTOMATIC_AUDIT_LOG` to durable storage or forward the structured records
to your central log/SIEM platform. The local `/app/logs` path in this fixture
is useful for verification but is not a substitute for a production log sink.

## What it checks

| Group | Covers |
|---|---|
| `platform` | `/health`, `/ready`, `/readiness`, `/status`, `/api/v1/status`, OpenAPI, Swagger, ReDoc, agent registry |
| `studio` | Every call the bundled Studio React client makes — info, agents, graph, schemas, config, runs, thread state/history, the SSE run stream, and the SPA bundle itself |
| `agent-rest` | `invoke`, `chat`, `invoke/stream` (SSE), `invoke/batch`, health, card, config, prompts, and the full thread lifecycle including fork, messages, summary, lineage, approvals and feedback |
| `a2a` | Agent-to-Agent task submit, poll and cancel |
| `plugins` | Registry, model card, health, `predict`, `predict/batch`, reload |
| `endpoints` | Registry, info, health, POST call, plus an optional browser-safe GET query contract (`--read-endpoint`) |
| `ingestion` | The always-available Studio `/api/v1/ingestors` registry, plus `/api/v1/ingestion` and per-ingestor routes when an ingestor is deployed |
| `pipelines` | Registry, config, validate, visualize, run, and `validate-draft` |
| `builder` | Opt-in Studio Builder lifecycle: save a visual endpoint→plugin field link, reload its persisted mapping, run it, then delete only the named disposable pipeline |
| `schema-contracts` | Discovers every deployed agent, plugin, endpoint, ingestor and pipeline, verifies its published input/output schema, and exercises the corresponding live operation using the OpenAPI contract |
| `pipelines-all` | Runs **every** published pipeline, not just the sampled one, and reports which of the nine step types actually executed |
| `isolation` | Fans out concurrent callers, each carrying a unique marker, and asserts no response or thread ever carries another caller's; it adapts its fan-out to an advertised rate-limit budget without weakening the concurrency proof |
| `tasks` | The task board, `invoke/async` submission, polling to a terminal state, and—with explicit `--plugin`, `--endpoint`, `--ingestor`, or `--pipeline` flags—the generic task dispatcher for each supplied resource type |
| `metrics` | Prometheus exposition and the presence of `agentomatic_*` series |
| `rate-limit` | That user routes *are* limited and probes and `/metrics` are *not* |
| `auth` | Anonymous and wrong-credential rejection, valid-credential acceptance, and that every probe path stays public |
| `errors` | Unknown agents, plugins, pipelines, endpoints and tasks return 4xx — never a 500 |
| `control-plane` | Every read route, plus disabling an agent, confirming it stops serving, re-enabling it, and toggling maintenance |

## Adapting to the deployment

The harness distinguishes *not configured* from *broken*, so a lean deployment
does not produce false failures:

- **No Studio** (`--profile minimal`): pass `--no-studio`.
- **No auth**: omit `--expect-auth`; the auth group is skipped.
- **No control plane / no metrics / no rate limiting**: detected from the
  response and reported as skipped.
- **No store**: thread and optimization-run routes answer `400`; the harness
  reports them skipped and names the variables that would enable them
  (`DATABASE_URL`, `AGENTOMATIC_LOGS_HISTORY`).

Rate limiting is handled rather than worked around: the harness is itself a
burst of traffic from one IP, so it honours `Retry-After` and retries — except
where a `429` is the property under test.

The dynamic `schema-contracts` group runs after the deliberate rate-limit
probe. This keeps its real-resource coverage from contaminating the concurrent
isolation test, while still proving that the schemas which drive Studio forms,
Builder mappings, and each deployed operation agree at runtime.

## Durability: does the data outlive the container?

Every check above runs against one live process, and a store that quietly fell
back to a file inside the container passes all of them — it writes, it reads
back, and only a restart tells the two apart. `durability_verify.py` splits the
proof across a restart so the difference shows:

```bash
python scripts/durability_verify.py write \
  --base-url http://localhost:8000 --api-key "$KEY" --agent my_chatbot \
  --studio-agent my_chatbot

# Replace the deployment: destroy the container and start a new one from the
# same image against the same database. A restart that keeps the writable
# layer proves nothing.
docker rm -f my-agent && docker run -d --name my-agent … my-image

python scripts/durability_verify.py verify \
  --base-url http://localhost:8000 --api-key "$KEY" --agent my_chatbot \
  --studio-agent my_chatbot
```

The `verify` phase reads the thread back, checks each message survived, and
appends one more to confirm the new process can *continue* the conversation
rather than merely read it. `--studio-agent` also writes a Studio execution
trace in the first phase, then proves its checkpoint and captured I/O state
survived and that replay uses the original persisted input after replacement.

!!! tip "Watch the boot log for a store you did not choose"
    Two things can silently redirect the store away from `DATABASE_URL`: a
    MEMORY-purpose connection (which outranks it, and says so with a warning
    naming both), and no configuration at all (which falls back to a local
    file). Both look identical until the container is replaced.

## What it does not cover

- **Model quality.** Agents are exercised for wiring, not for answer quality.
  An agent whose LLM is unreachable still passes if it degrades as designed;
  use `agentomatic optimize` and your own eval set for quality.
- **Your business logic.** The harness verifies the contract the platform
  publishes. Correctness of what your nodes compute is yours to test — see
  [Testing Your Agents](testing.md).
- **Load and soak behaviour.** It is a correctness check, not a benchmark.
