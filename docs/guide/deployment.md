# Production Deployment

This guide walks through deploying Agentomatic to production: **multiple
agents**, each with its **own inbound authentication**, its **own
authenticated databases and vector stores**, **custom APIs that call other
authenticated APIs**, plus **caching, observability, monitoring, and a control
plane** — all with as little glue code as possible.

It is written around a concrete target: **five independent agents** sharing one
deployment, where every agent authenticates callers, talks to different
backing services (each with its own credentials), and is fully observable.

!!! tip "Read these first"
    This guide ties together features documented in depth elsewhere:
    [Per-Agent Connections](connections.md), [Custom Endpoints](endpoints.md),
    [Security & Zero Trust](security.md), [Observability](observability.md), and
    the [Control Plane](control-plane.md). Here we focus on how they fit
    together for a real deployment.

## Architecture at a glance

```mermaid
flowchart TB
    Client[Clients / other services] -->|Bearer JWT| GW[Agentomatic app]

    subgraph GW[Agentomatic ASGI app]
        MW[Middleware chain:\nJWT → Zero-Trust → Rate-limit → Metrics → Maintenance]
        subgraph agents[Agents]
            A1[fraud_agent]
            A2[billing_agent]
            A3[rag_agent]
            A4[support_agent]
            A5[ops_agent]
        end
        EP[Custom endpoints\nmulti-model fan-out]
        CP[Control plane\n/api/v1/control]
    end

    A1 -->|scoped conn| DB1[(fraud DB)]
    A2 -->|scoped conn| DB2[(billing DB)]
    A3 -->|scoped conn| VS[(vector store)]
    A4 -->|scoped conn| MEM[(memory DB)]
    A3 -.->|cache| REDIS[(redis)]
    EP -->|OAuth2| ML1[model A]
    EP -->|OAuth2| ML2[model B]

    GW -->|/metrics| PROM[Prometheus]
    GW -->|OTLP traces| OTEL[OTel Collector]
    PROM --> GRAF[Grafana]
```

Everything above is configured declaratively — agents in folders, connections
in `connections.py`, endpoints in `endpoint.py`, and a handful of platform
flags. No custom FastAPI wiring is required.

## 1. Install for production

Install only the extras you use. Common production combinations:

```bash
# Core + Postgres (async) + JWT auth + metrics/tracing
pip install 'agentomatic[db,security,observability]'

# Add a vector store client for RAG agents (pick your provider)
pip install qdrant-client            # or chromadb, weaviate-client, pinecone-client

# Add redis for caching
pip install redis
```

| Extra | Brings in | Needed for |
| --- | --- | --- |
| `db` | SQLAlchemy async + drivers | Database connections, SQL memory store |
| `security` | PyJWT + JWKS | Inbound JWT/OAuth2, zero-trust |
| `observability` | prometheus-client, OpenTelemetry SDK | Metrics + tracing |
| `ui` | Chainlit | Optional chat debug UI |

!!! note "Async database drivers"
    Database URLs must use an **async** driver:
    `postgresql+asyncpg://…`, `mysql+aiomysql://…`, `sqlite+aiosqlite:///…`.

## 1b. Generate production containers (CLI)

Agentomatic is both a **dev platform** and a **production deploy target**.
Generate rootless / distroless images, compose, and a stack-derived env file::

```bash
agentomatic stack use remote
agentomatic deploy --stack remote --distroless --out deploy/generated
agentomatic stack export --env .env.production

# HTTPS + global auth at runtime
agentomatic run --stack remote \
  --ssl-certfile /certs/fullchain.pem \
  --ssl-keyfile /certs/privkey.pem \
  --require-auth-globally
```

### Deploy profiles: `--profile full|minimal`

`agentomatic deploy` picks how much of the platform the image runs. Both
profiles drive the **same** env-driven `main.py` (via `AGENTOMATIC_*` env vars
baked into the Dockerfile + compose) — there is one code path, not two images.

| Feature | `full` (default) | `minimal` |
| --- | :---: | :---: |
| Core agent REST API (`/api/v1/...`) | ✅ | ✅ |
| **Swagger / OpenAPI** (`/docs`, `/redoc`, `/openapi.json`) | ✅ | ✅ *(always on)* |
| Health / readiness (`/health`, `/readiness`) | ✅ | ✅ |
| Metrics / observability | ✅ | ✅ |
| Auth wiring (API-key / JWT / zero-trust) | ✅ | ✅ |
| Studio debug UI (`/studio/ui`) | ✅ | ❌ |
| Verbose/dev logging | ✅ (`INFO`) | ❌ (`WARNING`) |

```bash
# Production-lean image (Studio off, quieter logs; Swagger still served)
agentomatic deploy --profile minimal --stack remote --distroless
agentomatic deploy --minimal --stack remote          # shorthand
```

!!! warning "Swagger is always available"
    `--profile minimal` **never** disables `/docs`, `/redoc`, or
    `/openapi.json`. It only sets `AGENTOMATIC_ENABLE_STUDIO=0` and
    `AGENTOMATIC_LOG_LEVEL=WARNING`; the REST API, health, metrics, and auth
    stay enabled. The image still installs `agentomatic[all]`, so no required
    functionality is dropped — minimal just doesn't mount the Studio routes.

`deploy/` also includes a ready observability stack
(`deploy/observability/`) with Prometheus + OTel + Grafana.

!!! tip "Vendor backends stay yours"
    Register custom vector / embedding / store backends with
    `register_vector_provider`, `register_embedding_provider`, and
    `register_store_provider`. Agentomatic provides the ops; you bring the
    SDK (Cosmos, proprietary search, …).

## 2. Keep secrets out of code

Every config value that touches a credential supports `${ENV}` interpolation,
resolved at connection/endpoint build time — so your `connections.py`,
`endpoint.py`, and platform code stay secret-free and reviewable.

```python
DatabaseConnectionConfig(name="main", url="${FRAUD_DB_URL}")
```

Provide the values via your orchestrator (Kubernetes `Secret`, Docker
`--env-file`, Vault sidecar, etc.). A `.env` file is convenient for local and
staging; **never** commit real secrets.

```bash title=".env (staging example)"
# Inbound auth (JWKS from your IdP)
JWT_JWKS_URL=https://idp.example.com/.well-known/jwks.json
JWT_ISSUER=https://idp.example.com/
JWT_AUDIENCE=agentomatic

# Per-agent databases
FRAUD_DB_URL=postgresql+asyncpg://fraud:***@db-fraud:5432/fraud
BILLING_DB_URL=postgresql+asyncpg://billing:***@db-billing:5432/billing
SUPPORT_MEMORY_DB_URL=postgresql+asyncpg://support:***@db-support:5432/memory

# Vector store for the RAG agent
QDRANT_URL=https://vector.example.com:6333
QDRANT_API_KEY=***

# Cache
REDIS_URL=redis://cache:6379/0

# Outbound model APIs (OAuth2 client-credentials)
MODEL_A_URL=https://models.example.com/a
MODEL_B_URL=https://models.example.com/b
MODEL_TOKEN_URL=https://idp.example.com/oauth2/token
MODEL_CLIENT_ID=***
MODEL_CLIENT_SECRET=***

# Control plane
CONTROL_TOKEN=***
```

## 3. Inbound authentication (who may call an agent)

Agentomatic separates **authentication** (validating the caller's token) from
**authorization** (which agents/tools that caller may reach).

### JWT / OAuth2 validation

Enable JWT validation once at the platform level. Tokens are verified against
your IdP's JWKS; decoded claims land on `request.state.jwt_claims`.

```python
from agentomatic import AgentPlatform
from agentomatic.security import JWTConfig

platform = AgentPlatform.from_folder(
    "agents/",
    enable_jwt_auth=True,
    jwt_config=JWTConfig(
        enabled=True,
        jwks_url="${JWT_JWKS_URL}",   # resolved from env by your process
        issuer="${JWT_ISSUER}",
        audience="${JWT_AUDIENCE}",
        algorithms=["RS256"],
    ),
)
```

### Per-agent authorization (zero trust)

Turn on zero-trust and declare per-agent policies in each agent's manifest.
A caller must satisfy the policy (roles/scopes) for the specific agent — one
token does not grant access to all five agents.

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_jwt_auth=True,
    jwt_config=JWTConfig(enabled=True, jwks_url="${JWT_JWKS_URL}"),
    enable_zero_trust=True,
)
```

```yaml title="agents/billing_agent/manifest.yaml (excerpt)"
security:
  require_auth: true
  allowed_roles: ["billing", "admin"]
  allowed_scopes: ["billing:read", "billing:write"]
```

See [Security & Zero Trust](security.md) for tool-level policies and claim
mapping. Effective policies are visible at runtime via the
[control plane](control-plane.md) (`GET /api/v1/control/agents`).

## 4. Outbound connections (what an agent may reach)

Each agent declares its own authenticated resources in `connections.py`.
Connections are **scoped** to the agent, so credentials and pools never leak
across agents. Tag each with a `purpose` so features can find backends by
intent (`MEMORY`, `RAG`, `VECTOR`, `CACHE`, …).

=== "fraud_agent (SQL DB + scoring API)"
    ```python title="agents/fraud_agent/connections.py"
    from agentomatic.connections import (
        ConnectionPurpose,
        DatabaseConnectionConfig,
        HttpConnectionConfig,
    )
    from agentomatic.endpoints import AuthType, UpstreamAuthConfig

    CONNECTIONS = [
        DatabaseConnectionConfig(
            name="main",
            url="${FRAUD_DB_URL}",
            purpose=ConnectionPurpose.GENERAL,
            pool_size=10,
        ),
        HttpConnectionConfig(
            name="scoring_api",
            base_url="${FRAUD_SCORING_URL}",
            auth=UpstreamAuthConfig(
                type=AuthType.OAUTH2_CLIENT_CREDENTIALS,
                token_url="${MODEL_TOKEN_URL}",
                client_id="${MODEL_CLIENT_ID}",
                client_secret="${MODEL_CLIENT_SECRET}",
                scope="scoring:invoke",
            ),
        ),
    ]
    ```

=== "rag_agent (vector store + cache)"
    ```python title="agents/rag_agent/connections.py"
    from agentomatic.connections import (
        ConnectionPurpose,
        CustomConnectionConfig,
        VectorConnectionConfig,
    )

    CONNECTIONS = [
        VectorConnectionConfig(
            name="kb",
            provider="qdrant",
            url="${QDRANT_URL}",
            api_key="${QDRANT_API_KEY}",
            collection="knowledge_base",
            dimension=1536,
            purpose=ConnectionPurpose.RAG,
        ),
        # Zero-class cache: point at any factory, no wrapper needed
        CustomConnectionConfig(
            name="cache",
            factory="redis.asyncio.from_url",
            args=["${REDIS_URL}"],
            purpose=ConnectionPurpose.CACHE,
        ),
    ]
    ```

=== "support_agent (memory DB)"
    ```python title="agents/support_agent/connections.py"
    from agentomatic.connections import ConnectionPurpose, DatabaseConnectionConfig

    CONNECTIONS = [
        DatabaseConnectionConfig(
            name="memory",
            url="${SUPPORT_MEMORY_DB_URL}",
            purpose=ConnectionPurpose.MEMORY,
        ),
    ]
    ```

Use them at runtime with almost no code:

```python
from sqlalchemy import text
from agentomatic.connections import get_connections

conns = get_connections("fraud_agent")

# Authenticated SQL
async with conns.database("main").session() as session:
    total = (await session.execute(text("SELECT count(*) FROM cases"))).scalar_one()

# Authenticated HTTP (token acquired + cached automatically)
result = await conns.http("scoring_api").post("/score", payload={"amount": 42})

# Vector search for RAG
kb = get_connections("rag_agent").vector("kb")
hits = await kb.client.search(collection_name=kb.collection, query_vector=vec, limit=5)

# Cache (native redis client, initialised on demand)
redis = await get_connections("rag_agent").client("cache")
await redis.set("k", "v", ex=60)
```

!!! tip "Conversation memory on an agent's own database"
    Reuse an agent's authenticated DB as its memory store — the store shares
    the same engine/pool:

    ```python
    db = get_connections("support_agent").database("memory")
    store = await db.create_store()
    platform = AgentPlatform.from_folder("agents/", store=store)
    ```

See [Per-Agent Connections](connections.md) for vector providers, registering
custom types, and purpose-based lookups.

## 5. Custom APIs that call authenticated model APIs

When an agent needs context from one or more deployed models, expose a
**custom endpoint** that fans out over authenticated upstreams and aggregates
the results. The endpoint itself becomes a first-class route and is usable
from [pipelines](pipelines.md).

```python title="endpoints/ensemble/endpoint.py"
from agentomatic.endpoints import (
    AggregationStrategy,
    AuthType,
    BaseEndpoint,
    UpstreamAuthConfig,
    UpstreamConfig,
)


class EnsembleEndpoint(BaseEndpoint):
    name = "ensemble"
    description = "Fan out to two model APIs and aggregate."
    aggregation = AggregationStrategy.MAJORITY

    upstreams = [
        UpstreamConfig(
            name="model_a",
            base_url="${MODEL_A_URL}",
            auth=UpstreamAuthConfig(
                type=AuthType.OAUTH2_CLIENT_CREDENTIALS,
                token_url="${MODEL_TOKEN_URL}",
                client_id="${MODEL_CLIENT_ID}",
                client_secret="${MODEL_CLIENT_SECRET}",
            ),
        ),
        UpstreamConfig(
            name="model_b",
            base_url="${MODEL_B_URL}",
            auth=UpstreamAuthConfig(
                type=AuthType.OAUTH2_CLIENT_CREDENTIALS,
                token_url="${MODEL_TOKEN_URL}",
                client_id="${MODEL_CLIENT_ID}",
                client_secret="${MODEL_CLIENT_SECRET}",
            ),
        ),
    ]
```

Feed the aggregated result to an agent inside a pipeline:

```yaml title="pipelines/enrich.yaml"
name: enrich
steps:
  - endpoint: ensemble          # calls both models, aggregates
    input: {payload: "$.input"}
    output: context
  - agent: fraud_agent          # receives enriched context
    input: {query: "$.input.query", context: "$.context"}
```

See [Custom Endpoints](endpoints.md) for aggregation strategies, per-upstream
auth types, and schema control.

## 6. Wire the whole platform

Put it together in a single entrypoint. This is the complete production
configuration for the five-agent deployment:

```python title="main.py"
from agentomatic import AgentPlatform
from agentomatic.security import JWTConfig

platform = AgentPlatform.from_folder(
    "agents/",
    title="Acme Agents",
    version="1.0.0",
    # Discovery of custom endpoints (defaults to "endpoints/")
    endpoints_dir="endpoints/",
    # --- Inbound auth ---
    enable_jwt_auth=True,
    jwt_config=JWTConfig(
        enabled=True,
        jwks_url="${JWT_JWKS_URL}",
        issuer="${JWT_ISSUER}",
        audience="${JWT_AUDIENCE}",
    ),
    enable_zero_trust=True,
    # --- Hardening ---
    enable_rate_limit=True,
    rate_limit_requests=100,
    rate_limit_window=60,
    cors_origins=["https://app.example.com"],
    # --- Observability ---
    enable_metrics=True,       # /metrics (Prometheus)
    enable_telemetry=True,     # OpenTelemetry traces (OTLP)
    # --- Operations ---
    enable_control_plane=True,
    control_token="${CONTROL_TOKEN}",
)

app = platform.build()   # an ASGI app — serve with uvicorn/gunicorn
```

Run it:

=== "uvicorn (single process)"
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

=== "gunicorn (multiple workers)"
    ```bash
    gunicorn main:app \
      -k uvicorn.workers.UvicornWorker \
      -w 4 --bind 0.0.0.0:8000 \
      --timeout 120
    ```

=== "CLI"
    ```bash
    agentomatic run agents/ --host 0.0.0.0 --port 8000
    ```

!!! tip "Scaffolded `main.py` matches `agentomatic run`"
    Projects created with `agentomatic init --project` (and containers built by
    `agentomatic deploy`) ship a `main.py` whose module-level `app` is **feature-
    identical to `agentomatic run`**: it discovers every component directory and
    turns Studio, docs, health, and metrics on by default. Auth, control plane,
    and rate limiting are driven by `AGENTOMATIC_*` env vars
    (`AGENTOMATIC_ENABLE_AUTH`, `AGENTOMATIC_ENABLE_JWT`, `AGENTOMATIC_REQUIRE_AUTH`,
    `AGENTOMATIC_ENABLE_CONTROL_PLANE`, `AGENTOMATIC_ENABLE_RATE_LIMIT`,
    `AGENTOMATIC_TITLE`, `AGENTOMATIC_LOG_LEVEL`), so `uvicorn main:app` in the
    generated Dockerfile drops no functionality versus running the CLI.

!!! warning "Workers and in-memory state"
    Connection pools and per-process caches live **per worker**. Keep shared
    state (threads, memory, cache) in external services (Postgres, redis) so it
    is consistent across workers. Control-plane state (maintenance flag, agent
    drain) is per-process — front it with a shared store or drive it via your
    orchestrator if you need cross-worker consistency.

    **Background tasks** (async/batch/long-running jobs) are tracked in a
    `TaskStore`. The default is in-memory (per worker), so for `-w > 1` — or any
    deployment where task status must survive restarts — pass a durable
    [`SQLAlchemyTaskStore`](tasks.md#durable-sqlalchemy-store) so every worker
    reads/writes the same task board:

    ```python
    from agentomatic.tasks import SQLAlchemyTaskStore

    platform = AgentPlatform.from_folder(
        "agents/",
        task_store=SQLAlchemyTaskStore("${TASKS_DB_URL}"),  # e.g. postgresql+asyncpg://…
    )
    ```

## 7. Observability & monitoring

With `enable_metrics=True` the app exposes Prometheus metrics at `/metrics`,
and with `enable_telemetry=True` it emits OpenTelemetry traces (configure the
OTLP endpoint via standard `OTEL_*` env vars).

Key metrics emitted out of the box:

| Metric | What it tracks |
| --- | --- |
| `agentomatic_endpoint_calls_total` / `_duration_seconds` | Custom endpoint calls |
| `agentomatic_upstream_calls_total` / `_duration_seconds` | Per-upstream model calls |
| `agentomatic_connection_calls_total` | DB/vector/custom connection acquisitions |
| `agentomatic_registered_endpoints` | Number of registered endpoints |

A ready-to-run stack (Prometheus + OpenTelemetry Collector + Grafana with a
provisioned dashboard) lives in `deploy/observability/`:

```bash
cd deploy/observability
docker compose up -d
# Grafana → http://localhost:3000  (Agentomatic Overview dashboard)
```

Point the collector/Prometheus at your app's `/metrics` and OTLP endpoint (see
`deploy/observability/README.md`). Full details in the
[Observability guide](observability.md).

## 8. Operate at runtime (control plane)

The control plane (mounted at `{api_prefix}/control`) lets you inspect and
operate the platform without redeploying. Mutating calls require the
`X-Control-Token` header when `control_token` is set.

```bash
# Overview + per-agent health, effective auth policy, declared connections
curl -H "X-Control-Token: $CONTROL_TOKEN" https://api.example.com/api/v1/control
curl -H "X-Control-Token: $CONTROL_TOKEN" https://api.example.com/api/v1/control/agents

# Connection health per scope, and custom endpoints
curl https://api.example.com/api/v1/control/connections
curl https://api.example.com/api/v1/control/endpoints

# Drain a single agent (returns 503 for its routes), then re-enable
curl -X POST -H "X-Control-Token: $CONTROL_TOKEN" \
  https://api.example.com/api/v1/control/agents/fraud_agent/disable
curl -X POST -H "X-Control-Token: $CONTROL_TOKEN" \
  https://api.example.com/api/v1/control/agents/fraud_agent/enable

# Platform-wide maintenance mode
curl -X POST -H "X-Control-Token: $CONTROL_TOKEN" \
  -H "Content-Type: application/json" -d '{"enabled": true}' \
  https://api.example.com/api/v1/control/maintenance
```

All of this is also available visually in **Agentomatic Studio** (Control,
Endpoints, and Connections views) when it is enabled. See the
[Control Plane guide](control-plane.md).

## 9. Containerize

```dockerfile title="Dockerfile"
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml README.md ./
RUN pip install 'agentomatic[db,security,observability]' qdrant-client redis

# App code (agents/, endpoints/, pipelines/, main.py)
COPY . .

EXPOSE 8000
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", "--bind", "0.0.0.0:8000", "--timeout", "120"]
```

```yaml title="docker-compose.yml (app + backing services)"
services:
  agentomatic:
    build: .
    ports: ["8000:8000"]
    env_file: [.env]
    depends_on: [db-fraud, cache, vector]
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 15s
      timeout: 3s
      retries: 5

  db-fraud:
    image: postgres:16
    environment:
      POSTGRES_USER: fraud
      POSTGRES_PASSWORD: ${FRAUD_DB_PASSWORD}
      POSTGRES_DB: fraud
    volumes: ["fraud-data:/var/lib/postgresql/data"]

  cache:
    image: redis:7-alpine

  vector:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]

volumes:
  fraud-data:
```

## 10. Health checks & readiness

| Endpoint | Purpose | Auth |
| --- | --- | --- |
| `GET /health` | Liveness (always fast, unauthenticated) | none (skip-path) |
| `GET /api/v1/{agent}/health` | Per-agent liveness | per policy |
| `GET /api/v1/control/health` | Aggregate health (agents + connections) | control token |
| `GET /api/v1/status` | Whole-platform status JSON (all resources + task engine) | per policy |
| `GET /status` | Human-readable HTML status dashboard | per policy |

Use `/health` for Kubernetes liveness probes and
`/api/v1/control/health` for a deeper readiness gate. The unified
[`/status` dashboard](status.md) is the fastest way for operators to see the
health of every agent, plugin, pipeline, endpoint, ingestor, storage, and the
task engine at a glance.

```yaml title="Kubernetes probes (excerpt)"
livenessProbe:
  httpGet: { path: /health, port: 8000 }
  initialDelaySeconds: 10
readinessProbe:
  httpGet: { path: /health, port: 8000 }
  periodSeconds: 10
```

## Production checklist

- [ ] Installed only the extras you need (`db`, `security`, `observability`, provider clients).
- [ ] All credentials come from `${ENV}` — no secrets committed.
- [ ] `enable_jwt_auth=True` with a real `jwks_url`, `issuer`, and `audience`.
- [ ] `enable_zero_trust=True` and every agent has an explicit `security` policy.
- [ ] Rate limiting and a restrictive `cors_origins` list are set.
- [ ] Each agent declares its own scoped `connections.py`; pools sized for load.
- [ ] Shared state (threads/memory/cache) lives in external services, not process memory.
- [ ] For `workers > 1` or restart-durable tasks, a durable `task_store` (e.g. `SQLAlchemyTaskStore`) is configured.
- [ ] `/status` (and `/api/v1/status`) reviewed; all resources report healthy.
- [ ] `enable_metrics=True`, `/metrics` scraped, dashboard imported.
- [ ] `enable_telemetry=True` with the OTLP endpoint configured.
- [ ] `enable_control_plane=True` with a strong `control_token`.
- [ ] Liveness/readiness probes wired to `/health`.
- [ ] `uv run pytest`, `uv run ruff check`, and `mkdocs build --strict` pass in CI.
```
