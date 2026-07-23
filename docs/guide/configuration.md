# Configuration Reference

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Platform, Agent, and Environment Configuration</h3>
</div>

---

Agentomatic supports a layered configuration system. Settings can be defined via environment variables, `.env` files, Python code, or a combination. The platform resolves values using a clear priority hierarchy.

---

## 🔀 Configuration Hierarchy

Values are resolved in the following order of precedence (highest wins):

```mermaid
flowchart LR
    A["Environment<br/>Variables"] --> B["Merged<br/>Configuration"]
    C[".env File"] --> B
    D["Python Code<br/>(constructor args)"] --> B
    E["Built-in<br/>Defaults"] --> B

    style A fill:#c8e6c9,stroke:#388e3c
    style C fill:#e1f5fe,stroke:#0288d1
    style D fill:#fff3e0,stroke:#f57c00
    style E fill:#f3e5f5,stroke:#7b1fa2
```

| Priority | Source | Example |
|----------|--------|---------|
| **1 (highest)** | Environment variables | `export AGENTOMATIC_LOG_LEVEL=DEBUG` |
| **2** | `.env` file in project root | `AGENTOMATIC_LOG_LEVEL=DEBUG` |
| **3** | Python constructor arguments | `AgentPlatform(log_level="DEBUG")` |
| **4 (lowest)** | Built-in defaults | `"INFO"` |

!!! tip "Production Best Practice"
    Use environment variables or `.env` files for deployment-specific values (secrets, URLs, ports). Use Python constructor arguments for structural decisions (middleware toggles, storage backend selection).

---

## ⚙️ Platform Constructor Reference

When initializing `AgentPlatform.from_folder()` (or the direct `AgentPlatform()` constructor), you can pass the following keyword arguments:

```python
from agentomatic import AgentPlatform
from agentomatic.storage import SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    # -- Metadata --
    title="My Custom Agent Platform",
    description="Enterprise Assistant APIs",
    version="1.0.0",
    # -- Networking --
    api_prefix="/api/v1",
    package_prefix="",
    cors_origins=["https://dashboard.mycompany.com"],
    # -- Observability --
    log_level="INFO",
    enable_logging=True,
    enable_metrics=True,
    enable_telemetry=True,
    enable_feedback=True,
    # -- Storage --
    store=SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/db"),
    # -- Security --
    enable_auth=True,
    auth_api_key="sk_live_51hG...",
    enable_rate_limit=True,
    rate_limit_requests=100,
    rate_limit_window=60,
    # -- Memory --
    max_history_messages=50,
    summarize_after=30,
    # -- Studio --
    enable_studio=False,
)
```

### Complete Parameter Table

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents_dir` | `str \| Path` | `"agents/"` | Filesystem path to scan for agent packages. |
| `title` | `str` | `"Agentomatic Platform"` | Display title shown in Swagger docs (`/docs`) and Redoc. |
| `description` | `str` | `"Multi-agent API platform..."` | Description displayed in Swagger UI. |
| `version` | `str` | `"1.0.0"` | Semantic version string shown in API docs and root. |
| `api_prefix` | `str` | `"/api/v1"` | Global URL prefix for all agent endpoints. |
| `package_prefix` | `str` | `""` | Python import prefix for agent modules. Auto-detected from `agents_dir` name if empty. |
| `cors_origins` | `list[str] \| None` | `["*"]` | Allowed CORS origins. Defaults to allowing all. |
| `log_level` | `str` | `"INFO"` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `settings` | `PlatformSettings \| None` | `None` | Optional pre-configured `PlatformSettings` object. |
| `store` | `BaseStore \| None` | `None` | Storage backend instance (e.g. `MemoryStore`, `SQLAlchemyStore`). |
| `enable_logging` | `bool` | `True` | Add structured request-logging middleware. |
| `enable_auth` | `bool` | `False` | Add API-key authentication middleware. |
| `auth_api_key` | `str` | `""` | API key token (required when `enable_auth=True`). |
| `enable_rate_limit` | `bool` | `False` | Add rate-limiting middleware. |
| `rate_limit_requests` | `int` | `100` | Max requests per window per client IP. |
| `rate_limit_window` | `int` | `60` | Sliding window duration in seconds. |
| `enable_metrics` | `bool` | `False` | Mount Prometheus metrics at `/metrics`. |
| `enable_feedback` | `bool` | `True` | Enable feedback collection endpoints per agent. |
| `enable_telemetry` | `bool` | `True` | Auto-configure OpenTelemetry tracing. |
| `enable_studio` | `bool` | `False` | Mount the Studio debug API and UI at `/studio/`. |
| `middleware` | `list[tuple] \| None` | `None` | Custom middleware list: `[(MiddlewareCls, {kwargs}), ...]`. |
| `max_history_messages` | `int` | `50` | Maximum messages loaded into agent context. |
| `summarize_after` | `int` | `30` | Message threshold before auto-summarization kicks in. |

---

## 🌐 Environment Variables Reference

All platform settings can be overridden using environment variables. There are two systems:

### Direct `AGENTOMATIC_` Variables

These map directly to `AgentPlatform` constructor parameters:

| Variable | Maps To | Default | Description |
|----------|---------|---------|-------------|
| `AGENTOMATIC_PORT` | `run(port=...)` | `8000` | Server bind port |
| `AGENTOMATIC_HOST` | `run(host=...)` | `0.0.0.0` | Server bind address |
| `AGENTOMATIC_API_PREFIX` | `api_prefix` | `/api/v1` | Global URL prefix |
| `AGENTOMATIC_LOG_LEVEL` | `log_level` | `INFO` | Log verbosity |
| `AGENTOMATIC_ENABLE_AUTH` | `enable_auth` | `false` | Toggle authentication |
| `AGENTOMATIC_AUTH_API_KEY` | `auth_api_key` | `""` | Secret API key |
| `AGENTOMATIC_ENABLE_RATE_LIMIT` | `enable_rate_limit` | `false` | Toggle rate limiting |
| `AGENTOMATIC_RATE_LIMIT_REQUESTS` | `rate_limit_requests` | `100` | Max requests/window |
| `AGENTOMATIC_RATE_LIMIT_WINDOW` | `rate_limit_window` | `60` | Window duration (seconds) |
| `AGENTOMATIC_ENABLE_METRICS` | `enable_metrics` | `false` | Toggle Prometheus metrics |
| `AGENTOMATIC_CORS_ORIGINS` | `cors_origins` | `*` | Comma-separated origins |
| `AGENTOMATIC_ARTIFACT_ROOT` | `artifact_root` | `.local/artifacts` | Versioned plugin/model artifact bundles |
| `AGENTOMATIC_RUNS_ROOT` | `runs_root` | `.local/runs` | Scratch directory for pipeline/task outputs |
| `AGENTOMATIC_AUDIT_LOG` | `audit_log` | `""` (disabled) | JSONL op-audit sink path (non-PII metadata only) |
| `AGENTOMATIC_CHUNK_SIZE_TOKENS` | `chunk_size_tokens` | `1200` | Default ingestion chunk size |
| `AGENTOMATIC_CHUNK_OVERLAP_TOKENS` | `chunk_overlap_tokens` | `150` | Default ingestion chunk overlap |
| `AGENTOMATIC_MIN_QUALITY_SCORE` | `min_quality_score` | `0.70` | Ingestion quality warning threshold |

### `PlatformSettings` Nested Variables

The `PlatformSettings` Pydantic model supports **nested environment variables** using double-underscore (`__`) as delimiter:

=== "LLM Settings"

    ```bash
    # LLM provider configuration
    export LLM__PROVIDER=openai
    export LLM__MODEL=gpt-4o
    export LLM__TEMPERATURE=0.1
    export LLM__MAX_TOKENS=4096
    export LLM__OPENAI_API_KEY=sk-...

    # Azure OpenAI
    export LLM__PROVIDER=azure
    export LLM__AZURE_API_KEY=your-key
    export LLM__AZURE_API_BASE=https://your-resource.openai.azure.com
    export LLM__AZURE_DEPLOYMENT_NAME=gpt-4o

    # Ollama (default)
    export LLM__PROVIDER=ollama
    export LLM__MODEL=mistral:7b
    export LLM__OLLAMA_BASE_URL=http://localhost:11434

    # Vertex AI
    export LLM__PROVIDER=vertex
    export LLM__VERTEX_PROJECT=my-gcp-project
    export LLM__VERTEX_LOCATION=us-central1
    ```

=== "Database Settings"

    ```bash
    # Database connection
    export DB__URL=postgresql+asyncpg://user:pass@localhost:5432/agent_db
    export DB__POOL_SIZE=10
    export DB__MAX_OVERFLOW=20
    export DB__POOL_TIMEOUT=30
    export DB__ECHO=false
    ```

=== "Embedding Settings"

    ```bash
    # Embedding provider
    export EMBEDDING__PROVIDER=ollama
    export EMBEDDING__MODEL=nomic-embed-text
    export EMBEDDING__DIMENSION=768
    ```

=== "Feature Flags"

    ```bash
    # Feature toggles
    export FEATURES__ENABLE_STREAMING=true
    export FEATURES__ENABLE_A2A=true
    export FEATURES__ENABLE_METRICS=true
    export FEATURES__ENABLE_RATE_LIMIT=false
    export FEATURES__ENABLE_AUTH=false
    export FEATURES__ENABLE_DB=false
    export FEATURES__ENABLE_FEEDBACK=true
    export FEATURES__MAX_CONCURRENT_AGENTS=10
    export FEATURES__REQUEST_TIMEOUT=30.0
    export FEATURES__LLM_RETRY_COUNT=3
    export FEATURES__LLM_RETRY_DELAY=1.0
    export FEATURES__CIRCUIT_BREAKER_THRESHOLD=5
    export FEATURES__CIRCUIT_BREAKER_TIMEOUT=60.0
    ```

=== "Auth & Rate Limit"

    ```bash
    # Authentication
    export AUTH__API_KEY=sk_prod_your_secret_key

    # Rate limiting
    export RATE_LIMIT__REQUESTS=300
    export RATE_LIMIT__WINDOW_SECONDS=60
    ```

---

## 📋 `PlatformSettings` Class Reference

The `PlatformSettings` class is a Pydantic `BaseSettings` model that aggregates all nested configuration sections:

| Section | Class | Prefix | Description |
|---------|-------|--------|-------------|
| Root | `PlatformSettings` | — | App name, environment, log level, API version |
| `llm` | `LLMSettings` | `LLM__` | LLM provider, model, API keys |
| `embedding` | `EmbeddingSettings` | `EMBEDDING__` | Embedding provider and model |
| `db` | `DatabaseSettings` | `DB__` | Database URL and connection pool |
| `features` | `FeatureSettings` | `FEATURES__` | Feature flags and limits |
| `auth` | `AuthSettings` | `AUTH__` | API key authentication |
| `rate_limit` | `RateLimitSettings` | `RATE_LIMIT__` | Rate limiting configuration |

### Using `PlatformSettings` Directly

```python
from agentomatic.config.settings import PlatformSettings, get_settings

# Auto-loads from env vars and .env
settings = get_settings()

print(settings.llm.provider)          # "ollama"
print(settings.llm.model)             # "mistral:7b"
print(settings.features.enable_auth)  # False
print(settings.db.url)                # "sqlite+aiosqlite:///data/platform.db"

# Pass to platform
platform = AgentPlatform.from_folder("agents/", settings=settings)
```

### `.env` File Example

Create a `.env` file in your project root:

```bash
# .env — loaded automatically by PlatformSettings
APP_NAME=My Production Platform
APP_ENV=production
LOG_LEVEL=WARNING

# LLM Configuration
LLM__PROVIDER=openai
LLM__MODEL=gpt-4o
LLM__OPENAI_API_KEY=sk-proj-...
LLM__TEMPERATURE=0.1
LLM__MAX_TOKENS=4096

# Database
DB__URL=postgresql+asyncpg://postgres:secret@db:5432/agentomatic
DB__POOL_SIZE=20
DB__MAX_OVERFLOW=40

# Security
AUTH__API_KEY=sk_prod_super_secret
FEATURES__ENABLE_AUTH=true
FEATURES__ENABLE_RATE_LIMIT=true
RATE_LIMIT__REQUESTS=200
RATE_LIMIT__WINDOW_SECONDS=60
```

---

## 🤖 Per-Agent Configuration

Each agent can define its own configuration via `config.py`. This is independent of the platform-level settings and allows each agent to have unique parameters.

### Defining Agent Config

```python
# agents/my_agent/config.py
from pydantic import BaseModel, Field

class MyAgentConfig(BaseModel):
    """Agent-specific configuration."""

    prompt_version: str = Field("v1", description="Active prompt version")
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1)
    llm_model: str = Field("ollama/mistral:7b")
    enable_memory: bool = Field(True, description="Use conversation memory")
    top_k_documents: int = Field(5, description="RAG retrieval count")
```

### Accessing Agent Config at Runtime

```python
# Inside your agent's nodes.py
from agentomatic import AgentRegistry

async def process(state: dict) -> dict:
    config = AgentRegistry().get("my_agent").config
    temperature = config.temperature
    model = config.llm_model
    # Use in LLM call...
    return {"response": "..."}
```

### Config API Endpoint

When a config is detected, it's automatically exposed:

```http
GET /api/v1/my_agent/config
```

```json
{
  "prompt_version": "v1",
  "temperature": 0.2,
  "max_tokens": 2048,
  "llm_model": "ollama/mistral:7b",
  "enable_memory": true,
  "top_k_documents": 5
}
```

---

## 🚀 `platform.run()` Configuration

The `run()` method accepts server-level parameters:

```python
platform = AgentPlatform.from_folder("agents/")
app = platform.build()

# Or run directly with uvicorn
platform.run(
    host="0.0.0.0",
    port=8000,
    reload=False,
    workers=4,
    # Extra kwargs passed to uvicorn.run()
    ssl_keyfile="/path/to/key.pem",
    ssl_certfile="/path/to/cert.pem",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"0.0.0.0"` | Bind address |
| `port` | `int` | `8000` | Bind port |
| `reload` | `bool` | `False` | Auto-reload on code changes (dev only) |
| `workers` | `int` | `1` | Number of uvicorn worker processes |
| `**kwargs` | `Any` | — | Extra arguments passed to `uvicorn.run()` |

---

## 📦 Docker / Production Example

=== "docker-compose.yml"

    ```yaml
    services:
      agentomatic:
        build: .
        ports:
          - "8000:8000"
        environment:
          - APP_ENV=production
          - LOG_LEVEL=WARNING
          - LLM__PROVIDER=openai
          - LLM__MODEL=gpt-4o
          - LLM__OPENAI_API_KEY=${OPENAI_API_KEY}
          - DB__URL=postgresql+asyncpg://postgres:secret@db:5432/agents
          - FEATURES__ENABLE_AUTH=true
          - AUTH__API_KEY=${API_KEY}
          - FEATURES__ENABLE_RATE_LIMIT=true
          - RATE_LIMIT__REQUESTS=200
        depends_on:
          - db

      db:
        image: postgres:16-alpine
        environment:
          POSTGRES_DB: agents
          POSTGRES_PASSWORD: secret
        volumes:
          - pgdata:/var/lib/postgresql/data

    volumes:
      pgdata:
    ```

=== "Kubernetes ConfigMap"

    ```yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: agentomatic-config
    data:
      APP_ENV: "production"
      LOG_LEVEL: "WARNING"
      LLM__PROVIDER: "vertex"
      LLM__VERTEX_PROJECT: "my-gcp-project"
      DB__URL: "postgresql+asyncpg://user:pass@pg-service:5432/agents"
      FEATURES__ENABLE_AUTH: "true"
      FEATURES__ENABLE_METRICS: "true"
    ```

---

## ❓ Troubleshooting

??? question "My environment variables aren't being loaded"
    Check these common causes:
    
    1. **`.env` file location**: Must be in the project root (same directory as `main.py`)
    2. **Variable prefix**: Platform variables must start with `AGENTOMATIC_` (e.g., `AGENTOMATIC_LOG_LEVEL`)
    3. **Priority**: Environment variables always override `.env` file values
    4. **Restart required**: `.env` changes require a server restart (no hot-reload)

??? question "CORS errors from my frontend"
    Pass your frontend's origin to `cors_origins`:
    
    ```python
    platform = AgentPlatform.from_folder(
        "agents/",
        cors_origins=["http://localhost:3000", "https://app.example.com"],
    )
    ```

---

## 📚 Related Documentation

| Topic | Link |
|-------|------|
| Middleware (auth, rate limiting) | [Middleware](middleware.md) |
| Storage backends | [Storage Backends](storage.md) |
| Configuration stacks | [Stacks](stacks.md) |
| Security & JWT | [Security](security.md) |
| LLM providers & failover | [LLM Providers](llm-providers.md) |
| CLI `run` command options | [CLI Reference](../cli/commands.md) |
