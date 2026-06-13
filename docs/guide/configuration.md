# Configuration Reference

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Platform and Settings Customization</h3>
</div>

---

Agentomatic offers global settings configuration at the `AgentPlatform` initialization level. Config settings can be defined in Python code or overridden using system environment variables.

---

## ⚙️ Platform Parameters Reference

When initializing `AgentPlatform.from_folder()` (or the direct `AgentPlatform()` constructor), you can pass the following keyword arguments:

```python
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore

platform = AgentPlatform.from_folder(
    agents_dir="agents/",
    title="My Custom Agent Platform",
    description="Enterprise Assistant APIs",
    version="1.0.0",
    api_prefix="/api/v1",
    cors_origins=["https://dashboard.mycompany.com"],
    log_level="INFO",
    store=MemoryStore(),
    enable_logging=True,
    enable_auth=True,
    auth_api_key="sk_live_51hG...",
    enable_rate_limit=True,
    rate_limit_requests=100,
    rate_limit_window=60,
    enable_metrics=True,
    enable_feedback=True,
    enable_telemetry=True,
)
```

### 1. Metadata Properties

- **`agents_dir`** (type: `str | Path`, default: `"agents/"`): Filesystem path to scan for discovered agent packages.
- **`title`** (type: `str`, default: `"Agentomatic Platform"`): Display title of the API platform, visible in OpenAPI Swagger documentation (`/docs`) and Redoc.
- **`description`** (type: `str`, default: `"Multi-agent API platform powered by Agentomatic"`): Detailed description displayed in Swagger UI.
- **`version`** (type: `str`, default: `"1.0.0"`): Semantic version string shown in API docs and root responses.

### 2. Networking and CORS Settings

- **`api_prefix`** (type: `str`, default: `"/api/v1"`): Global URL prefix prepended to all auto-generated and custom agent routers (e.g. `/api/v1/{agent_name}/invoke`).
- **`cors_origins`** (type: `list[str]`, default: `["*"]`): List of allowed origins for Cross-Origin Resource Sharing (CORS). Defaults to allowing all origins.

### 3. Observability and Debugging

- **`log_level`** (type: `str`, default: `"INFO"`): Verbosity level for console logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- **`enable_logging`** (type: `bool`, default: `True`): Toggles requests/responses lifecycle log printouts via structured `loguru` middleware.
- **`enable_metrics`** (type: `bool`, default: `False`): Mounts the Prometheus instrumentation exporter at the `/metrics` endpoint.
- **`enable_telemetry`** (type: `bool`, default: `True`): Configures OpenTelemetry spans tracking LLMs, RAG, and execution runtimes.
- **`enable_feedback`** (type: `bool`, default: `True`): Enables thumbs-up/correction endpoints per agent to build optimization datasets.

### 4. Storage Backend

- **`store`** (type: `BaseStore | None`, default: `None`): Instance of a storage adapter (e.g. `MemoryStore` or `SQLAlchemyStore`) used to save message threads and rating logs.

### 5. Security & Rate Limiting

- **`enable_auth`** (type: `bool`, default: `False`): If `True`, mounts Key Authentication middleware on all agent routes.
- **`auth_api_key`** (type: `str`, default: `""`): The secret API key token required by clients when `enable_auth` is enabled.
- **`enable_rate_limit`** (type: `bool`, default: `False`): Enables rate limit checks for client requests.
- **`rate_limit_requests`** (type: `int`, default: `100`): Maximum allowed queries a single IP/client can make within the rate limit window.
- **`rate_limit_window`** (type: `int`, default: `60`): Sliding window duration (in seconds) for rate limiting.

---

## 🌐 Environment Variables Override

All platform settings can be dynamically overridden using environment variables prefixed with `AGENTOMATIC_`. This is ideal for production deployments (e.g. Docker, Kubernetes, or serverless envs):

```bash
# Network Bind Configs
export AGENTOMATIC_PORT=9000
export AGENTOMATIC_HOST="127.0.0.1"

# Platform Overrides
export AGENTOMATIC_API_PREFIX="/api/v2"
export AGENTOMATIC_LOG_LEVEL="DEBUG"

# Security Configurations
export AGENTOMATIC_ENABLE_AUTH="true"
export AGENTOMATIC_AUTH_API_KEY="sk_prod_992kjh1..."

# Rate Limiting
export AGENTOMATIC_ENABLE_RATE_LIMIT="true"
export AGENTOMATIC_RATE_LIMIT_REQUESTS=300
```
