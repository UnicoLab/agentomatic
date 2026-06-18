# Middleware Stack

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Pipeline Security, Scaling, Monitoring, and Extensibility</h3>
</div>

---

Agentomatic wraps every incoming request and outgoing response in a pluggable middleware pipeline. You can configure and toggle authentication, rate limiting, logging, metrics collection, and feedback recording globally on the `AgentPlatform`.

---

## 🏗️ Middleware Architecture

Middleware components are added in **reverse order** — the last middleware added is the first to execute. The diagram below shows the actual execution flow for an incoming request:

```mermaid
flowchart LR
    Client["🌐 Client"] --> CORS["CORS"]
    CORS --> Logging["📝 Logging"]
    Logging --> Auth["🔐 Auth"]
    Auth --> RateLimit["⚡ Rate Limit"]
    RateLimit --> Metrics["📊 Metrics"]
    Metrics --> Custom["🔌 Custom"]
    Custom --> Handler["🤖 Agent<br/>Handler"]
    Handler --> Custom
    Custom --> Metrics
    Metrics --> RateLimit
    RateLimit --> Auth
    Auth --> Logging
    Logging --> CORS
    CORS --> Client

    style Auth fill:#ffcdd2
    style RateLimit fill:#fff9c4
    style Metrics fill:#c8e6c9
    style Logging fill:#e1f5fe
    style Custom fill:#f3e5f5
```

### Pipeline Registration Order

The platform adds middleware in this specific order during `build()`:

| Order | Middleware | Toggle | Always Active |
|-------|-----------|--------|---------------|
| 1 | **CORS** | — | ✅ Yes |
| 2 | **LoggingMiddleware** | `enable_logging=True` | Default on |
| 3 | **AuthMiddleware** | `enable_auth=True` | Default off |
| 4 | **RateLimitMiddleware** | `enable_rate_limit=True` | Default off |
| 5 | **MetricsMiddleware** | `enable_metrics=True` | Default off |
| 6 | **Custom middleware** | `middleware=[...]` | User-defined |
| 7 | **FeedbackCollector** | `enable_feedback=True` | Default on |
| 8 | **OpenTelemetry** | `enable_telemetry=True` | Default on |

!!! info "Starlette Ordering"
    FastAPI/Starlette adds middleware in **reverse order**: the last `app.add_middleware()` call wraps the outermost layer. This means CORS (added first) is the outermost wrapper, and custom middleware (added last) is closest to the handler.

### Full Stack Activation

```python
from agentomatic import AgentPlatform
from agentomatic.storage import SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    # Enable all built-in middleware
    enable_logging=True,
    enable_auth=True,
    auth_api_key="sk_live_51hG782k...",
    enable_rate_limit=True,
    rate_limit_requests=100,
    rate_limit_window=60,
    enable_metrics=True,
    enable_feedback=True,
    enable_telemetry=True,
    store=SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/db"),
)
```

---

## 🔐 1. Authentication Middleware

Secures your API endpoints against unauthorized traffic. Inspects every request (except health checks and documentation) for a valid API key.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | **(required)** | The expected secret key value |
| `header_name` | `str` | `"X-API-Key"` | HTTP header to check for the key |
| `query_param` | `str` | `"api_key"` | Query parameter fallback name |
| `skip_paths` | `set[str] \| None` | See below | Paths that bypass authentication |

**Default skip paths:** `/health`, `/healthz`, `/readiness`, `/docs`, `/openapi.json`, `/redoc`, `/`

### Enabling Authentication

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_auth=True,
    auth_api_key="sk_live_51hG782k...",
)
```

### Authentication Channels

Clients can authenticate using either of these methods:

=== "HTTP Header (Recommended)"

    ```bash
    curl -X POST http://localhost:8000/api/v1/my_agent/invoke \
      -H "Content-Type: application/json" \
      -H "X-API-Key: sk_live_51hG782k..." \
      -d '{"query": "Hello!"}'
    ```

=== "Query Parameter"

    ```bash
    # Useful for browser testing and WebSocket connections
    curl "http://localhost:8000/api/v1/my_agent/health?api_key=sk_live_51hG782k..."
    ```

### Error Responses

When the key is missing or invalid:

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{
  "detail": "Invalid or missing API key"
}
```

!!! warning "Security Best Practices"
    - Never commit API keys to version control
    - Use environment variables: `export AGENTOMATIC_AUTH_API_KEY=sk_prod_...`
    - Rotate keys regularly
    - Use different keys for development, staging, and production

---

## ⚡ 2. Rate Limiting Middleware

Protects your backend and LLM models from burst traffic and denial-of-service attacks. Implements an in-memory **sliding-window** rate limiter per client IP.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_requests` | `int` | `100` | Maximum requests allowed per window |
| `window_seconds` | `int` | `60` | Sliding window duration in seconds |

**Skip paths:** `/health`, `/healthz`, `/readiness`

### Client Identification

The middleware identifies clients using:

1. **`X-Forwarded-For` header** — first IP in the chain (for proxied requests)
2. **`request.client.host`** — direct connection IP (fallback)

### Enabling Rate Limiting

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_rate_limit=True,
    rate_limit_requests=60,    # 60 requests
    rate_limit_window=60,      # per 60-second window
)
```

### Rate Limit Response Headers

Every successful response includes rate status headers:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed within the window |
| `X-RateLimit-Remaining` | Remaining requests the client can make |
| `Retry-After` | *(On 429 only)* Seconds until the client can retry |

### Limit Exceeded Response

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 12
Content-Type: application/json

{
  "detail": "Rate limit exceeded",
  "retry_after": 12
}
```

### Client-Side Handling Example

```python
import httpx
import asyncio

async def call_with_retry(url: str, data: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))
            print(f"Rate limited. Retrying in {retry_after}s...")
            await asyncio.sleep(retry_after)
            response = await client.post(url, json=data)

        return response.json()
```

---

## 📊 3. Prometheus Metrics Middleware

Exposes real-time API telemetry for scraping with Prometheus and visualizing in Grafana. Automatically serves a `/metrics` endpoint.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str` | `"agentomatic"` | Prefix for all Prometheus metric names |

**Skip paths:** `/health`, `/healthz`, `/readiness`, `/metrics`

**Dependency:** Requires `prometheus-client` package. If not installed, the middleware degrades gracefully (no metrics recorded).

### Enabling Metrics

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_metrics=True,
)
```

### Exposed Prometheus Metrics

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `agentomatic_http_requests_total` | Counter | `method`, `path`, `status` | Total HTTP requests processed |
| `agentomatic_http_request_duration_seconds` | Histogram | `method`, `path` | Request processing latency |
| `agentomatic_http_requests_active` | Gauge | — | Number of concurrent in-flight requests |

### Histogram Buckets

The duration histogram uses these bucket boundaries (in seconds):

```
0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
```

### Cardinality Control

To prevent metric explosion, the middleware **normalizes high-cardinality paths** by collapsing UUIDs and hex IDs:

```
/api/v1/my_agent/threads/thread_a1b2c3d4e5f6/messages
→ /api/v1/my_agent/threads/{id}/messages
```

### Scraping with Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'agentomatic'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Sample `/metrics` Output

```text
# HELP agentomatic_http_requests_total Total HTTP requests
# TYPE agentomatic_http_requests_total counter
agentomatic_http_requests_total{method="POST",path="/api/v1/qa_agent/invoke",status="200"} 1847.0

# HELP agentomatic_http_request_duration_seconds HTTP request duration
# TYPE agentomatic_http_request_duration_seconds histogram
agentomatic_http_request_duration_seconds_bucket{le="0.1",method="POST",path="/api/v1/qa_agent/invoke"} 52.0
agentomatic_http_request_duration_seconds_bucket{le="0.5",method="POST",path="/api/v1/qa_agent/invoke"} 1803.0

# HELP agentomatic_http_requests_active Active HTTP requests
# TYPE agentomatic_http_requests_active gauge
agentomatic_http_requests_active 3.0
```

---

## 📝 4. Logging Middleware

Monitors API lifecycle events with structured, color-coded console output. Tracks request/response timing and assigns unique request IDs for traceability.

### Enabling Logging

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_logging=True,   # Enabled by default
    log_level="DEBUG",     # DEBUG | INFO | WARNING | ERROR
)
```

**Skip paths:** `/health`, `/healthz`, `/readiness`, `/metrics`, `/favicon.ico`

### Features

| Feature | Description |
|---------|-------------|
| **Request ID tracking** | Uses `X-Request-ID` header if provided, otherwise generates a unique 12-character hex ID |
| **Response timing** | Measures precise processing time in milliseconds |
| **Response headers** | Adds `X-Request-ID` and `X-Process-Time-Ms` to every response |
| **Structured format** | Uses `loguru` for clean, color-coded, structured output |

### Response Headers Added

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier (from client header or auto-generated) |
| `X-Process-Time-Ms` | Processing time in milliseconds |

### Console Output Example

```
2026-06-13 22:16:10.123 | INFO | → POST /api/v1/my_agent/invoke [a1b2c3d4e5f6]
2026-06-13 22:16:10.547 | INFO | ← POST /api/v1/my_agent/invoke → 200 (424.3ms) [a1b2c3d4e5f6]
2026-06-13 22:16:15.541 | INFO | → POST /api/v1/qa_agent/chat [f7e8d9c0b1a2]
2026-06-13 22:16:18.102 | INFO | ← POST /api/v1/qa_agent/chat → 200 (2561.0ms) [f7e8d9c0b1a2]
```

### Correlating Requests

Pass a custom request ID from the client for end-to-end traceability:

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/invoke \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-trace-12345" \
  -d '{"query": "Hello!"}'
```

---

## 👍 5. Feedback Middleware

Automatically records user feedback (thumbs up/down, ratings, corrections, comments) for every agent. Feedback data feeds into optimization datasets for prompt tuning.

### Components

The feedback system has two main components:

| Component | Purpose |
|-----------|---------|
| `FeedbackCollector` | Async collector with background storage and in-memory buffer |
| `@collect_feedback` | Decorator for auto-recording agent inputs/outputs |

### Enabling Feedback

```python
from agentomatic.storage import SQLAlchemyStore

platform = AgentPlatform.from_folder(
    "agents/",
    enable_feedback=True,    # Default: True
    store=SQLAlchemyStore("postgresql+asyncpg://..."),  # Persists feedback
)
```

### FeedbackRecord Fields

| Field | Type | Description |
|-------|------|-------------|
| `feedback_id` | `str` | Auto-generated unique ID |
| `agent_name` | `str` | Name of the agent |
| `user_id` | `str` | User who submitted the feedback |
| `thread_id` | `str \| None` | Associated conversation thread |
| `message_id` | `int \| None` | Specific message being rated |
| `query` | `str` | Original user query |
| `response` | `str` | Agent's response |
| `rating` | `int \| None` | 1 (thumbs down) to 5 (thumbs up) |
| `comment` | `str \| None` | Free-text comment |
| `correction` | `str \| None` | User-provided correct answer |
| `feedback_type` | `str` | Type: `thumbs`, `rating`, `correction`, `comment` |
| `timestamp` | `str` | ISO 8601 timestamp |
| `metadata` | `dict` | Arbitrary metadata |

### Using the Decorator

```python
from agentomatic.middleware.feedback import collect_feedback

@collect_feedback(store=True, log=True)
async def node_fn(state: dict) -> dict:
    """Every invocation is automatically recorded for feedback."""
    query = state.get("current_query", "")
    return {"response": f"Processed: {query}"}
```

### Programmatic Feedback Recording

```python
from agentomatic.middleware.feedback import get_collector

collector = get_collector()
record = await collector.record(
    agent_name="qa_agent",
    user_id="user_123",
    query="What is the return policy?",
    response="Our return policy allows...",
    rating=5,
    feedback_type="thumbs",
    thread_id="thread_abc123",
)
```

### Exporting Feedback as JSONL

Export feedback data for prompt optimization pipelines:

```python
collector = get_collector()
jsonl_data = await collector.export_jsonl(agent_name="qa_agent")

# Save to file
with open("feedback_export.jsonl", "w") as f:
    f.write(jsonl_data)
```

Output format (one JSON object per line):

```json
{"query": "What is the return policy?", "expected_answer": "Our return policy...", "metadata": {"rating": 5, "comment": null, "feedback_type": "thumbs"}}
```

---

## 🔌 Custom Middleware

You can inject custom ASGI or Starlette middlewares into the pipeline using the `middleware` parameter.

### Creating Custom Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    def __init__(self, app, *, csp_policy: str = "default-src 'self'"):
        super().__init__(app)
        self.csp_policy = csp_policy

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = self.csp_policy
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response
```

### Registering Custom Middleware

Pass a list of `(MiddlewareClass, kwargs)` tuples:

```python
platform = AgentPlatform.from_folder(
    "agents/",
    middleware=[
        (SecurityHeadersMiddleware, {"csp_policy": "default-src 'self'"}),
        (AnotherMiddleware, {"option": "value"}),
    ],
)
```

### Example: Request Timing Middleware

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware

class DetailedTimingMiddleware(BaseHTTPMiddleware):
    """Add detailed timing headers for performance monitoring."""

    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        response.headers["X-Response-Time"] = f"{duration:.4f}s"
        response.headers["Server-Timing"] = f"total;dur={duration * 1000:.1f}"
        return response
```

### Example: IP Allowlist Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class IPAllowlistMiddleware(BaseHTTPMiddleware):
    """Restrict access to specific IP addresses."""

    def __init__(self, app, *, allowed_ips: list[str]):
        super().__init__(app)
        self.allowed_ips = set(allowed_ips)

    async def dispatch(self, request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        if client_ip not in self.allowed_ips:
            return JSONResponse(
                {"detail": "Access denied"},
                status_code=403,
            )
        return await call_next(request)
```

---

## 📋 Middleware Best Practices

### Ordering Guidelines

!!! tip "Recommended Order"
    1. **Security middleware** (auth, IP allowlist) — reject bad requests early
    2. **Rate limiting** — protect against abuse before processing
    3. **Logging/tracing** — capture all requests including rejected ones
    4. **Metrics** — measure everything that passes security
    5. **Business logic middleware** — custom transformations

### Performance Considerations

| Practice | Rationale |
|----------|-----------|
| Keep `skip_paths` updated | Avoid unnecessary processing on health/probe endpoints |
| Use `time.perf_counter()` | More precise than `time.time()` for latency measurements |
| Avoid blocking I/O in middleware | All middleware is async — don't use synchronous database calls |
| Minimize response body inspection | Reading the body forces buffering and hurts streaming |

### Middleware Interaction with Streaming

!!! warning "SSE / Streaming Responses"
    The logging and metrics middleware measure time until the **first byte** is sent, not until the full streaming response completes. For streaming endpoints (`/chat/stream`), the `X-Process-Time-Ms` header reflects initial response time only.

### Disabling All Middleware

For maximum performance in trusted environments:

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_logging=False,
    enable_auth=False,
    enable_rate_limit=False,
    enable_metrics=False,
    enable_feedback=False,
    enable_telemetry=False,
)
```
