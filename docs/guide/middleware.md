# Middleware Stack

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Pipeline Security, Scaling, and Monitoring</h3>
</div>

---

Agentomatic wraps every incoming request and outgoing response in a pluggable middleware pipeline. You can configure and toggle authentication, rate limiting, logging, and metrics collecting globally on the `AgentPlatform` wrapper.

---

## 🔐 1. Authentication Middleware

Secure your API endpoints against unauthorized traffic. When enabled, Agentomatic inspects every request (except health checks, documentation, and the global card) for a valid API token.

```python
platform = AgentPlatform.from_folder(
    folder_path="agents/",
    enable_auth=True,
    auth_api_key="sk_live_51hG782k...",
)
```

### Authentication Channels
API clients can authenticate using one of the following two channels:

1. **HTTP Headers (Recommended)**: Pass the API key using the `X-API-Key` header.
   ```http
   X-API-Key: sk_live_51hG782k...
   ```
2. **Query Parameters**: Pass the API key as a query string parameter (useful for testing in browsers).
   ```http
   GET /api/v1/my_agent/invoke?api_key=sk_live_51hG782k...
   ```

### Error Handling
If the key is missing or invalid, the middleware intercepts the call and returns:
- **Status Code**: `401 Unauthorized`
- **Response Payload**:
  ```json
  {
    "detail": "Invalid or missing API Key"
  }
  ```

---

## ⚡ 2. Rate Limiting Middleware

Protect your backend and LLM models from burst traffic and denial of service (DoS) attacks. Agentomatic features an in-memory token-bucket style rate limiter.

```python
platform = AgentPlatform.from_folder(
    folder_path="agents/",
    enable_rate_limit=True,
    rate_limit_requests=60,   # Allowed requests
    rate_limit_window=60,     # Time window (seconds)
)
```

### Rate Limit Headers
Every response includes headers letting the client know their current rate status:

| Header | Description |
|---|---|
| `X-RateLimit-Limit` | The maximum number of requests allowed within the window. |
| `X-RateLimit-Remaining` | The remaining number of requests the client can make in the current window. |
| `Retry-After` | *(Included on 429 errors)* The number of seconds the client must wait before retrying. |

### Limit Exceeded
When a client exceeds their allowance, the request is blocked:
- **Status Code**: `429 Too Many Requests`
- **Response Payload**:
  ```json
  {
    "detail": "Rate limit exceeded. Try again in 12 seconds."
  }
  ```

---

## 📊 3. Prometheus Metrics Middleware

Expose real-time API telemetry to scrape with Prometheus and visualize in Grafana. Mounting this middleware automatically exposes a `/metrics` route at the root level of your platform application.

```python
platform = AgentPlatform.from_folder(
    folder_path="agents/",
    enable_metrics=True,
)
```

### Exposed HTTP Metrics
The middleware instrumentations record the following standard web application metrics:

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `agentomatic_requests_total` | Counter | `method`, `endpoint`, `status_code` | Total HTTP requests processed. |
| `agentomatic_request_duration_seconds` | Histogram | `method`, `endpoint` | Processing latency for requests. |
| `agentomatic_active_requests` | Gauge | None | Number of concurrent requests currently running. |

> 📊 *For custom agent metrics (LLM durations, database connections, agent logic timings), see the [Telemetry & Feedback Guide](telemetry.md).*

---

## 📝 4. Structured Logging Middleware

Monitors API lifecycle events, measures precise processing speeds, and isolates exceptions.

```python
platform = AgentPlatform.from_folder(
    folder_path="agents/",
    enable_logging=True,
    log_level="DEBUG",  # DEBUG | INFO | WARNING | ERROR
)
```

Agentomatic standardizes console outputs using the `loguru` framework, printing clean, structured, and color-coded message lines:

```
2026-06-13 22:16:10.123 | INFO     | agentomatic.middleware.logging:dispatch:18 - GET /api/v1/my_agent/health - 200 OK - 4.21ms
2026-06-13 22:16:15.541 | ERROR    | agentomatic.middleware.logging:dispatch:24 - POST /api/v1/qa_agent/invoke - 500 Internal Error - exception: ConnectionRefusedError
```

---

## 🔌 5. Injecting Custom Middlewares

You can append custom ASGI or FastAPI middlewares to the platform stack using the `middleware` parameter. Pass a list of tuples containing the middleware class and its configuration keyword arguments:

```python
from starlette.middleware.base import BaseHTTPMiddleware
from agentomatic import AgentPlatform

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, custom_header_value: str):
        super().__init__(app)
        self.value = custom_header_value

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Security-Level"] = self.value
        return response

# Register in the platform stack
platform = AgentPlatform.from_folder(
    folder_path="agents/",
    middleware=[
        (SecurityHeadersMiddleware, {"custom_header_value": "Maximum"})
    ],
)
```
