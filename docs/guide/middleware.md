# Middleware

Agentomatic provides a pluggable middleware pipeline. All middleware is toggleable.

## Built-in Middleware

### Authentication

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_auth=True,
    auth_api_key="your-secret-key",
)
```

Clients authenticate via:
- Header: `X-API-Key: your-secret-key`
- Query: `?api_key=your-secret-key`

### Rate Limiting

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_rate_limit=True,
    rate_limit_requests=100,  # per window
    rate_limit_window=60,     # seconds
)
```

Response headers include: `X-RateLimit-Limit`, `X-RateLimit-Remaining`.

### Prometheus Metrics

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_metrics=True,
)
```

Exposes `/metrics` endpoint with:
- `agentomatic_http_requests_total`
- `agentomatic_http_request_duration_seconds`
- `agentomatic_http_requests_active`

### Custom Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware

class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Custom"] = "value"
        return response

platform = AgentPlatform.from_folder(
    "agents/",
    middleware=[(MyMiddleware, {})],
)
```
