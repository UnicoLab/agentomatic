# Status Dashboard

Every Agentomatic platform exposes a **unified status page** that shows the
health of *everything* it manages — agents, ML plugins, custom endpoints,
ingestors, and pipelines — plus the task executor and storage backend, in one
place.

| Route | Purpose |
|-------|---------|
| `GET /status` | Self-contained, auto-refreshing HTML dashboard (no external assets) |
| `GET /api/v1/status` | The same data as structured JSON (for your own tooling) |

Both are mounted automatically — nothing to configure.

## The dashboard

Open [`http://localhost:8000/status`](http://localhost:8000/status) after
starting the platform. It polls the JSON endpoint every few seconds and renders:

- an **overall health** indicator + platform version and uptime,
- **summary cards** with healthy/total counts per resource type,
- a **task panel** (counts by status, currently running vs. max concurrency,
  and which target types are runnable), and
- a **storage panel** plus **per-resource tables** with each item's health and
  metadata.

## JSON shape

```jsonc
{
  "status": "healthy",
  "platform": { "name": "...", "version": "1.0.0", "uptime_seconds": 42.1, "maintenance_mode": false },
  "summary": {
    "agents":    { "total": 3, "healthy": 3 },
    "plugins":   { "total": 1, "healthy": 1 },
    "endpoints": { "total": 2, "healthy": 2 },
    "ingestors": { "total": 1, "healthy": 1 },
    "pipelines": { "total": 1, "healthy": 1 }
  },
  "resources": { "agents": { "total": 3, "healthy": 3, "degraded": 0, "items": { "...": { "status": "healthy" } } } },
  "tasks": {
    "enabled": true,
    "total": 12,
    "by_status": { "queued": 0, "running": 1, "succeeded": 10, "failed": 1, "cancelled": 0 },
    "running": 1,
    "max_concurrency": 8,
    "supported_targets": ["agent", "endpoint", "ingestion", "pipeline", "plugin"]
  },
  "storage": { "status": "healthy" },
  "generated_at": 1718900000.0
}
```

The top-level `status` is `degraded` if any resource is unhealthy or storage is
unhealthy, otherwise `healthy` — handy for a single uptime check.

## Related probes

- `GET /health` — aggregate health (agents, plugins, endpoints, ingestors,
  pipelines, storage) for load balancers.
- `GET /readiness` — Kubernetes-style readiness probe.
- `GET /api/v1/status` — the rich snapshot used by the dashboard.

## Programmatic use

Build the same payload yourself (e.g. to push into your own monitoring):

```python
from agentomatic.core.status import build_status_payload

snapshot = await build_status_payload(platform)
```
