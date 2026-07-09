# Agentomatic Observability Stack

A ready-to-run monitoring stack for Agentomatic: **Prometheus** (metrics),
**OpenTelemetry Collector** (traces), and **Grafana** (pre-provisioned
dashboards).

## Quick start

```bash
cd deploy/observability
docker compose up -d
```

| Service    | URL                     | Credentials     |
| ---------- | ----------------------- | --------------- |
| Grafana    | http://localhost:3000   | `admin` / `admin` |
| Prometheus | http://localhost:9090   | –               |
| OTLP gRPC  | `localhost:4317`        | –               |
| OTLP HTTP  | `localhost:4318`        | –               |

Grafana loads the **Agentomatic Overview** dashboard automatically (folder
_Agentomatic_).

## Point your app at the stack

Enable metrics and tracing on the platform and export OTLP to the collector:

```python
from agentomatic import AgentPlatform

platform = AgentPlatform(
    title="My Agents",
    enable_metrics=True,   # exposes GET /metrics for Prometheus
    enable_tracing=True,   # emits OTLP spans
)
app = platform.build()
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
uvicorn app:app --host 0.0.0.0 --port 8000
```

Prometheus scrapes `host.docker.internal:8000/metrics` by default (see
`prometheus/prometheus.yml`). Change the target if your app runs elsewhere.

## Metrics reference

| Metric | Type | Labels |
| ------ | ---- | ------ |
| `agentomatic_requests_total` | counter | `method`, `endpoint`, `status_code` |
| `agentomatic_request_duration_seconds` | histogram | `method`, `endpoint` |
| `agentomatic_agent_invocations_total` | counter | `agent_name`, `status` |
| `agentomatic_agent_duration_seconds` | histogram | `agent_name` |
| `agentomatic_errors_total` | counter | `error_type`, `agent_name` |
| `agentomatic_endpoint_calls_total` | counter | `endpoint`, `status` |
| `agentomatic_endpoint_duration_seconds` | histogram | `endpoint` |
| `agentomatic_upstream_calls_total` | counter | `status` |
| `agentomatic_upstream_duration_seconds` | histogram | – |
| `agentomatic_connection_calls_total` | counter | `connection`, `status` |
| `agentomatic_active_requests` | gauge | – |
| `agentomatic_active_agents` | gauge | – |
| `agentomatic_registered_agents` | gauge | – |
| `agentomatic_registered_endpoints` | gauge | – |

## Wiring a trace backend

The collector logs spans via the `debug` exporter out of the box. To ship to
Tempo/Jaeger/Honeycomb, add an exporter in `otel-collector/config.yaml` and
include it in the `traces` pipeline.

## Tear down

```bash
docker compose down
```
