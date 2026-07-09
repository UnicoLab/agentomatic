# Observability & Monitoring

Agentomatic ships automatic logging, Prometheus metrics, OpenTelemetry
tracing, and a **ready-to-run** monitoring stack (Prometheus + OpenTelemetry
Collector + Grafana) with a pre-provisioned dashboard.

## Enabling metrics & tracing

```python
from agentomatic import AgentPlatform

platform = AgentPlatform(
    title="My Agents",
    enable_metrics=True,   # exposes GET /metrics for Prometheus
    enable_tracing=True,   # emits OTLP spans
)
app = platform.build()
```

Point tracing at any OTLP collector via the standard environment variable:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
uvicorn app:app --host 0.0.0.0 --port 8000
```

## The ready-made stack

A complete stack lives in `deploy/observability/`:

```text
deploy/observability/
├── docker-compose.yml            # Prometheus + OTel Collector + Grafana
├── prometheus/prometheus.yml     # Scrape config for your app
├── otel-collector/config.yaml    # OTLP receiver + pipelines
├── grafana/
│   ├── provisioning/             # Datasource + dashboard providers
│   └── dashboards/               # Agentomatic Overview dashboard
└── README.md
```

Bring it up:

```bash
cd deploy/observability
docker compose up -d
```

| Service    | URL                     | Credentials       |
| ---------- | ----------------------- | ----------------- |
| Grafana    | http://localhost:3000   | `admin` / `admin` |
| Prometheus | http://localhost:9090   | –                 |
| OTLP gRPC  | `localhost:4317`        | –                 |
| OTLP HTTP  | `localhost:4318`        | –                 |

Grafana automatically loads the **Agentomatic Overview** dashboard (folder
_Agentomatic_) with panels for request throughput/latency, agent invocations,
custom endpoint and upstream calls, connection acquisitions, and error rates.

!!! tip "Scrape target"
    Prometheus scrapes `host.docker.internal:8000/metrics` by default. Adjust
    the target in `prometheus/prometheus.yml` if your app runs elsewhere.

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

These cover the full request path — including [custom endpoints](endpoints.md),
their upstream model calls, and [per-agent connections](connections.md) — so
you get end-to-end visibility with zero extra code.

## Wiring a trace backend

The bundled collector logs spans via the `debug` exporter. To ship traces to
Tempo, Jaeger, or Honeycomb, add an exporter in `otel-collector/config.yaml`
and include it in the `traces` pipeline:

```yaml
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug, otlp/tempo]
```

## Logging

Structured logging (via `loguru`) is configured automatically. Every request,
agent invocation, endpoint call, and connection acquisition is logged with
contextual detail, so local development and production share the same
observable behaviour. See [Telemetry & Feedback](telemetry.md) for
request-level telemetry and user feedback capture.
