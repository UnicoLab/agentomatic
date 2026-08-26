# Observability & Monitoring

Agentomatic ships automatic logging, Prometheus metrics, OpenTelemetry
tracing, and a **ready-to-run** monitoring stack (Prometheus + OpenTelemetry
Collector + Grafana) with a pre-provisioned dashboard.

!!! tip "At-a-glance health"
    For a single, human-readable roll-up of every agent, plugin, pipeline,
    endpoint, ingestor, the storage backend, and the task engine, open the
    unified [`/status` dashboard](status.md) (or `GET /api/v1/status` for JSON).
    It complements the metrics/tracing below with an instant control-plane view.

## Enabling metrics & tracing

```python
from agentomatic import AgentPlatform

platform = AgentPlatform(
    title="My Agents",
    enable_metrics=True,   # exposes GET /metrics for Prometheus
    enable_telemetry=True, # emits OTLP spans
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
    Prometheus scrapes `host.docker.internal:8010/metrics` by default, matching
    the repository's packaged Docker platform. Adjust the target to `:8000`
    when Agentomatic itself runs directly on the host with that port, or to
    your service DNS name in a shared container network.

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

### Durable operation audit trail

When invocation history is enabled (`AGENTOMATIC_LOGS_HISTORY=1` with a
persistent store), every persisted agent, pipeline, plugin, endpoint, and
ingestion invocation also emits a safe audit event. It includes the generated
run correlation ID, resource, operation, outcome, and latency — never raw
request payloads or credentials. Conversation IDs are represented by a
one-way short `thread_ref`, not their original values.

Set `AGENTOMATIC_AUDIT_HASH_KEY` to a separately managed, high-entropy secret
when audit references must remain stable across API-key rotation. Otherwise
the configured API key is used as the HMAC key; with neither configured, the
reference is process-local only.

Configure a JSONL sink on a durable mounted volume or a log-forwarder-managed
path; do not point it at a container-only directory if the audit trail must
survive replacement:

```bash
AGENTOMATIC_AUDIT_LOG=/var/log/agentomatic/audit.jsonl
AGENTOMATIC_LOGS_HISTORY=1
DATABASE_URL=postgresql+asyncpg://user:password@db/agentomatic
```

The sink rotates at 20 MB and retains 90 days locally. Production deployments
should also collect stdout/audit records in their central logging or SIEM
system; filesystem retention is a safety layer, not a replacement for one.
