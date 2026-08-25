# Verifying a Deployment

`scripts/e2e_verify.py` drives every surface the platform publishes against a
**running server** and reports pass/fail per group. It is deployment-agnostic:
point it at `agentomatic run`, at `uvicorn main:app`, or at a container built
by `agentomatic deploy`, and it adapts to what that deployment actually has
switched on.

Use it to answer the question a test suite cannot: *does this container, with
this configuration, behave correctly right now?*

```bash
python scripts/e2e_verify.py \
  --base-url http://localhost:8000 \
  --agent my_agent --plugin my_plugin --pipeline my_pipeline \
  --endpoint my_endpoint --ingestor my_ingestor \
  --api-key "$AGENTOMATIC_API_KEY" \
  --control-token "$AGENTOMATIC_CONTROL_TOKEN" \
  --expect-auth \
  --json report.json
```

The exit code is `0` only when every check passes, so it drops straight into
CI or a post-deploy gate.

## What it checks

| Group | Covers |
|---|---|
| `platform` | `/health`, `/ready`, `/readiness`, `/status`, `/api/v1/status`, OpenAPI, Swagger, ReDoc, agent registry |
| `studio` | Every call the bundled Studio React client makes — info, agents, graph, schemas, config, runs, thread state/history, the SSE run stream, and the SPA bundle itself |
| `agent-rest` | `invoke`, `chat`, `invoke/stream` (SSE), `invoke/batch`, health, card, config, prompts, and the full thread lifecycle including fork, messages, summary, lineage, approvals and feedback |
| `a2a` | Agent-to-Agent task submit, poll and cancel |
| `plugins` | Registry, model card, health, `predict`, `predict/batch`, reload |
| `endpoints` | Registry, info, health, call |
| `ingestion` | Both `/api/v1/ingestion` and the `/api/v1/ingestors` alias the Studio bundle uses, plus per-ingestor info and health |
| `pipelines` | Registry, config, validate, visualize, run, and `validate-draft` |
| `tasks` | The task board, `invoke/async` submission, and polling a task to a terminal state |
| `metrics` | Prometheus exposition and the presence of `agentomatic_*` series |
| `rate-limit` | That user routes *are* limited and probes and `/metrics` are *not* |
| `auth` | Anonymous and wrong-credential rejection, valid-credential acceptance, and that every probe path stays public |
| `errors` | Unknown agents, plugins, pipelines, endpoints and tasks return 4xx — never a 500 |
| `control-plane` | Every read route, plus disabling an agent, confirming it stops serving, re-enabling it, and toggling maintenance |

## Adapting to the deployment

The harness distinguishes *not configured* from *broken*, so a lean deployment
does not produce false failures:

- **No Studio** (`--profile minimal`): pass `--no-studio`.
- **No auth**: omit `--expect-auth`; the auth group is skipped.
- **No control plane / no metrics / no rate limiting**: detected from the
  response and reported as skipped.
- **No store**: thread and optimization-run routes answer `400`; the harness
  reports them skipped and names the variables that would enable them
  (`DATABASE_URL`, `AGENTOMATIC_LOGS_HISTORY`).

Rate limiting is handled rather than worked around: the harness is itself a
burst of traffic from one IP, so it honours `Retry-After` and retries — except
where a `429` is the property under test.

## What it does not cover

- **Model quality.** Agents are exercised for wiring, not for answer quality.
  An agent whose LLM is unreachable still passes if it degrades as designed;
  use `agentomatic optimize` and your own eval set for quality.
- **Your business logic.** The harness verifies the contract the platform
  publishes. Correctness of what your nodes compute is yours to test — see
  [Testing Your Agents](testing.md).
- **Load and soak behaviour.** It is a correctness check, not a benchmark.
