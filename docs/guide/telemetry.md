# Telemetry & Feedback

Agentomatic provides built-in **OpenTelemetry tracing**, **Prometheus metrics**, and **async feedback collection** — all auto-configured and zero-effort.

## OpenTelemetry Auto-Instrumentation

Tracing is auto-configured when you build the platform:

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_telemetry=True,   # default: True
)
app = platform.build()  # ← OTEL auto-configured here
```

### What Gets Traced

| Span | Attributes | Description |
|------|-----------|-------------|
| `agentomatic.agent.invoke` | `agent_name`, `duration_ms` | Every agent invocation |
| `agentomatic.llm.call` | `model`, `tokens` | LLM API calls |
| `agentomatic.rag.retrieve` | `doc_count`, `latency_ms` | RAG retrieval |
| `http.request` | `method`, `url`, `status_code` | Auto-instrumented httpx |

### Environment Variables

```bash
# Required for production
OTEL_SERVICE_NAME=my-platform
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317

# Optional
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer token
OTEL_TRACES_SAMPLER=parentbased_traceidratio
```

If no env vars are set, traces go to console (development mode).

### Custom Spans in Agent Code

```python
from agentomatic.observability.telemetry import traced

@traced("my_agent.retrieve_docs")
async def retrieve_docs(query: str) -> list[str]:
    # This creates a span with timing + error tracking
    docs = await vector_store.search(query)
    return docs

@traced("my_agent.generate_answer")
async def generate(query: str, context: list[str]) -> str:
    return await llm.generate(query, context=context)
```

### Graceful Degradation

Everything works without OpenTelemetry installed — decorators become pass-throughs, setup is a no-op:

```bash
# Full tracing
pip install agentomatic[telemetry]

# Without — everything still works, just no traces
pip install agentomatic
```

---

## Feedback Collection

Auto-collects user feedback on agent responses. Feedback is stored via your `BaseStore` backend and can be exported as JSONL for optimization.

### Auto-Enabled

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_feedback=True,   # default: True
    store=MemoryStore(),     # or SQLAlchemyStore
)
```

This auto-adds to **every agent**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/{agent}/feedback` | `POST` | Submit feedback (thumbs, rating, correction) |
| `/{agent}/feedback` | `GET` | List feedback entries |
| `/{agent}/feedback/export` | `GET` | Export as JSONL for optimization |

### Submit Feedback

```python
import httpx

# Thumbs up
await httpx.AsyncClient().post(
    "http://localhost:8000/api/v1/hr_bot/feedback",
    json={
        "query": "What is PTO?",
        "response": "PTO is paid time off.",
        "rating": 5,
        "feedback_type": "thumbs",
    }
)

# Correction (used as training signal)
await httpx.AsyncClient().post(
    "http://localhost:8000/api/v1/hr_bot/feedback",
    json={
        "query": "What is PTO?",
        "response": "PTO is paid time off.",
        "correction": "PTO = 25 days/year for full-time employees.",
        "rating": 2,
        "feedback_type": "correction",
    }
)
```

### Export for Optimization

```bash
# Get JSONL dataset from feedback
curl http://localhost:8000/api/v1/hr_bot/feedback/export
```

Returns:
```json
{
    "agent": "hr_bot",
    "format": "jsonl",
    "data": "{\"query\": \"What is PTO?\", \"expected_answer\": \"PTO = 25 days/year...\"}",
    "count": 42
}
```

### Decorator for Auto-Recording

```python
from agentomatic.middleware.feedback import collect_feedback

@collect_feedback(store=True, log=True)
async def my_agent(state):
    # Input/output auto-recorded for dataset building
    return {"response": "..."}
```

---

## Optimization Endpoint

The dedicated `POST /{agent}/optimize/invoke` endpoint returns **full pipeline context** — not just the response text:

```python
import httpx

resp = await httpx.AsyncClient().post(
    "http://localhost:8000/api/v1/hr_bot/optimize/invoke",
    json={
        "query": "What is the PTO policy?",
        "system_prompt_override": "You are a precise HR assistant...",
    }
)
data = resp.json()
```

Response includes:

```json
{
    "response": "The PTO policy grants 25 days per year...",
    "retrieval_context": ["Section 3.2: PTO Policy..."],
    "tool_calls": [{"name": "search_kb", "result": "..."}],
    "steps_taken": ["retrieve_docs", "generate_answer"],
    "reasoning": "I retrieved the PTO section from the handbook...",
    "citations": [{"source": "handbook.pdf", "page": 12}],
    "duration_ms": 1234.5
}
```

This gives the optimizer full visibility for DeepEval metrics like `faithfulness`, `contextual_relevancy`, and `contextual_precision`.

The `PromptOptimizer` automatically uses this endpoint when available, falling back to `/invoke`:

```python
from agentomatic.optimize import PromptOptimizer

optimizer = PromptOptimizer(
    agent="hr_bot",
    metrics=["faithfulness", "contextual_relevancy", "geval:Is the answer accurate?"],
)
# Automatically uses /optimize/invoke → full context → accurate metrics
result = await optimizer.optimize(dataset, max_iterations=10)
```
