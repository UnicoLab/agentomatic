# Telemetry & Feedback

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Observability and Reinforcement Loop</h3>
</div>

---

Agentomatic provides built-in, production-grade **OpenTelemetry tracing**, **Prometheus metrics**, and **feedback collection** loops. This ensures you can monitor agent behaviors, track latency, audit costs, and capture user corrections to continuously improve your prompts.

---

## 📡 1. OpenTelemetry Auto-Instrumentation

When enabled, Agentomatic auto-instruments the execution stack. It tracks requests, trace spans, database queries, and downstream API calls (e.g. LLM invocations).

```python
platform = AgentPlatform.from_folder(
    "agents/",
    enable_telemetry=True,   # Enabled by default
)
app = platform.build()
```

### Trace Spans & Metadata

Agentomatic automatically generates spans for every phase of execution:

| Span Name | Attributes | Description |
|---|---|---|
| `agentomatic.agent.invoke` | `agent_name`, `user_id`, `thread_id`, `version` | Overall time spent inside the agent execution. |
| `agentomatic.llm.call` | `provider`, `model`, `prompt_tokens`, `completion_tokens` | Time spent executing LLM completions. |
| `agentomatic.rag.retrieve` | `query`, `doc_count`, `latency_ms` | Vector database document retrieval span. |
| `http.request` | `http.method`, `http.status_code`, `http.target` | Client HTTP requests and outgoing requests made via `httpx`. |

### Configuring Exporters

To route traces to APM backends (like **Jaeger**, **Datadog**, **Honeycomb**, or **Google Cloud Trace**), configure OTLP environment variables:

```bash
# Production Exporter Setup
export OTEL_SERVICE_NAME="agentomatic-platform"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger-collector:4317"
export OTEL_TRACES_SAMPLER="parentbased_traceidratio"
export OTEL_TRACES_SAMPLER_ARG="0.1"  # Sample 10% of traces in production
```

> 💡 *If no OTLP variables are defined, Agentomatic defaults to console logging, dumping trace summaries into standard output for development.*

### Adding Custom Spans in Agent Code

You can instrument custom processing nodes or helper functions in your agent folder (e.g., `nodes.py` or `tools.py`) using the `@traced` decorator:

```python
from agentomatic.observability.telemetry import traced

@traced("support_agent.query_kb")
async def query_knowledge_base(query: str) -> list[str]:
    # This automatically records a span named 'support_agent.query_kb'
    # and tracks any exceptions raised during execution.
    return await vector_store.similarity_search(query)
```

---

## 📊 2. Agent Metrics & Instrumentation

Agentomatic comes equipped with Prometheus instruments for tracking agent-specific metrics. If `prometheus_client` is not installed, it falls back to a graceful no-op dummy state.

### Core Metrics

The framework records the following counters, gauges, and histograms:

```python
from agentomatic.observability.metrics import (
    AGENT_INVOCATION_COUNT,  # Counter: ["agent_name", "status"]
    AGENT_DURATION,          # Histogram: ["agent_name"]
    LLM_DURATION,            # Histogram: ["provider", "model"]
    ERROR_COUNT,             # Counter: ["error_type", "agent_name"]
    ACTIVE_AGENTS,           # Gauge: Concurrent active agent runs
)
```

### Context-Based Tracking

Track agent run durations and statuses in your code using the async context manager `track_agent_invocation`:

```python
from agentomatic.observability.metrics import track_agent_invocation

async def run_analytics(agent_name: str):
    async with track_agent_invocation(agent_name):
        # ACTIVE_AGENTS increments
        await process_graph()
        # SUCCESS increments
    # DURATION is observed and ACTIVE_AGENTS decrements
```

### Concurrency Primitives & Circuit Breakers

Protect external APIs and rate-limited LLM providers using built-in circuit breakers and semaphores:

- **AgentSemaphore**: Limit concurrent executions per agent.
- **CircuitBreaker**: Automatically trip and fail fast when external services experience consecutive failures.

```python
from agentomatic.observability.concurrency import CircuitBreaker, AgentSemaphore

# 1. Initialize a circuit breaker
llm_breaker = CircuitBreaker(
    name="llm_provider",
    failure_threshold=5,   # Trip after 5 failures
    reset_timeout=60.0,    # Reset connection attempt after 60s
)

# 2. Wrap LLM calls
async with llm_breaker.call():
    response = await chat_llm.ainvoke(query)
```

---

## 💬 3. Feedback Collection Loop

Gather real-user feedback directly through your agent API. 

```python
platform = AgentPlatform.from_folder(
    folder_path="agents/",
    enable_feedback=True,  # Auto-registers feedback endpoints
    store=SQLAlchemyStore("sqlite+aiosqlite:///data.db")
)
```

When enabled, Agentomatic mounts these three endpoints for **every registered agent**:
1. `POST /api/v1/{agent}/feedback` — Submit a rating (1-5), comment, or correct answer.
2. `GET /api/v1/{agent}/feedback` — List historical feedback entries.
3. `GET /api/v1/{agent}/feedback/export` — Export logs as a JSONL dataset.

### Submitting Feedback (Client Example)

```python
import httpx

# Submit a thumbs-down correction to train the optimizer
async with httpx.AsyncClient() as client:
    await client.post(
        "http://localhost:8000/api/v1/support_agent/feedback",
        json={
            "query": "How do I reset my password?",
            "response": "Use the settings panel.",
            "correction": "Click 'Forgot Password' on the login screen.",
            "rating": 1,
            "feedback_type": "correction",
        }
    )
```

---

## ⚡ 4. The `/optimize/invoke` Endpoint

For automated evaluation and prompt tuning, standard response text is not enough. Evaluation metrics like `faithfulness` or `context_recall` require access to intermediate contexts and steps.

Agentomatic mounts a dedicated `POST /api/v1/{agent}/optimize/invoke` endpoint which returns **the full pipeline context**:

### Request Payload
```json
{
  "query": "What are the working hours?",
  "system_prompt_override": "You are a concise office assistant..."
}
```

### Response Payload
```json
{
  "response": "Working hours are 9:00 AM to 5:00 PM.",
  "retrieval_context": [
    "Office Policy: Standard business hours are 9 AM to 5 PM."
  ],
  "tool_calls": [
    {
      "name": "search_handbook",
      "args": {"query": "hours"},
      "result": "..."
    }
  ],
  "steps_taken": ["search_db", "format_reply"],
  "reasoning": "Retrieved working hours from office policy docs.",
  "citations": [{"source": "handbook_policy.pdf", "page": 4}],
  "duration_ms": 341.2,
  "metadata": {}
}
```

The `PromptOptimizer` automatically utilizes this endpoint when running optimization loops. It fetches intermediate retrieval context and tool logs, enabling deep evaluation scores.
