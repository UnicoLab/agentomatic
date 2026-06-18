# API Reference

Complete reference for all HTTP endpoints exposed by the Agentomatic platform. All per-agent endpoints are prefixed with `/api/v1/{agent_name}/`.

---

## Platform Endpoints

These endpoints are global — not scoped to a specific agent.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Platform info and version |
| `GET` | `/health` | Aggregated health check |
| `GET` | `/readiness` | Kubernetes readiness probe |
| `GET` | `/docs` | OpenAPI interactive docs |
| `GET` | `/.well-known/agent.json` | A2A protocol discovery |
| `GET` | `/api/v1/agents` | List all registered agents |
| `GET` | `/api/v1/storage/stats` | Storage backend statistics |
| `GET` | `/metrics` | Prometheus metrics endpoint |

---

### `GET /`

Platform information.

**Response:**
```json
{
  "platform": "Agentomatic",
  "version": "1.0.0",
  "agents": 3
}
```

---

### `GET /health`

Aggregated health check across all agents.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "agents": {
    "my_agent": {"status": "healthy"},
    "rag_agent": {"status": "healthy"}
  }
}
```

---

### `GET /readiness`

Kubernetes readiness probe. Returns `200` when the platform is ready to accept traffic.

```bash
curl http://localhost:8000/readiness
```

---

### `GET /api/v1/agents`

List all registered agents with metadata.

```bash
curl http://localhost:8000/api/v1/agents
```

**Response:**
```json
{
  "agents": [
    {
      "name": "my_agent",
      "slug": "my-agent",
      "description": "A helpful assistant",
      "version": "1.0.0",
      "framework": "langgraph"
    }
  ]
}
```

---

## Per-Agent Endpoints

All prefixed with `/api/v1/{agent_name}/`.

### Execution

#### `POST /invoke`

Invoke agent synchronously. Returns the full response after execution completes.

**Request Body — `AgentInvokeRequest`:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | ✅ | — | User query or input |
| `user_id` | `string` | | `"default-user"` | User identifier |
| `context` | `object` | | `{}` | Additional context for the agent |
| `thread_id` | `string \| null` | | auto-generated | Thread ID for continuity |
| `prompt_version` | `string` | | `"v1"` | Prompt version to use |
| `temperature` | `float \| null` | | `null` | Temperature override (0.0–2.0) |
| `max_tokens` | `int \| null` | | `null` | Max tokens override |
| `metadata` | `object` | | `{}` | Extra metadata |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "user_id": "user-123",
    "thread_id": "thread-abc",
    "temperature": 0.7
  }'
```

**Response Body — `AgentInvokeResponse`:**

| Field | Type | Description |
|---|---|---|
| `response` | `string` | Agent response text |
| `agent_type` | `string` | Agent slug identifier |
| `thread_id` | `string \| null` | Thread ID used |
| `suggestions` | `string[]` | Follow-up suggestions |
| `citations` | `object[]` | Source citations |
| `steps_taken` | `string[]` | Processing steps |
| `context` | `any` | Context data (RAG docs, search results) |
| `metadata` | `object` | Response metadata incl. `prompt_version` |
| `duration_ms` | `float` | Processing time in milliseconds |

**Response Example:**
```json
{
  "response": "The capital of France is Paris.",
  "agent_type": "my-agent",
  "thread_id": "thread-abc",
  "suggestions": ["Tell me about Paris landmarks"],
  "citations": [],
  "steps_taken": ["retrieve", "generate"],
  "context": {},
  "metadata": {"prompt_version": "v1"},
  "duration_ms": 142.5
}
```

**Error Responses:**

| Status | Description |
|---|---|
| `404` | Agent not found |
| `202` | Execution suspended (HITL) — returns `approval_id` |
| `500` | Agent invocation failed |

---

#### `POST /invoke/stream`

Invoke agent with Server-Sent Events (SSE) streaming.

**Request Body:** Same as `POST /invoke` (`AgentInvokeRequest`).

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/invoke/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain quantum computing"}'
```

**SSE Response:**

```
data: {"response": "", "steps_taken": ["retrieve"]}

data: {"response": "Quantum computing uses...", "steps_taken": ["retrieve", "generate"]}

data: [DONE]
```

**Headers:**

| Header | Value |
|---|---|
| `Content-Type` | `text/event-stream` |
| `X-Agent` | Agent name |
| `Cache-Control` | `no-cache` |

---

#### `POST /chat`

Session-aware chat with automatic conversation memory. When a thread store is configured, this endpoint automatically loads history, invokes the agent, and persists messages.

**Request Body — `AgentChatRequest`:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `content` | `string` | ✅ | — | User message |
| `user_id` | `string` | | `"default-user"` | User identifier |
| `thread_id` | `string \| null` | | auto-generated | Thread ID |
| `context` | `object` | | `{}` | Context for agent code |
| `metadata` | `object` | | `{}` | Extra metadata |
| `messages` | `object[] \| null` | | `null` | Override: supply own message history |
| `include_history` | `boolean` | | `true` | Load history from store |
| `max_history` | `int \| null` | | `null` | Max messages to load |
| `persist` | `boolean` | | `true` | Auto-save messages to store |
| `prompt_version` | `string` | | `"v1"` | Prompt version |

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What is our PTO policy?",
    "thread_id": "thread_abc123",
    "user_id": "user_42",
    "context": {"department": "engineering"}
  }'
```

**Response:**

```json
{
  "response": "Based on our handbook, the PTO policy is...",
  "thread_id": "thread_abc123",
  "agent_type": "hr-agent",
  "suggestions": ["Ask about sick days"],
  "citations": [],
  "steps_taken": ["search_policy", "generate"],
  "context": {},
  "duration_ms": 234.5,
  "metadata": {"prompt_version": "v1"},
  "history_loaded": 12
}
```

---

### Inspection

#### `GET /health`

Per-agent health check.

```bash
curl http://localhost:8000/api/v1/my_agent/health
```

```json
{
  "status": "healthy",
  "agent": "my_agent",
  "framework": "langgraph"
}
```

---

#### `GET /config`

Get agent configuration.

```bash
curl http://localhost:8000/api/v1/my_agent/config
```

```json
{
  "agent": "my_agent",
  "config": {
    "model": "gpt-4o",
    "temperature": 0.7,
    "prompt_version": "v1"
  }
}
```

---

#### `GET /prompts`

List available prompt versions and the active version.

```bash
curl http://localhost:8000/api/v1/my_agent/prompts
```

```json
{
  "agent": "my_agent",
  "versions": ["v1", "v2"],
  "active": "v1"
}
```

---

### A2A Protocol

#### `GET /card`

A2A Agent Card — machine-readable agent description for agent-to-agent discovery.

```bash
curl http://localhost:8000/api/v1/my_agent/card
```

```json
{
  "name": "my-agent",
  "description": "A helpful assistant",
  "version": "1.0.0",
  "framework": "langgraph",
  "capabilities": {
    "streaming": true,
    "chat": true,
    "invoke": true,
    "a2a": true
  },
  "endpoints": {
    "invoke": "/api/v1/my_agent/invoke",
    "chat": "/api/v1/my_agent/chat",
    "stream": "/api/v1/my_agent/invoke/stream",
    "health": "/api/v1/my_agent/health"
  },
  "metadata": {}
}
```

---

#### `POST /a2a/tasks`

Submit an A2A protocol task.

**Request Body — `A2ATaskRequest`:**

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | `object` | ✅ | A2A message with `content` field |
| `metadata` | `object` | | Extra metadata |

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/a2a/tasks \
  -H "Content-Type: application/json" \
  -d '{"message": {"content": "Analyze this dataset"}}'
```

```json
{
  "task_id": "task_a1b2c3d4e5f6",
  "status": "completed",
  "result": "Analysis complete..."
}
```

---

#### `GET /a2a/tasks/{task_id}`

Get A2A task status.

```bash
curl http://localhost:8000/api/v1/my_agent/a2a/tasks/task_a1b2c3d4e5f6
```

```json
{
  "task_id": "task_a1b2c3d4e5f6",
  "status": "completed",
  "message": "Task tracking requires storage backend"
}
```

---

### Thread Management

#### `POST /threads`

Create a new conversation thread.

**Request Body — `CreateThreadRequest`:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `thread_id` | `string \| null` | | auto-generated | Custom thread ID |
| `user_id` | `string` | | `"default-user"` | User identifier |
| `title` | `string \| null` | | `null` | Thread title |
| `metadata` | `object` | | `{}` | Extra metadata |

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/threads \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_42", "title": "Onboarding questions"}'
```

---

#### `GET /threads`

List threads, optionally filtered by user.

```bash
# All threads for this agent
curl http://localhost:8000/api/v1/my_agent/threads

# Filter by user
curl http://localhost:8000/api/v1/my_agent/threads?user_id=user_42
```

```json
{
  "threads": [
    {
      "id": "thread_abc123",
      "user_id": "user_42",
      "agent_name": "my_agent",
      "title": "Onboarding questions",
      "created_at": "2026-06-18T20:00:00Z"
    }
  ],
  "count": 1
}
```

---

#### `GET /threads/{thread_id}`

Get a specific thread.

```bash
curl http://localhost:8000/api/v1/my_agent/threads/thread_abc123
```

---

#### `PATCH /threads/{thread_id}`

Update thread title or metadata.

**Request Body — `UpdateThreadRequest`:**

| Field | Type | Description |
|---|---|---|
| `title` | `string \| null` | New title |
| `metadata` | `object \| null` | New metadata (replaces existing) |

```bash
curl -X PATCH http://localhost:8000/api/v1/my_agent/threads/thread_abc123 \
  -H "Content-Type: application/json" \
  -d '{"title": "Updated title"}'
```

---

#### `DELETE /threads/{thread_id}`

Delete a thread and all its messages.

```bash
curl -X DELETE http://localhost:8000/api/v1/my_agent/threads/thread_abc123
```

```json
{"status": "deleted", "thread_id": "thread_abc123"}
```

---

#### `GET /threads/{thread_id}/messages`

Get messages with pagination.

| Query Param | Type | Default | Description |
|---|---|---|---|
| `limit` | `int` | `100` | Max messages to return |
| `offset` | `int` | `0` | Offset for pagination |

```bash
curl "http://localhost:8000/api/v1/my_agent/threads/thread_abc/messages?limit=20&offset=0"
```

```json
{
  "thread_id": "thread_abc",
  "messages": [
    {"role": "user", "content": "Hello", "timestamp": "..."},
    {"role": "assistant", "content": "Hi there!", "timestamp": "..."}
  ],
  "count": 2,
  "limit": 20,
  "offset": 0
}
```

---

#### `DELETE /threads/{thread_id}/messages`

Clear all messages in a thread (keeps the thread itself).

```bash
curl -X DELETE http://localhost:8000/api/v1/my_agent/threads/thread_abc/messages
```

```json
{"status": "cleared", "thread_id": "thread_abc"}
```

---

#### `GET /threads/{thread_id}/summary`

Get or generate a conversation summary.

```bash
curl http://localhost:8000/api/v1/my_agent/threads/thread_abc/summary
```

```json
{
  "thread_id": "thread_abc",
  "summary": "User asked about PTO policy and sick days. Agent provided handbook references."
}
```

---

### Thread Forking & Lineage

#### `POST /threads/{thread_id}/fork`

Fork a thread at a specific message index.

**Request Body — `ForkThreadRequest`:**

| Field | Type | Required | Description |
|---|---|---|---|
| `message_index` | `int` | ✅ | Fork point (0-indexed) |
| `new_thread_id` | `string \| null` | | Custom ID for fork |
| `title` | `string \| null` | | Title for forked thread |

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/threads/thread_abc/fork \
  -H "Content-Type: application/json" \
  -d '{"message_index": 2, "title": "A/B Test Variant"}'
```

```json
{
  "id": "thread_xyz789",
  "parent_thread_id": "thread_abc",
  "fork_message_index": 2,
  "title": "A/B Test Variant"
}
```

---

#### `GET /threads/{thread_id}/lineage`

Get the full ancestry tree (ancestors and descendants).

```bash
curl http://localhost:8000/api/v1/my_agent/threads/thread_xyz789/lineage
```

```json
{
  "thread_id": "thread_xyz789",
  "ancestors": [{"id": "thread_abc", "title": "Original"}],
  "descendants": []
}
```

---

### Human-in-the-Loop (HITL)

#### `GET /threads/{thread_id}/pending`

List pending HITL approvals for a thread.

```bash
curl http://localhost:8000/api/v1/my_agent/threads/thread_001/pending
```

```json
{
  "thread_id": "thread_001",
  "pending": [
    {
      "id": "tx_984712",
      "node_name": "financial_transfer",
      "state_snapshot": {"amount": 1000, "recipient": "..."},
      "created_at": "2026-06-18T20:00:00Z"
    }
  ],
  "count": 1
}
```

---

#### `POST /threads/{thread_id}/approve`

Approve a suspended state and resume execution.

**Request Body — `ApproveSuspendedRequest`:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `approval_id` | `string` | ✅ | — | ID of the suspended state |
| `approved` | `boolean` | | `true` | Approval flag |
| `context` | `object` | | `{}` | Additional context to merge |

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/threads/thread_001/approve \
  -H "Content-Type: application/json" \
  -d '{
    "approval_id": "tx_984712",
    "context": {"approved_limit": 500}
  }'
```

**Behavior:**

1. Retrieves the suspended state snapshot
2. Deletes the pending record
3. Merges `context` into `state.metadata`
4. Sets `state.metadata.hitl_approved = true`
5. Resumes graph execution with the updated state
6. Returns the final `AgentInvokeResponse`

!!! note "Re-suspension"
    If the resumed execution hits another HITL node, a new `202 Accepted` response is returned with a new `approval_id`.

---

#### `POST /threads/{thread_id}/reject`

Reject a suspended state and discard the execution context.

**Request Body — `RejectSuspendedRequest`:**

| Field | Type | Required | Description |
|---|---|---|---|
| `approval_id` | `string` | ✅ | ID of the suspended state |
| `reason` | `string \| null` | | Rejection reason |

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/threads/thread_001/reject \
  -H "Content-Type: application/json" \
  -d '{"approval_id": "tx_984712", "reason": "Amount exceeds policy"}'
```

```json
{
  "status": "rejected",
  "approval_id": "tx_984712",
  "reason": "Amount exceeds policy"
}
```

---

### Feedback

#### `POST /feedback`

Submit user feedback on an agent response.

**Request Body — `FeedbackRequest`:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `user_id` | `string` | | `"anonymous"` | User identifier |
| `rating` | `int \| null` | | `null` | 1–5 star rating |
| `comment` | `string \| null` | | `null` | Text comment |
| `correction` | `string \| null` | | `null` | Correct answer |
| `feedback_type` | `string` | | `"thumbs"` | `thumbs \| rating \| correction \| comment` |
| `query` | `string` | | `""` | Original query |
| `response` | `string` | | `""` | Agent response being rated |
| `thread_id` | `string \| null` | | `null` | Thread ID |
| `metadata` | `object` | | `{}` | Extra metadata |

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 5,
    "comment": "Very helpful!",
    "query": "What is PTO policy?",
    "response": "Our PTO policy states...",
    "thread_id": "thread_abc"
  }'
```

```json
{"status": "recorded", "feedback_id": "fb_abc123"}
```

---

#### `GET /feedback`

List feedback for this agent.

| Query Param | Type | Default | Description |
|---|---|---|---|
| `limit` | `int` | `50` | Max records to return |

```bash
curl "http://localhost:8000/api/v1/my_agent/feedback?limit=20"
```

---

#### `GET /feedback/export`

Export feedback as JSONL for optimization datasets (DeepEval, fine-tuning).

```bash
curl http://localhost:8000/api/v1/my_agent/feedback/export
```

```json
{
  "agent": "my_agent",
  "format": "jsonl",
  "data": "{\"query\":\"...\",\"response\":\"...\",\"rating\":5}\n{...}",
  "count": 42
}
```

---

### Optimization

#### `POST /optimize/invoke`

Full-context invocation for optimization pipelines. Returns retrieval context, tool calls, reasoning steps — everything needed for DeepEval metrics.

**Request Body — `OptimizeInvokeRequest`:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | ✅ | — | User query |
| `system_prompt_override` | `string \| null` | | `null` | System prompt to inject |
| `user_id` | `string` | | `"optimizer"` | User ID |
| `context` | `object` | | `{}` | Additional context |
| `include_retrieval_context` | `boolean` | | `true` | Return RAG context |
| `include_steps` | `boolean` | | `true` | Return execution steps |

```bash
curl -X POST http://localhost:8000/api/v1/my_agent/optimize/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum computing?"}'
```

**Response Body — `OptimizeInvokeResponse`:**

```json
{
  "response": "Quantum computing is...",
  "retrieval_context": ["Document 1 content...", "Document 2 content..."],
  "tool_calls": [{"name": "search", "args": {"query": "quantum"}}],
  "steps_taken": ["retrieve", "rerank", "generate"],
  "reasoning": "First I searched for...",
  "citations": [{"source": "doc1.pdf", "page": 3}],
  "duration_ms": 456.7,
  "metadata": {}
}
```

---

## Studio Debug API

All Studio endpoints are prefixed with `/studio/`.

### Discovery

#### `GET /studio/info`

Platform-level metadata and capabilities.

```bash
curl http://localhost:8000/studio/info
```

```json
{
  "version": "1.0.0",
  "platform_title": "Agentomatic Platform",
  "agent_count": 3,
  "capabilities": ["studio", "storage", "streaming"]
}
```

---

#### `GET /studio/agents`

List all agents with debugging capabilities.

```bash
curl http://localhost:8000/studio/agents
```

```json
[
  {
    "name": "my_agent",
    "slug": "my-agent",
    "description": "A helpful assistant",
    "version": "1.0.0",
    "framework": "langgraph",
    "capabilities": ["graph", "streaming", "checkpoints", "state", "breakpoints", "hitl"],
    "has_graph": true,
    "has_config": true,
    "has_prompts": true
  }
]
```

---

### Graph Inspection

#### `GET /studio/agents/{name}/graph`

Get the agent's execution graph topology.

```bash
curl http://localhost:8000/studio/agents/my_agent/graph
```

```json
{
  "agent_name": "my_agent",
  "nodes": [
    {"id": "__start__", "name": "__start__", "type": "start", "metadata": {}},
    {"id": "retrieve", "name": "retrieve", "type": "agent", "metadata": {}},
    {"id": "__end__", "name": "__end__", "type": "end", "metadata": {}}
  ],
  "edges": [
    {"id": "edge_0", "source": "__start__", "target": "retrieve", "condition": null}
  ],
  "entry_point": "__start__",
  "end_points": ["__end__"]
}
```

---

#### `GET /studio/agents/{name}/schemas`

Get JSON schemas for agent input/output models.

```bash
curl http://localhost:8000/studio/agents/my_agent/schemas
```

---

#### `GET /studio/agents/{name}/config`

Get agent configuration.

```bash
curl http://localhost:8000/studio/agents/my_agent/config
```

---

### Runs

#### `POST /studio/agents/{name}/runs`

Create and execute a run synchronously.

**Request Body — `StudioRunRequest`:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | ✅ | — | User query |
| `user_id` | `string` | | `"default-user"` | User ID |
| `thread_id` | `string \| null` | | auto-generated | Thread ID |
| `context` | `object` | | `{}` | Additional context |
| `metadata` | `object` | | `{}` | Extra metadata |
| `prompt_version` | `string` | | `"v1"` | Prompt version |
| `breakpoints` | `string[]` | | `[]` | Node names to break at |
| `checkpoint_id` | `string \| null` | | `null` | Checkpoint to resume from |

```bash
curl -X POST http://localhost:8000/studio/agents/my_agent/runs \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "breakpoints": ["human_review"]}'
```

---

#### `POST /studio/agents/{name}/runs/stream`

Create and stream a run via SSE with full event detail.

```bash
curl -X POST http://localhost:8000/studio/agents/my_agent/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze this data"}'
```

**Headers:**

| Header | Value |
|---|---|
| `Content-Type` | `text/event-stream` |
| `X-Studio-Run-Id` | Run identifier |
| `X-Accel-Buffering` | `no` |

---

#### `GET /studio/agents/{name}/runs`

List recent runs for an agent.

| Query Param | Type | Default | Description |
|---|---|---|---|
| `limit` | `int` | `50` | Max runs to return |

```bash
curl "http://localhost:8000/studio/agents/my_agent/runs?limit=10"
```

---

#### `GET /studio/agents/{name}/runs/{run_id}`

Get a specific run with all events.

```bash
curl http://localhost:8000/studio/agents/my_agent/runs/run_abc123
```

---

### State & Checkpoints

#### `GET /studio/agents/{name}/threads/{thread_id}/state`

Get the latest thread state.

```bash
curl http://localhost:8000/studio/agents/my_agent/threads/thread_001/state
```

```json
{
  "thread_id": "thread_001",
  "agent_name": "my_agent",
  "state": {
    "messages": [...],
    "response": "Last response",
    "steps_taken": ["retrieve", "generate"]
  },
  "timestamp": "2026-06-18T20:00:00Z",
  "checkpoint_id": "cp_abc123"
}
```

---

#### `POST /studio/agents/{name}/threads/{thread_id}/state`

Apply a partial state update.

**Request Body — `StudioStateUpdate`:**

| Field | Type | Required | Description |
|---|---|---|---|
| `updates` | `object` | ✅ | Key-value pairs to merge |

```bash
curl -X POST http://localhost:8000/studio/agents/my_agent/threads/thread_001/state \
  -H "Content-Type: application/json" \
  -d '{"updates": {"response": "Manually overridden response"}}'
```

---

#### `GET /studio/agents/{name}/threads/{thread_id}/history`

List checkpoint history for a thread.

```bash
curl http://localhost:8000/studio/agents/my_agent/threads/thread_001/history
```

```json
[
  {
    "id": "cp_003",
    "thread_id": "thread_001",
    "step": 2,
    "state": {...},
    "metadata": {},
    "parent_id": "cp_002",
    "timestamp": "2026-06-18T20:00:03Z"
  },
  {
    "id": "cp_002",
    "thread_id": "thread_001",
    "step": 1,
    "state": {...},
    "metadata": {},
    "parent_id": "cp_001",
    "timestamp": "2026-06-18T20:00:02Z"
  }
]
```

---

### Resume (HITL / Interrupt)

#### `POST /studio/agents/{name}/threads/{thread_id}/resume`

Resume a LangGraph execution that was paused by an interrupt.

**Request Body — `StudioResumeRequest`:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `value` | `any` | | `null` | Human response or approval value |
| `action` | `string` | | `"approve"` | `"approve"` or `"reject"` |

```bash
curl -X POST \
  http://localhost:8000/studio/agents/my_agent/threads/thread_001/resume \
  -H "Content-Type: application/json" \
  -d '{"value": {"approved": true, "notes": "Looks good"}, "action": "approve"}'
```

Uses LangGraph's `Command(resume=value)` to continue execution from the interrupt point. Returns an SSE stream of continued execution events.

---

## Error Codes Reference

| HTTP Status | Meaning | Common Causes |
|---|---|---|
| `200` | Success | Normal response |
| `202` | Accepted (Suspended) | HITL suspension — execution paused for approval |
| `400` | Bad Request | Thread storage not configured, no fields to update |
| `404` | Not Found | Agent, thread, run, or suspended state not found |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Agent invocation failed, graph error |

### Error Response Format

```json
{
  "detail": "Error description string"
}
```

### HITL Suspension Response (202)

```json
{
  "detail": {
    "status": "suspended",
    "approval_id": "tx_984712",
    "node_name": "financial_transfer",
    "message": "Transaction requires human approval."
  }
}
```

---

## SSE Event Types Reference

Events streamed via Studio's `/runs/stream` endpoint:

| Event Type | Node Field | Description |
|---|---|---|
| `run_start` | — | Run execution begins |
| `node_start` | Node name | Graph node begins execution |
| `node_end` | Node name | Graph node completes (includes output) |
| `message_chunk` | LLM name | Token streamed from chat model |
| `subagent_start` | Subagent name | Subgraph execution begins |
| `subagent_end` | Subagent name | Subgraph execution completes |
| `task_update` | `planning:{tool}` | Planning tool (write_todos) detected |
| `breakpoint_hit` | Interrupted node | Execution paused at breakpoint |
| `run_complete` | — | Run execution finishes successfully |
| `run_error` | — | Run execution failed (includes error) |
| `done` | — | Stream complete |
