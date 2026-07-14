# Frontend API Usage Guide

This guide covers every endpoint that Agentomatic exposes **per agent** and
shows how to consume them from a frontend application. All examples use the
actual request / response models defined in `router_factory.py`.

!!! info "Base URL convention"
    Every agent is mounted at **`/api/v1/{agent_name}/`**. There is **no**
    `/agents/` segment in the path. For an agent called `alpha` the invoke
    endpoint is `/api/v1/alpha/invoke`.

---

## TypeScript Interfaces

The interfaces below map **1-to-1** to the Pydantic models on the server.

```typescript
// ── Invoke ──────────────────────────────────────────────────────

/** Maps to AgentInvokeRequest (router_factory.py) */
interface AgentInvokeRequest {
  query: string;                        // required
  user_id?: string;                     // default: "default-user"
  context?: Record<string, any>;        // default: {}
  thread_id?: string | null;            // default: null
  prompt_version?: string;              // default: "v1"
  temperature?: number | null;          // 0.0 – 2.0
  max_tokens?: number | null;           // >= 1
  metadata?: Record<string, any>;       // default: {}
}

/** Maps to AgentInvokeResponse (router_factory.py) */
interface AgentInvokeResponse {
  response: string;
  agent_type: string;
  thread_id: string | null;
  suggestions: string[];
  citations: Record<string, any>[];
  steps_taken: string[];
  context: any;
  metadata: Record<string, any>;
  duration_ms: number;
}

// ── Chat ────────────────────────────────────────────────────────

/** Maps to AgentChatRequest (router_factory.py) */
interface AgentChatRequest {
  content: string;                      // required – user message
  user_id?: string;                     // default: "default-user"
  thread_id?: string | null;            // default: null
  context?: Record<string, any>;        // default: {}
  metadata?: Record<string, any>;       // default: {}
  messages?: { role: string; content: string }[] | null;  // override history
  include_history?: boolean;            // default: true
  max_history?: number | null;          // override agent default
  persist?: boolean;                    // default: true
  prompt_version?: string;              // default: "v1"
}

// ── Threads ─────────────────────────────────────────────────────

interface CreateThreadRequest {
  thread_id?: string | null;            // auto-generated if omitted
  user_id?: string;                     // default: "default-user"
  title?: string | null;
  metadata?: Record<string, any>;
}

interface UpdateThreadRequest {
  title?: string | null;
  metadata?: Record<string, any> | null;
}

// ── Feedback ────────────────────────────────────────────────────

interface FeedbackPayload {
  thread_id: string;
  rating: number;                       // e.g. 1–5
  comment?: string;
  metadata?: Record<string, any>;
}

// ── Tasks (async / batch execution) ─────────────────────────────

/** Maps to TaskProgress (tasks/models.py) */
interface TaskProgress {
  percent: number;                      // 0–100
  message: string;
  current?: number | null;
  total?: number | null;
  stage?: string | null;
}

/** Maps to TaskRecord.public_dict() (tasks/models.py) */
interface TaskRecord {
  id: string;                           // "task_..."
  target_type: "agent" | "plugin" | "pipeline" | "endpoint" | "ingestion";
  target: string;                       // resource name
  mode: "sync" | "async" | "batch" | "stream";
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: TaskProgress;
  result?: any;                         // populated once succeeded
  error?: string | null;                // populated once failed
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  duration_ms?: number | null;
  links?: { status: string; events: string; result: string; cancel: string };
}
```

---

## Core Endpoints

### 1. Invoke — synchronous request / response 🎯

**`POST /api/v1/{agent_name}/invoke`**

Use this for the vast majority of frontend interactions.

```typescript
const res = await fetch("/api/v1/alpha/invoke", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "What is machine learning?",
    user_id: "user-42",
    context: { domain: "education" },
    temperature: 0.7,
  }),
});

const data: AgentInvokeResponse = await res.json();
console.log(data.response);       // "Machine learning is …"
console.log(data.duration_ms);    // 432.1
console.log(data.steps_taken);    // ["retrieve", "generate"]
```

### 2. Streaming — SSE (Server-Sent Events) ⚡

**`POST /api/v1/{agent_name}/invoke/stream`**

Same request body as `/invoke`. The response is an `text/event-stream`.

```typescript
const res = await fetch("/api/v1/alpha/invoke/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "Explain quantum computing step by step",
  }),
});

const reader = res.body!.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value, { stream: true });
  const lines = chunk.split("\n");

  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const payload = line.slice(6);
      if (payload === "[DONE]") return;
      if (payload.startsWith("[ERROR]")) {
        console.error("Stream error:", payload);
        return;
      }
      // Append token to the UI
      appendToken(payload);
    }
  }
}
```

### 3. Chat — session-aware conversation 💬

**`POST /api/v1/{agent_name}/chat`**

Automatically manages conversation history, thread creation, and message
persistence when a thread store is configured.

```typescript
const res = await fetch("/api/v1/alpha/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    content: "Explain quantum computing",
    user_id: "user-42",
    thread_id: "thread-abc",          // reuse an existing thread
    include_history: true,            // auto-load prior messages
    persist: true,                    // save this exchange
  }),
});

const data = await res.json();
console.log(data.response);
console.log(data.thread_id);         // "thread-abc"
```

### 4. Async Tasks — long-running jobs with progress ⏳ { #async-tasks }

Any resource (agent, plugin, pipeline, endpoint, ingestor) can run as a tracked
background task. This is ideal for long jobs — e.g. document ingestion — where
the frontend submits work, then **polls** or **streams** progress.

**Submit** (returns immediately with `202` and a task record):

- Agent: `POST /api/v1/{agent}/invoke/async` (single) · `/invoke/batch` (many)
- Plugin: `POST /api/v1/{plugin}/predict/async` · `/predict/batch`
- Pipeline: `POST /api/v1/pipelines/{name}/run/async` · `/run/batch`
- Endpoint: `POST /api/v1/endpoints/{name}{path}/async` · `.../batch`
- Ingestor: `POST /api/v1/ingestion/{name}/run/async` · `/run/batch`

```typescript
// 1. Submit an async task
const submit = await fetch("/api/v1/alpha/invoke/async", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "Summarise this 400-page report" }),
});
const task: TaskRecord = await submit.json();
// task.status === "queued"; task.links.status === "/api/v1/tasks/task_ab12..."

// 2a. Poll the task board until terminal
async function pollTask(taskId: string): Promise<TaskRecord> {
  while (true) {
    const res = await fetch(`/api/v1/tasks/${taskId}`);
    const rec: TaskRecord = await res.json();
    updateProgressBar(rec.progress.percent, rec.progress.message);
    if (["succeeded", "failed", "cancelled"].includes(rec.status)) return rec;
    await new Promise((r) => setTimeout(r, 750));
  }
}
const final = await pollTask(task.id);
console.log(final.result);            // once succeeded
```

**Or stream progress live via SSE** instead of polling:

```typescript
// 2b. Stream progress events
const es = new EventSource(`/api/v1/tasks/${task.id}/events`);
es.onmessage = (e) => {
  const evt = JSON.parse(e.data);      // { status, progress, ... }
  updateProgressBar(evt.progress.percent, evt.progress.message);
  if (["succeeded", "failed", "cancelled"].includes(evt.status)) es.close();
};
```

**Cancel** an in-flight task:

```typescript
await fetch(`/api/v1/tasks/${task.id}/cancel`, { method: "POST" });
```

The **task board** endpoints (not agent-scoped):

| Method | Path | Description |
| ------ | ---- | ----------- |
| `POST` | `/api/v1/tasks` | Submit a task for any target (`{ target_type, target, input }`) |
| `GET` | `/api/v1/tasks` | List/filter tasks (`status`, `target_type`, `target`, `limit`, `offset`) |
| `GET` | `/api/v1/tasks/{id}` | Task status, progress, result |
| `GET` | `/api/v1/tasks/{id}/result` | Terminal result payload |
| `GET` | `/api/v1/tasks/{id}/events` | SSE progress stream |
| `POST` | `/api/v1/tasks/{id}/cancel` | Request cancellation |
| `DELETE` | `/api/v1/tasks/{id}` | Delete a task record |

!!! tip "Batch jobs"
    `POST .../batch` accepts `{ "inputs": [...], "batch_concurrency"?, "callback_url"? }`
    and returns one task whose `progress` tracks per-item completion.

### React task-polling hook

```tsx
import { useCallback, useState } from "react";

export function useAsyncTask(agentName: string) {
  const [task, setTask] = useState<TaskRecord | null>(null);
  const [running, setRunning] = useState(false);

  const run = useCallback(
    async (request: AgentInvokeRequest) => {
      setRunning(true);
      const submit = await fetch(`/api/v1/${agentName}/invoke/async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      let rec: TaskRecord = await submit.json();
      setTask(rec);

      while (!["succeeded", "failed", "cancelled"].includes(rec.status)) {
        await new Promise((r) => setTimeout(r, 750));
        rec = await fetch(`/api/v1/tasks/${rec.id}`).then((r) => r.json());
        setTask(rec);
      }
      setRunning(false);
      return rec;
    },
    [agentName],
  );

  return { run, task, running };
}
```

---

## Agent Metadata Endpoints

### Health

**`GET /api/v1/{agent_name}/health`**

```typescript
const res = await fetch("/api/v1/alpha/health");
const health = await res.json();
// { "status": "ok", "agent": "alpha" }
```

### Configuration

**`GET /api/v1/{agent_name}/config`**

Returns the agent's manifest and runtime configuration.

```typescript
const res = await fetch("/api/v1/alpha/config");
const config = await res.json();
```

### Prompt Versions

**`GET /api/v1/{agent_name}/prompts`**

Lists all available prompt versions for the agent.

```typescript
const res = await fetch("/api/v1/alpha/prompts");
const prompts = await res.json();
```

### Agent Card (A2A)

**`GET /api/v1/{agent_name}/card`**

Returns the agent's A2A-compatible card with capability declarations and
endpoint URLs.

```typescript
const res = await fetch("/api/v1/alpha/card");
const card = await res.json();
// {
//   "name": "alpha",
//   "description": "...",
//   "version": "1.0.0",
//   "capabilities": { "streaming": true, "chat": true, ... },
//   "endpoints": {
//     "invoke": "/api/v1/alpha/invoke",
//     "chat":   "/api/v1/alpha/chat",
//     "stream": "/api/v1/alpha/invoke/stream",
//     "health": "/api/v1/alpha/health"
//   }
// }
```

---

## Thread Management

All thread endpoints require a thread store to be configured on the platform.

| Method   | Path                                            | Description                  |
| -------- | ----------------------------------------------- | ---------------------------- |
| `POST`   | `/api/v1/{name}/threads`                        | Create a new thread          |
| `GET`    | `/api/v1/{name}/threads`                        | List threads                 |
| `GET`    | `/api/v1/{name}/threads/{tid}`                  | Get thread details           |
| `PATCH`  | `/api/v1/{name}/threads/{tid}`                  | Update thread title/metadata |
| `DELETE` | `/api/v1/{name}/threads/{tid}`                  | Delete a thread              |
| `GET`    | `/api/v1/{name}/threads/{tid}/messages`         | Get thread messages          |
| `DELETE` | `/api/v1/{name}/threads/{tid}/messages`         | Clear thread messages        |
| `GET`    | `/api/v1/{name}/threads/{tid}/summary`          | Get conversation summary     |
| `POST`   | `/api/v1/{name}/threads/{tid}/fork`             | Fork a thread                |
| `GET`    | `/api/v1/{name}/threads/{tid}/lineage`          | Get thread lineage           |

### Example: create & use a thread

```typescript
// 1. Create a thread
const createRes = await fetch("/api/v1/alpha/threads", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    user_id: "user-42",
    title: "ML Questions",
  }),
});
const thread = await createRes.json();
const threadId = thread.thread_id;

// 2. Chat within that thread (history auto-loads)
const chatRes = await fetch("/api/v1/alpha/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    content: "What is gradient descent?",
    thread_id: threadId,
    user_id: "user-42",
  }),
});

// 3. Retrieve messages later
const msgsRes = await fetch(
  `/api/v1/alpha/threads/${threadId}/messages`
);
const messages = await msgsRes.json();
```

---

## Human-in-the-Loop

| Method | Path                                           | Description                 |
| ------ | ---------------------------------------------- | --------------------------- |
| `GET`  | `/api/v1/{name}/threads/{tid}/pending`         | Check for pending approvals |
| `POST` | `/api/v1/{name}/threads/{tid}/approve`         | Approve a pending action    |
| `POST` | `/api/v1/{name}/threads/{tid}/reject`          | Reject a pending action     |

---

## Feedback

| Method | Path                               | Description                     |
| ------ | ---------------------------------- | ------------------------------- |
| `POST` | `/api/v1/{name}/feedback`          | Submit feedback                 |
| `GET`  | `/api/v1/{name}/feedback`          | List feedback for the agent     |
| `GET`  | `/api/v1/{name}/feedback/export`   | Export feedback as JSONL        |

---

## A2A (Agent-to-Agent) Protocol

| Method | Path                                    | Description          |
| ------ | --------------------------------------- | -------------------- |
| `POST` | `/api/v1/{name}/a2a/tasks`              | Submit an A2A task   |
| `GET`  | `/api/v1/{name}/a2a/tasks/{task_id}`    | Get A2A task status  |

---

## Optimization

| Method | Path                                 | Description                              |
| ------ | ------------------------------------ | ---------------------------------------- |
| `POST` | `/api/v1/{name}/optimize/invoke`     | Invoke with full optimization context    |

---

## Full Endpoint Reference

Every agent exposes **26 endpoints**. All paths below are relative to the
agent prefix `/api/v1/{agent_name}`.

| # | Method   | Path                             | Request Body            | Description                       |
|---|----------|----------------------------------|-------------------------|-----------------------------------|
| 1 | `POST`   | `/invoke`                        | `AgentInvokeRequest`    | Synchronous invocation            |
| 2 | `POST`   | `/invoke/stream`                 | `AgentInvokeRequest`    | SSE streaming invocation          |
| 3 | `POST`   | `/chat`                          | `AgentChatRequest`      | Session-aware chat                |
| 4 | `GET`    | `/health`                        | —                       | Agent health check                |
| 5 | `GET`    | `/config`                        | —                       | Agent configuration               |
| 6 | `GET`    | `/prompts`                       | —                       | Available prompt versions         |
| 7 | `GET`    | `/card`                          | —                       | A2A agent card                    |
| 8 | `POST`   | `/a2a/tasks`                     | `A2ATaskRequest`        | Submit A2A task                   |
| 9 | `GET`    | `/a2a/tasks/{task_id}`           | —                       | Get A2A task status               |
| 10 | `POST`  | `/threads`                       | `CreateThreadRequest`   | Create thread                     |
| 11 | `GET`   | `/threads`                       | —                       | List threads                      |
| 12 | `GET`   | `/threads/{tid}`                 | —                       | Get thread                        |
| 13 | `PATCH` | `/threads/{tid}`                 | `UpdateThreadRequest`   | Update thread                     |
| 14 | `DELETE`| `/threads/{tid}`                 | —                       | Delete thread                     |
| 15 | `GET`   | `/threads/{tid}/messages`        | —                       | Get messages                      |
| 16 | `DELETE`| `/threads/{tid}/messages`        | —                       | Clear messages                    |
| 17 | `GET`   | `/threads/{tid}/summary`         | —                       | Conversation summary              |
| 18 | `POST`  | `/optimize/invoke`               | `OptimizeInvokeRequest` | Optimization invocation           |
| 19 | `POST`  | `/feedback`                      | Feedback payload        | Submit feedback                   |
| 20 | `GET`   | `/feedback`                      | —                       | List feedback                     |
| 21 | `GET`   | `/feedback/export`               | —                       | Export feedback (JSONL)           |
| 22 | `GET`   | `/threads/{tid}/pending`         | —                       | Pending approvals                 |
| 23 | `POST`  | `/threads/{tid}/approve`         | —                       | Approve pending action            |
| 24 | `POST`  | `/threads/{tid}/reject`          | —                       | Reject pending action             |
| 25 | `POST`  | `/threads/{tid}/fork`            | —                       | Fork thread                       |
| 26 | `GET`   | `/threads/{tid}/lineage`         | —                       | Thread lineage                    |

!!! info "Plus execution-mode routes"
    When the task engine is enabled (default), each agent also exposes
    `POST /invoke/async` and `POST /invoke/batch` that submit work to the
    [task board](#async-tasks). Plugins, pipelines,
    endpoints, and ingestors get the equivalent `/async` and `/batch` variants
    of their primary route.

---

## Platform Surfaces (beyond a single agent)

Besides the per-agent endpoints above, the platform exposes cross-cutting
surfaces a frontend can consume. These power the **Agentomatic Studio** views
(Control, Endpoints, Connections, Pipelines, Plugins).

### Control Plane

Mounted at `{api_prefix}/control` when `enable_control_plane=True`. Mutating
calls require the `X-Control-Token` header if a `control_token` is configured.

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/api/v1/control` | Platform overview (counts, uptime, maintenance flag) |
| `GET` | `/api/v1/control/agents` | Agents with health, effective auth policy, connections |
| `GET` | `/api/v1/control/agents/{name}` | Single-agent operational detail |
| `GET` | `/api/v1/control/endpoints` | Registered custom endpoints |
| `GET` | `/api/v1/control/connections` | Connection health by scope |
| `GET` | `/api/v1/control/health` | Aggregate health (agents + connections) |
| `GET` | `/api/v1/control/metrics/summary` | Coarse counters for dashboards |
| `GET` | `/api/v1/control/config` | Sanitised feature/config snapshot |
| `POST` | `/api/v1/control/agents/{name}/disable` | Drain an agent (503 for its routes) |
| `POST` | `/api/v1/control/agents/{name}/enable` | Re-enable an agent |
| `POST` | `/api/v1/control/maintenance` | Toggle maintenance mode (`{ "enabled": bool }`) |

```typescript
// Overview + agent list
const info = await fetch("/api/v1/control").then((r) => r.json());
const agents = await fetch("/api/v1/control/agents").then((r) => r.json());

// Drain an agent (requires control token when configured)
await fetch("/api/v1/control/agents/fraud_agent/disable", {
  method: "POST",
  headers: { "X-Control-Token": CONTROL_TOKEN },
});
```

```typescript
/** Maps to ControlAgentInfo (control/models.py) */
interface ControlAgentInfo {
  name: string;
  slug: string;
  description: string;
  version: string;
  framework: string;
  enabled: boolean;
  requires_auth: boolean;
  allowed_roles: string[];
  allowed_scopes: string[];
  connections: string[];
  health: Record<string, any>;
}

/** Maps to ControlEndpointInfo */
interface ControlEndpointInfo {
  name: string;
  description: string;
  version: string;
  path: string;
  methods: string[];
  aggregation: string;
  upstreams: string[];
  ready: boolean;
}

/** Maps to ControlConnectionInfo */
interface ControlConnectionInfo {
  scope: string;
  connections: Record<string, any>;
}
```

### Custom Endpoints

Each custom endpoint is mounted under `{api_prefix}/endpoints/{name}`. The main
handler lives at the endpoint's configurable `path` (default `/call`), with
`/health` and `/info` alongside it. List all endpoints via `GET
/api/v1/endpoints` or the control plane. Use these to enrich a UI with
model-ensemble results.

```typescript
// Default handler path is `/call` (override via the endpoint's `path` attr)
const result = await fetch("/api/v1/endpoints/ensemble/call", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ payload: { text: "classify me" } }),
}).then((r) => r.json());
```

### Pipelines & Plugins

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/api/v1/pipelines` | List pipelines |
| `POST` | `/api/v1/pipelines/{name}/run` | Run a pipeline (`{ input, metadata? }`) |
| `GET` | `/api/v1/pipelines/{name}/validate` | Validate a pipeline |
| `GET` | `/api/v1/pipelines/{name}/visualize` | Mermaid diagram of the flow |
| `GET` | `/api/v1/plugins` | List ML plugins |
| `GET` | `/api/v1/plugins/{name}/model_card` | Plugin model card |
| `POST` | `/api/v1/plugins/{name}/predict` | Run a plugin prediction |

### Ingestion / RAG

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/api/v1/ingestion` | List registered ingestors |
| `POST` | `/api/v1/ingestion/{name}/run` | Run an ingestion job (sync) |
| `POST` | `/api/v1/ingestion/{name}/run/async` | Run as a tracked task |
| `GET` | `/api/v1/ingestion/{name}/info` | Ingestor metadata / readiness |

### Status Dashboard

A single roll-up of the whole platform's health — ideal for an ops/admin view.

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/status` | HTML dashboard (all resources + task engine) |
| `GET` | `/api/v1/status` | JSON: overall + per-resource health |

```typescript
const status = await fetch("/api/v1/status").then((r) => r.json());
// {
//   status: "healthy",                        // overall rollup
//   platform: { name, version, uptime_seconds, maintenance_mode },
//   summary: { agents: { total, healthy }, plugins: {...}, ... },
//   resources: {
//     agents:    { total, healthy, degraded, items: { ...per-agent health } },
//     plugins:   { ... }, endpoints: { ... },
//     ingestors: { ... }, pipelines: { ... },
//   },
//   tasks:   { enabled, running, queued, ... },   // task-engine stats
//   storage: { status, backend, ... },
// }
```

!!! tip "Auth for platform surfaces"
    When JWT auth is enabled, send the same `Authorization: Bearer <token>`
    header used for agent calls. The control plane's **mutating** routes
    additionally require `X-Control-Token` when a control token is configured.

---

## Framework Integration Examples

### React Hook

```tsx
import { useState, useCallback } from "react";

interface UseAgentOptions {
  agentName: string;
  baseUrl?: string;
}

export function useAgent({ agentName, baseUrl = "" }: UseAgentOptions) {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<AgentInvokeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const invoke = useCallback(
    async (request: AgentInvokeRequest) => {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(
          `${baseUrl}/api/v1/${agentName}/invoke`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(request),
          },
        );

        if (!res.ok) {
          const err = await res.json();
          throw new Error(
            typeof err.detail === "string"
              ? err.detail
              : err.detail?.error ?? `Request failed (${res.status})`,
          );
        }

        const data: AgentInvokeResponse = await res.json();
        setResponse(data);
        return data;
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [agentName, baseUrl],
  );

  return { invoke, loading, response, error };
}

// ── Usage ─────────────────────────────────────────────────────

function ChatComponent() {
  const { invoke, loading, response, error } = useAgent({
    agentName: "alpha",
  });

  const handleSubmit = async (query: string) => {
    await invoke({ query, user_id: "user-42" });
  };

  return (
    <div>
      {loading && <div>Processing…</div>}
      {error && <div className="error">Error: {error}</div>}
      {response && (
        <div>
          <p>{response.response}</p>
          <small>{response.duration_ms.toFixed(0)} ms</small>
        </div>
      )}
    </div>
  );
}
```

### React Streaming Hook

```tsx
import { useState, useCallback, useRef } from "react";

export function useAgentStream(agentName: string) {
  const [streaming, setStreaming] = useState(false);
  const [tokens, setTokens] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const stream = useCallback(
    async (request: AgentInvokeRequest) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStreaming(true);
      setTokens("");

      const res = await fetch(
        `/api/v1/${agentName}/invoke/stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
          signal: controller.signal,
        },
      );

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          for (const line of chunk.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6);
            if (data === "[DONE]") return;
            if (data.startsWith("[ERROR]")) throw new Error(data);
            setTokens((prev) => prev + data);
          }
        }
      } finally {
        setStreaming(false);
      }
    },
    [agentName],
  );

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  return { stream, streaming, tokens, cancel };
}
```

### Vue.js Composable

```typescript
import { ref, readonly } from "vue";

export function useAgent(agentName: string, baseUrl = "") {
  const loading = ref(false);
  const response = ref<AgentInvokeResponse | null>(null);
  const error = ref<string | null>(null);

  async function invoke(request: AgentInvokeRequest) {
    loading.value = true;
    error.value = null;

    try {
      const res = await fetch(
        `${baseUrl}/api/v1/${agentName}/invoke`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
        },
      );

      if (!res.ok) {
        const err = await res.json();
        throw new Error(
          typeof err.detail === "string"
            ? err.detail
            : err.detail?.error ?? `Request failed (${res.status})`,
        );
      }

      const data: AgentInvokeResponse = await res.json();
      response.value = data;
      return data;
    } catch (err: any) {
      error.value = err.message ?? "Unknown error";
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function chat(request: AgentChatRequest) {
    loading.value = true;
    error.value = null;

    try {
      const res = await fetch(
        `${baseUrl}/api/v1/${agentName}/chat`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
        },
      );

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail?.error ?? `Request failed`);
      }

      const data = await res.json();
      response.value = data;
      return data;
    } catch (err: any) {
      error.value = err.message ?? "Unknown error";
      throw err;
    } finally {
      loading.value = false;
    }
  }

  return {
    invoke,
    chat,
    loading: readonly(loading),
    response: readonly(response),
    error: readonly(error),
  };
}
```

---

## Error Handling

Agentomatic returns standard FastAPI error responses. All errors use the
`detail` field.

```typescript
interface APIError {
  detail:
    | string
    | {
        error: string;
        agent: string;
        validation_errors?: any[];
        expected_schema?: string;
      };
}

async function safeInvoke(
  agentName: string,
  request: AgentInvokeRequest,
): Promise<AgentInvokeResponse> {
  const res = await fetch(`/api/v1/${agentName}/invoke`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err: APIError = await res.json();

    switch (res.status) {
      case 404:
        throw new Error(`Agent "${agentName}" not found`);
      case 422:
        if (typeof err.detail === "object") {
          console.error("Validation errors:", err.detail.validation_errors);
          console.error("Expected schema:", err.detail.expected_schema);
        }
        throw new Error("Invalid request body");
      case 500:
        throw new Error("Internal server error");
      default:
        throw new Error(
          typeof err.detail === "string"
            ? err.detail
            : JSON.stringify(err.detail),
        );
    }
  }

  return res.json();
}
```

---

## Quick Reference

!!! tip "Rules of thumb"
    - **Use `/invoke`** for standard request → response interactions (95% of
      use cases).
    - **Use `/invoke/stream`** when you need token-by-token streaming in a
      chat UI.
    - **Use `/chat`** when you want automatic conversation history, thread
      management, and message persistence.
    - **All three endpoints work with every agent** — the platform generates
      them automatically.

!!! warning "Common mistakes"
    - ❌ `/api/v1/agents/alpha/invoke` — wrong, there is no `/agents/` segment.
    - ✅ `/api/v1/alpha/invoke` — correct.
    - ❌ `{ payload: { query: "..." } }` — wrong, no `payload` wrapping.
    - ✅ `{ query: "..." }` — correct, fields go at the top level.
    - ❌ `{ input: "..." }` for chat — wrong field name.
    - ✅ `{ content: "..." }` for chat — correct field name.
    - ❌ `{ streaming: true }` — wrong, there is no `streaming` field.
    - ✅ Use the `/invoke/stream` endpoint instead for streaming.
