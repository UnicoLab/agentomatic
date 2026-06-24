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
