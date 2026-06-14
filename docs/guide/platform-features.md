# Advanced Platform Features

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Enterprise-Grade Orchestration & Core Capabilities</h3>
</div>

---

Agentomatic provides a suite of advanced platform capabilities designed to make multi-agent systems production-ready. These features solve common real-world challenges around human-in-the-loop validation, schema enforcement, prompt routing, checkpointer persistence, and service resilience.

---

## 🚦 1. Human-in-the-Loop (HITL) Protocol

In complex multi-agent architectures, agents often require human validation before executing sensitive actions (e.g., executing a transaction, sending an email, or performing a destructive write).

Agentomatic provides a built-in suspension/resume mechanism that allows nodes or graphs to halt execution, serialize their state snapshots to the storage backend, and resume or abort dynamically when approved/rejected via HTTP.

```
                  ┌──────────────────────┐
                  │  Agent Execution     │
                  └──────────┬───────────┘
                             │
                  Node raises AgentSuspendedException
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Serializes state snapshot     │
                  │ Saves to database    │
                  │ Returns HTTP 202     │
                  └──────────────────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
       POST /approve           POST /reject
                 │                       │
      Merges human context       Deletes snapshot
      Resumes execution          Aborts execution
                 │                       │
                 ▼                       ▼
       ┌──────────────────┐    ┌──────────────────┐
       │   Completes graph│    │  Returns 200 OK  │
       └──────────────────┘    └──────────────────┘
```

### Suspending Execution

To suspend execution, raise `AgentSuspendedException` from anywhere inside your node function or LangGraph node:

```python
from agentomatic.core.router_factory import AgentSuspendedException

async def financial_transfer_node(state: dict):
    metadata = state.get("metadata") or {}
    
    # Check if we already have approval
    if metadata.get("hitl_approved"):
        return {
            "response": f"Successfully transferred ${state['amount']}",
            "metadata": metadata
        }
    
    # Otherwise, suspend and wait for human confirmation
    raise AgentSuspendedException(
        approval_id=f"tx_{state['transaction_id']}",
        node_name="financial_transfer_node",
        state_snapshot=state,
        message="Transaction requires human approval."
    )
```

When this exception is thrown:
1. Agentomatic intercepts the execution.
2. The current `state_snapshot` is stored in the persistent database under `SuspendedStateModel`.
3. The API immediately returns a `202 Accepted` status code with the approval details.

**Response payload:**
```json
{
  "detail": {
    "status": "suspended",
    "approval_id": "tx_984712",
    "node_name": "financial_transfer_node",
    "message": "Transaction requires human approval."
  }
}
```

### REST Endpoints for Approvals

Each registered agent automatically exposes three endpoints to manage suspended states:

#### A. List Pending Approvals
Retrieve all currently suspended states for a specific thread.

* **Method:** `GET`
* **Path:** `/api/v1/{agent_slug}/threads/{thread_id}/pending`
* **Response:**
  ```json
  {
    "thread_id": "thread_abc123",
    "count": 1,
    "pending": [
      {
        "id": "tx_984712",
        "node_name": "financial_transfer_node",
        "state_snapshot": { ... },
        "created_at": "2026-06-13T22:50:34"
      }
    ]
  }
  ```

#### B. Approve and Resume Execution
Approve the suspension, optionally merge new parameters (e.g., corrections or decisions), and resume the agent's execution from that exact step.

* **Method:** `POST`
* **Path:** `/api/v1/{agent_slug}/threads/{thread_id}/approve`
* **Payload:**
  ```json
  {
    "approval_id": "tx_984712",
    "context": {
      "approved_limit": 500
    }
  }
  ```
* **Behavior:** Deletes the pending suspended snapshot, merges the request's `context` into the state, marks `state.metadata.hitl_approved = True`, and resumes the graph execution synchronously. It returns the final agent execution output.

#### C. Reject and Abort Execution
Reject the transaction and discard the execution context.

* **Method:** `POST`
* **Path:** `/api/v1/{agent_slug}/threads/{thread_id}/reject`
* **Payload:**
  ```json
  {
    "approval_id": "tx_984712",
    "reason": "Risk score too high"
  }
  ```
* **Response:**
  ```json
  {
    "status": "rejected",
    "approval_id": "tx_984712",
    "reason": "Risk score too high"
  }
  ```

---

## 💾 2. LangGraph Checkpointer (`AgentomaticCheckpointer`)

If you are building your agents using **LangGraph**, you need a checkpoint saver to persist the graph's memory across invocations. Agentomatic provides a native adapter: `AgentomaticCheckpointer`.

This checkpointer implements LangGraph's `BaseCheckpointSaver` and delegates storage operations directly to your configured Agentomatic `BaseStore` (e.g., `SQLAlchemyStore` or `MemoryStore`). This ensures that thread states, checkpoint namespaces, and historical tuples are stored consistently without needing a separate database.

### Usage Example

```python
from langgraph.graph import StateGraph
from agentomatic.storage import SQLAlchemyStore
from agentomatic.storage.checkpointer import AgentomaticCheckpointer

# 1. Setup your database store
store = SQLAlchemyStore("postgresql+asyncpg://postgres:secret@localhost:5432/agent_db")

# 2. Wrap it with the LangGraph checkpointer adapter
checkpointer = AgentomaticCheckpointer(store)

# 3. Create and compile your LangGraph with the checkpointer
builder = StateGraph(MyStateClass)
# ... build graph nodes ...

graph = builder.compile(checkpointer=checkpointer)
```

All LangGraph checkpoints are automatically persisted in your database and can be fetched or updated using the standard LangGraph runtime config.

---

## 🎯 3. Structured Output Enforcer

Ensuring that LLM outputs strictly match validation schemas is critical for reliable API integration. Agentomatic makes structured output enforcement seamless:

1. **Auto-bind Schemas**: Under the hood, Agentomatic leverages LangChain's `.with_structured_output()` to bind Pydantic schemas directly to the LLM runtime.
2. **Fallback Parser**: If the provider or mock model does not support native schema enforcement (e.g., mock/fake models in test environments), Agentomatic automatically intercepts the output and applies a robust parsing and default-fallback generator wrapper.

### Factory Method

Use `get_structured_llm` to create a model bound to a Pydantic schema:

```python
from pydantic import BaseModel, Field
from agentomatic.providers.llm import get_structured_llm

class TranslationOutput(BaseModel):
    detected_language: str
    translated_text: str
    confidence: float = Field(default=0.0)

# Build an LLM instance that returns parsed instances of TranslationOutput
structured_model = get_structured_llm(
    response_model=TranslationOutput,
    provider="openai",
    model="gpt-4o",
    temperature=0.0
)

result = structured_model.invoke("Translate 'Bonjour tout le monde' to English")
# 'result' is guaranteed to be a TranslationOutput object
print(result.translated_text)  # "Hello everyone"
```

---

## 🗃️ 4. Thread Forking & Cloning

Debugging agent failures or performing A/B evaluation of system prompt prompts requires the ability to fork history. Agentomatic supports **Thread Forking** at the API and database levels.

You can clone a parent thread starting at a specific message index. This creates a brand new thread containing a copy of all messages up to and including that index, allowing subsequent interactions to diverge without altering the parent thread's history.

* **Method:** `POST`
* **Path:** `/api/v1/{agent_slug}/threads/{thread_id}/fork`
* **Payload:**
  ```json
  {
    "message_index": 2,
    "new_thread_id": "fork_thread_99",
    "title": "A/B Test Variant B"
  }
  ```
* **Response:** Returns the new thread dictionary:
  ```json
  {
    "id": "fork_thread_99",
    "user_id": "user_id_12",
    "agent_name": "support_agent",
    "title": "A/B Test Variant B",
    "message_count": 3,
    "metadata": {}
  }
  ```

---

## 📊 5. A/B Test Prompt Router

Agentomatic enables you to easily conduct prompt version testing in production. You can configure routing splits between different system prompts directly inside your agent configuration.

### Configuration

Add `prompt_ab_tests` with fractional weights inside your agent's config settings:

```json
{
  "prompt_ab_tests": {
    "v1": 0.7,
    "v2": 0.3
  }
}
```

### Execution Flow

1. When a client invokes the agent without specifying a `prompt_version` (or when set to `"v1"`), Agentomatic performs a **weighted random choice** based on the configured split (in the example above: 70% traffic to `v1`, 30% traffic to `v2`).
2. The chosen version is automatically stored in the invocation state under `state["prompt_version"]`.
3. Your agent nodes can read `prompt_version` from the state to load the correct prompt template.
4. The chosen version is returned in the response metadata (`metadata.prompt_version`) so that you can trace and analyze performance.

### Telemetry & Feedback Correlation

To measure the performance and quality of different prompt variants:
1. **Response Tracking**: Read `metadata.prompt_version` from the `/invoke` or `/chat` JSON response payload.
2. **Feedback Logging**: When calling the `POST /api/v1/{agent}/feedback` endpoint, pass the active prompt version in the request metadata payload:
   ```json
   {
     "thread_id": "thread_123",
     "rating": 5,
     "comment": "Highly accurate answer!",
     "metadata": {
       "prompt_version": "v2"
     }
   }
   ```
   This associates ratings and comments directly with each version, allowing you to run analytical breakdowns of user satisfaction metrics per prompt variant.

> [!NOTE]
> Clients can explicitly override the A/B router by passing a specific `prompt_version` parameter in the invoke request (e.g. `{"prompt_version": "v2"}`).

---

## 🛡️ 6. LLM Failover & Fallbacks

LLM APIs can suffer from outages, rate limiting (HTTP 429), or transient network errors. To ensure high availability, Agentomatic lets you define chainable fallback providers.

### Configuration

Pass a list of provider slugs in the `fallbacks` parameter of `get_llm()`:

```python
from agentomatic.providers.llm import get_llm

llm = get_llm(
    provider="openai",
    model="gpt-4",
    fallbacks=["azure", "ollama"]
)
```

If the primary provider (`openai`) raises an exception during execution, LangChain's fallback mechanism automatically routes the request to the next available backup in the list (`azure`, then `ollama`), shielding clients from upstream downtime.

---

## 🔌 7. State-Level Hook Middleware (Interceptors)

For custom telemetry, audit logging, or real-time security scanning, you can register **state-level node hooks**. These are run instantly before and after every agent node invocation.

```python
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder("agents/")

# 1. Register a before-node interceptor
def audit_input_hook(agent_name: str, state: dict):
    print(f"[Hook] Agent '{agent_name}' is about to run. Query: {state.get('current_query')}")
    # You can inspect or modify the state inline before execution starts
    state["metadata"]["audit_logged"] = True

platform.register_before_node_hook(audit_input_hook)

# 2. Register an after-node interceptor
def audit_output_hook(agent_name: str, result: dict):
    print(f"[Hook] Agent '{agent_name}' completed. Response snippet: {result.get('response')[:30]}")

platform.register_after_node_hook(audit_output_hook)
```

---

## 🛡️ 8. Safe Checkpoint Serialization

LangGraph checkpoints can contain non-JSON-serializable Python objects (datetimes, bytes, custom classes). The `AgentomaticCheckpointer` automatically handles this via a safe JSON round-trip:

```python
from agentomatic.storage.checkpointer import AgentomaticCheckpointer, _ensure_json_serializable

# Objects like datetimes, bytes, and custom classes are safely converted
data = _ensure_json_serializable({"ts": datetime.now(), "raw": b"bytes"})
# → {"ts": "2026-06-14 12:00:00", "raw": "b'bytes'"}
```

This happens transparently inside `aput()` — no user configuration needed. All values are round-tripped through `json.dumps(obj, default=str)` → `json.loads()`.

---

## ⏰ 9. HITL TTL Expiry & Cleanup

Suspended states automatically receive a 7-day TTL (`expires_at`). This prevents stale suspended states from accumulating in the database when humans forget to approve or reject them.

### Automatic TTL

When `save_suspended_state()` is called, the `expires_at` field defaults to `now + 7 days`:

```python
state = await store.save_suspended_state(
    approval_id="app_xyz",
    thread_id="thread_1",
    agent_name="my_agent",
    node_name="approval_node",
    state_json={"action": "delete_account"},
)
print(state["expires_at"])  # 7 days from now
```

### Cleanup

Call `cleanup_expired_states()` on a schedule (e.g. cron, background task) to purge expired states:

```python
# Clean up expired suspended states
count = await store.cleanup_expired_states()
print(f"Cleaned up {count} expired states")
```

This works identically on `MemoryStore` and `SQLAlchemyStore`.

---

## 🌳 10. Thread Lineage Tracking

Forked threads now have **first-class lineage fields** instead of metadata-based tracking:

| Field | Type | Description |
|---|---|---|
| `parent_thread_id` | `str \| None` | ID of the parent thread (null for root threads) |
| `fork_message_index` | `int \| None` | Message index where the fork occurred |

### Querying Lineage

Use `get_thread_lineage()` to traverse the full ancestor/descendant tree:

```python
# Fork a thread
forked = await store.fork_thread("root_thread", 2, "child_thread")
print(forked["parent_thread_id"])  # "root_thread"

# Get full lineage
lineage = await store.get_thread_lineage("child_thread")
print(lineage["ancestors"])    # [root_thread_dict]
print(lineage["descendants"])  # []
```

### REST Endpoint

```http
GET /api/v1/{agent}/threads/{thread_id}/lineage
```

**Response:**
```json
{
  "thread_id": "child_thread",
  "ancestors": [{"id": "root_thread", ...}],
  "descendants": [{"id": "grandchild_thread", ...}]
}
```

---

## 📊 11. LLM Failover Telemetry

Track failover events for observability and alerting:

```python
from agentomatic.providers.llm import record_failover, get_failover_count, reset_llm

# Record a failover event
record_failover("openai", "azure", "RateLimitError")
# Emits: 🔄 LLM failover #1: openai -> azure | Error: RateLimitError

# Check current count
print(get_failover_count())  # 1

# Reset (typically in tests)
reset_llm()  # Also resets failover counter
```

The failover chain now also passes `exceptions_to_handle=(Exception,)` to `with_fallbacks()`, ensuring **any** exception triggers the fallback chain, not just specific ones.

---

## 🧠 12. Conversation Memory & Session Management

Agentomatic provides **automatic conversation memory** for all deployed agents. When a thread store is configured, every `/chat` and `/invoke` call automatically:

1. **Loads prior conversation history** into the agent's `messages` state
2. **Invokes the agent** with full conversational context
3. **Persists** both user and assistant messages to the store
4. **Summarises** older messages when the conversation grows long

```
          Frontend                    Agentomatic                    Store
             │                            │                           │
             │  POST /chat                │                           │
             │  {content, thread_id}      │                           │
             │───────────────────────────>│                           │
             │                            │  get_messages(thread_id)  │
             │                            │──────────────────────────>│
             │                            │  [msg1, msg2, ...]        │
             │                            │<──────────────────────────│
             │                            │                           │
             │                            │  ── Windowing ──          │
             │                            │  ── Summarization ──      │
             │                            │                           │
             │                            │  ainvoke(state)           │
             │                            │  (state.messages =        │
             │                            │   history + current)      │
             │                            │                           │
             │                            │  add_message(user)        │
             │                            │──────────────────────────>│
             │                            │  add_message(assistant)   │
             │                            │──────────────────────────>│
             │                            │                           │
             │  {response, thread_id,     │                           │
             │   history_loaded: 12}      │                           │
             │<───────────────────────────│                           │
```

### Configuration

Memory is automatically enabled when a `store` is provided to `AgentPlatform`:

```python
from agentomatic import AgentPlatform
from agentomatic.storage import MemoryStore

platform = AgentPlatform.from_folder(
    "agents/",
    store=MemoryStore(),  # enables memory + thread management
    max_history_messages=50,  # max messages in context window
    summarize_after=30,       # summarize when exceeding this count
)
```

### Chat Request with Memory

```python
# Frontend sends a message with thread_id
POST /api/v1/my_agent/chat
{
    "content": "What's our PTO policy?",
    "thread_id": "thread_abc123",   # reuse for multi-turn
    "user_id": "user_42",
    "context": {                    # arbitrary data for agent code
        "user_role": "manager",
        "department": "engineering"
    },
    "include_history": true,        # default: true
    "max_history": 20,              # optional override
    "persist": true,                # default: true — auto-save messages
    "prompt_version": "v1"          # optional explicit prompt version
}
```

#### Override: Supply Your Own Messages

If the frontend manages its own conversation state, it can pass `messages` directly — this **skips** automatic history loading from the store:

```python
POST /api/v1/my_agent/chat
{
    "content": "Follow-up question",
    "messages": [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ],
    "persist": false    # optional: don't save to store
}
```

#### Response

The response includes all agent output fields plus conversation metadata:

```json
{
    "response": "Based on our company handbook...",
    "thread_id": "thread_abc123",
    "agent_type": "hr-agent",
    "suggestions": ["Ask about sick days", "View calendar"],
    "citations": [...],
    "steps_taken": ["search_policy_db", "generate_response"],
    "context": {"retrieved_documents": [...]},
    "duration_ms": 234.5,
    "metadata": {"prompt_version": "v1", "source": "mobile_app"},
    "history_loaded": 12
}
```

| Response Field | Description |
|----------------|-------------|
| `response` | Agent's text response |
| `thread_id` | Thread ID (auto-created if not provided) |
| `agent_type` | Agent slug identifier |
| `suggestions` | Follow-up suggestions from agent |
| `citations` | Source citations from agent |
| `steps_taken` | Processing steps the agent took |
| `context` | Context data returned by agent (RAG docs, search results, etc.) |
| `metadata` | Merged metadata (request + agent + prompt_version) |
| `history_loaded` | Number of prior messages loaded into context |
| `duration_ms` | Processing time in milliseconds |

### Windowing & Summarization

When conversations grow long, agentomatic automatically:

1. **Windows** — keeps only the latest `max_messages` in context
2. **Summarises** — compresses older messages into a summary via the LLM
3. **Prepends** — adds `[Conversation Summary]` as a system message

```python
# Example: After 50 messages, the agent receives:
state["messages"] = [
    SystemMessage("[Conversation Summary]\nUser discussed PTO policy..."),
    HumanMessage("Question 48"),
    AIMessage("Answer 48"),
    HumanMessage("Question 49"),
    AIMessage("Answer 49"),
    HumanMessage("Current question"),  # ← latest
]
```

If no LLM is available, a fallback summary is used (simple truncated concatenation).

### Using `ConversationMemoryManager` Directly

For advanced use cases, you can use the manager directly:

```python
from agentomatic.core.memory_manager import ConversationMemoryManager
from agentomatic.storage import MemoryStore

store = MemoryStore()
mgr = ConversationMemoryManager(
    store,
    max_messages=50,
    summarize_after=30,
    summary_token_target=200,
)

# Load history
thread_id = await mgr.get_or_create_thread("thread_123", "user_1", "my-agent")
messages = await mgr.load_history(thread_id, "What's the weather?")

# Save a turn
await mgr.save_turn(thread_id, "What's the weather?", "It's sunny!", agent_name="weather-bot")

# Get full conversation summary
summary = await mgr.get_conversation_summary(thread_id)
```

---

## 📋 13. Thread Management API

Agentomatic provides a full CRUD API for conversation threads.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/threads` | Create a thread explicitly |
| `GET` | `/threads` | List threads (filterable by `user_id`) |
| `GET` | `/threads/{id}` | Get a specific thread |
| `PATCH` | `/threads/{id}` | Update thread title/metadata |
| `DELETE` | `/threads/{id}` | Delete thread and all messages |
| `GET` | `/threads/{id}/messages` | Get messages with pagination |
| `DELETE` | `/threads/{id}/messages` | Clear messages (keep thread) |
| `GET` | `/threads/{id}/summary` | Generate conversation summary |
| `POST` | `/threads/{id}/fork` | Fork thread at a message index |
| `GET` | `/threads/{id}/lineage` | Get thread ancestry tree |

### Examples

```python
# Create a thread
POST /api/v1/my_agent/threads
{"user_id": "user_42", "title": "Onboarding questions"}

# Get paginated messages
GET /api/v1/my_agent/threads/thread_abc/messages?limit=20&offset=0

# Clear conversation history
DELETE /api/v1/my_agent/threads/thread_abc/messages

# Update thread title
PATCH /api/v1/my_agent/threads/thread_abc
{"title": "Updated title"}

# Get conversation summary
GET /api/v1/my_agent/threads/thread_abc/summary
# → {"thread_id": "thread_abc", "summary": "User asked about..."}
```

### Frontend Integration Guide

For frontend applications (React, Vue, mobile), here's the recommended flow:

```
1. Start Session
   POST /threads → get thread_id

2. Send Messages
   POST /chat {content, thread_id} → get response

3. Load History (on page reload)
   GET /threads/{id}/messages?limit=50 → display messages

4. Clear Conversation
   DELETE /threads/{id}/messages → reset

5. List User's Conversations
   GET /threads?user_id=xxx → sidebar list
```

All thread data is persisted in the configured store (MemoryStore for dev, SQLAlchemyStore for production).
