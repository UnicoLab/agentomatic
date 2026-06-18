# Chat Interface (Chainlit)

Agentomatic includes a built-in ChatGPT-like conversational interface powered by **Chainlit**. It provides an interactive chat playground to test agent responses, compare prompt versions, inspect tool calls, and collect user feedback — without writing any frontend code.

!!! info "Chat UI vs Agentomatic Studio"
    Agentomatic offers **two** debug interfaces for different workflows:

    | | Chat Interface (this page) | [Agentomatic Studio](studio.md) |
    |---|---|---|
    | **Purpose** | Conversational testing | Visual debugging & inspection |
    | **Launch flag** | `--with-ui` | `--studio` |
    | **URL** | `/chat` | `/studio/ui/` |
    | **Best for** | Testing agent responses, prompt A/B testing, user feedback | Graph visualization, state inspection, time-travel, breakpoints |
    | **Interface** | Chat bubbles (ChatGPT-like) | Node graph + debug panels |
    | **Framework** | Chainlit | React |

    **Use the Chat Interface when** you want to have a conversation with your agent and evaluate response quality. **Use [Studio](studio.md) when** you need to debug execution flow, inspect state, or trace node-by-node behavior.

---

## Installation & Launch

### Install the UI Extra

```bash
pip install agentomatic[ui]
```

### Launch Modes

=== "Embedded Mode (Recommended)"

    Mounts the Chainlit interface directly into your FastAPI application. API requests and chat sessions share the same persistence backend and middleware stack.

    ```bash
    agentomatic run --with-ui
    ```

    - **Platform API**: `http://localhost:8000`
    - **Chat UI**: `http://localhost:8000/chat`
    - **API Docs**: `http://localhost:8000/docs`

=== "Standalone Mode"

    Launches the chat console as a separate process, pointing to an already-running platform backend.

    ```bash
    # Start the platform first
    agentomatic run

    # Then launch the UI separately
    agentomatic ui --port 8000 --ui-port 8001
    ```

    - **Chat UI**: `http://localhost:8001`

=== "Combined with Studio"

    Run both debug interfaces simultaneously:

    ```bash
    agentomatic run --with-ui --studio --reload
    ```

    - **Chat UI**: `http://localhost:8000/chat`
    - **Studio**: `http://localhost:8000/studio/ui/`

!!! tip "Development workflow"
    During development, combine `--with-ui` with `--reload` for live reloading:
    ```bash
    agentomatic run --with-ui --reload
    ```

---

## Interface Features

### :material-robot: Agent Selector

A top-navigation dropdown lists all registered agents discovered by the platform registry. Select an agent to dynamically load its input form, configuration, and documentation.

### :material-tag: Prompt Version Selector

Inspect and switch between prompt versions (e.g., `v1`, `v2`, `v1_formal`) on the fly. Chat queries execute against the selected version, enabling manual A/B comparison of prompt behaviors.

```json
// agents/my_agent/prompts.json
{
  "v1": {
    "system": "You are a concise assistant.",
    "user_template": "Query: {query}"
  },
  "v2": {
    "system": "You are a creative, detailed assistant.",
    "user_template": "Please elaborate on: {query}"
  }
}
```

### :material-waves: Token-by-Token Streaming

If your agent supports streaming (via SSE), response completions stream onto the screen in real-time, matching the experience of ChatGPT and similar interfaces.

### :material-hammer-wrench: Tool Call Visualizations

Intermediate agent actions — tool calls, function invocations, retrieval steps — are captured and rendered as clean, expandable panels in the chat flow. Click a panel to inspect the exact input arguments and JSON output returned by the tool.

### :material-brain: Chain-of-Thought & Reasoning

If your agent returns reasoning or step-by-step logs, the UI highlights these in collapsible cards showing the agent's thought process before the final answer.

### :material-book-open-variant: Citations & Sources

Citations returned by RAG pipelines are rendered as clickable badges at the bottom of messages, referencing PDFs, web links, or documentation files.

### :material-thumb-up: User Feedback Collection

Every response includes thumbs-up and thumbs-down icons. Users can submit rating scores and commentary directly from the UI. Feedback is:

- Immediately saved to the platform's database (SQL or Memory)
- Available via the `/api/v1/{agent}/feedback` endpoint
- Exportable as training data for prompt optimization

!!! note "Feedback-driven optimization"
    Feedback collected through the Chat UI can be exported and used as evaluation datasets for the [Prompt Optimization](optimization.md) pipeline:
    ```bash
    agentomatic optimize my_agent --dataset feedback_export.jsonl --metrics relevancy
    ```

---

## Customization

### Theme & Layout

When running `agentomatic run --with-ui` for the first time, Agentomatic generates a default `.chainlit/config.toml` file. Customize it to match your brand:

```toml
[theme]
# Custom brand colors
primary = "#7c3aed"          # Deep purple (matches Agentomatic theme)
background = "#1a202c"       # Dark background
paper = "#2d3748"            # Card backgrounds
font_family = "Inter, sans-serif"

[UI]
name = "My AI Assistant"     # Title shown in the header
show_readme = false          # Hide the README panel
default_expand_messages = true
```

### Custom Welcome Message

Edit the `.chainlit/README.md` file to customize the welcome screen shown when users open a new session:

```markdown
# Welcome to My Agent Platform 🚀

Select an agent from the dropdown above and start chatting.

**Available agents:**
- **Support Bot** — Answer customer questions
- **Code Assistant** — Help with programming tasks
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAINLIT_AUTH_SECRET` | — | Secret for session authentication |
| `AGENTOMATIC_API_URL` | `http://localhost:8000` | Backend API URL (standalone mode) |

---

## Programmatic Integration

You can also mount the Chat UI programmatically from Python:

```python
from agentomatic import AgentPlatform

platform = AgentPlatform.from_folder("agents/")
app = platform.build()

# Mount Chainlit UI
from agentomatic.ui import mount
mount(app)  # Chat UI available at /chat
```

---

## When to Use Chat UI vs Studio

| Scenario | Use Chat UI | Use Studio |
|----------|:-----------:|:----------:|
| Testing agent response quality | ✅ | |
| Comparing prompt versions side-by-side | ✅ | |
| Collecting user feedback | ✅ | |
| Demonstrating agents to stakeholders | ✅ | |
| Debugging graph execution flow | | ✅ |
| Inspecting intermediate node state | | ✅ |
| Time-travel debugging (replay from checkpoint) | | ✅ |
| Setting breakpoints on nodes | | ✅ |
| Live state editing during execution | | ✅ |

!!! tip "Recommendation"
    For **development and debugging**, use [Agentomatic Studio](studio.md). For **testing and evaluation**, use the Chat Interface. Both can run simultaneously with `--with-ui --studio`.
