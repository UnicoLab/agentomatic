# Chat Interface (Chainlit)

Agentomatic includes a built-in Chainlit **debug chat**. It lets an operator
select a deployed agent, send a synchronous invoke request, and inspect the
returned response, execution-step labels, citations, and suggestions — without
writing a frontend.

!!! info "Chat UI vs Agentomatic Studio"
    Agentomatic offers **two** debug interfaces for different workflows:

    | | Chat Interface (this page) | [Agentomatic Studio](studio.md) |
    |---|---|---|
    | **Purpose** | Conversational testing | Visual debugging & inspection |
    | **Launch flag** | `--with-ui` | `--studio` |
    | **URL** | `/chat` | `/studio/ui/` |
    | **Best for** | Quick local response checks | Graph visualization, state inspection, time-travel, breakpoints |
    | **Interface** | Chat bubbles (ChatGPT-like) | Node graph + debug panels |
    | **Framework** | Chainlit | React |

    **Use the Chat Interface when** you want to have a conversation with your agent and evaluate response quality. **Use [Studio](studio.md) when** you need to debug execution flow, inspect state, or trace node-by-node behavior.

---

## Installation & Launch

### Install the UI Extra

```bash
pip install "agentomatic[ui]"
```

### Launch Modes

=== "Embedded Mode (Recommended)"

    Mounts the Chainlit interface into the FastAPI application. The debug chat
    invokes the platform's normal `/invoke` API, so that API request follows
    the platform middleware.

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

!!! warning "Local debug surface, not an authenticated operator console"
    The bundled UI currently sends plain `/invoke` requests and does not
    forward an API key, JWT, browser identity, or a conversation `thread_id`.
    It is intended for a local or otherwise trusted debugging setup. For a
    production frontend, use the documented REST/Studio APIs and your own
    authentication and session handling.

---

## Interface Features

### :material-robot: Agent Selector

A Chainlit settings selector lists agents returned by the platform registry.
Select an agent to send subsequent messages to that agent's `/invoke` route.

### :material-hammer-wrench: Invocation inspection

Each message creates an expandable Chainlit step containing the `/invoke`
payload and complete JSON response. If the response contains `steps_taken`,
the UI displays those agent-supplied execution labels in a second step. It does
not stream `/invoke/stream`, inspect provider tool calls independently, or
infer hidden chain-of-thought.

### :material-book-open-variant: Citations & Sources

Citations returned by the agent are displayed as an expandable JSON retrieval
step. The bundled UI does not submit feedback; use the agent feedback API or a
production client for that workflow.

---

## Customization

### Theme & Layout

Chainlit controls its own theme and layout files. If your project has a
`.chainlit/config.toml`, customize it according to your installed Chainlit
version; Agentomatic does not generate or manage that file.

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

The bundled debug chat sends its welcome message from
`agentomatic.ui.chat`; customize that module or provide a dedicated Chainlit
application when you need branded copy and behavior.

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
| Comparing prompt versions side-by-side | | ✅ (use a run payload) |
| Collecting user feedback | | ✅ (use the agent feedback API) |
| Demonstrating agents to stakeholders | ✅ | |
| Debugging graph execution flow | | ✅ |
| Inspecting intermediate node state | | ✅ |
| Time-travel debugging (replay from checkpoint) | | ✅ |
| Setting breakpoints on nodes | | ✅ |
| Live state editing during execution | | ✅ |

!!! tip "Recommendation"
    For **development and debugging**, use [Agentomatic Studio](studio.md). For **testing and evaluation**, use the Chat Interface. Both can run simultaneously with `--with-ui --studio`.
