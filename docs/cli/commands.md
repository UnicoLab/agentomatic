# CLI Commands

Agentomatic ships with a **click-based CLI** for scaffolding, running, testing, and inspecting agents.

!!! tip "Install CLI extras for the best experience"
    ```bash
    pip install agentomatic[cli]   # Rich tables + interactive prompts
    ```

---

## `agentomatic --help`

```
Usage: agentomatic [OPTIONS] COMMAND [ARGS]...

  Agentomatic — Drop agents, not code ⚡

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  init     Scaffold a new agent from a template.
  run      Start the platform server.
  list     List discovered agents.
  test     Interactive agent testing.
  inspect  Show agent structure and configuration.
  doctor   Check environment health.
  ui       Launch the Chainlit debug UI.
```

---

## `agentomatic init`

Scaffold a new agent from a template.

```
Usage: agentomatic init [OPTIONS] NAME

  Scaffold a new agent from a template.

Arguments:
  NAME  Agent name (snake_case)  [required]

Options:
  -t, --template [basic|full|rag|chatbot|custom]
                          Template to use (interactive if omitted)
  --dir TEXT              Agents directory  [default: agents]
  -f, --force             Overwrite existing agent
  --help                  Show this message and exit.
```

**Examples:**

```bash
# Interactive template selection (requires questionary)
agentomatic init my_agent

# Non-interactive with specific template
agentomatic init hr_bot --template rag

# Force overwrite existing agent
agentomatic init hr_bot --template full --force

# Custom agents directory
agentomatic init my_agent --template basic --dir src/agents
```

---

## `agentomatic run`

Start the platform server with uvicorn.

```
Usage: agentomatic run [OPTIONS]

  Start the platform server.

Options:
  --agents-dir TEXT  Agents directory  [default: agents]
  --host TEXT        Bind address  [default: 0.0.0.0]
  --port INTEGER     Bind port  [default: 8000]
  --reload           Auto-reload on file changes
  --with-ui          Enable Chainlit debug UI at /chat
  --help             Show this message and exit.
```

**Examples:**

```bash
# Basic start
agentomatic run

# Development with auto-reload + debug UI
agentomatic run --reload --with-ui

# Custom host and port
agentomatic run --host 127.0.0.1 --port 9000

# Different agents directory
agentomatic run --agents-dir src/agents
```

---

## `agentomatic list`

List all discovered agents in a rich table.

```
Usage: agentomatic list [OPTIONS]

  List discovered agents.

Options:
  --agents-dir TEXT  Agents directory  [default: agents]
  --help             Show this message and exit.
```

**Example output:**

```
╭────────────────────── Discovered Agents ──────────────────────╮
│ Name         │ Framework  │ Version │ Endpoints │ Status      │
├──────────────┼────────────┼─────────┼───────────┼─────────────┤
│ hr_bot       │ langgraph  │ 1.0.0   │ 12        │ ✅ Ready    │
│ rag_agent    │ langgraph  │ 1.0.0   │ 12        │ ✅ Ready    │
│ classifier   │ custom     │ 0.1.0   │ 12        │ ✅ Ready    │
╰──────────────┴────────────┴─────────┴───────────┴─────────────╯
```

---

## `agentomatic test`

Interactive agent testing in the terminal. Sends queries and displays responses in real-time.

```
Usage: agentomatic test [OPTIONS] NAME

  Interactive agent testing.

Arguments:
  NAME  Agent name to test  [required]

Options:
  --host TEXT      Server host  [default: localhost]
  --port INTEGER   Server port  [default: 8000]
  --help           Show this message and exit.
```

**Example:**

```bash
agentomatic test hr_bot
# 🤖 Connected to hr_bot at http://localhost:8000
# You: What is our PTO policy?
# hr_bot: Our PTO policy provides 25 days per year...
# You: /quit
```

---

## `agentomatic inspect`

Show detailed agent structure, manifest fields, and configuration.

```
Usage: agentomatic inspect [OPTIONS] NAME

  Show agent structure and configuration.

Arguments:
  NAME  Agent name to inspect  [required]

Options:
  --agents-dir TEXT  Agents directory  [default: agents]
  --help             Show this message and exit.
```

**Example output:**

```
╭──────────────────── Agent: hr_bot ────────────────────╮
│ Manifest                                              │
│   name:        hr_bot                                 │
│   slug:        hr-bot                                 │
│   framework:   langgraph                              │
│   version:     1.0.0                                  │
│   keywords:    hr, policy, benefits                   │
│                                                       │
│ Files                                                 │
│   ✅ __init__.py    (manifest + entry)                │
│   ✅ graph.py       (state graph)                     │
│   ✅ nodes.py       (processing logic)                │
│   ✅ config.py      (pydantic config)                 │
│   ✅ prompts.json   (2 versions: v1, v2)              │
│   ⬚  api.py        (not present — using auto-gen)    │
╰───────────────────────────────────────────────────────╯
```

---

## `agentomatic doctor`

Check environment health — Python version, installed extras, connectivity.

```
Usage: agentomatic doctor [OPTIONS]

  Check environment health.

Options:
  --agents-dir TEXT  Agents directory  [default: agents]
  --help             Show this message and exit.
```

**Example output:**

```
╭──────────────────── Agentomatic Doctor ────────────────────╮
│ ✅ Python         3.12.0                                   │
│ ✅ agentomatic    0.1.0                                    │
│ ✅ FastAPI        0.115.x                                  │
│ ✅ click          8.1.x                                    │
│ ✅ LangGraph      0.4.x                                   │
│ ✅ Ollama         connected                                │
│ ⬚  OpenAI        not installed                             │
│ ✅ Prometheus     0.21.x                                   │
│ ✅ SQLAlchemy     2.0.x                                    │
│ ✅ Chainlit       2.0.x                                    │
│ ✅ DeepEval       2.0.x                                    │
│ ✅ OpenTelemetry  1.20.x                                   │
╰────────────────────────────────────────────────────────────╯
```

---

## `agentomatic ui`

Launch the Chainlit debug UI as a standalone process.

```
Usage: agentomatic ui [OPTIONS]

  Launch the Chainlit debug UI.

Options:
  --host TEXT        API server host  [default: localhost]
  --port INTEGER     API server port  [default: 8000]
  --ui-port INTEGER  Chainlit UI port  [default: 8001]
  --help             Show this message and exit.
```

**Example:**

```bash
# Launch UI pointing to running platform
agentomatic ui
# → Chainlit UI at http://localhost:8001
# → Connecting to API at http://localhost:8000

# Custom ports
agentomatic ui --port 9000 --ui-port 9001
```

!!! note "Alternative: embedded mode"
    You can also run the UI embedded in the platform server:
    ```bash
    agentomatic run --with-ui
    # → UI available at http://localhost:8000/chat
    ```
