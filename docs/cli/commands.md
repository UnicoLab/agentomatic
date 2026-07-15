# CLI Reference

Agentomatic ships with a comprehensive CLI built on [Click](https://click.palletsprojects.com/) with [Rich](https://rich.readthedocs.io/) terminal output. Every stage of the agent lifecycle — scaffolding, running, testing, debugging, inspecting, and optimizing — is accessible from the command line.

```bash
agentomatic --help
```

```text
Usage: agentomatic [OPTIONS] COMMAND [ARGS]...

  ⚡ Agentomatic — Drop agents, not code.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  init          Scaffold a new agent from a template.
  new           Scaffold a full project (alias for init --project).
  run           Start the platform server.
  deploy        Generate Dockerfile/compose/.env (full|minimal profiles).
  list          List discovered agents.
  test          Interactive agent testing in the console.
  inspect       Show agent folder structure and configs.
  doctor        Verify environment health and packages.
  demo          Launch a quick demo with Studio.
  ui            Launch the Chainlit debug UI.
  optimize      Run automatic prompt tuning loops.
  agents-guide  Emit an agent primer for a target project.
  stack         Manage environment stacks.
  pipeline      Manage and execute pipelines.
```

!!! tip "Rich terminal output"
    Install the `cli` extra for beautiful Rich tables, trees, and panels:
    ```bash
    pip install agentomatic[cli]
    ```
    Without Rich, the CLI falls back to plain text output.

---

## Command Categories

| Category | Commands | Purpose |
|----------|----------|---------|
| **Scaffold** | [`init`](#agentomatic-init) | Create new agents from templates |
| **Run** | [`run`](#agentomatic-run), [`demo`](#agentomatic-demo) | Start the platform server |
| **Debug** | [`test`](#agentomatic-test), [`ui`](#agentomatic-ui) | Interactive testing and chat UI |
| **Inspect** | [`list`](#agentomatic-list), [`inspect`](#agentomatic-inspect), [`doctor`](#agentomatic-doctor) | Discover, validate, and diagnose |
| **Optimize** | [`optimize`](#agentomatic-optimize) | Automatic prompt tuning |
| **Deploy** | [`deploy`](#agentomatic-deploy) | Generate container artefacts (full/minimal profiles) |
| **Agents** | [`agents-guide`](#agentomatic-agents-guide) | Bootstrap a coding agent with an Agentomatic primer |
| **Stacks** | [`stack init`](#agentomatic-stack-init), [`stack list`](#agentomatic-stack-list) | Multi-environment stack management |

---

## Scaffold Commands

### `agentomatic init`

Scaffold a new agent directory from one of the pre-built templates.

```text
Usage: agentomatic init [OPTIONS] NAME

Arguments:
  NAME  Agent name (snake_case)  [required]

Options:
  -t, --template [basic|full|rag|chatbot|deepagent|custom|legacy_dict|plugin]
                          Template to use (interactive picker if omitted)
  -d, --dir TEXT          Agents parent directory  [default: agents]
  -f, --force             Overwrite existing agent directory
```

#### Templates

| Template | Description |
|----------|-------------|
| `basic` | Minimal class-based agent (recommended) — quick start |
| `chatbot` | Conversational class-based agent with memory |
| `rag` | RAG class-based agent — retrieve → generate pipeline |
| `full` | All features: class agent with config, schemas, api, tools, prompts |
| `deepagent` | Deep Agent with planning, tools, subagents |
| `custom` | Framework-agnostic — no LangGraph dependency |
| `legacy_dict` | Legacy functional agent — `__init__.py` with `manifest` + `node_fn` |
| `plugin` | ML Model Plugin — wrap classical ML models with REST endpoints |

#### Examples

```bash
# Interactive template selection (requires questionary)
agentomatic init support_agent

# Standard scaffolding with the basic template
agentomatic init helper_bot --template basic

# RAG template in a custom directory
agentomatic init knowledge_bot --template rag --dir my_agents

# Force overwrite an existing agent
agentomatic init helper_bot --template full --force
```

#### Generated Output

```text
📁 agents/helper_bot
├── 📄 __init__.py
├── 📄 agent.py
├── 📄 llm.py
├── 📄 config.py
├── 📄 prompts.json
├── 📄 langgraph.json
├── 📄 .env.example
└── 📄 README.md

🚀 What's next?
  1. Edit helper_bot/agent.py with your logic
  2. agentomatic run to start
  3. agentomatic test helper_bot to test
  4. Open http://localhost:8000/docs for API docs
```

---

## Run Commands

### `agentomatic run`

Start the platform microservice. Runs a local `uvicorn` web server hosting the FastAPI routing stack with auto-discovered agents.

```text
Usage: agentomatic run [OPTIONS]

Options:
  --agents-dir TEXT    Agents folder to scan  [default: agents]
  --plugins-dir TEXT   Plugins folder to scan  [default: plugins]
  --host TEXT          Bind address  [default: 0.0.0.0]
  --port INTEGER       Bind port  [default: 8000]
  --reload             Auto-reload on code or config file changes
  --title TEXT         Platform title
  --log-level TEXT     Log level (DEBUG, INFO, WARNING, ERROR)  [default: INFO]
  --with-ui, --ui     Mount the Chainlit chat UI at /chat
  --studio/--no-studio Enable/disable Agentomatic Studio at /studio/ui/ (default: on)
```

#### Examples

=== "Development (Studio + reload)"

    Studio is **on by default** — no extra flag needed:

    ```bash
    agentomatic run --reload
    ```

    | Service | URL |
    |---------|-----|
    | API | `http://localhost:8000` |
    | Swagger Docs | `http://localhost:8000/docs` |
    | Studio | `http://localhost:8000/studio/ui/` |

=== "Development (Chat UI + reload)"

    ```bash
    agentomatic run --with-ui --reload
    ```

    | Service | URL |
    |---------|-----|
    | API | `http://localhost:8000` |
    | Chat UI | `http://localhost:8000/chat` |

=== "Full Debug (Both UIs)"

    ```bash
    agentomatic run --studio --with-ui --reload
    ```

=== "Production-like"

    ```bash
    agentomatic run --host 0.0.0.0 --port 9000 --log-level WARNING
    ```

=== "Custom Directory"

    ```bash
    agentomatic run --agents-dir my_agents --title "My Platform"
    ```

!!! warning "Studio requires the `studio` extra"
    ```bash
    pip install agentomatic[studio]
    ```

---

### `agentomatic demo`

Launch a quick demo with a pre-built agent and Studio enabled. Perfect for first-time exploration or demonstrations.

```text
Usage: agentomatic demo [OPTIONS]

Options:
  --port INTEGER   Bind port  [default: 8000]
```

#### Example

```bash
agentomatic demo
```

This command:

1. Scaffolds a temporary demo agent with a pre-built LangGraph workflow
2. Starts the platform with Studio enabled
3. Opens `http://localhost:8000/studio/ui/` in your browser

!!! note "Demo agents are temporary"
    The demo agent lives in a temporary directory and is cleaned up when the server stops. Use `agentomatic init` to create permanent agents.

---

## Debug & Test Commands

### `agentomatic test`

Interactive, chat-like terminal session to test agent completions and streaming directly in your shell. Requires a running platform instance.

```text
Usage: agentomatic test [OPTIONS] NAME

Arguments:
  NAME  Agent name to test  [required]

Options:
  --host TEXT         Server host address  [default: localhost]
  --port INTEGER      Server port  [default: 8000]
  --agents-dir TEXT   Agents directory  [default: agents]
```

#### Example Session

```bash
agentomatic test my_chatbot
```

```text
⚡ agentomatic
🧪 Testing agent: my_chatbot
   API: http://localhost:8000/api/v1/my_chatbot/invoke
   Type 'quit' or 'exit' to stop

🗣️  You: Hello!
🤖 my_chatbot: Hello! How can I assist you today?
   Steps: greeting_node
   ⏱ 114ms

🗣️  You: What is machine learning?
🤖 my_chatbot: Machine learning is a subset of AI that enables...
   Steps: retrieval_node → response_node
   Suggestions: Tell me more, Show examples
   ⏱ 1,204ms

🗣️  You: quit

👋 Test session ended
```

!!! info "Multi-turn conversations"
    The test session automatically tracks `thread_id` across messages, so you can test multi-turn conversations with full context persistence.

---

### `agentomatic ui`

Launch the Chainlit debug Chat console as a standalone process (points to a running backend application).

```text
Usage: agentomatic ui [OPTIONS]

Options:
  --host TEXT         FastAPI server address  [default: localhost]
  --port INTEGER      FastAPI server port  [default: 8000]
  --ui-port INTEGER   Chainlit UI port to bind  [default: 8001]
```

#### Example

```bash
# Start the platform
agentomatic run

# In a separate terminal, launch the Chat UI
agentomatic ui --port 8000 --ui-port 9000
```

!!! tip "Prefer embedded mode"
    For most use cases, `agentomatic run --with-ui` is simpler than running the UI standalone. Use standalone mode when you need the UI on a separate port or host.

---

## Inspect & Diagnose Commands

### `agentomatic list`

Scans your agents folder and prints a rich table of discovered agent packages and their statuses.

```text
Usage: agentomatic list [OPTIONS]

Options:
  --agents-dir TEXT   Agents folder to scan  [default: agents]
```

#### Example Output

```text
╭────────────────────── 🤖 Agents in agents ──────────────────────╮
│ Name          │ Files │ Manifest │ Graph │
├───────────────┼───────┼──────────┼───────┤
│ my_chatbot    │ 8     │ ✅       │ ✅    │
│ rag_agent     │ 9     │ ✅       │ ✅    │
│ classifier    │ 4     │ ✅       │ —     │
╰───────────────┴───────┴──────────┴───────╯

   Total: 3 agent(s)
```

---

### `agentomatic inspect`

Validates an agent package structure, reads manifest properties, and displays present/absent files with their contents.

```text
Usage: agentomatic inspect [OPTIONS] NAME

Arguments:
  NAME  Agent name to inspect  [required]

Options:
  --agents-dir TEXT   Agents parent folder  [default: agents]
```

#### Example Output

```text
╭────────────── 🔍 Agent Inspector ──────────────╮
│ my_chatbot                                      │
│ agents/my_chatbot                               │
╰─────────────────────────────────────────────────╯

📁 my_chatbot/
├── 📄 __init__.py (1,245 bytes)
├── 📄 agent.py (2,891 bytes)
├── 📄 llm.py (890 bytes)
├── 📄 config.py (512 bytes)
├── 📄 prompts.json (678 bytes)
├── 📄 tools.py (1,024 bytes)
├── 📄 .env.example (156 bytes)
└── 📄 README.md (890 bytes)

╭── __init__.py ──────────────────────────────────╮
│ from agentomatic import AgentManifest           │
│ from .agent import MyChatbotAgent               │
│                                                 │
│ manifest = AgentManifest(                       │
│     name="my_chatbot",                          │
│     slug="my-chatbot",                          │
│     framework="langgraph",                      │
│     ...                                         │
│ )                                               │
│ graph_fn = get_graph                            │
╰─────────────────────────────────────────────────╯

╭── prompts.json (2 versions) ────────────────────╮
│ {                                               │
│   "v1": { "system": "You are a concise..." },   │
│   "v2": { "system": "You are a detailed..." }   │
│ }                                               │
╰─────────────────────────────────────────────────╯
```

---

### `agentomatic doctor`

Diagnostic tool that audits your Python environment, installed packages, optional extras, and external service connections.

```text
Usage: agentomatic doctor [OPTIONS]

Options:
  --agents-dir TEXT   Agents folder to inspect  [default: agents]
```

#### Example Output

```text
╭──────────────── 🩺 Environment Health Check ────────────────╮
│ Component              │ Status │ Details                    │
├────────────────────────┼────────┼────────────────────────────┤
│ Python                 │ ✅     │ 3.12.0                     │
│ fastapi                │ ✅     │ 0.115.x                    │
│ uvicorn                │ ✅     │ 0.34.x                     │
│ pydantic               │ ✅     │ 2.10.x                     │
│ loguru                 │ ✅     │ 0.7.x                      │
│ httpx                  │ ✅     │ 0.28.x                     │
│ langgraph [langgraph]  │ ✅     │ 0.4.x                      │
│ langchain_core [lc]    │ ✅     │ 0.3.x                      │
│ rich [cli]             │ ✅     │ 13.x                       │
│ chainlit [ui]          │ ✅     │ 2.0.x                      │
│ sqlalchemy [db]        │ ✅     │ 2.0.x                      │
│ prometheus [metrics]   │ ✅     │ 0.21.x                     │
│ Agents directory       │ ✅     │ 3 agent(s) in agents       │
╰────────────────────────┴────────┴────────────────────────────╯

✅ All core dependencies satisfied!
```

!!! tip "Run doctor first"
    If anything is not working, `agentomatic doctor` is always the first diagnostic step. It identifies missing packages, incompatible versions, and unreachable services.

---

## Optimize Commands

### `agentomatic optimize`

Execute the DSPy-inspired prompt optimization pipeline. Evaluates your agent's prompts against a dataset, iteratively rewrites them, and saves the best-performing version.

```text
Usage: agentomatic optimize [OPTIONS] AGENT

Arguments:
  AGENT  Agent name to optimize  [required]

Options:
  -d, --dataset TEXT       Path to evaluation dataset (JSONL/CSV)  [required]
  -m, --metrics TEXT       Comma-separated metrics  [default: exact_match]
  -s, --strategy TEXT      Optimization strategy  [default: iterative_rewrite]
                           Choices: iterative_rewrite, few_shot, chain_of_thought
  --max-iterations INT     Maximum optimization steps  [default: 10]
  --target-score FLOAT     Stop when this avg score is reached  [default: 0.9]
  --rewrite-llm TEXT       LLM for prompt rewriting
  --eval-llm TEXT          LLM for evaluation grading
  --llm TEXT               Default fallback LLM (env: AGENTOMATIC_TASK_MODEL /
                           LLM__MODEL; else ollama/mistral:7b)
  --patience INT           Early stopping patience  [default: 3]
  --prompt TEXT            Initial system prompt (overrides prompts.json)
  --no-report              Skip generating HTML report
  --apply                  Auto-save the best-performing prompt
  --host TEXT              Platform API base URL  [default: http://localhost:8000]
```

#### Optimization Strategies

| Strategy | Description |
|----------|-------------|
| `iterative_rewrite` | LLM rewrites the prompt based on failure analysis each iteration |
| `few_shot` | Bootstraps few-shot examples from successful evaluation pairs |
| `chain_of_thought` | Adds chain-of-thought reasoning steps to the prompt |

#### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | Response exactly matches expected output |
| `contains` | Response contains the expected substring |
| `relevancy` | LLM-graded answer relevancy (requires eval LLM) |
| `faithfulness` | LLM-graded factual faithfulness |
| `completeness` | LLM-graded answer completeness |
| `coherence` | LLM-graded response coherence |
| `toxicity` | LLM-graded toxic speech detection |
| `bias` | LLM-graded bias detection |

#### Examples

=== "Basic Optimization"

    ```bash
    agentomatic optimize my_chatbot \
      --dataset eval_qa.jsonl \
      --metrics exact_match,contains \
      --apply
    ```

=== "Advanced with Custom LLMs"

    ```bash
    agentomatic optimize rag_agent \
      --dataset eval_rag.jsonl \
      --metrics relevancy,faithfulness,completeness \
      --strategy iterative_rewrite \
      --max-iterations 20 \
      --target-score 0.95 \
      --rewrite-llm openai/gpt-4o \
      --eval-llm openai/gpt-4o-mini \
      --patience 5 \
      --apply
    ```

=== "Few-Shot Bootstrap"

    ```bash
    agentomatic optimize support_bot \
      --dataset support_tickets.csv \
      --metrics relevancy,coherence \
      --strategy few_shot \
      --max-iterations 5 \
      --apply
    ```

#### Dataset Format

=== "JSONL"

    ```jsonl
    {"query": "What is Python?", "expected": "Python is a programming language"}
    {"query": "Explain REST APIs", "expected": "REST is an architectural style"}
    ```

=== "CSV"

    ```csv
    query,expected
    "What is Python?","Python is a programming language"
    "Explain REST APIs","REST is an architectural style"
    ```

!!! info "HTML Reports"
    By default, each optimization run generates an interactive HTML report in `optimization_reports/` showing score progression, prompt diffs, and per-sample results. Disable with `--no-report`.

---

## Stack Commands

### `agentomatic stack init`

Create default stack configuration files for multi-environment management.

```text
Usage: agentomatic stack init [OPTIONS]

Options:
  -d, --dir TEXT   Stacks directory  [default: stacks]
```

#### Example

```bash
agentomatic stack init
```

This creates default `local.yaml` and `remote.yaml` stack files in the `stacks/` directory.

---

### `agentomatic stack list`

List all available stack configurations and show the currently active stack.

```text
Usage: agentomatic stack list [OPTIONS]

Options:
  -d, --dir TEXT   Stacks directory  [default: stacks]
```

#### Example Output

```bash
agentomatic stack list
```

```text
╭─────────────── 📚 Stacks ───────────────╮
│ Name    │ Active │
├─────────┼────────┤
│ local   │ ✅     │
│ remote  │        │
╰─────────┴────────╯
```

---

## Deploy Commands

### `agentomatic deploy`

Generate production container artefacts (Dockerfile, `docker-compose.yml`,
`.env.example`, optional `nginx.conf`) that run the project via `uvicorn main:app`.

```text
Usage: agentomatic deploy [OPTIONS]

Options:
  --stack TEXT              Stack name to derive env from.
  --distroless              Emit Dockerfile.distroless (minimal attack surface).
  --profile [full|minimal]  Deploy profile  [default: full]
  --minimal                 Shorthand for --profile minimal.
  --out TEXT                Output directory  [default: deploy/generated]
  --with-nginx/--no-nginx   Emit an nginx.conf reverse proxy template.
  --with-agent-stubs        Emit one compose service per discovered agent.
```

**Profiles**

| | `full` (default) | `minimal` |
| --- | :---: | :---: |
| REST API, health, metrics, auth | ✅ | ✅ |
| **Swagger** (`/docs`, `/redoc`, `/openapi.json`) | ✅ | ✅ *(always on)* |
| Studio UI (`/studio/ui`) | ✅ | ❌ |
| Verbose logging | `INFO` | `WARNING` |

Both profiles drive the **same** env-driven `main.py`; `minimal` just bakes
`AGENTOMATIC_ENABLE_STUDIO=0` + `AGENTOMATIC_LOG_LEVEL=WARNING` into the image and
compose. See the [deployment guide](../guide/deployment.md) for details.

```bash
agentomatic deploy --stack remote --distroless          # full
agentomatic deploy --profile minimal --stack remote     # production-lean
agentomatic deploy --minimal --stack remote             # shorthand
```

---

## Agent Bootstrap Commands

### `agentomatic agents-guide`

Emit an Agentomatic primer so any coding agent (Cursor, Claude, etc.) can be
bootstrapped in a target project. The content comes from a single source of truth
(`agentomatic.cli.agent_guide`) so it stays in sync with the platform.

```text
Usage: agentomatic agents-guide [OPTIONS]

Options:
  --write [AGENTS.md|CLAUDE.md|.cursor/skills/agentomatic/SKILL.md]
                Write the primer into the current project at the given path.
  -f, --force   Overwrite the target file if it already exists.
```

```bash
# Print the primer to stdout
agentomatic agents-guide

# Write it into a project (refuses to overwrite without --force)
agentomatic agents-guide --write AGENTS.md
agentomatic agents-guide --write CLAUDE.md --force
agentomatic agents-guide --write .cursor/skills/agentomatic/SKILL.md
```

The primer covers what Agentomatic is, the install, the develop → optimize →
deploy loop, key CLI commands, deploy profiles, the `AGENTOMATIC_*` env vars, the
class-agent flow, and the provider-agnostic principles.

---

## Common Workflows

### Development Workflow

```bash
# 1. Scaffold a new agent
agentomatic init my_agent --template full

# 2. Check environment is healthy
agentomatic doctor

# 3. Start with Studio for visual debugging
agentomatic run --studio --reload

# 4. Test interactively from another terminal
agentomatic test my_agent

# 5. Inspect agent structure
agentomatic inspect my_agent
```

### Prompt Optimization Workflow

```bash
# 1. Start the platform
agentomatic run

# 2. Run optimization (in another terminal)
agentomatic optimize my_agent \
  --dataset eval_data.jsonl \
  --metrics relevancy,faithfulness \
  --strategy iterative_rewrite \
  --max-iterations 15 \
  --apply

# 3. Verify the new prompt in the Chat UI
agentomatic run --with-ui
```

### Production Deployment

```bash
# Minimal production server (no debug UIs)
agentomatic run --host 0.0.0.0 --port 8000 --log-level WARNING
```
