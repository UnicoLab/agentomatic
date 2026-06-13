# CLI Reference

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Interactive Command-Line Toolchain</h3>
</div>

---

Agentomatic ships with a robust, click-based CLI for scaffolding, serving, listing, inspecting, testing, and optimizing your agents.

---

## ⚡ Global Help Overview

```bash
agentomatic --help
```

```text
Usage: agentomatic [OPTIONS] COMMAND [ARGS]...

  Agentomatic — Drop agents, not code ⚡

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  init      Scaffold a new agent from a template.
  run       Start the platform server.
  list      List discovered agents.
  test      Interactive agent testing in the console.
  inspect   Show agent folder structure and configs.
  doctor    Verify environment health and packages.
  ui        Launch the Chainlit debug UI.
  optimize  Run automatic prompt tuning loops.
```

---

## 🏗️ `agentomatic init`

Scaffold a new agent directory from one of our 5 pre-built conventions.

### Usage
```text
Usage: agentomatic init [OPTIONS] NAME

Arguments:
  NAME  Agent name (snake_case)  [required]

Options:
  -t, --template [basic|full|rag|chatbot|custom]
                          Template to use (interactive list if omitted)
  --dir TEXT              Agents parent directory  [default: agents]
  -f, --force             Overwrite existing agent directory
```

### Examples
```bash
# Select templates interactively (requires pip install questionary)
agentomatic init support_agent

# Standard scaffolding (basic template)
agentomatic init helper_bot --template basic

# RAG template forced overwrite
agentomatic init knowledge_bot --template rag --force
```

---

## 🚀 `agentomatic run`

Start the platform microservice. Runs a local `uvicorn` web server hosting the FastAPI routing stack.

### Usage
```text
Usage: agentomatic run [OPTIONS]

Options:
  --agents-dir TEXT  Agents folder to scan  [default: agents]
  --host TEXT        Bind address  [default: 0.0.0.0]
  --port INTEGER     Bind port  [default: 8000]
  --reload           Auto-reload on code or config file changes
  --with-ui          Mount the Chainlit chat UI at /chat
```

### Examples
```bash
# Standard launch
agentomatic run

# Dev launch (live reload + debugging Chat UI at http://localhost:8000/chat)
agentomatic run --reload --with-ui

# Bound to specific port
agentomatic run --port 9000
```

---

## 📊 `agentomatic list`

Scans your agents folder and prints a rich table of discovered agent packages and their statuses.

### Usage
```text
Usage: agentomatic list [OPTIONS]

Options:
  --agents-dir TEXT  Agents folder to scan  [default: agents]
```

### Example Output
```text
╭────────────────────── Discovered Agents ──────────────────────╮
│ Name         │ Framework  │ Version │ Endpoints │ Status      │
├──────────────┼────────────┼─────────┼───────────┼─────────────┤
│ hr_bot       │ langgraph  │ 1.0.0   │ 12        │ Ready       │
│ rag_agent    │ langgraph  │ 1.0.0   │ 12        │ Ready       │
│ classifier   │ custom     │ 0.1.0   │ 12        │ Ready       │
╰──────────────┴────────────┴─────────┴───────────┴─────────────╯
```

---

## 🔍 `agentomatic inspect`

Validates your agent package structure, reading manifest properties and listing present/absent conventions.

### Usage
```text
Usage: agentomatic inspect [OPTIONS] NAME

Arguments:
  NAME  Agent name to inspect  [required]

Options:
  --agents-dir TEXT  Agents parent folder  [default: agents]
```

### Example Output
```text
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

## 💬 `agentomatic test`

Interactive, chat-like terminal interface to test agent completions and streaming directly inside your shell.

### Usage
```text
Usage: agentomatic test [OPTIONS] NAME

Arguments:
  NAME  Agent name to test  [required]

Options:
  --host TEXT      Server host address  [default: localhost]
  --port INTEGER   Server host port  [default: 8000]
```

### Example Session
```bash
agentomatic test hr_bot
# 🤖 Connected to hr_bot at http://localhost:8000
# You: Hello!
# hr_bot: Hello! How can I assist you with HR policies today?
# You: /quit
```

---

## 🎯 `agentomatic optimize`

Executes the prompt optimization tuning loop over a dataset, evaluating outputs and saving improvements.

### Usage
```text
Usage: agentomatic optimize [OPTIONS] AGENT

Arguments:
  AGENT  Agent name to optimize  [required]

Options:
  -d, --dataset TEXT      Path to evaluation dataset (JSONL/CSV)  [required]
  -m, --metrics TEXT      Comma-separated metrics (e.g., exact_match,contains)  [required]
  -s, --strategy [iterative_rewrite|few_shot|chain_of_thought]
                          Optimization strategy  [default: iterative_rewrite]
  --max-iterations INTEGER
                          Maximum optimization steps  [default: 10]
  --target-score FLOAT    Stop optimization once this average score is hit  [default: 0.9]
  --rewrite-llm TEXT      Override LLM used to rewrite prompt instructions
  --eval-llm TEXT         Override LLM used to grade responses
  --llm TEXT              Default LLM to fallback on  [default: ollama/mistral:7b]
  --patience INTEGER      Stop early if score doesn't improve for N steps  [default: 3]
  --prompt TEXT           Initial system prompt override (bypasses prompts.json)
  --no-report             Skip generating interactive HTML report logs
  --apply                 Automatically save and promote the best-performing prompt
  --host TEXT             Platform API base host url  [default: http://localhost:8000]
```

### Examples
```bash
# Run basic optimization loop and auto-save the best prompt
agentomatic optimize hr_bot \
  --dataset eval_qa.jsonl \
  --metrics exact_match,contains \
  --strategy iterative_rewrite \
  --max-iterations 10 \
  --apply
```

---

## 🩺 `agentomatic doctor`

Diagnostic diagnostic tool that audits your local Python virtualenv, installs, dependencies, package versions, and external service connections (Ollama / OpenAI).

### Usage
```text
Usage: agentomatic doctor [OPTIONS]

Options:
  --agents-dir TEXT  Agents folder to inspect  [default: agents]
```

### Example Output
```text
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

## 🎨 `agentomatic ui`

Launch the Chainlit debug Chat console as a standalone process (points to a running backend application).

### Usage
```text
Usage: agentomatic ui [OPTIONS]

Options:
  --host TEXT        FastAPI server address  [default: localhost]
  --port INTEGER     FastAPI server port  [default: 8000]
  --ui-port INTEGER  Chainlit UI port to bind  [default: 8001]
```

### Example
```bash
# Launch standalone debug UI on port 9000
agentomatic ui --port 8000 --ui-port 9000
```
