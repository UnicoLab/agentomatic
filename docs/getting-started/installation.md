# Installation

## Requirements

- **Python 3.11+** (3.12 and 3.13 also supported)
- **pip**, **uv**, or **poetry** as package manager

!!! tip "Check your Python version"
    ```bash
    python --version  # Must be ≥ 3.11
    ```

---

## Quick Install

=== "pip"
    ```bash
    pip install agentomatic
    ```

=== "uv"
    ```bash
    uv add agentomatic
    ```

=== "poetry"
    ```bash
    poetry add agentomatic
    ```

## Full Install (Recommended)

Install with all optional extras for the complete experience:

```bash
pip install agentomatic[all]
```

---

## Optional Extras

Install only what you need:

| Extra | Command | What It Adds |
|---|---|---|
| `langgraph` | `pip install agentomatic[langgraph]` | LangGraph agent support |
| `ollama` | `pip install agentomatic[ollama]` | Ollama LLM provider |
| `openai` | `pip install agentomatic[openai]` | OpenAI LLM provider |
| `metrics` | `pip install agentomatic[metrics]` | Prometheus metrics |
| `db` | `pip install agentomatic[db]` | SQLAlchemy + SQLite storage |
| `db-postgres` | `pip install agentomatic[db-postgres]` | PostgreSQL support |
| `cli` | `pip install agentomatic[cli]` | Rich CLI + interactive prompts |
| `ui` | `pip install agentomatic[ui]` | Chainlit debug UI |
| `optimize` | `pip install agentomatic[optimize]` | Prompt optimization + DeepEval |
| `telemetry` | `pip install agentomatic[telemetry]` | OpenTelemetry tracing |
| `all` | `pip install agentomatic[all]` | Everything above |

!!! note "Combining extras"
    You can combine multiple extras in a single install:
    ```bash
    pip install agentomatic[langgraph,ollama,db,metrics]
    ```

---

## Development Installation

Clone and install from source for contributing or local development:

```bash
git clone https://github.com/UnicoLab/agentomatic.git
cd agentomatic
pip install -e ".[all,dev]"
pre-commit install  # Set up commit hooks
```

Or with **uv**:

```bash
git clone https://github.com/UnicoLab/agentomatic.git
cd agentomatic
uv sync --all-extras
pre-commit install
```

---

## Verify Installation

Run the built-in health check to confirm everything is working:

```bash
agentomatic doctor
```

Expected output:

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

!!! tip "Next Step"
    Ready to go? Head to the [Quick Start](quickstart.md) guide.
