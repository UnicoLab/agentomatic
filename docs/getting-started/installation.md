# Installation

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Getting Started Stack</h3>
</div>

---

## Requirements

- **Python 3.11+** (Python 3.12 and 3.13 are fully supported).
- **Package Manager**: `pip`, `uv`, or `poetry`.

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

---

## Full Install (Recommended)

To enable all production features (including prompt optimization, database persistence, OpenTelemetry tracing, and the graphical Chainlit chat playground), install with the `all` extras flag:

```bash
pip install agentomatic[all]

# Or with uv
uv add agentomatic --extra all
```

---

## Optional Extras

If you prefer a lightweight install, you can select only the modules and dependencies you need:

| Extra Flag | Install Command | What It Enables |
|---|---|---|
| `langgraph` | `pip install agentomatic[langgraph]` | Direct support for LangGraph StateGraphs |
| `ollama` | `pip install agentomatic[ollama]` | Local Ollama LLM provider integrations |
| `openai` | `pip install agentomatic[openai]` | OpenAI API provider integrations |
| `metrics` | `pip install agentomatic[metrics]` | Prometheus exporter metrics |
| `db` | `pip install agentomatic[db]` | SQLAlchemy engines + local SQLite support |
| `db-postgres` | `pip install agentomatic[db-postgres]` | SQLAlchemy async PostgreSQL client driver |
| `cli` | `pip install agentomatic[cli]` | Rich terminal formatting + interactive select prompt controls |
| `ui` | `pip install agentomatic[ui]` | Graphical Chainlit chat debug console |
| `optimize` | `pip install agentomatic[optimize]` | DSPy-style optimizer loop + DeepEval validation |
| `telemetry` | `pip install agentomatic[telemetry]` | OpenTelemetry APM tracing exporters |
| `all` | `pip install agentomatic[all]` | Installs all components and drivers above |

!!! note "Combining extras"
    You can combine multiple extras in a single install command:
    ```bash
    pip install agentomatic[langgraph,db,metrics]
    ```

---

## Development Installation (From Source)

To contribute to the framework or run custom builds:

=== "pip"
    ```bash
    git clone https://github.com/UnicoLab/agentomatic.git
    cd agentomatic
    pip install -e ".[all,dev]"
    pre-commit install  # Installs git commit linter hooks
    ```

=== "uv"
    ```bash
    git clone https://github.com/UnicoLab/agentomatic.git
    cd agentomatic
    uv sync --all-extras
    pre-commit install
    ```

---

## Verify Installation

Verify your local environment health, connectivity, and dependencies using the built-in diagnostic test:

```bash
agentomatic doctor
```

### Expected Diagnostic Output

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

!!! tip "Next Step"
    Now that the installation is complete, proceed to the [Quick Start](quickstart.md) guide!
