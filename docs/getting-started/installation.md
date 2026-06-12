# Installation

## Requirements

- Python 3.11+
- pip or uv

## Core Installation

```bash
pip install agentomatic
```

## Optional Extras

| Extra | Command | What It Adds |
|---|---|---|
| `langgraph` | `pip install agentomatic[langgraph]` | LangGraph integration |
| `ollama` | `pip install agentomatic[ollama]` | Ollama LLM provider |
| `openai` | `pip install agentomatic[openai]` | OpenAI provider |
| `db` | `pip install agentomatic[db]` | SQLAlchemy + SQLite storage |
| `db-postgres` | `pip install agentomatic[db-postgres]` | PostgreSQL storage |
| `metrics` | `pip install agentomatic[metrics]` | Prometheus metrics |
| `cli` | `pip install agentomatic[cli]` | Rich CLI + interactive prompts |
| `ui` | `pip install agentomatic[ui]` | Chainlit debug UI |
| `all` | `pip install agentomatic[all]` | Everything (recommended) |

## Development Installation

```bash
git clone https://github.com/UnicoLab/agentomatic.git
cd agentomatic
pip install -e ".[all,dev]"
```

## Verify Installation

```bash
agentomatic doctor
```

This shows all installed components and their versions.
