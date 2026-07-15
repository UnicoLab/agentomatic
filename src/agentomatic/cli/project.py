"""Project-level scaffolding for a new Agentomatic workspace.

Used by ``agentomatic init --project NAME`` / ``agentomatic new NAME`` to
create a runnable layout with an explicit ``AgentPlatform`` config, stacks,
``.env.example``, and empty component directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentomatic._version import __version__


def _requirements() -> str:
    """Return a ``requirements.txt`` pinning agentomatic for the image build.

    ``agentomatic deploy`` installs from PyPI directly, but this file gives the
    project a concrete dependency source for local installs and any build that
    prefers ``pip install -r requirements.txt``.
    """
    return f"# Agentomatic project dependencies\nagentomatic[all]=={__version__}\n"


def _main_py(name: str) -> str:
    """Return a starter ``main.py`` with an explicit AgentPlatform config."""
    display = name.replace("_", " ").title()
    return f'''"""{name} — Agentomatic platform entrypoint.

Edit the AgentPlatform kwargs below to enable auth, metrics, stores, etc.
Stacks in ``stacks/`` supply LLM / embedding / DB defaults — switch with::

    agentomatic stack use local   # or remote
"""
from __future__ import annotations

from agentomatic import AgentPlatform

# Optional: SQLAlchemyStore, MemoryStore, connection configs, JWTConfig, …


def create_platform() -> AgentPlatform:
    """Build the platform from discovered agents/plugins/endpoints/ingestion."""
    return AgentPlatform.from_folder(
        "agents/",
        plugins_dir="plugins/",
        endpoints_dir="endpoints/",
        ingestion_dir="ingestion/",
        stacks_dir="stacks/",
        # Pipelines are auto-discovered from ../pipelines/ (sibling of agents/).
        # stack="local",              # or set via .agentomatic-stack / STACK env
        title="{display} Platform",
        description="Agentomatic multi-agent platform for {name}",
        enable_studio=True,
        enable_metrics=True,
        # enable_auth=True,
        # auth_api_key="${{API_KEY}}",  # prefer env / stack auth
        # enable_jwt_auth=True,        # set jwt_config / JWKS via stack
        # enable_zero_trust=True,
        # require_auth_globally=False,  # needs JWT or API-key auth wired
        # enable_control_plane=True,
        # control_token="${{CONTROL_TOKEN}}",
        # enable_rate_limit=True,
    )


_platform = create_platform()
app = _platform.build()


if __name__ == "__main__":
    _platform.run(host="0.0.0.0", port=8000)
'''


def _readme(name: str) -> str:
    """Return a project README."""
    return f"""# {name}

Agentomatic multi-agent platform project.

## Layout

```
{name}/
├── main.py              # Explicit AgentPlatform config (edit this!)
├── agents/              # Agent packages (agentomatic init <agent>)
├── plugins/             # ML plugins
├── endpoints/           # Custom HTTP endpoints to model services
├── ingestion/           # Document ingestors
├── pipelines/           # YAML / Python pipelines
├── stacks/              # LLM / embedding / DB / auth profiles
│   ├── local.yaml
│   └── remote.yaml
├── .env.example         # Copy to .env and fill secrets
└── README.md
```

## Quick start

```bash
cp .env.example .env
agentomatic stack use local
agentomatic init hello --template basic
agentomatic run --studio
# open http://127.0.0.1:8000/studio/ui/  and  /docs
```

## Agent cards / manifests

Each agent under ``agents/`` exports an ``AgentManifest`` in ``__init__.py``
(name, slug, description, intent keywords). Class agents also get a stack-driven
``llm.py`` so the LLM comes from your stack config — not hardcoded.

## Production

```bash
agentomatic deploy --stack remote --distroless
agentomatic stack export --env .env.production
```
"""


def _env_example() -> str:
    """Return a project-level ``.env.example``."""
    return """# Agentomatic project environment
# Copy to .env and fill in values. Stacks reference ${VAR} placeholders.

# Active stack (optional; prefer: agentomatic stack use local)
# AGENTOMATIC_STACK=local

# --- LLM providers ---
OPENAI_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
OLLAMA_BASE_URL=http://localhost:11434

# --- Auth (enable in main.py / remote stack) ---
API_KEY=
CONTROL_TOKEN=
JWT_JWKS_URL=
JWT_ISSUER=
JWT_AUDIENCE=

# --- Databases ---
DATABASE_URL=sqlite+aiosqlite:///./agentomatic.db
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/agentomatic

# --- Vector / RAG (register your provider; any SDK) ---
# VECTOR_URL=
# VECTOR_API_KEY=
# VECTOR_COLLECTION=chunks
QDRANT_URL=
QDRANT_API_KEY=

# --- Observability ---
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
"""


def _gitignore() -> str:
    """Return a sensible project ``.gitignore``."""
    return """__pycache__/
*.py[cod]
.venv/
venv/
.env
.agentomatic-stack
*.egg-info/
dist/
build/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/
*.db
.DS_Store
deploy/generated/
compiled/
"""


def get_project_files(name: str) -> dict[str, str]:
    """Return relative path → content for a new Agentomatic project.

    Args:
        name: Project directory / display name.

    Returns:
        Mapping of relative file paths to file contents.
    """
    from agentomatic.stacks.defaults import get_default_stack_yaml

    files: dict[str, str] = {
        "main.py": _main_py(name),
        "requirements.txt": _requirements(),
        "README.md": _readme(name),
        ".env.example": _env_example(),
        ".gitignore": _gitignore(),
        "agents/.gitkeep": "",
        "agents/__init__.py": '"""Agentomatic agents package."""\n',
        "plugins/.gitkeep": "",
        "plugins/__init__.py": '"""Agentomatic plugins package."""\n',
        "endpoints/.gitkeep": "",
        "ingestion/.gitkeep": "",
        "pipelines/.gitkeep": "",
        "stacks/local.yaml": get_default_stack_yaml("local"),
        "stacks/remote.yaml": get_default_stack_yaml("remote"),
        ".agentomatic-stack": "local\n",
    }
    return files


def scaffold_project(target: Path, name: str, *, force: bool = False) -> dict[str, Any]:
    """Write a project scaffold into *target*.

    Args:
        target: Destination directory.
        name: Project name (used in titles / README).
        force: Overwrite existing files when True.

    Returns:
        Summary dict with ``written``, ``skipped``, and ``path``.
    """
    target.mkdir(parents=True, exist_ok=True)
    files = get_project_files(name)
    written: list[str] = []
    skipped: list[str] = []
    for rel, content in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            skipped.append(rel)
            continue
        path.write_text(content)
        written.append(rel)
    return {"path": str(target), "written": written, "skipped": skipped}
