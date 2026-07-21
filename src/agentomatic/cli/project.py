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
    """Return a starter ``main.py`` whose module-level ``app`` matches ``run``.

    The generated module exposes ``app = platform.build()`` so a deployed
    container (``uvicorn main:app``) serves the exact same feature set as
    ``agentomatic run`` (Studio, docs, health, metrics, all component dirs).
    Every feature is driven from ``AGENTOMATIC_*`` env vars, so the same file
    works unchanged in dev and in production without code edits.

    Args:
        name: Project directory / display name.

    Returns:
        Rendered ``main.py`` source.
    """
    display = name.replace("_", " ").title()
    return f'''"""{name} — Agentomatic platform entrypoint.

Serves an identical feature set whether launched with ``agentomatic run``
(dev) or ``uvicorn main:app`` (container). Toggle features with
``AGENTOMATIC_*`` env vars — no code edits required. Stacks in ``stacks/``
supply LLM / embedding / DB defaults — switch with::

    agentomatic stack use local   # or remote
"""
from __future__ import annotations

import os

from agentomatic import AgentPlatform

# Optional: SQLAlchemyStore, MemoryStore, connection configs, JWTConfig, …


def _env_bool(var: str, default: bool) -> bool:
    """Parse a boolean feature flag from the environment.

    Args:
        var: Environment variable name.
        default: Value used when the variable is unset or empty.

    Returns:
        ``True`` for ``1/true/yes/on`` (case-insensitive), else ``False``.
    """
    raw = os.getenv(var)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {{"1", "true", "yes", "on"}}


def create_platform() -> AgentPlatform:
    """Build a fully-featured platform matching ``agentomatic run`` defaults.

    Discovers every component directory (agents, plugins, endpoints, ingestion,
    pipelines, stacks) and enables Studio, docs, health, and metrics by default.
    Auth, control plane, and rate limiting stay opt-in via ``AGENTOMATIC_*``
    env vars so the deployed container is configurable without code changes.

    Returns:
        A configured :class:`AgentPlatform` ready to :meth:`build`.
    """
    # ``require_auth`` mirrors ``agentomatic run --require-auth-globally``:
    # it implies zero-trust + JWT unless those are overridden individually.
    require_auth = _env_bool("AGENTOMATIC_REQUIRE_AUTH", False)
    return AgentPlatform.from_folder(
        "agents/",
        plugins_dir="plugins/",
        endpoints_dir="endpoints/",
        ingestion_dir="ingestion/",
        stacks_dir="stacks/",
        # Pipelines are auto-discovered from ../pipelines/ (sibling of agents/).
        # stack="local",              # or set via .agentomatic-stack / STACK env
        title=os.getenv("AGENTOMATIC_TITLE", "{display} Platform"),
        description="Agentomatic multi-agent platform for {name}",
        log_level=os.getenv("AGENTOMATIC_LOG_LEVEL", "INFO"),
        # On by default — matches `agentomatic run` (Studio) + prod observability.
        enable_studio=_env_bool("AGENTOMATIC_ENABLE_STUDIO", True),
        enable_metrics=_env_bool("AGENTOMATIC_ENABLE_METRICS", True),
        # Opt-in hardening — drive from env / stack; no code edits needed.
        enable_auth=_env_bool("AGENTOMATIC_ENABLE_AUTH", False),
        auth_api_key=os.getenv("AGENTOMATIC_API_KEY", ""),
        enable_jwt_auth=_env_bool("AGENTOMATIC_ENABLE_JWT", require_auth),
        enable_zero_trust=_env_bool("AGENTOMATIC_ENABLE_ZERO_TRUST", require_auth),
        require_auth_globally=require_auth,
        # On with Studio so Control / Endpoints / Connections tabs work out of the box.
        enable_control_plane=_env_bool("AGENTOMATIC_ENABLE_CONTROL_PLANE", True),
        control_token=os.getenv("AGENTOMATIC_CONTROL_TOKEN", ""),
        enable_rate_limit=_env_bool("AGENTOMATIC_ENABLE_RATE_LIMIT", False),
    )


_platform = create_platform()
app = _platform.build()


if __name__ == "__main__":
    _platform.run(
        host=os.getenv("AGENTOMATIC_HOST", "0.0.0.0"),
        port=int(os.getenv("AGENTOMATIC_PORT", "8000")),
    )
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

# --- Deploy feature flags (main.py reads these; parity with `agentomatic run`) ---
# Studio + metrics are ON by default; auth/control-plane/rate-limit are opt-in.
# AGENTOMATIC_TITLE=My Platform
# AGENTOMATIC_LOG_LEVEL=INFO
# AGENTOMATIC_ENABLE_STUDIO=1
# AGENTOMATIC_ENABLE_METRICS=1
# AGENTOMATIC_ENABLE_AUTH=0            # API-key auth (needs AGENTOMATIC_API_KEY)
# AGENTOMATIC_ENABLE_JWT=0             # JWT auth (set JWKS via stack)
# AGENTOMATIC_REQUIRE_AUTH=0           # implies JWT + zero-trust; needs auth wired
# AGENTOMATIC_ENABLE_CONTROL_PLANE=0   # needs AGENTOMATIC_CONTROL_TOKEN
# AGENTOMATIC_ENABLE_RATE_LIMIT=0
# AGENTOMATIC_API_KEY=
# AGENTOMATIC_CONTROL_TOKEN=

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
