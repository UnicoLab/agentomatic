# Agentomatic Project Rules

## Code Style
- Always use `from __future__ import annotations` as the first import in every Python file.
- Follow Google-style docstrings for all public functions and classes.
- Maximum line length: 99 characters (configured in `pyproject.toml`).
- Use `ruff` for both linting and formatting. Rules: E, F, I, W, UP.
- Target Python 3.11+; do not use features exclusive to 3.12+.

## Imports
- Use absolute imports from `agentomatic` (e.g., `from agentomatic.core.manifest import AgentManifest`).
- Use `TYPE_CHECKING` blocks for type-only imports to avoid circular dependencies.
- Sort imports with `isort` via ruff (rule `I`).

## Testing
- Test framework: `pytest` with `pytest-asyncio` in auto mode.
- Test files live in `tests/` and follow `test_*.py` naming.
- Use `MagicMock` and `AsyncMock` from `unittest.mock` for mocking.
- **Important**: When testing `_classify_node()` or any method that accesses `.name`, use `SimpleNamespace(name="...")` instead of `MagicMock(name="...")` because MagicMock's `name` parameter sets the mock's internal name, not a `.name` attribute.
- Run tests: `uv run pytest tests/ --override-ini='addopts='`
- Always verify existing tests pass after changes: `uv run pytest tests/ -q`

## Git Conventions
- Commit messages follow Conventional Commits: `type(scope): description`
- Types: feat, fix, docs, chore, test, ci, refactor, perf, style, build
- Examples:
  - `feat(studio): add deep_agent subagent event mapping`
  - `fix(adapter): correct node classification for planning nodes`
  - `docs: extend LangGraph integration guide`

## Documentation
- Documentation uses MkDocs Material and lives in `docs/`.
- Use material admonitions (`!!!`), tabbed code blocks (`=== "Tab"`), and mermaid diagrams.
- Always verify docs build: `uv run mkdocs build --strict`

## Package Structure
- Source code in `src/agentomatic/`.
- Each subpackage has an `__init__.py` that re-exports the public API.
- Production subpackages: `endpoints/` (custom APIs to deployed ML models),
  `connections/` (per-agent databases / vector stores / HTTP / any backend),
  `control/` (control plane), `security/` (JWT + zero-trust), plus the
  `middleware/` stack and `observability/`.
- New features should include: implementation, tests, documentation, and changelog entry.

## CI/CD
- CI runs on every push to `main` and every PR.
- All PRs must pass: lint, format check, tests (Python 3.11-3.13), type check, and build.
- Releases use Python Semantic Release triggered by conventional commits.
