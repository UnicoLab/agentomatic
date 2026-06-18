# Contributing to Agentomatic

Thank you for your interest in contributing to Agentomatic! This guide will help you get set up for development and understand our conventions.

---

## Development Setup

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package manager
- **Git** — Version control

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/UnicoLab/agentomatic.git
cd agentomatic

# Install all development dependencies
make dev

# Verify everything works
make check-all
```

The `make dev` command installs:

- Core package in editable mode
- All optional extras (`[all,dev,docs]`)
- Pre-commit hooks for automated quality checks

---

## Code Style

### Linting & Formatting

We use **[Ruff](https://docs.astral.sh/ruff/)** for both linting and formatting:

```bash
# Check for lint errors
make lint

# Auto-fix and format
make format
```

!!! important "CI Enforcement"
    The CI pipeline runs `ruff check` and `ruff format --check`. PRs with lint errors will fail.

### Conventions

- **Imports**: Use `from __future__ import annotations` in all files
- **Type hints**: All public functions must have type annotations
- **Docstrings**: Google-style docstrings for all public APIs
- **Logging**: Use `loguru.logger` — never `print()` in library code

---

## Testing

### Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov

# Studio-specific tests
make test-studio

# Quick mode (no verbose)
make test-quick
```

### Writing Tests

- Tests live in `tests/` mirroring the source structure
- Use `pytest` with `MagicMock` / `AsyncMock` for mocking
- Use `@pytest.mark.asyncio` for async tests
- Use `FastAPI TestClient` for API integration tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestMyFeature:
    def test_basic_behavior(self):
        """Test description matching the assertion."""
        result = my_function()
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_behavior(self):
        mock_fn = AsyncMock(return_value={"response": "ok"})
        result = await mock_fn({"query": "test"})
        assert result["response"] == "ok"
```

### Test Categories

| Category | Location | Coverage |
|---|---|---|
| Unit tests | `tests/test_agentomatic.py` | Core platform, registry, manifest |
| CLI tests | `tests/test_cli.py` | All CLI commands |
| Studio tests | `tests/test_studio.py` | Adapters, router, decorators |
| Integration | `tests/test_integration.py` | Full request flow |
| Optimization | `tests/test_optimize.py`, `tests/test_fitter.py` | Prompt optimizer |
| Platform features | `tests/test_platform_features.py` | HITL, threads, forking |

---

## Project Structure

```
src/agentomatic/
├── core/               # Platform, registry, manifest, router factory
├── cli/                # Click-based CLI commands
├── demo/               # Built-in demo agent for testing
├── middleware/          # Auth, rate limiting, metrics, logging
├── optimize/           # Prompt optimization (fitter, metrics, search space)
├── storage/            # BaseStore ABC, MemoryStore, SQLAlchemyStore
├── studio/             # Studio API, adapters, decorators
│   ├── adapter.py      # StudioAdapter ABC
│   ├── adapters/       # LangGraph, LangChain, Generic
│   ├── decorators.py   # @studio_graph, @studio_state, @studio_stream
│   ├── models.py       # Pydantic models for Studio API
│   └── router.py       # FastAPI endpoints
├── telemetry/          # OpenTelemetry integration
├── ui/                 # Chainlit debug UI
└── templates/          # Agent scaffolding templates
```

---

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Write tests** for any new functionality
3. **Run quality checks**: `make check-all`
4. **Update docs** if adding user-facing features
5. **Submit PR** with a clear description of changes

### Commit Messages

We follow conventional commits:

```
feat: add LangChain adapter with LCEL graph extraction
fix: resolve thread creation race condition
docs: update Studio architecture section
test: add adapter resolution tests
chore: update ruff configuration
```

---

## Making Changes

### Adding a New Adapter

1. Create `src/agentomatic/studio/adapters/my_adapter.py`
2. Extend `StudioAdapter` ABC
3. Add detection logic to `resolve_adapter()` in `adapters/__init__.py`
4. Add tests to `tests/test_studio.py`
5. Update `docs/guide/studio.md` with the new adapter

### Adding a CLI Command

1. Add the command to `src/agentomatic/cli/commands.py`
2. Follow existing patterns (Rich output, `_echo()`, `_print_banner()`)
3. Add to `docs/cli/commands.md`
4. Add a Makefile shortcut if useful

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
