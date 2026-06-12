# Contributing to Agentomatic

Thank you for your interest in contributing! 🎉

## Development Setup

```bash
git clone https://github.com/UnicoLab/agentomatic.git
cd agentomatic

# Install all dependencies + pre-commit hooks
make dev
```

## Code Quality

We enforce strict code quality via pre-commit hooks and CI:

```bash
make lint          # Ruff linter
make format        # Auto-format
make typecheck     # Mypy type checking
make test          # Run all tests
make test-cov      # Tests with coverage
make check-all     # All checks at once
```

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) enforced by pre-commit:

```
feat: add new storage backend
fix: resolve thread ID collision
docs: update storage guide
chore: update dependencies
test: add middleware tests
ci: fix release workflow
refactor: simplify router factory
perf: optimize registry lookup
```

## Pull Request Process

1. **Fork** the repository
2. **Branch** from `main` (`feat/my-feature` or `fix/my-fix`)
3. **Write tests** for any new functionality
4. **Ensure** `make check-all` passes locally
5. **Submit** a PR with a clear description

## Architecture

```
src/agentomatic/
├── core/          # Platform, Registry, Router, Manifest
├── cli/           # CLI commands + templates
├── config/        # Settings and configuration
├── middleware/     # Auth, rate limit, metrics, logging
├── observability/  # Circuit breaker, metrics
├── providers/     # LLM and embedding providers
├── protocols/     # API decorators and response models
├── storage/       # BaseStore ABC + implementations
└── ui/            # Chainlit debug UI
```

## Adding a New Storage Backend

1. Subclass `BaseStore` from `agentomatic.storage.base`
2. Implement all abstract methods
3. Add tests in `tests/test_storage_<name>.py`
4. Update docs in `docs/guide/storage.md`

## Adding a New Template

1. Add template functions in `src/agentomatic/cli/templates.py`
2. Register in the `TEMPLATES` dict
3. Add the template case in `get_template_files()`
4. Add tests in `tests/test_cli.py`

## Release Process

Releases are automated via [python-semantic-release](https://python-semantic-release.readthedocs.io/):

1. Merge PR to `main`
2. Semantic release analyzes commit messages
3. If releasable commits found → bumps version, creates tag, publishes to PyPI
4. Docs are auto-deployed via mike

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
