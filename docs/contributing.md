# Contributing

## Development Setup

```bash
git clone https://github.com/UnicoLab/agentomatic.git
cd agentomatic
pip install -e ".[all,dev]"
pre-commit install
```

## Code Quality

```bash
make lint       # Ruff linter
make format     # Auto-format
make typecheck  # Mypy
make test       # All tests
make test-cov   # With coverage
```

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: fix a bug
docs: documentation changes
chore: maintenance
test: add tests
ci: CI/CD changes
```

## Pull Request Process

1. Fork the repo
2. Create a feature branch (`feat/my-feature`)
3. Write tests
4. Ensure `make lint test` passes
5. Submit PR
