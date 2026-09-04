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
make audit         # Known-vulnerability audit of locked dependencies
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

When creating commits via a HEREDOC, use **`/bin/cat`** (not bare `cat`). Many
developer shells alias `cat` to `bat`, which injects ANSI colors and line numbers
into the message and breaks Python Semantic Release / GitHub CI titles. Example:

```bash
git commit -m "$(/bin/cat <<'EOF'
ci: fix lint and typing

EOF
)"
```

A `commit-msg` pre-commit hook (`scripts/strip_commit_msg_ansi.py`) also strips
ANSI/line-number garbage if it slips through — install hooks with
`pre-commit install --hook-type commit-msg`.

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

Releases are automated via [python-semantic-release](https://python-semantic-release.readthedocs.io/)
and are dispatched from the **Release** GitHub Actions workflow:

1. Merge the release candidate to `main` and wait for every CI job to pass.
2. Run the Release workflow with `DRY_RUN=true`. Confirm the calculated version
   and that `CHANGELOG.md` contains `<!-- version list -->`.
3. Run the same workflow with `DRY_RUN=false`. Semantic release analyzes the
   conventional commits, updates both version files and `CHANGELOG.md`, then
   creates and pushes the release commit and tag.
4. The workflow downloads the released Studio bundle, builds and verifies the
   wheel, publishes it to PyPI, verifies the GitHub release, and deploys the
   versioned docs with mike.
5. Install the published wheel in a clean environment and run the deployment
   verifier from `docs/guide/verifying-a-deployment.md` against staging before
   promoting production traffic.

The release wrapper intentionally fails before semantic release when the
changelog insertion marker is missing; otherwise a release can bump and tag a
version without recording its changes.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
