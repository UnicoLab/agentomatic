# ==============================================================================
# Agentomatic — Drop agents, not code
# Framework Lifecycle Makefile
# ==============================================================================
.DEFAULT_GOAL := help
.PHONY: help install dev test test-live lint audit format clean build publish docs

# === Setup ===
install: ## Install agentomatic (core)
	uv sync

dev: ## Install with ALL dev dependencies
	uv sync --all-extras --group dev --group docs
	uv run pre-commit install

# === Quality ===
lint: ## Run ruff linter
	uv run ruff check src/agentomatic/ tests/ scripts/

audit: ## Fail on known vulnerabilities in the locked dependency set
	uv run pip-audit --progress-spinner off

format: ## Auto-format code
	uv run ruff check --fix src/agentomatic/ tests/ scripts/
	uv run ruff format src/agentomatic/ tests/ scripts/

format-check: ## Verify formatting (read-only, for CI)
	uv run ruff format --check src/agentomatic/ tests/ scripts/

typecheck: ## Run mypy type checking
	uv run mypy src/agentomatic/ --ignore-missing-imports

precommit: ## Run all pre-commit hooks
	uv run pre-commit run --all-files

# === Testing ===
test: ## Run all tests
	uv run pytest tests/ -m "not live" -v

test-live: ## Run tests requiring a configured live model service
	uv run pytest tests/ -m live -v --override-ini="addopts="

test-cov: ## Run tests with coverage report
	uv run pytest tests/ --cov=agentomatic --cov-report=html --cov-report=term --cov-report=xml -v

test-quick: ## Run tests (fast, no verbose)
	uv run pytest tests/ -m "not live" -q

test-watch: ## Run tests in watch mode (requires pytest-watch)
	uv run ptw tests/ -- -v

test-studio: ## Run studio-specific tests
	uv run pytest tests/test_studio.py -v --override-ini="addopts="

# === Build & Publish ===
build: clean ## Build the package
	uv build

publish: build ## Publish to PyPI
	uv run twine upload dist/*

publish-test: build ## Publish to Test PyPI
	uv run twine upload --repository testpypi dist/*

# === Documentation ===
docs-serve: ## Serve docs locally (live reload)
	uv run mkdocs serve

docs-build: ## Build static docs site
	uv run mkdocs build

docs-check: ## Verify docs build (strict mode)
	uv run mkdocs build --strict

docs-deploy: ## Deploy docs to GitHub Pages (via mike)
	uv run mike deploy --push --update-aliases $$(uv run python -c "from agentomatic import __version__; print(__version__)") latest
	uv run mike set-default --push latest

# === CLI ===
init: ## Scaffold a new agent (usage: make init AGENT=my_agent TEMPLATE=basic)
	uv run agentomatic init $(AGENT) --template $(or $(TEMPLATE),basic) --dir agents

run: ## Run the platform
	uv run agentomatic run --agents-dir agents --reload

run-ui: ## Run the platform with Chainlit chat UI
	uv run agentomatic run --agents-dir agents --reload --with-ui

run-studio: ## Run the platform with Studio debug UI
	uv run agentomatic run --agents-dir agents --reload --studio

demo: ## Launch demo platform with Studio for E2E testing
	uv run agentomatic demo

list-agents: ## List discovered agents
	uv run agentomatic list --agents-dir agents

doctor: ## Check environment health
	uv run agentomatic doctor

# === Clean ===
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml site/

# === Info ===
version: ## Show package version
	@uv run python -c "from agentomatic import __version__; print(f'agentomatic v{__version__}')"

structure: ## Show package structure
	@find src/agentomatic -type f -name "*.py" | sort

check-all: lint audit typecheck test docs-check ## Run all quality checks
	@echo "✅ All checks passed!"

check-ci: lint format-check typecheck test-cov docs-check ## Full CI parity check (lint + format-check + typecheck + coverage)
	@echo "✅ CI check passed!"

# === Help ===
help: ## Show this help
	@echo "⚡ Agentomatic — Drop agents, not code"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
