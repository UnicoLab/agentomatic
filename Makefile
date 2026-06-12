# ==============================================================================
# Agentomatic — Framework Lifecycle Makefile
# ==============================================================================
.DEFAULT_GOAL := help
.PHONY: help install dev test lint format clean build publish

# === Setup ===
install: ## Install agentomatic (core)
	uv pip install -e .

dev: ## Install with all dev dependencies
	uv pip install -e ".[all,dev]"

# === Quality ===
lint: ## Run ruff linter
	uv run ruff check src/agentomatic/ tests/

format: ## Auto-format code
	uv run ruff check --fix src/agentomatic/ tests/
	uv run ruff format src/agentomatic/ tests/

typecheck: ## Run mypy type checking
	uv run mypy src/agentomatic/ --ignore-missing-imports

# === Testing ===
test: ## Run all tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ --cov=agentomatic --cov-report=html --cov-report=term -v

test-quick: ## Run tests (fast, no verbose)
	uv run pytest tests/ -q

# === Build & Publish ===
build: clean ## Build the package
	uv run python -m build

publish: build ## Publish to PyPI (requires credentials)
	uv run twine upload dist/*

publish-test: build ## Publish to Test PyPI
	uv run twine upload --repository testpypi dist/*

# === CLI ===
init: ## Scaffold a new agent (usage: make init AGENT=my_agent)
	uv run agentomatic init $(AGENT) --dir agents

run: ## Run the example platform
	cd examples/hello_agent && uv run uvicorn main:app --reload --port 8000

list-agents: ## List agents in examples/hello_agent
	uv run agentomatic list --agents-dir examples/hello_agent/agents

# === Clean ===
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

# === Info ===
version: ## Show package version
	@uv run python -c "from agentomatic import __version__; print(f'agentomatic v{__version__}')"

structure: ## Show package structure
	@find src/agentomatic -type f -name "*.py" | sort | head -40

# === Help ===
help: ## Show this help
	@echo "🤖 Agentomatic — Drop agents, not code"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'