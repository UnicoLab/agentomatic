# Vision Backend Makefile
# Comprehensive development, testing, and deployment commands

.PHONY: help install dev test lint format build docker-build docker-run clean
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Configuration
PROJECT_NAME := vision-backend
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest
COMPOSE_FILE := docker-compose.yml
TEST_TIMEOUT := 300
PARALLEL_WORKERS := auto

# Poetry command prefix
POETRY_RUN := poetry run

# Help target
help: ## Show this help message
	@echo "$(CYAN)Vision Backend - Available Commands:$(RESET)"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(install|dev|format|lint)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(test|check)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker & Deployment:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(docker|build|deploy)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Utilities:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(clean|logs|monitoring)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'

# Development commands
install: ## Install dependencies and setup development environment
	@echo "$(CYAN)Installing dependencies with Poetry...$(RESET)"
	@poetry install --with dev
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

install-poetry: ## Install Poetry package manager
	@echo "$(CYAN)Installing Poetry...$(RESET)"
	@curl -sSL https://install.python-poetry.org | python3 -
	@echo "$(GREEN)✓ Poetry installed$(RESET)"

dev: ## Start development server with hot reload
	@echo "$(CYAN)Starting development server...$(RESET)"
	@$(POETRY_RUN) uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

dev-debug: ## Start development server with debug logging
	@echo "$(CYAN)Starting development server in debug mode...$(RESET)"
	@PYTHONPATH=. $(POETRY_RUN) uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Testing commands
test: ## Run all tests with parallel execution
	@echo "$(CYAN)Running all tests in parallel...$(RESET)"
	@$(POETRY_RUN) pytest tests/ -v -n $(PARALLEL_WORKERS) --dist=worksteal --tb=short --cov=src --cov-report=term-missing --cov-report=html --timeout=$(TEST_TIMEOUT)

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/ -v -m "unit" -n $(PARALLEL_WORKERS) --tb=short --cov=src --cov-report=term

test-integration: ## Run integration tests only
	@echo "$(CYAN)Running integration tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/ -v -m "integration" -n $(PARALLEL_WORKERS) --tb=short --timeout=$(TEST_TIMEOUT)

test-e2e: ## Run end-to-end tests
	@echo "$(CYAN)Running end-to-end tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/test_e2e.py -v -m "e2e" -s --tb=short --timeout=$(TEST_TIMEOUT)

test-load: ## Run load and performance tests
	@echo "$(CYAN)Running load tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/test_load_performance.py -v -m "load" -s --tb=short --timeout=600

test-stress: ## Run stress tests
	@echo "$(CYAN)Running stress tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/test_load_performance.py -v -m "stress" -s --tb=short --timeout=900

test-docker: ## Run Docker integration tests
	@echo "$(CYAN)Running Docker tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/test_docker.py -v -m "docker" -s --tb=short --timeout=$(TEST_TIMEOUT)

test-parallel: ## Run parallel agent tests
	@echo "$(CYAN)Running parallel agent tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/test_parallel_agents.py -v -m "parallel" -n $(PARALLEL_WORKERS) --tb=short

test-fast: ## Run fast tests only (exclude slow tests)
	@echo "$(CYAN)Running fast tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/ -v -m "not slow" -n $(PARALLEL_WORKERS) --tb=short --maxfail=5

test-coverage: ## Run tests with detailed coverage report
	@echo "$(CYAN)Running tests with coverage analysis...$(RESET)"
	@$(POETRY_RUN) pytest tests/ -v -n $(PARALLEL_WORKERS) --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml --cov-fail-under=70

test-report: ## Generate comprehensive test report
	@echo "$(CYAN)Generating comprehensive test report...$(RESET)"
	@$(POETRY_RUN) pytest tests/ -v -n $(PARALLEL_WORKERS) --html=reports/report.html --json-report --json-report-file=reports/report.json --cov=src --cov-report=html:reports/coverage

# Code quality commands
lint: ## Run all linting checks
	@echo "$(CYAN)Running linting checks...$(RESET)"
	@$(POETRY_RUN) flake8 src tests --max-line-length=88 --extend-ignore=E203,W503
	@$(POETRY_RUN) mypy src --ignore-missing-imports
	@$(POETRY_RUN) bandit -r src -f json -o reports/bandit.json || $(POETRY_RUN) bandit -r src

lint-fix: ## Run linting with auto-fix where possible
	@echo "$(CYAN)Running linting with auto-fix...$(RESET)"
	@$(POETRY_RUN) black src tests
	@$(POETRY_RUN) isort src tests
	@echo "$(GREEN)✓ Code formatted$(RESET)"

format: ## Format code with black and isort
	@echo "$(CYAN)Formatting code...$(RESET)"
	@$(POETRY_RUN) black src tests
	@$(POETRY_RUN) isort src tests
	@echo "$(GREEN)✓ Code formatted$(RESET)"

format-check: ## Check code formatting without making changes
	@echo "$(CYAN)Checking code formatting...$(RESET)"
	@$(POETRY_RUN) black --check src tests
	@$(POETRY_RUN) isort --check-only src tests

type-check: ## Run type checking with mypy
	@echo "$(CYAN)Running type checks...$(RESET)"
	@$(POETRY_RUN) mypy src --ignore-missing-imports --strict-optional

security-check: ## Run security analysis with bandit
	@echo "$(CYAN)Running security analysis...$(RESET)"
	@$(POETRY_RUN) bandit -r src

check-all: lint type-check security-check ## Run all code quality checks

# Docker commands
docker-build: ## Build Docker image
	@echo "$(CYAN)Building Docker image...$(RESET)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(RESET)"

docker-build-distroless: ## Build distroless Docker image
	@echo "$(CYAN)Building distroless Docker image...$(RESET)"
	@docker build -f Dockerfile.distroless -t $(DOCKER_IMAGE):distroless .
	@echo "$(GREEN)✓ Distroless Docker image built: $(DOCKER_IMAGE):distroless$(RESET)"

docker-build-all: docker-build docker-build-distroless ## Build all Docker images

docker-run: ## Run Docker container
	@echo "$(CYAN)Running Docker container...$(RESET)"
	@docker run -d --name $(PROJECT_NAME)-dev -p 8000:8000 --env-file .env $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✓ Container started: $(PROJECT_NAME)-dev$(RESET)"

docker-run-distroless: ## Run distroless Docker container
	@echo "$(CYAN)Running distroless Docker container...$(RESET)"
	@docker run -d --name $(PROJECT_NAME)-distroless -p 8001:8000 $(DOCKER_IMAGE):distroless
	@echo "$(GREEN)✓ Distroless container started: $(PROJECT_NAME)-distroless$(RESET)"

docker-stop: ## Stop Docker container
	@echo "$(CYAN)Stopping Docker containers...$(RESET)"
	@docker stop $(PROJECT_NAME)-dev $(PROJECT_NAME)-distroless 2>/dev/null || true
	@docker rm $(PROJECT_NAME)-dev $(PROJECT_NAME)-distroless 2>/dev/null || true
	@echo "$(GREEN)✓ Containers stopped$(RESET)"

docker-logs: ## Show Docker container logs
	@echo "$(CYAN)Showing Docker container logs...$(RESET)"
	@docker logs $(PROJECT_NAME)-dev 2>/dev/null || echo "$(YELLOW)No dev container running$(RESET)"

# Docker Compose commands
compose-build: ## Build services with docker-compose
	@echo "$(CYAN)Building services with docker-compose...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE) build
	@echo "$(GREEN)✓ Services built$(RESET)"

compose-up: ## Start all services with docker-compose
	@echo "$(CYAN)Starting services with docker-compose...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)✓ Services started$(RESET)"
	@echo "$(YELLOW)Alpha Agent: http://localhost:8001$(RESET)"
	@echo "$(YELLOW)Beta Agent: http://localhost:8002$(RESET)"
	@echo "$(YELLOW)Nginx Proxy: http://localhost:80$(RESET)"

compose-down: ## Stop all services with docker-compose
	@echo "$(CYAN)Stopping services with docker-compose...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE) down
	@echo "$(GREEN)✓ Services stopped$(RESET)"

compose-logs: ## Show docker-compose logs
	@echo "$(CYAN)Showing docker-compose logs...$(RESET)"
	@docker-compose -f $(COMPOSE_FILE) logs -f

compose-ps: ## Show docker-compose service status
	@echo "$(CYAN)Docker-compose service status:$(RESET)"
	@docker-compose -f $(COMPOSE_FILE) ps

# Health check commands
health-check: ## Check application health
	@echo "$(CYAN)Checking application health...$(RESET)"
	@curl -f http://localhost:8000/healthz || echo "$(RED)Health check failed$(RESET)"

health-check-compose: ## Check docker-compose services health
	@echo "$(CYAN)Checking docker-compose services health...$(RESET)"
	@echo "Alpha Agent:"
	@curl -f http://localhost:8001/healthz 2>/dev/null && echo "$(GREEN)✓ Healthy$(RESET)" || echo "$(RED)✗ Unhealthy$(RESET)"
	@echo "Beta Agent:"
	@curl -f http://localhost:8002/healthz 2>/dev/null && echo "$(GREEN)✓ Healthy$(RESET)" || echo "$(RED)✗ Unhealthy$(RESET)"
	@echo "Nginx Proxy:"
	@curl -f http://localhost:80/healthz 2>/dev/null && echo "$(GREEN)✓ Healthy$(RESET)" || echo "$(RED)✗ Unhealthy$(RESET)"

# Agent management commands
test-agents: ## Test agent endpoints
	@echo "$(CYAN)Testing agent endpoints...$(RESET)"
	@echo "Testing Alpha agent..."
	@curl -X POST http://localhost:8000/api/v1/agents/alpha/chat \
		-H "Content-Type: application/json" \
		-d '{"input":"What is AI?","context":"test"}' 2>/dev/null || echo "$(YELLOW)Alpha agent not available$(RESET)"
	@echo "Testing Beta agent..."
	@curl -X POST http://localhost:8000/api/v1/agents/beta/chat \
		-H "Content-Type: application/json" \
		-d '{"input":"Design a system","context":"test"}' 2>/dev/null || echo "$(YELLOW)Beta agent not available$(RESET)"

# Performance and monitoring
benchmark: ## Run performance benchmarks
	@echo "$(CYAN)Running performance benchmarks...$(RESET)"
	@$(POETRY_RUN) pytest tests/test_load_performance.py -v -m "performance" -s --tb=short

# CI/CD helpers
ci-install: ## Install dependencies for CI environment
	@echo "$(CYAN)Installing CI dependencies with Poetry...$(RESET)"
	@poetry install --with dev

ci-test: ## Run tests for CI environment
	@echo "$(CYAN)Running CI tests...$(RESET)"
	@$(POETRY_RUN) pytest tests/ -v -n auto --tb=short --cov=src --cov-report=xml --cov-report=term --timeout=300

ci-lint: ## Run linting for CI environment
	@echo "$(CYAN)Running CI linting...$(RESET)"
	@$(POETRY_RUN) black --check src tests
	@$(POETRY_RUN) isort --check-only src tests
	@$(POETRY_RUN) flake8 src tests --max-line-length=88
	@$(POETRY_RUN) mypy src --ignore-missing-imports

# Quick setup commands
quick-setup: install setup-dirs ## Quick development setup
	@echo "$(GREEN)✓ Quick setup complete!$(RESET)"
	@echo "$(YELLOW)Run 'make dev' to start the development server$(RESET)"
	@echo "$(YELLOW)Run 'make test' to run tests$(RESET)"
	@echo "$(YELLOW)Run 'make compose-up' to start with Docker$(RESET)"

# Comprehensive test suite
test-all: test-unit test-integration test-e2e test-docker ## Run all test suites
	@echo "$(GREEN)✓ All test suites completed$(RESET)"

# Verification workflow
verify: format-check lint type-check test-fast ## Quick verification workflow
	@echo "$(GREEN)✓ Verification complete$(RESET)"

verify-full: check-all test docker-build ## Full verification workflow
	@echo "$(GREEN)✓ Full verification complete$(RESET)"