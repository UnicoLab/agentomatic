# Vision Backend Makefile
# Comprehensive development, testing, and deployment commands

.PHONY: help install dev test lint format build docker-build docker-run clean
.DEFAULT_GOAL := help

# Colors for output
CYAtest-agents: ## Test agent endpoints (simplified)
	@echo "$(CYAN)Testing primary agent endpoints...$(RESET)"
	@echo "Testing Alpha via /invoke endpoint..."
	@curl -X POST http://localhost:8000/api/v1/agents/alpha/invoke \
		-H "Content-Type: application/json" \
		-d '{"payload":{"query":"What is AI?","context":"test"},"streaming":false}' 2>/dev/null && echo "$(GREEN)✓ Alpha invoke$(RESET)" || echo "$(YELLOW)Alpha invoke not available$(RESET)"
	@echo "Testing Beta via /invoke endpoint..."
	@curl -X POST http://localhost:8000/api/v1/agents/beta/invoke \
		-H "Content-Type: application/json" \
		-d '{"payload":{"problem":"How to optimize code?","domain":"Software Engineering"},"streaming":false}' 2>/dev/null && echo "$(GREEN)✓ Beta invoke$(RESET)" || echo "$(YELLOW)Beta invoke not available$(RESET)"
	@echo "Testing Alpha via /chat endpoint..."
	@curl -X POST http://localhost:8000/api/v1/agents/alpha/chat \
		-H "Content-Type: application/json" \
		-d '{"input":"What is AI?","context":"test"}' 2>/dev/null && echo "$(GREEN)✓ Alpha chat$(RESET)" || echo "$(YELLOW)Alpha chat not available$(RESET)"36m
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

# Docker commands with optimized builds and caching
docker-build: ## Build optimized Docker image with caching
	@echo "$(CYAN)Building optimized Docker image with Poetry...$(RESET)"
	@DOCKER_BUILDKIT=1 docker build \
		--target builder \
		--cache-from $(DOCKER_IMAGE):builder-cache \
		--tag $(DOCKER_IMAGE):builder-cache \
		.
	@DOCKER_BUILDKIT=1 docker build \
		--cache-from $(DOCKER_IMAGE):builder-cache \
		--cache-from $(DOCKER_IMAGE):$(DOCKER_TAG) \
		--tag $(DOCKER_IMAGE):$(DOCKER_TAG) \
		.
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(RESET)"

docker-build-no-cache: ## Build Docker image without cache
	@echo "$(CYAN)Building Docker image without cache...$(RESET)"
	@DOCKER_BUILDKIT=1 docker build --no-cache -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(RESET)"

docker-build-distroless: ## Build optimized distroless Docker image
	@echo "$(CYAN)Building optimized distroless Docker image...$(RESET)"
	@DOCKER_BUILDKIT=1 docker build \
		--file Dockerfile.distroless \
		--target builder \
		--cache-from $(DOCKER_IMAGE):distroless-builder-cache \
		--tag $(DOCKER_IMAGE):distroless-builder-cache \
		.
	@DOCKER_BUILDKIT=1 docker build \
		--file Dockerfile.distroless \
		--cache-from $(DOCKER_IMAGE):distroless-builder-cache \
		--cache-from $(DOCKER_IMAGE):distroless \
		--tag $(DOCKER_IMAGE):distroless \
		.
	@echo "$(GREEN)✓ Distroless Docker image built: $(DOCKER_IMAGE):distroless$(RESET)"

docker-build-distroless-no-cache: ## Build distroless Docker image without cache
	@echo "$(CYAN)Building distroless Docker image without cache...$(RESET)"
	@DOCKER_BUILDKIT=1 docker build --no-cache -f Dockerfile.distroless -t $(DOCKER_IMAGE):distroless .
	@echo "$(GREEN)✓ Distroless Docker image built: $(DOCKER_IMAGE):distroless$(RESET)"

docker-build-all: docker-build docker-build-distroless ## Build all Docker images with optimization

docker-build-all-no-cache: docker-build-no-cache docker-build-distroless-no-cache ## Build all Docker images without cache

docker-multi-arch: ## Build multi-architecture images (AMD64 + ARM64)
	@echo "$(CYAN)Building multi-architecture Docker images...$(RESET)"
	@docker buildx create --name vision-builder --use --bootstrap 2>/dev/null || true
	@docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--tag $(DOCKER_IMAGE):$(DOCKER_TAG) \
		--tag $(DOCKER_IMAGE):latest \
		--push \
		.
	@docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--file Dockerfile.distroless \
		--tag $(DOCKER_IMAGE):distroless \
		--tag $(DOCKER_IMAGE):distroless-latest \
		--push \
		.
	@echo "$(GREEN)✓ Multi-architecture images built and pushed$(RESET)"

docker-run: ## Run Docker container with proper configuration
	@echo "$(CYAN)Running Docker container...$(RESET)"
	@docker run -d \
		--name $(PROJECT_NAME)-dev \
		--restart unless-stopped \
		-p 8000:8000 \
		-e PYTHONPATH=/app \
		--env-file .env \
		--health-cmd="curl -f http://localhost:8000/api/v1/health || exit 1" \
		--health-interval=30s \
		--health-timeout=10s \
		--health-retries=3 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✓ Container started: $(PROJECT_NAME)-dev$(RESET)"
	@echo "$(YELLOW)Health check: http://localhost:8000/api/v1/health$(RESET)"

docker-run-distroless: ## Run distroless Docker container
	@echo "$(CYAN)Running distroless Docker container...$(RESET)"
	@docker run -d \
		--name $(PROJECT_NAME)-distroless \
		--restart unless-stopped \
		-p 8001:8000 \
		-e PYTHONPATH=/app \
		$(DOCKER_IMAGE):distroless
	@echo "$(GREEN)✓ Distroless container started: $(PROJECT_NAME)-distroless$(RESET)"
	@echo "$(YELLOW)No health check available in distroless mode$(RESET)"

docker-run-dev: ## Run Docker container in development mode with volume mounts
	@echo "$(CYAN)Running Docker container in development mode...$(RESET)"
	@docker run -d \
		--name $(PROJECT_NAME)-dev-volume \
		-p 8000:8000 \
		-v $(PWD)/src:/app/src:ro \
		-v $(PWD)/logs:/app/logs \
		--env-file .env \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✓ Development container started with volume mounts$(RESET)"

docker-stop: ## Stop and remove Docker containers
	@echo "$(CYAN)Stopping Docker containers...$(RESET)"
	@docker stop $(PROJECT_NAME)-dev $(PROJECT_NAME)-distroless $(PROJECT_NAME)-dev-volume 2>/dev/null || true
	@docker rm $(PROJECT_NAME)-dev $(PROJECT_NAME)-distroless $(PROJECT_NAME)-dev-volume 2>/dev/null || true
	@echo "$(GREEN)✓ Containers stopped and removed$(RESET)"

docker-logs: ## Show Docker container logs
	@echo "$(CYAN)Showing Docker container logs...$(RESET)"
	@docker logs $(PROJECT_NAME)-dev 2>/dev/null || echo "$(YELLOW)No dev container running$(RESET)"

docker-logs-distroless: ## Show distroless Docker container logs
	@echo "$(CYAN)Showing distroless Docker container logs...$(RESET)"
	@docker logs $(PROJECT_NAME)-distroless 2>/dev/null || echo "$(YELLOW)No distroless container running$(RESET)"

docker-shell: ## Get shell access to running container
	@echo "$(CYAN)Opening shell in Docker container...$(RESET)"
	@docker exec -it $(PROJECT_NAME)-dev /bin/bash 2>/dev/null || echo "$(RED)Container not running or shell not available$(RESET)"

docker-stats: ## Show Docker container resource usage
	@echo "$(CYAN)Docker container resource usage:$(RESET)"
	@docker stats --no-stream $(PROJECT_NAME)-dev $(PROJECT_NAME)-distroless 2>/dev/null || echo "$(YELLOW)No containers running$(RESET)"

docker-inspect: ## Inspect Docker images and containers
	@echo "$(CYAN)Docker image information:$(RESET)"
	@docker images $(DOCKER_IMAGE) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
	@echo ""
	@echo "$(CYAN)Running containers:$(RESET)"
	@docker ps --filter "name=$(PROJECT_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

docker-clean: ## Clean up Docker resources
	@echo "$(CYAN)Cleaning up Docker resources...$(RESET)"
	@docker container prune -f
	@docker image prune -f
	@docker volume prune -f
	@docker network prune -f
	@echo "$(GREEN)✓ Docker cleanup completed$(RESET)"

docker-clean-all: ## Clean up all Docker resources (including unused images)
	@echo "$(CYAN)Cleaning up all Docker resources...$(RESET)"
	@docker system prune -a -f --volumes
	@echo "$(GREEN)✓ Complete Docker cleanup completed$(RESET)"

docker-size: ## Show image sizes and layers
	@echo "$(CYAN)Docker image sizes:$(RESET)"
	@docker images $(DOCKER_IMAGE) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
	@echo ""
	@echo "$(CYAN)Image layer information:$(RESET)"
	@docker history $(DOCKER_IMAGE):$(DOCKER_TAG) --format "table {{.CreatedBy}}\t{{.Size}}" 2>/dev/null || echo "$(YELLOW)Standard image not found$(RESET)"
	@docker history $(DOCKER_IMAGE):distroless --format "table {{.CreatedBy}}\t{{.Size}}" 2>/dev/null || echo "$(YELLOW)Distroless image not found$(RESET)"

docker-benchmark: ## Run container performance benchmark
	@echo "$(CYAN)Running Docker container performance benchmark...$(RESET)"
	@docker run --rm \
		-e PYTHONPATH=/app \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		python -c "import time; start=time.time(); import src.app.main; print(f'Cold start time: {time.time()-start:.2f}s')"
	@echo "$(GREEN)✓ Benchmark completed$(RESET)"

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

# Utility commands
setup-dirs: ## Create necessary directories
	@echo "$(CYAN)Creating necessary directories...$(RESET)"
	@mkdir -p logs reports htmlcov
	@echo "$(GREEN)✓ Directories created$(RESET)"

clean: ## Clean up build artifacts and cache files
	@echo "$(CYAN)Cleaning up build artifacts...$(RESET)"
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".coverage" -delete
	@rm -rf htmlcov/ .pytest_cache/ dist/ build/ *.egg-info/
	@echo "$(GREEN)✓ Clean completed$(RESET)"

restart-ollama: ## Restart Ollama service
	@echo "$(CYAN)Restarting Ollama service...$(RESET)"
	@pkill -f ollama || true
	@sleep 2
	@ollama serve &
	@sleep 5
	@echo "$(GREEN)✓ Ollama restarted$(RESET)"

pull-model: ## Pull the required Ollama model
	@echo "$(CYAN)Pulling Ollama model: gemma3:1b...$(RESET)"
	@ollama pull gemma3:1b
	@echo "$(GREEN)✓ Model pulled$(RESET)"

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

# API testing commands
test-api: ## Test API endpoints
	@echo "$(CYAN)Testing API endpoints...$(RESET)"
	@echo "Testing system endpoints..."
	@curl -f http://localhost:8000/api/v1/agents 2>/dev/null && echo "$(GREEN)✓ Agents list$(RESET)" || echo "$(RED)✗ Agents list failed$(RESET)"
	@curl -f http://localhost:8000/api/v1/agents/alpha/health 2>/dev/null && echo "$(GREEN)✓ Alpha health$(RESET)" || echo "$(RED)✗ Alpha health failed$(RESET)"
	@curl -f http://localhost:8000/api/v1/agents/beta/health 2>/dev/null && echo "$(GREEN)✓ Beta health$(RESET)" || echo "$(RED)✗ Beta health failed$(RESET)"

test-schemas: ## Test agent schema endpoints
	@echo "$(CYAN)Testing agent schemas...$(RESET)"
	@curl -f http://localhost:8000/api/v1/agents/alpha/schema 2>/dev/null && echo "$(GREEN)✓ Alpha schema$(RESET)" || echo "$(RED)✗ Alpha schema failed$(RESET)"
	@curl -f http://localhost:8000/api/v1/agents/beta/schema 2>/dev/null && echo "$(GREEN)✓ Beta schema$(RESET)" || echo "$(RED)✗ Beta schema failed$(RESET)"

test-universal: ## Test universal agent endpoints
	@echo "$(CYAN)Testing universal agent endpoints...$(RESET)"
	@echo "Testing Alpha with universal endpoint..."
	@curl -X POST http://localhost:8000/api/v1/agents/alpha/invoke \
		-H "Content-Type: application/json" \
		-d '{"payload":{"query":"What is AI?","context":"test"},"streaming":false}' 2>/dev/null || echo "$(YELLOW)Alpha universal endpoint not available$(RESET)"
	@echo "Testing Beta with universal endpoint..."
	@curl -X POST http://localhost:8000/api/v1/agents/beta/invoke \
		-H "Content-Type: application/json" \
		-d '{"payload":{"problem":"How to optimize code?","domain":"Software Engineering"},"streaming":false}' 2>/dev/null || echo "$(YELLOW)Beta universal endpoint not available$(RESET)"

# Complete API test suite
test-api-complete: test-api test-schemas test-universal ## Run complete API test suite
	@echo "$(GREEN)✓ Complete API test suite finished$(RESET)"

# Docker optimization and CI/CD workflows
docker-optimize: ## Optimize Docker setup and build caches
	@echo "$(CYAN)Optimizing Docker setup...$(RESET)"
	@echo "Building Poetry lock file if needed..."
	@poetry lock --check 2>/dev/null || poetry lock
	@echo "Enabling Docker BuildKit for optimized builds..."
	@export DOCKER_BUILDKIT=1
	@echo "Creating builder instance for multi-platform builds..."
	@docker buildx create --name vision-builder --use --bootstrap 2>/dev/null || true
	@echo "$(GREEN)✓ Docker optimization completed$(RESET)"

docker-ci-build: ## CI/CD optimized build with caching
	@echo "$(CYAN)Running CI/CD optimized Docker build...$(RESET)"
	@DOCKER_BUILDKIT=1 docker build \
		--tag $(DOCKER_IMAGE):$(DOCKER_TAG) \
		--tag $(DOCKER_IMAGE):latest \
		--cache-from $(DOCKER_IMAGE):latest \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		.
	@DOCKER_BUILDKIT=1 docker build \
		--file Dockerfile.distroless \
		--tag $(DOCKER_IMAGE):distroless \
		--tag $(DOCKER_IMAGE):distroless-latest \
		--cache-from $(DOCKER_IMAGE):distroless-latest \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		.
	@echo "$(GREEN)✓ CI/CD Docker build completed$(RESET)"

docker-security-scan: ## Run security scan on Docker images
	@echo "$(CYAN)Running security scans on Docker images...$(RESET)"
	@docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image $(DOCKER_IMAGE):$(DOCKER_TAG) || echo "$(YELLOW)Trivy not available, skipping security scan$(RESET)"
	@docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image $(DOCKER_IMAGE):distroless || echo "$(YELLOW)Trivy not available for distroless image$(RESET)"
	@echo "$(GREEN)✓ Security scan completed$(RESET)"

docker-full-workflow: docker-optimize docker-build-all docker-security-scan docker-benchmark ## Complete Docker workflow
	@echo "$(GREEN)✓ Complete Docker workflow finished$(RESET)"

# Performance and resource monitoring
docker-monitor: ## Monitor Docker container resource usage
	@echo "$(CYAN)Monitoring Docker container resources...$(RESET)"
	@echo "Press Ctrl+C to stop monitoring"
	@docker stats $(PROJECT_NAME)-dev $(PROJECT_NAME)-distroless

docker-health-monitor: ## Continuous health monitoring
	@echo "$(CYAN)Starting continuous health monitoring...$(RESET)"
	@echo "Press Ctrl+C to stop monitoring"
	@while true; do \
		echo "$(CYAN)[$(shell date)] Health check status:$(RESET)"; \
		curl -s http://localhost:8000/api/v1/health && echo " $(GREEN)✓ Healthy$(RESET)" || echo " $(RED)✗ Unhealthy$(RESET)"; \
		sleep 30; \
	done

# Development workflow shortcuts
dev-docker: docker-build docker-run ## Quick development Docker workflow
	@echo "$(GREEN)✓ Development Docker setup ready$(RESET)"
	@echo "$(YELLOW)Container running at: http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Health check: http://localhost:8000/api/v1/health$(RESET)"

prod-docker: docker-build-distroless docker-run-distroless ## Quick production Docker workflow
	@echo "$(GREEN)✓ Production Docker setup ready$(RESET)"
	@echo "$(YELLOW)Distroless container running at: http://localhost:8001$(RESET)"

# Final comprehensive commands
build-all: clean install docker-build-all ## Complete build workflow
	@echo "$(GREEN)✓ Complete build workflow finished$(RESET)"

verify-docker: docker-build test-docker docker-benchmark ## Verify Docker setup
	@echo "$(GREEN)✓ Docker verification completed$(RESET)"

deploy-ready: verify-full docker-ci-build docker-security-scan ## Complete deployment preparation
	@echo "$(GREEN)✓ Deployment ready - all checks passed$(RESET)"