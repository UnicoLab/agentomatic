# Vision Backend Makefile
# Streamlines all development and deployment commands

.PHONY: help install dev test lint format build docker-build docker-run clean

# Default target
help:
	@echo "Vision Backend - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server with hot reload"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build        - Build Docker image"
	@echo "  docker-build-dist   - Build distroless Docker image"
	@echo "  docker-run          - Run Docker container"
	@echo "  docker-compose-up   - Run with docker-compose"
	@echo "  docker-compose-down - Stop docker-compose"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       - Clean temporary files"
	@echo "  logs        - Show application logs"

# Development commands
install:
	@echo "Installing dependencies..."
	pip install -e .
	pip install -r requirements-dev.txt

dev:
	@echo "Starting development server..."
	uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "Running linting checks..."
	flake8 src tests
	mypy src
	bandit -r src

format:
	@echo "Formatting code..."
	black src tests
	isort src tests

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t vision-backend:latest .

docker-build-dist:
	@echo "Building distroless Docker image..."
	docker build -f Dockerfile.distroless -t vision-backend:distroless .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 --env-file .env vision-backend:latest

docker-compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up -d

docker-compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

# Utility commands
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ dist/ build/

logs:
	@echo "Showing application logs..."
	docker-compose logs -f app

# Agent management
create-agent:
	@echo "Creating new agent..."
	@read -p "Enter agent name: " agent_name; \
	mkdir -p src/agents/agent_$$agent_name; \
	echo "Agent $$agent_name created in src/agents/agent_$$agent_name"

# LLM provider setup
setup-ollama:
	@echo "Setting up Ollama..."
	@echo "Make sure Ollama is installed and running on http://localhost:11434"
	@echo "Pull models with: ollama pull gemma2:1b"

setup-gemini:
	@echo "Setting up Gemini..."
	@echo "Make sure you have Google Cloud credentials configured"
	@echo "Set GOOGLE_APPLICATION_CREDENTIALS environment variable"

# Quick development setup
quick-setup: install
	@echo "Quick setup complete!"
	@echo "Run 'make dev' to start the development server"
	@echo "Run 'make docker-compose-up' to start with Docker"