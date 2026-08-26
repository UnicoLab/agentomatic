# Multi-stage Docker build for Agentomatic
# Optimized for quick builds using uv package manager

# Build stage
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from PyPI at a pinned version.
#
# This used to be `COPY --from=ghcr.io/astral-sh/uv:latest`, which pulled
# an unpinned tag: image contents changed under you between builds, and a
# breaking uv release could break the build with no diff to show for it.
# PyPI is already required by every other layer here, so sourcing uv from
# it also drops a second registry from the build's dependency set.
ARG UV_VERSION=0.8.17
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# Set working directory
WORKDIR /app

# Copy dependency files first for cache efficiency.
# README.md is required: pyproject.toml declares it as the project readme,
# so the build backend fails without it when uv installs the project below.
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (without the project itself)
# Extras matter here: a bare `uv sync` installs only the core dependencies, so
# the image shipped without sqlalchemy, langgraph, prometheus-client or pyjwt —
# /metrics served nothing, DATABASE_URL failed with "No module named
# 'sqlalchemy'", and JWT auth could not be enabled at all. `all` restores those
# and matches what `agentomatic deploy` builds. `db-postgres` and `openai` are
# named separately because `all` deliberately omits provider SDKs and the
# PostgreSQL driver. The local compose stack supports both Postgres and an
# OpenAI-compatible host oMLX endpoint.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra all --extra db-postgres --extra openai

# Copy source code and install the project
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra all --extra db-postgres --extra openai

# Production stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./

# Create only the writable directories. The virtual environment and source
# tree are already copied with the right owner above; recursively chowning
# all of /app walks thousands of dependency files and turns each production
# rebuild into a multi-minute no-op layer.
RUN mkdir -p /app/logs /app/tmp /app/agents \
    && chown appuser:appuser /app/logs /app/tmp /app/agents

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application using the CLI
CMD ["agentomatic", "run", "--agents-dir", "agents", "--host", "0.0.0.0", "--port", "8000"]
