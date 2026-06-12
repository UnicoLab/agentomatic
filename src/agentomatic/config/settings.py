"""Platform settings with feature flags and nested configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    """LLM provider configuration."""

    provider: str = Field("ollama", description="LLM provider: ollama|azure|openai|vertex|dummy")
    model: str = Field("mistral:7b", description="Model name")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1)
    ollama_base_url: str = Field("http://localhost:11434")
    azure_api_key: str = Field("", description="Azure OpenAI API key")
    azure_api_base: str = Field("")
    azure_api_version: str = Field("2024-02-15-preview")
    azure_deployment_name: str = Field("")
    openai_api_key: str = Field("")
    vertex_project: str = Field("")
    vertex_location: str = Field("us-central1")


class EmbeddingSettings(BaseModel):
    """Embedding provider configuration."""

    provider: str = Field("dummy", description="Embedding provider: ollama|dummy")
    model: str = Field("nomic-embed-text")
    dimension: int = Field(768)


class DatabaseSettings(BaseModel):
    """Database configuration."""

    url: str = Field("sqlite+aiosqlite:///data/platform.db")
    pool_size: int = Field(10, ge=1)
    max_overflow: int = Field(20, ge=0)
    pool_timeout: int = Field(30, ge=1)
    echo: bool = Field(False)


class FeatureSettings(BaseModel):
    """Feature flags for the platform."""

    enable_streaming: bool = Field(True, description="Enable SSE streaming endpoints")
    enable_a2a: bool = Field(True, description="Enable A2A protocol")
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_rate_limit: bool = Field(False, description="Enable rate limiting")
    enable_auth: bool = Field(False, description="Enable API key authentication")
    enable_db: bool = Field(False, description="Enable database storage")
    enable_feedback: bool = Field(True, description="Enable feedback collection")
    max_concurrent_agents: int = Field(10, ge=1)
    request_timeout: float = Field(30.0, gt=0)
    llm_retry_count: int = Field(3, ge=0)
    llm_retry_delay: float = Field(1.0, ge=0)
    circuit_breaker_threshold: int = Field(5, ge=1)
    circuit_breaker_timeout: float = Field(60.0, gt=0)


class AuthSettings(BaseModel):
    """Authentication configuration."""

    api_key: str = Field("", description="API key for authentication")


class RateLimitSettings(BaseModel):
    """Rate limiting configuration."""

    requests: int = Field(100, ge=1)
    window_seconds: int = Field(60, ge=1)


class PlatformSettings(BaseSettings):
    """Root platform configuration.

    Priority: env vars > .env > YAML > defaults.
    Nested delimiter: __ (double underscore).
    """

    app_name: str = Field("Agentomatic Platform")
    app_env: str = Field("development")
    log_level: str = Field("INFO")
    api_version: str = Field("v1")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)

    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


_settings: PlatformSettings | None = None


def get_settings() -> PlatformSettings:
    """Get or create the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = PlatformSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (for testing)."""
    global _settings
    _settings = None
