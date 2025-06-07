"""Application settings and configuration."""

from pydantic_settings import BaseSettings
from typing import Optional
from enum import Enum

from ..common.llm_factory import LLMProvider


class AppConfig(BaseSettings):
    """Global application configuration.

    Example:
        config = AppConfig()
        print(config.host)  # "0.0.0.0"
    """

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    api_version: str = "v1"
    agents_package: str = "src.agents"
    log_level: str = "INFO"

    # LLM Provider settings
    default_llm_provider: LLMProvider = LLMProvider.OLLAMA

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:1b"

    # Gemini settings
    gemini_project_id: str = "ai-vision-innovation-ski"
    gemini_location: str = "europe-west4"
    gemini_model: str = "gemini-2.0-flash"

    # Agent settings
    default_temperature: float = 0.7
    default_max_tokens: int = 1000
    default_timeout: float = 30.0
    enable_streaming: bool = True

    # Rate limiting
    rate_limit_calls: int = 100
    rate_limit_window: int = 60

    # Queue settings
    max_queue_size: int = 100
    enable_queue: bool = True

    # Security
    cors_origins: list[str] = ["*"]
    api_key: Optional[str] = None

    class Config:
        env_prefix = "VISION_"
        env_file = ".env"


# Global config instance
config = AppConfig()