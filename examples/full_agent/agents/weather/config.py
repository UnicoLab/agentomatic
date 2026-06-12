"""Agent-specific configuration.

Optional overwrite: if this file exists, the registry loads the config
and makes it available via agent.config and GET /config endpoint.

Naming convention: {AgentName}Config class (e.g., WeatherConfig).
Alternatively, export a 'config' variable.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class WeatherConfig(BaseModel):
    """Weather agent configuration."""
    
    # Prompt version to use
    prompt_version: str = Field("v1", description="Active prompt version")
    
    # API settings
    api_url: str = Field(
        "https://api.openweathermap.org/data/2.5",
        description="Weather API base URL",
    )
    api_key: str = Field("", description="Weather API key")
    
    # Agent behavior
    default_location: str = Field("Paris, France", description="Default location")
    units: str = Field("metric", description="Temperature units: metric|imperial")
    forecast_days: int = Field(5, ge=1, le=16, description="Number of forecast days")
    
    # LLM overrides
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1)
    
    # Feature flags (per-agent)
    enable_alerts: bool = Field(True, description="Enable weather alerts")
    enable_history: bool = Field(False, description="Enable historical data")
