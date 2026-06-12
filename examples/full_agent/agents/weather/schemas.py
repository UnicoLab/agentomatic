"""Custom request/response schemas.

Optional overwrite: define custom Pydantic models for your agent's
API endpoints. These can be used in custom api.py routers.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class WeatherRequest(BaseModel):
    """Custom weather request."""
    location: str = Field(..., description="City name or coordinates")
    days: int = Field(1, ge=1, le=16, description="Forecast days")
    units: str = Field("metric", description="Temperature units")
    include_alerts: bool = Field(True, description="Include weather alerts")


class WeatherForecast(BaseModel):
    """Weather forecast response."""
    location: str
    temperature: float
    conditions: str
    humidity: int
    wind_speed: float
    forecast_date: str


class WeatherResponse(BaseModel):
    """Full weather response."""
    location: str
    current: WeatherForecast
    forecasts: list[WeatherForecast] = []
    alerts: list[str] = []
    source: str = "weather_api"
