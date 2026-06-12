"""Custom API router for weather agent.

Optional overwrite: if this file exports a 'router', the registry
will use it INSTEAD of auto-generating endpoints. This gives you
full control over the agent's API.

You can also ADD endpoints alongside the auto-generated ones by
using a different approach (see platform.include_router()).
"""
from __future__ import annotations

from fastapi import APIRouter

from .schemas import WeatherRequest, WeatherResponse, WeatherForecast

# NOTE: This router is discovered by the registry and REPLACES
# the auto-generated router for this agent.
router = APIRouter()


@router.get("/locations")
async def list_locations() -> dict:
    """Custom endpoint: list supported locations."""
    return {
        "locations": [
            {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
            {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
            {"name": "New York", "country": "US", "lat": 40.7128, "lon": -74.0060},
        ]
    }


@router.post("/forecast")
async def custom_forecast(request: WeatherRequest) -> WeatherResponse:
    """Custom endpoint with typed request/response."""
    return WeatherResponse(
        location=request.location,
        current=WeatherForecast(
            location=request.location,
            temperature=24.0,
            conditions="Sunny",
            humidity=60,
            wind_speed=15.0,
            forecast_date="2026-06-12",
        ),
        alerts=[],
        source="custom_endpoint",
    )
