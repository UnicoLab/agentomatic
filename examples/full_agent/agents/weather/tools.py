"""LangChain-compatible tools for the weather agent.

Optional: define custom tools that can be bound to LLMs.
These are standard @tool decorated functions.
"""

from __future__ import annotations


def get_temperature(location: str) -> str:
    """Get the current temperature for a location.

    Args:
        location: City name (e.g., 'Paris, France')

    Returns:
        Temperature string
    """
    # In production, call a real weather API
    return f"The temperature in {location} is 24°C (75°F)"


def get_weather_alerts(region: str) -> str:
    """Get active weather alerts for a region.

    Args:
        region: Region or country name

    Returns:
        Alert information
    """
    return f"No active weather alerts for {region}"
