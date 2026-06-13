"""Node functions for weather agent.

Each function receives and returns a state dict.
"""

from __future__ import annotations

from typing import Any


async def classify_query(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the user query into forecast or alert."""
    query = state.get("current_query", "").lower()
    if any(word in query for word in ["alert", "warning", "storm", "danger"]):
        decision = "alert"
    else:
        decision = "forecast"
    return {
        "routing_decision": decision,
        "steps_taken": [f"classified_as_{decision}"],
    }


def route_query(state: dict[str, Any]) -> str:
    """Route to the appropriate node based on classification."""
    return state.get("routing_decision", "forecast")


async def get_forecast(state: dict[str, Any]) -> dict[str, Any]:
    """Get weather forecast."""
    query = state.get("current_query", "")
    return {
        "response": f"🌤️ Weather forecast for your query: '{query}'\n"
        f"Today: Sunny, 24°C | Tomorrow: Partly cloudy, 22°C",
        "metadata": {"source": "weather_api", "confidence": 0.95},
        "steps_taken": ["fetched_forecast"],
    }


async def get_alerts(state: dict[str, Any]) -> dict[str, Any]:
    """Get weather alerts."""
    return {
        "response": "⚠️ Weather Alert: No active weather alerts in your area.",
        "metadata": {"source": "alert_system", "alert_level": "none"},
        "steps_taken": ["checked_alerts"],
    }


async def format_response(state: dict[str, Any]) -> dict[str, Any]:
    """Format the final response."""
    response = state.get("response", "No data available")
    return {
        "response": response,
        "agent_type": "demo-weather-agent",
        "suggestions": [
            "What's tomorrow's forecast?",
            "Any weather alerts?",
            "Weekly forecast",
        ],
    }
