# Input & Output Schemas

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Declaring Strong Types and Boundary Validation for AI Agents</h3>
</div>

---

One of the core features of Agentomatic is **automated boundary validation**. In standard LLM frameworks, inputs and outputs are typically unstructured dictionaries (`dict[str, Any]`), making them prone to runtime type mismatches, missing fields, or parameter injection.

Agentomatic solves this by allowing developers to define strong API contracts via standard **Pydantic** schemas. These schemas are parsed dynamically at startup to:

1. Validate incoming HTTP request payloads at the FastAPI boundary.
2. Generate interactive Swagger UI (`/docs`) and Redoc API documentation.
3. Shape and serialize agent responses before returning them to the client.
4. Auto-generate compatibility-ready **A2A (Agent-to-Agent)** cards.

---

## 🔍 Schema Resolution Conventions

By default, every agent in Agentomatic uses generic, fallback models:
* **Default Input**: `AgentInvokeRequest`
* **Default Output**: `AgentInvokeResponse`

To override these schemas, create a `schemas.py` file inside your agent's directory (e.g. `agents/my_agent/schemas.py`). At startup, Agentomatic scans this module and dynamically resolves the models using the following priority queue:

### Input Schema Resolution
Agentomatic searches `schemas.py` for class definitions matching one of these names (in order):
1. **`CustomInvokeRequest`**
2. **`{AgentNameCamel}Request`** (e.g. `SearchBotRequest` for an agent named `search_bot`)
3. **`AgentInvokeRequest`**

### Output Schema Resolution
Agentomatic searches `schemas.py` for class definitions matching one of these names (in order):
1. **`CustomInvokeResponse`**
2. **`{AgentNameCamel}Response`** (e.g. `SearchBotResponse` for an agent named `search_bot`)
3. **`AgentInvokeResponse`**

> [!NOTE]
> If no matching classes are found in `schemas.py`, or if the file does not exist, Agentomatic falls back to the default `AgentInvokeRequest` and `AgentInvokeResponse` models.

---

## 🛠️ Defining Custom Schemas

To define custom models, simply import Pydantic's `BaseModel` and declare your fields using type annotations and metadata helpers like `Field`.

### Step 1: Create `schemas.py`

Here is a typical `schemas.py` implementation for a weather retrieval agent:

```python
# agents/weather_agent/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class WeatherAgentRequest(BaseModel):
    """Custom input parameters for the weather search agent."""

    location: str = Field(
        ...,
        description="The city name or zip code to check weather for",
        examples=["Paris", "San Francisco"],
    )
    units: str = Field(
        default="celsius",
        description="Temperature unit of measurement",
        examples=["celsius", "fahrenheit"],
    )
    include_forecast: bool = Field(
        default=False,
        description="Whether to include a 5-day forecast in the output",
    )


class WeatherAgentResponse(BaseModel):
    """Custom output response format returned to the client."""

    forecast_summary: str = Field(
        ...,
        description="Natural language summary of current weather conditions",
    )
    temperature_val: float = Field(..., description="Numerical temperature value")
    wind_speed_kmh: float = Field(..., description="Wind speed in km/h")
    is_rainy: bool = Field(default=False, description="Flag for rain indicators")
    steps_taken: list[str] = Field(
        default_factory=list,
        description="Steps executed during retrieval",
    )
```

### Step 2: Access Schemas in Your Execution Node

When the input payload is validated by FastAPI, it is passed directly into your node function or state graph. If you are using a standard node function (`node_fn`), you can access fields from the `state` dict.

```python
# agents/weather_agent/__init__.py
from agentomatic import AgentManifest
from .schemas import WeatherAgentRequest, WeatherAgentResponse

manifest = AgentManifest(
    name="weather_agent",
    slug="weather-agent",
    description="Fetches live weather data and alerts.",
    version="1.0.0",
)

async def node_fn(state: dict) -> dict:
    # 1. Retrieve the validated fields from the initial state
    # (By default, fields are mapped to state.get("current_query") or state.get("metadata"))
    location = state.get("metadata", {}).get("location")
    units = state.get("metadata", {}).get("units", "celsius")
    
    # 2. Process and fetch data
    # (mocking execution logic here)
    temp = 22.5 if units == "celsius" else 72.5
    
    # 3. Return a dict matching the WeatherAgentResponse schema structure
    return {
        "forecast_summary": f"Sunny and warm in {location}",
        "temperature_val": temp,
        "wind_speed_kmh": 15.4,
        "is_rainy": False,
        "steps_taken": ["fetch_weather_api", "format_response"]
    }
```

---

## ⚡ Runtime Validation & Error Handling

FastAPI enforces constraints at the HTTP boundary. When a client submits invalid parameters, the application automatically rejects the request before it reaches your agent execution code, returning an `HTTP 422 Unprocessable Entity` response.

### Validation Example

If a request is sent to `POST /api/v1/weather_agent/invoke` with a missing `location` field:

```bash
curl -X POST http://localhost:8000/api/v1/weather_agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"units": "celsius"}'
```

The server responds immediately with:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "location"],
      "msg": "Field required",
      "input": {
        "units": "celsius"
      }
    }
  ]
}
```

This ensures that your agent nodes only receive clean, well-formed data that adheres to the contract.

---

## 🌐 Swagger UI Integration

When you register custom schemas, Agentomatic updates the OpenAPI definition for that specific agent. 

To view and interact with the custom schemas:
1. Start the server: `agentomatic run --reload`
2. Open your browser and navigate to `http://localhost:8000/docs`
3. Expand the route group corresponding to your agent (e.g., `Weather_Agent`).
4. You will see your exact custom Request and Response models documented with field types, descriptions, and examples!

---

## 🤝 Agent-to-Agent (A2A) Discovery

Custom schemas are also automatically serialized into the agent's **A2A Agent Card** at `GET /api/v1/agents/{agent_name}/card` (and the universal `.well-known/agent.json` directory). 

This allows other agents running on the platform or external client orchestrators to programmatically read the exact types, required fields, and descriptions of what your agent expects and outputs, paving the way for autonomous collaboration.
