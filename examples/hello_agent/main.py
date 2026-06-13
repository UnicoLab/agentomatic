"""Hello Agent — minimal agentomatic example.

Usage:
    pip install agentomatic[langgraph]
    cd examples/hello_agent
    uvicorn main:app --reload

    # Test:
    curl -X POST http://localhost:8000/api/v1/hello/invoke \
      -H "Content-Type: application/json" \
      -d '{"query": "Hi there!"}'
"""

from agentomatic import AgentPlatform

# One line to create the platform
platform = AgentPlatform.from_folder(
    "agents/",
    title="Hello Agent Platform",
    package_prefix="agents",
)

# One line to build the FastAPI app
app = platform.build()

# Run: uvicorn main:app --reload
if __name__ == "__main__":
    platform.run(reload=True)
