# Vision Backend - Multi-Agent Architecture

A scalable multi-agent architecture built with LangGraph and FastAPI, featuring dynamic agent discovery, versioned prompts, and comprehensive input/output validation using Pydantic.

## Features

- **Multi-Agent System**: Pluggable agents with auto-discovery
- **LangGraph Integration**: Complex workflows with state management
- **Versioned Prompts**: JSON-based prompt versioning with LangChain
- **API Versioning**: RESTful APIs with automatic route generation
- **Dynamic Discovery**: Agents automatically discovered and registered
- **Pydantic Validation**: Comprehensive input/output validation
- **Ollama Integration**: Local LLM support with retry mechanisms
- **Docker Support**: Containerized deployment ready

## Architecture

```
src/
├── app/                    # FastAPI application
│   ├── main.py            # Application factory
│   ├── settings.py        # Configuration management
│   ├── api.py             # Route auto-discovery
│   └── dependencies.py    # Agent registry & DI
├── common/                # Shared components
│   ├── base_agent.py      # Abstract agent base class
│   ├── utilities.py       # Common utilities
│   └── nodes/             # Reusable LangGraph nodes
└── agents/                # Agent implementations
    ├── agent_alpha/       # Example: General purpose agent
    └── agent_beta/        # Example: Reasoning specialist
```

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Ollama with gemma2:1b model

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd vision-backend
   poetry install
   ```

2. **Install and start Ollama**:
   ```bash
   # Install Ollama (see https://ollama.ai)
   ollama pull gemma2:1b
   ollama serve
   ```

3. **Run the application**:
   ```bash
   poetry run uvicorn src.app.main:app --reload
   ```

4. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/healthz

### Docker Deployment

```bash
# Build and run
docker build -t vision-backend .
docker run -p 8000:8000 vision-backend

# Or use docker-compose
docker-compose up --build
```

## API Usage

### Alpha Agent (General Purpose)

```bash
curl -X POST "http://localhost:8000/api/v1/agents/alpha/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "context": "Educational context for beginners"
  }'
```

### Beta Agent (Reasoning & Analysis)

```bash
curl -X POST "http://localhost:8000/api/v1/agents/beta/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "How to optimize database performance?",
    "domain": "Software Engineering",
    "requirements": ["Low latency", "High throughput"],
    "constraints": "Limited memory budget"
  }'
```

## Adding New Agents

1. **Create agent directory**:
   ```
   src/agents/agent_gamma/
   ├── config.py          # Agent configuration
   ├── prompts.json       # Versioned prompts
   ├── schemas.py         # Input/Output models
   ├── llm.py            # LLM wrapper
   ├── state.py          # Workflow state
   ├── graph.py          # LangGraph implementation
   └── api.py            # FastAPI router
   ```

2. **Implement required components**:
   - Extend `BaseAgent` in `graph.py`
   - Define Pydantic models in `schemas.py`
   - Create FastAPI router in `api.py`

3. **Agent auto-discovery**: The system automatically discovers and registers your agent!

## Configuration

Environment variables (prefix with `VISION_`):

- `VISION_HOST`: Server host (default: "0.0.0.0")
- `VISION_PORT`: Server port (default: 8000)
- `VISION_DEBUG`: Debug mode (default: False)
- `VISION_API_VERSION`: API version (default: "v1")
- `VISION_OLLAMA_BASE_URL`: Ollama URL (default: "http://localhost:11434")
- `VISION_LOG_LEVEL`: Log level (default: "INFO")

## Development

### Code Quality

```bash
# Format code
poetry run black src/
poetry run isort src/

# Type checking
poetry run mypy src/

# Linting
poetry run flake8 src/
```

### Testing

```bash
# Run tests
poetry run pytest

# With coverage
poetry run pytest --cov=src/
```

## Project Structure Details

### Agent Scaffold

Each agent follows a consistent pattern:

- **config.py**: Pydantic configuration model
- **prompts.json**: Versioned prompt templates
- **schemas.py**: Input/Output Pydantic models
- **llm.py**: LLM wrapper with retry logic
- **state.py**: Workflow state management
- **graph.py**: LangGraph workflow implementation
- **api.py**: FastAPI endpoints

### Core Components

- **BaseAgent**: Abstract base class for all agents
- **AgentRegistry**: Dynamic agent discovery and management
- **ValidationNode**: Reusable input/output validation
- **Retry Utilities**: Robust error handling and retries

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the agent scaffold pattern
4. Add comprehensive tests
5. Submit a pull request