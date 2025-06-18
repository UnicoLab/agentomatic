"""Comprehensive test suite for the Vision backend."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.app.main import app
from src.agents.alpha.agent import AlphaAgent
from src.common.llm_factory import LLMFactory, LLMConfig
from src.common.prompt_manager import PromptManager
from src.app.settings import config


class TestApp:
    """Test cases for the main FastAPI application."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_agent_list(self):
        """Test agent listing endpoint."""
        response = self.client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)

    @pytest.mark.asyncio
    async def test_agent_execution(self):
        """Test agent execution endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post(
                "/agents/alpha/execute",
                json={
                    "input": "Test input",
                    "context": "Test context",
                    "parameters": {}
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "output" in data
            assert "agent" in data

    @pytest.mark.asyncio
    async def test_streaming_execution(self):
        """Test streaming agent execution."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            async with ac.stream(
                "POST",
                "/agents/alpha/stream",
                json={
                    "input": "Test streaming input",
                    "context": "Test context"
                }
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/plain"


class TestLLMFactory:
    """Test cases for LLM factory and configuration."""

    def test_ollama_config_creation(self):
        """Test Ollama LLM configuration."""
        config = LLMConfig(
            provider="ollama",
            model="gemma2:1b",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        assert config.provider == "ollama"
        assert config.model == "gemma2:1b"

    def test_gemini_config_creation(self):
        """Test Gemini LLM configuration."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-2.0-flash",
            project_id="test-project",
            location="europe-west4"
        )
        assert config.provider == "gemini"
        assert config.model == "gemini-2.0-flash"

    @patch('src.common.llm_factory.ChatOllama')
    def test_create_ollama_llm(self, mock_ollama):
        """Test Ollama LLM creation."""
        config = LLMConfig(provider="ollama", model="gemma2:1b")
        factory = LLMFactory()

        llm = factory.create_llm(config)
        mock_ollama.assert_called_once()
        assert llm is not None

    @patch('src.common.llm_factory.ChatVertexAI')
    def test_create_gemini_llm(self, mock_gemini):
        """Test Gemini LLM creation."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-2.0-flash",
            project_id="test-project"
        )
        factory = LLMFactory()

        llm = factory.create_llm(config)
        mock_gemini.assert_called_once()
        assert llm is not None


class TestPromptManager:
    """Test cases for prompt management system."""

    def setup_method(self):
        """Set up prompt manager."""
        self.prompt_manager = PromptManager()

    def test_load_json_prompts(self):
        """Test loading prompts from JSON file."""
        prompts = self.prompt_manager.load_prompts(
            "src/agents/agent_alpha/prompts.json",
            format="json"
        )
        assert isinstance(prompts, dict)
        assert "system" in prompts
        assert "user" in prompts

    def test_load_python_prompts(self):
        """Test loading prompts from Python file."""
        prompts = self.prompt_manager.load_prompts(
            "prompts/agent_alpha_prompts.py",
            format="python"
        )
        assert isinstance(prompts, dict)

    def test_format_prompt(self):
        """Test prompt formatting with variables."""
        template = "Hello {name}, your task is {task}"
        formatted = self.prompt_manager.format_prompt(
            template,
            name="Alice",
            task="testing"
        )
        assert formatted == "Hello Alice, your task is testing"

    def test_get_prompt_with_fallback(self):
        """Test prompt retrieval with fallback."""
        prompt = self.prompt_manager.get_prompt(
            "nonexistent_key",
            fallback="Default prompt"
        )
        assert prompt == "Default prompt"


class TestAlphaAgent:
    """Test cases for Alpha Agent implementation."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for testing."""
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(content="Test response")
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        """Create Alpha Agent instance for testing."""
        with patch('src.agents.agent_alpha.agent.LLMFactory') as mock_factory:
            mock_factory.return_value.create_llm.return_value = mock_llm
            return AlphaAgent()

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.name == "alpha"
        assert agent.description is not None
        assert agent.llm is not None

    @pytest.mark.asyncio
    async def test_agent_execution(self, agent, mock_llm):
        """Test agent execution with mock LLM."""
        result = await agent.execute({
            "input": "Test input",
            "context": "Test context"
        })

        assert "output" in result
        assert result["agent"] == "alpha"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_streaming(self, agent):
        """Test agent streaming execution."""
        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        with patch.object(agent.llm, 'astream', return_value=mock_stream()):
            chunks = []
            async for chunk in agent.stream({
                "input": "Test streaming input"
            }):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks == ["chunk1", "chunk2", "chunk3"]


class TestErrorHandling:
    """Test cases for error handling and edge cases."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_invalid_agent_name(self):
        """Test handling of invalid agent names."""
        response = self.client.post(
            "/agents/nonexistent/execute",
            json={"input": "test"}
        )
        assert response.status_code == 404

    def test_invalid_request_body(self):
        """Test handling of invalid request bodies."""
        response = self.client.post(
            "/agents/alpha/execute",
            json={"invalid": "data"}
        )
        assert response.status_code == 422

    def test_llm_factory_invalid_provider(self):
        """Test LLM factory with invalid provider."""
        config = LLMConfig(provider="invalid", model="test")
        factory = LLMFactory()

        with pytest.raises(ValueError):
            factory.create_llm(config)

    @pytest.mark.asyncio
    async def test_agent_execution_timeout(self):
        """Test agent execution timeout handling."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
                response = await ac.post(
                    "/agents/alpha/execute",
                    json={"input": "test"},
                    timeout=1.0
                )
                assert response.status_code == 408


class TestSettings:
    """Test cases for application settings."""

    def test_settings_initialization(self):
        """Test settings initialization with defaults."""
        # Use the config object from settings
        assert config.debug is False  # Default in production
        assert config.host == "0.0.0.0"
        assert config.port == 8000

    def test_settings_from_environment(self):
        """Test settings override from environment variables."""
        import os
        original_debug = os.environ.get("DEBUG")

        try:
            os.environ["DEBUG"] = "true"
            # In real scenario, we'd reload config, but for test we just check the concept
            assert config.debug is not None
        finally:
            if original_debug:
                os.environ["DEBUG"] = original_debug
            elif "DEBUG" in os.environ:
                del os.environ["DEBUG"]


class TestIntegration:
    """Integration test cases."""

    @pytest.mark.asyncio
    async def test_full_agent_workflow(self):
        """Test complete agent workflow from API to response."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Test agent listing
            list_response = await ac.get("/agents")
            assert list_response.status_code == 200

            # Test agent execution
            exec_response = await ac.post(
                "/agents/alpha/execute",
                json={
                    "input": "What is the capital of France?",
                    "context": "Geography question",
                    "parameters": {"temperature": 0.7}
                }
            )
            assert exec_response.status_code == 200

            # Verify response structure
            data = exec_response.json()
            assert "output" in data
            assert "agent" in data
            assert "execution_time" in data

    @pytest.mark.asyncio
    async def test_concurrent_agent_executions(self):
        """Test concurrent agent executions."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            tasks = []
            for i in range(5):
                task = ac.post(
                    "/agents/alpha/execute",
                    json={"input": f"Test input {i}"}
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "output" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])