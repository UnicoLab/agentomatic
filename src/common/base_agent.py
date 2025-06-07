"""Base agent class for all LangGraph agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, AsyncIterator
from langgraph.graph import StateGraph
from loguru import logger
from pydantic import BaseModel

from .llm_factory import LLMFactory, LLMConfig, LLMProvider, BaseLLMWrapper
from .prompt_manager import PromptManager


class BaseAgent(ABC):
    """Abstract base class for all agents with unified LLM and prompt management.

    Example:
        class MyAgent(BaseAgent):
            def __init__(self):
                super().__init__(
                    agent_name="my_agent",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA, model_name="gemma2:1b")
                )

            def build_graph(self) -> StateGraph:
                # Implementation here
                pass

            async def run(self, input_data: BaseModel) -> BaseModel:
                # Implementation here
                pass
    """

    def __init__(self, name: str, llm: BaseLLMWrapper, prompt_manager: PromptManager) -> None:
        self.name = name
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.graph: Optional[StateGraph] = None

        # Build the graph after initialization
        self.graph = self.build_graph()
        logger.info(f"Initialized {name} agent")

    def get_prompt(self, version: str = "v1") -> Optional[str]:
        """Get a prompt template string by version.

        Args:
            version: The prompt version to retrieve

        Returns:
            The prompt template string or None if not found
        """
        return self.prompt_manager.get_prompt(version)

    def format_prompt(self, version: str = "v1", **kwargs) -> Optional[str]:
        """Format a prompt with given variables.

        Args:
            version: The prompt version to use
            **kwargs: Variables to format the prompt with

        Returns:
            Formatted prompt string
        """
        return self.prompt_manager.format_prompt(version, **kwargs)

    async def generate_response(
        self,
        prompt: str,
        streaming: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate a response using the configured LLM.

        Args:
            prompt: The formatted prompt
            streaming: Whether to stream the response
            **kwargs: Additional generation parameters

        Returns:
            Generated response (string or async iterator for streaming)
        """
        return await self.llm.generate(prompt, streaming=streaming, **kwargs)

    async def health_check(self) -> Dict[str, Any]:
        """Check agent health including LLM and prompt availability.

        Returns:
            Health status information
        """
        try:
            llm_healthy = await self.llm.health_check()
            prompt_versions = self.prompt_manager.list_versions()

            return {
                "agent": self.name,
                "status": "healthy" if llm_healthy and prompt_versions else "degraded",
                "llm_healthy": llm_healthy,
                "llm_provider": self.llm.config.provider.value,
                "model_name": self.llm.config.model_name,
                "prompt_versions": prompt_versions,
                "graph_ready": self.graph is not None
            }
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            return {
                "agent": self.name,
                "status": "unhealthy",
                "error": str(e)
            }

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for this agent.

        Returns:
            A configured StateGraph instance
        """
        pass

    @abstractmethod
    async def run(
        self,
        input_data: BaseModel,
        streaming: bool = False
    ) -> Union[BaseModel, AsyncIterator[str]]:
        """Run the agent with the given input.

        Args:
            input_data: The input data model
            streaming: Whether to stream the response

        Returns:
            The output data model or async iterator for streaming
        """
        pass