"""LLM factory for creating different LLM providers with unified interface."""

from enum import Enum
from typing import Optional, Dict, Any, AsyncGenerator, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
from loguru import logger

try:
    from langchain_ollama import ChatOllama
    from langchain_google_vertexai import ChatVertexAI
    from langchain_core.messages import HumanMessage
except ImportError as e:
    logger.warning(f"Some LLM dependencies not installed: {e}")


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    GEMINI = "gemini"


@dataclass
class LLMConfig:
    """Configuration for LLM instances."""
    provider: LLMProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    streaming: bool = False
    timeout: int = 30

    # Provider-specific configs
    base_url: Optional[str] = None  # For Ollama
    project_id: Optional[str] = None  # For Gemini
    location: Optional[str] = None    # For Gemini
    api_key: Optional[str] = None     # For any provider that needs it

    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
            "timeout": self.timeout,
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else None,
            "project_id": self.project_id,
            "location": self.location
        }


class BaseLLMWrapper(ABC):
    """Abstract base class for LLM wrappers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._llm = None
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM client."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        streaming: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response from the LLM."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM is healthy and responsive."""
        pass

    async def ensure_initialized(self) -> None:
        """Ensure the LLM is initialized before use."""
        if not self._initialized:
            await self.initialize()


class OllamaWrapper(BaseLLMWrapper):
    """Wrapper for Ollama LLM."""

    async def initialize(self) -> None:
        """Initialize Ollama client."""
        try:
            self._llm = ChatOllama(
                model=self.config.model_name,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
                timeout=self.config.timeout
            )
            self._initialized = True
            logger.info(f"Initialized Ollama with model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        streaming: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response from Ollama."""
        await self.ensure_initialized()

        try:
            message = HumanMessage(content=prompt)

            if streaming:
                return self._stream_generate(message)
            else:
                response = await self._llm.ainvoke([message])
                return response.content

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def _stream_generate(self, message) -> AsyncGenerator[str, None]:
        """Stream generate responses from Ollama."""
        try:
            async for chunk in self._llm.astream([message]):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"Error: {str(e)}"

    async def health_check(self) -> bool:
        """Check Ollama health."""
        try:
            await self.ensure_initialized()
            # Simple test generation
            test_response = await self.generate("Hello", streaming=False)
            return bool(test_response)
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


class GeminiWrapper(BaseLLMWrapper):
    """Wrapper for Google Gemini LLM."""

    async def initialize(self) -> None:
        """Initialize Gemini client."""
        try:
            self._llm = ChatVertexAI(
                model_name=self.config.model_name,
                project=self.config.project_id,
                location=self.config.location or "us-central1",
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens
            )
            self._initialized = True
            logger.info(f"Initialized Gemini with model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        streaming: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate response from Gemini."""
        await self.ensure_initialized()

        try:
            message = HumanMessage(content=prompt)

            if streaming:
                return self._stream_generate(message)
            else:
                response = await self._llm.ainvoke([message])
                return response.content

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    async def _stream_generate(self, message) -> AsyncGenerator[str, None]:
        """Stream generate responses from Gemini."""
        try:
            async for chunk in self._llm.astream([message]):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            yield f"Error: {str(e)}"

    async def health_check(self) -> bool:
        """Check Gemini health."""
        try:
            await self.ensure_initialized()
            # Simple test generation
            test_response = await self.generate("Hello", streaming=False)
            return bool(test_response)
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False


class LLMFactory:
    """Factory for creating LLM instances."""

    _instances: Dict[str, BaseLLMWrapper] = {}

    @classmethod
    async def create_llm(cls, config: LLMConfig) -> BaseLLMWrapper:
        """Create or get cached LLM instance."""
        cache_key = f"{config.provider.value}:{config.model_name}:{config.base_url or ''}"

        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create new instance
        if config.provider == LLMProvider.OLLAMA:
            instance = OllamaWrapper(config)
        elif config.provider == LLMProvider.GEMINI:
            instance = GeminiWrapper(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        # Initialize and cache
        await instance.initialize()
        cls._instances[cache_key] = instance

        logger.info(f"Created and cached LLM instance: {cache_key}")
        return instance

    @classmethod
    def create_llm_sync(cls, config: LLMConfig) -> BaseLLMWrapper:
        """Create LLM instance synchronously (for use in __init__ methods)."""
        cache_key = f"{config.provider.value}:{config.model_name}:{config.base_url or ''}"

        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create new instance without initializing
        if config.provider == LLMProvider.OLLAMA:
            instance = OllamaWrapper(config)
        elif config.provider == LLMProvider.GEMINI:
            instance = GeminiWrapper(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        cls._instances[cache_key] = instance
        logger.info(f"Created (not initialized) LLM instance: {cache_key}")
        return instance

    @classmethod
    def get_instance(cls, config: LLMConfig) -> Optional[BaseLLMWrapper]:
        """Get cached LLM instance without creating new one."""
        cache_key = f"{config.provider.value}:{config.model_name}:{config.base_url or ''}"
        return cls._instances.get(cache_key)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached LLM instances."""
        cls._instances.clear()
        logger.info("Cleared LLM factory cache")

    @classmethod
    def list_instances(cls) -> Dict[str, Dict[str, Any]]:
        """List all cached LLM instances."""
        return {
            key: {
                "provider": instance.config.provider.value,
                "model_name": instance.config.model_name,
                "initialized": instance._initialized
            }
            for key, instance in cls._instances.items()
        }