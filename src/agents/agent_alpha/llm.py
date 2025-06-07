"""LLM wrapper for the Alpha agent."""

import time
from typing import Optional
import ollama
from loguru import logger

from ...app.settings import config
from ...common.utilities import retry
from .config import AlphaConfig


class AlphaLLM:
    """LLM wrapper for Alpha agent using Ollama.
    
    Example:
        llm = AlphaLLM()
        response = await llm.generate("What is AI?", max_tokens=500)
    """
    
    def __init__(self, agent_config: Optional[AlphaConfig] = None) -> None:
        self.config = agent_config or AlphaConfig()
        self.client = ollama.AsyncClient(host=config.ollama_base_url)
    
    @retry(max_attempts=3, delay=1.0)
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text using the Ollama model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If generation fails after retries
        """
        start_time = time.time()
        
        try:
            # Use provided values or fall back to config defaults
            generation_params = {
                "model": self.config.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens or self.config.max_tokens,
                    "temperature": temperature or self.config.temperature,
                    **kwargs
                }
            }
            
            logger.debug(f"Generating with model {self.config.model_name}")
            
            response = await self.client.generate(**generation_params)
            
            # Extract the response text
            response_text = response.get("response", "")
            
            generation_time = time.time() - start_time
            logger.debug(f"Generated {len(response_text)} chars in {generation_time:.2f}s")
            
            return response_text
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the LLM is available and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.client.generate(
                model=self.config.model_name,
                prompt="Hello",
                options={"num_predict": 10}
            )
            return bool(response.get("response"))
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False