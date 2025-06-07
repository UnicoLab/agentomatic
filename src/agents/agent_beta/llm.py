"""LLM wrapper for the Beta agent."""

import time
from typing import Optional
import ollama
from loguru import logger

from ...app.settings import config
from ...common.utilities import retry
from .config import BetaConfig


class BetaLLM:
    """LLM wrapper for Beta agent using Ollama with reasoning focus.
    
    Example:
        llm = BetaLLM()
        response = await llm.generate("Analyze this problem...", max_tokens=1500)
    """
    
    def __init__(self, agent_config: Optional[BetaConfig] = None) -> None:
        self.config = agent_config or BetaConfig()
        self.client = ollama.AsyncClient(host=config.ollama_base_url)
    
    @retry(max_attempts=3, delay=1.0)
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate analytical text using the Ollama model.
        
        Args:
            prompt: The input prompt for analysis
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower for more focused reasoning)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated analytical response
        """
        start_time = time.time()
        
        try:
            generation_params = {
                "model": self.config.model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens or self.config.max_tokens,
                    "temperature": temperature or self.config.temperature,
                    "top_p": 0.9,  # Focused sampling for reasoning
                    "repeat_penalty": 1.1,
                    **kwargs
                }
            }
            
            logger.debug(f"Beta generating with model {self.config.model_name}")
            
            response = await self.client.generate(**generation_params)
            response_text = response.get("response", "")
            
            generation_time = time.time() - start_time
            logger.debug(f"Beta generated {len(response_text)} chars in {generation_time:.2f}s")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Beta LLM generation failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the LLM is available for reasoning tasks."""
        try:
            response = await self.client.generate(
                model=self.config.model_name,
                prompt="Analyze: 1+1=?",
                options={"num_predict": 20}
            )
            return bool(response.get("response"))
        except Exception as e:
            logger.warning(f"Beta LLM health check failed: {e}")
            return False