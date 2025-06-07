"""Configuration for the Beta agent."""

from pydantic import BaseModel


class BetaConfig(BaseModel):
    """Configuration for the Beta agent."""

    name: str = "beta"
    model_name: str = "gemma3:1b"
    prompt_version: str = "v1"
    prompts_file: str = "prompts.json"
    max_tokens: int = 1500
    temperature: float = 0.5
    timeout: float = 45.0
    enable_reasoning: bool = True