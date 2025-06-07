"""Configuration for the Alpha agent."""

from pydantic import BaseModel


class AlphaConfig(BaseModel):
    """Configuration for the Alpha agent."""

    name: str = "alpha"
    model_name: str = "gemma3:1b"
    prompt_version: str = "v1"
    prompts_file: str = "prompts.json"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: float = 30.0