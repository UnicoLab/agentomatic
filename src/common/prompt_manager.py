"""Simplified prompt management for the new agent architecture."""

import json
from pathlib import Path
from typing import Any

from loguru import logger


class PromptManager:
    """Simplified prompt manager for agent prompts."""

    def __init__(self, agent_name: str):
        """Initialize prompt manager for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., 'alpha', 'beta')
        """
        self.agent_name = agent_name
        self.prompts_file = Path(f"src/agents/{agent_name}/prompts.json")
        self._prompts: dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from the agent's prompts.json file."""
        try:
            if self.prompts_file.exists():
                with open(self.prompts_file) as f:
                    self._prompts = json.load(f)
                logger.info(f"Loaded {len(self._prompts)} prompts for {self.agent_name}")
            else:
                logger.warning(f"Prompts file not found: {self.prompts_file}")
                self._prompts = {}
        except Exception as e:
            logger.error(f"Failed to load prompts for {self.agent_name}: {e}")
            self._prompts = {}

    def get_prompt(self, version: str = "v1") -> str | None:
        """Get prompt template by version.

        Args:
            version: Prompt version to retrieve

        Returns:
            Prompt template string or None if not found
        """
        prompt_data = self._prompts.get(version)
        if prompt_data and isinstance(prompt_data, dict):
            return prompt_data.get("template")
        return None

    def format_prompt(self, version: str = "v1", **kwargs) -> str | None:
        """Format prompt with context variables.

        Args:
            version: Prompt version to use
            **kwargs: Variables to substitute in the prompt

        Returns:
            Formatted prompt string or None if not found
        """
        template = self.get_prompt(version)
        if not template:
            return None

        try:
            return template.format(**kwargs)
        except Exception as e:
            logger.error(f"Failed to format prompt {self.agent_name}/{version}: {e}")
            return None

    def list_versions(self) -> list[str]:
        """List all available prompt versions."""
        return list(self._prompts.keys())

    def reload(self) -> None:
        """Reload prompts from disk."""
        self._load_prompts()
