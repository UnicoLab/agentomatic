"""JSON-based prompt version manager."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


class PromptManager:
    """Manages versioned prompt templates loaded from JSON files."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._prompts: dict[str, dict[str, str]] = {}

    def load_from_file(self, path: Path) -> None:
        """Load prompts from a JSON file."""
        try:
            with open(path) as f:
                self._prompts = json.load(f)
            count = len(self._prompts)
            logger.debug(f"Loaded {count} prompt version(s) for '{self.agent_name}'")
        except Exception as exc:
            logger.error(f"Failed to load prompts for '{self.agent_name}': {exc}")

    def get_prompt(self, version: str = "v1", prompt_type: str = "system") -> str | None:
        """Get a raw prompt template by version and type."""
        version_data = self._prompts.get(version)
        if version_data is None:
            return None
        return version_data.get(prompt_type)

    def format_prompt(
        self,
        version: str = "v1",
        prompt_type: str = "user_template",
        **kwargs: Any,
    ) -> str | None:
        """Get and format a prompt with the given variables."""
        template = self.get_prompt(version, prompt_type)
        if template is None:
            return None
        try:
            return template.format_map(kwargs)
        except KeyError:
            return template

    def list_versions(self) -> list[str]:
        """List available prompt versions."""
        return list(self._prompts.keys())

    def reload(self, path: Path) -> None:
        """Reload prompts from disk."""
        self.load_from_file(path)
