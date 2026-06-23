"""JSON-based prompt version manager."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


class PromptManager:
    """Manages versioned prompt templates loaded from JSON files."""

    def __init__(self, agent_name: str, prompts_file: Path | str | None = None) -> None:
        self.agent_name = agent_name
        self._prompts: dict[str, dict[str, Any]] = {}
        if prompts_file:
            self.load_from_file(Path(prompts_file))

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

    # -----------------------------------------------------------------
    # LangChain integration
    # -----------------------------------------------------------------

    def as_langchain_template(
        self,
        version: str = "v1",
        prompt_type: str = "system",
    ) -> Any:
        """Convert a prompt version to a LangChain ``PromptTemplate``.

        Requires ``langchain-core`` to be installed.

        Args:
            version: Prompt version key (e.g. ``"v1"``).
            prompt_type: Prompt type within the version (e.g. ``"system"``).

        Returns:
            A ``PromptTemplate`` instance, or ``None`` if the version/type
            is not found or ``langchain-core`` is not installed.
        """
        raw = self.get_prompt(version, prompt_type)
        if raw is None:
            return None
        try:
            from langchain_core.prompts import PromptTemplate

            return PromptTemplate.from_template(raw)
        except ImportError:
            logger.warning(
                "langchain-core is not installed — returning None. "
                "Install with: pip install langchain-core"
            )
            return None

    def as_chat_template(
        self,
        version: str = "v1",
    ) -> Any:
        """Convert a prompt version to a LangChain ``ChatPromptTemplate``.

        Builds a chat template from the ``"system"`` and ``"user_template"``
        entries in the given version.  Requires ``langchain-core``.

        Args:
            version: Prompt version key.

        Returns:
            A ``ChatPromptTemplate`` instance, or ``None`` if the version
            is missing or ``langchain-core`` is not installed.
        """
        version_data = self._prompts.get(version)
        if version_data is None:
            return None
        try:
            from langchain_core.prompts import ChatPromptTemplate

            messages: list[tuple[str, str]] = []
            if "system" in version_data:
                messages.append(("system", version_data["system"]))
            if "user_template" in version_data:
                messages.append(("human", version_data["user_template"]))
            if not messages:
                return None
            return ChatPromptTemplate.from_messages(messages)
        except ImportError:
            logger.warning(
                "langchain-core is not installed — returning None. "
                "Install with: pip install langchain-core"
            )
            return None
