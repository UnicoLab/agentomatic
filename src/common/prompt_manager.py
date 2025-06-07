"""Prompt management system with versioning and Python file support."""

import os
import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class PromptVersion:
    """Represents a versioned prompt."""
    version: str
    content: str
    variables: List[str]
    created_at: datetime
    description: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class PromptManager:
    """Manages prompts with versioning support.

    Supports both JSON and Python file formats:
    - JSON: Traditional format for simple prompts
    - Python: Dynamic prompts with logic and variables

    Example Python prompt file (prompts/agent_alpha/v1.py):
        def get_prompt(context):
            return f"You are an AI assistant. User: {context['user_input']}"

        VARIABLES = ["user_input"]
        DESCRIPTION = "Main prompt for agent alpha"
        TAGS = ["assistant", "main"]
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Dict[str, PromptVersion]] = {}
        self._ensure_prompts_dir()
        self._load_all_prompts()

    def _ensure_prompts_dir(self) -> None:
        """Create prompts directory if it doesn't exist."""
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # Create example prompts for agents
        for agent in ["agent_alpha", "agent_beta"]:
            agent_dir = self.prompts_dir / agent
            agent_dir.mkdir(exist_ok=True)

            # Create example JSON prompt
            json_file = agent_dir / "prompts.json"
            if not json_file.exists():
                example_prompts = {
                    "v1": {
                        "content": f"You are {agent}. Help the user with their request: {{user_input}}",
                        "variables": ["user_input"],
                        "description": f"Main prompt for {agent}",
                        "tags": ["main", "assistant"]
                    }
                }
                with open(json_file, 'w') as f:
                    json.dump(example_prompts, f, indent=2)

            # Create example Python prompt
            py_file = agent_dir / "v1.py"
            if not py_file.exists():
                py_content = f'''"""Dynamic prompt for {agent} v1."""

def get_prompt(context):
    """Generate prompt based on context."""
    user_input = context.get("user_input", "")
    mood = context.get("mood", "helpful")

    return f"""You are {agent}, a {mood} AI assistant.

User request: {user_input}

Please provide a helpful response."""

VARIABLES = ["user_input", "mood"]
DESCRIPTION = "Dynamic prompt for {agent} with mood context"
TAGS = ["dynamic", "mood", "assistant"]
'''
                with open(py_file, 'w') as f:
                    f.write(py_content)

    def _load_all_prompts(self) -> None:
        """Load all prompts from the prompts directory."""
        self._cache.clear()

        for agent_dir in self.prompts_dir.iterdir():
            if agent_dir.is_dir():
                agent_name = agent_dir.name
                self._cache[agent_name] = {}

                # Load JSON prompts
                json_file = agent_dir / "prompts.json"
                if json_file.exists():
                    self._load_json_prompts(agent_name, json_file)

                # Load Python prompts
                for py_file in agent_dir.glob("*.py"):
                    if py_file.stem not in ["__init__", "__pycache__"]:
                        self._load_python_prompt(agent_name, py_file)

        logger.info(f"Loaded prompts for {len(self._cache)} agents")

    def _load_json_prompts(self, agent_name: str, json_file: Path) -> None:
        """Load prompts from JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            for version, prompt_data in data.items():
                prompt_version = PromptVersion(
                    version=version,
                    content=prompt_data["content"],
                    variables=prompt_data.get("variables", []),
                    created_at=datetime.now(),
                    description=prompt_data.get("description"),
                    tags=prompt_data.get("tags", [])
                )
                self._cache[agent_name][version] = prompt_version

        except Exception as e:
            logger.error(f"Failed to load JSON prompts from {json_file}: {e}")

    def _load_python_prompt(self, agent_name: str, py_file: Path) -> None:
        """Load prompt from Python file."""
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Validate required function exists
            if not hasattr(module, 'get_prompt'):
                logger.warning(f"Python prompt {py_file} missing get_prompt function")
                return

            version = py_file.stem
            variables = getattr(module, 'VARIABLES', [])
            description = getattr(module, 'DESCRIPTION', None)
            tags = getattr(module, 'TAGS', [])

            # Store the module for dynamic execution
            prompt_version = PromptVersion(
                version=version,
                content=str(py_file),  # Store file path for Python prompts
                variables=variables,
                created_at=datetime.now(),
                description=description,
                tags=tags
            )

            # Add reference to the module for execution
            prompt_version._module = module

            self._cache[agent_name][version] = prompt_version

        except Exception as e:
            logger.error(f"Failed to load Python prompt from {py_file}: {e}")

    def get_prompt(self, agent_name: str, version: str = "v1") -> Optional[PromptVersion]:
        """Get a specific prompt version.

        Args:
            agent_name: Name of the agent
            version: Version of the prompt (default: "v1")

        Returns:
            PromptVersion object or None if not found
        """
        return self._cache.get(agent_name, {}).get(version)

    def format_prompt(
        self,
        agent_name: str,
        version: str = "v1",
        **context
    ) -> Optional[str]:
        """Format a prompt with context variables.

        Args:
            agent_name: Name of the agent
            version: Version of the prompt
            **context: Variables to substitute in the prompt

        Returns:
            Formatted prompt string or None if not found

        Example:
            formatted = manager.format_prompt(
                "agent_alpha",
                "v1",
                user_input="Hello world",
                mood="friendly"
            )
        """
        prompt_version = self.get_prompt(agent_name, version)
        if not prompt_version:
            logger.warning(f"Prompt not found: {agent_name}/{version}")
            return None

        try:
            # Handle Python prompts
            if hasattr(prompt_version, '_module'):
                return prompt_version._module.get_prompt(context)

            # Handle JSON prompts (string formatting)
            return prompt_version.content.format(**context)

        except Exception as e:
            logger.error(f"Failed to format prompt {agent_name}/{version}: {e}")
            return None

    def list_agents(self) -> List[str]:
        """List all available agents."""
        return list(self._cache.keys())

    def list_versions(self, agent_name: str) -> List[str]:
        """List all versions for an agent."""
        return list(self._cache.get(agent_name, {}).keys())

    def get_agent_prompts(self, agent_name: str) -> Dict[str, PromptVersion]:
        """Get all prompts for an agent."""
        return self._cache.get(agent_name, {}).copy()

    def add_prompt(
        self,
        agent_name: str,
        version: str,
        content: str,
        variables: List[str] = None,
        description: str = None,
        tags: List[str] = None,
        save_to_file: bool = True
    ) -> bool:
        """Add a new prompt version.

        Args:
            agent_name: Name of the agent
            version: Version identifier
            content: Prompt content
            variables: List of variables used in the prompt
            description: Optional description
            tags: Optional tags
            save_to_file: Whether to save to JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            if agent_name not in self._cache:
                self._cache[agent_name] = {}

            prompt_version = PromptVersion(
                version=version,
                content=content,
                variables=variables or [],
                created_at=datetime.now(),
                description=description,
                tags=tags or []
            )

            self._cache[agent_name][version] = prompt_version

            if save_to_file:
                self._save_json_prompts(agent_name)

            logger.info(f"Added prompt {agent_name}/{version}")
            return True

        except Exception as e:
            logger.error(f"Failed to add prompt {agent_name}/{version}: {e}")
            return False

    def _save_json_prompts(self, agent_name: str) -> None:
        """Save agent prompts to JSON file."""
        agent_dir = self.prompts_dir / agent_name
        agent_dir.mkdir(exist_ok=True)

        json_file = agent_dir / "prompts.json"
        prompts_data = {}

        for version, prompt_version in self._cache[agent_name].items():
            # Only save JSON prompts (not Python ones)
            if not hasattr(prompt_version, '_module'):
                prompts_data[version] = {
                    "content": prompt_version.content,
                    "variables": prompt_version.variables,
                    "description": prompt_version.description,
                    "tags": prompt_version.tags
                }

        if prompts_data:
            with open(json_file, 'w') as f:
                json.dump(prompts_data, f, indent=2)

    def reload_prompts(self) -> None:
        """Reload all prompts from disk."""
        logger.info("Reloading prompts from disk")
        self._load_all_prompts()

    def get_prompt_info(self, agent_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a prompt."""
        prompt_version = self.get_prompt(agent_name, version)
        if not prompt_version:
            return None

        info = asdict(prompt_version)
        info["created_at"] = prompt_version.created_at.isoformat()
        info["is_python"] = hasattr(prompt_version, '_module')

        return info

    def search_prompts(
        self,
        query: str = None,
        tags: List[str] = None,
        agent_name: str = None
    ) -> Dict[str, Dict[str, PromptVersion]]:
        """Search prompts by query, tags, or agent name.

        Args:
            query: Text to search in content and description
            tags: Tags to filter by
            agent_name: Specific agent to search in

        Returns:
            Dictionary of matching prompts grouped by agent
        """
        results = {}

        agents_to_search = [agent_name] if agent_name else self._cache.keys()

        for agent in agents_to_search:
            if agent not in self._cache:
                continue

            agent_results = {}

            for version, prompt_version in self._cache[agent].items():
                match = True

                # Check query match
                if query:
                    query_lower = query.lower()
                    if not (
                        query_lower in prompt_version.content.lower() or
                        (prompt_version.description and query_lower in prompt_version.description.lower())
                    ):
                        match = False

                # Check tags match
                if tags and match:
                    if not any(tag in prompt_version.tags for tag in tags):
                        match = False

                if match:
                    agent_results[version] = prompt_version

            if agent_results:
                results[agent] = agent_results

        return results