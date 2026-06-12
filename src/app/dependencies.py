"""Application dependencies and agent registry."""

import importlib
from pathlib import Path

from loguru import logger

from ..common.base_agent import BaseAgent
from ..common.llm_factory import LLMConfig, LLMProvider
from .settings import config


class AgentRegistry:
    """Registry for discovering and managing agents."""

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        self._agent_configs: dict[str, LLMConfig] = {}

    def discover_agents(self) -> None:
        """Discover all agents in the agents package automatically."""
        agents_dir = Path("src/agents")

        if not agents_dir.exists():
            logger.warning("Agents directory not found")
            return

        # Look for agent directories - both old pattern (agent_*) and new simplified pattern
        for agent_dir in agents_dir.iterdir():
            if agent_dir.is_dir() and (agent_dir / "__init__.py").exists():
                # Handle old pattern: agent_alpha, agent_beta
                if agent_dir.name.startswith("agent_"):
                    agent_name = agent_dir.name.replace("agent_", "")
                    self._discover_agent_old_pattern(agent_name, agent_dir)
                # Handle new simplified pattern: any valid directory name
                elif not agent_dir.name.startswith("_"):  # Skip private directories
                    agent_name = agent_dir.name
                    self._discover_agent_new_pattern(agent_name, agent_dir)

    def _discover_agent_old_pattern(self, agent_name: str, agent_dir: Path) -> None:
        """Discover and register an agent using the old pattern."""
        try:
            # Import the agent module (old pattern)
            module_path = f"src.agents.agent_{agent_name}"

            # Try to import the main agent class
            try:
                agent_module = importlib.import_module(f"{module_path}.agent")
                agent_class = getattr(agent_module, f"{agent_name.title()}Agent", None)

                if agent_class and issubclass(agent_class, BaseAgent):
                    # Create default LLM config for this agent
                    llm_config = self._create_default_llm_config(agent_name)

                    # Initialize the agent
                    agent_instance = agent_class(agent_name, llm_config)
                    self._agents[agent_name] = agent_instance
                    self._agent_configs[agent_name] = llm_config

                    logger.info(f"Registered agent (old pattern): {agent_name}")
                else:
                    logger.warning(f"No valid agent class found in {module_path}.agent")

            except ImportError as e:
                logger.warning(f"Could not import agent {agent_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover agent {agent_name}: {e}")

    def _discover_agent_new_pattern(self, agent_name: str, agent_dir: Path) -> None:
        """Discover and register an agent using the new simplified pattern."""
        try:
            # Import the agent module (new pattern)
            module_path = f"src.agents.{agent_name}"

            # Try to import the agent instance directly
            try:
                agent_module = importlib.import_module(f"{module_path}.agent")
                agent_instance = getattr(agent_module, "agent", None)

                if agent_instance and isinstance(agent_instance, BaseAgent):
                    # Register the pre-instantiated agent
                    self._agents[agent_name] = agent_instance
                    self._agent_configs[agent_name] = agent_instance.llm.config

                    logger.info(f"Registered agent (new pattern): {agent_name}")
                else:
                    logger.warning(f"No valid agent instance found in {module_path}.agent")

            except ImportError as e:
                logger.warning(f"Could not import agent {agent_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover agent {agent_name}: {e}")

    def _discover_agent(self, agent_name: str, agent_dir: Path) -> None:
        """Discover and register a specific agent (legacy method)."""
        # This method is kept for backward compatibility but now calls the old pattern method
        self._discover_agent_old_pattern(agent_name, agent_dir)

    def _create_default_llm_config(self, agent_name: str) -> LLMConfig:
        """Create default LLM configuration for an agent."""
        if config.default_llm_provider == LLMProvider.OLLAMA:
            return LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name=config.ollama_model,
                temperature=config.default_temperature,
                max_tokens=config.default_max_tokens,
                streaming=config.enable_streaming,
                timeout=config.default_timeout,
                base_url=config.ollama_base_url,
            )
        elif config.default_llm_provider == LLMProvider.GEMINI:
            return LLMConfig(
                provider=LLMProvider.GEMINI,
                model_name=config.gemini_model,
                temperature=config.default_temperature,
                max_tokens=config.default_max_tokens,
                streaming=config.enable_streaming,
                timeout=config.default_timeout,
                project_id=config.gemini_project_id,
                location=config.gemini_location,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.default_llm_provider}")

    def register_agent(self, agent_name: str, agent: BaseAgent) -> None:
        """Manually register an agent."""
        self._agents[agent_name] = agent
        logger.info(f"Manually registered agent: {agent_name}")

    def register_agent_by_name(self, agent_name: str) -> None:
        """Register a specific agent by name."""
        agents_dir = Path("src/agents")

        if not agents_dir.exists():
            logger.warning("Agents directory not found")
            return

        # Look for the specific agent directory
        agent_dir = agents_dir / agent_name
        if agent_dir.exists() and agent_dir.is_dir() and (agent_dir / "__init__.py").exists():
            # Handle new simplified pattern: alpha, beta
            self._discover_agent_new_pattern(agent_name, agent_dir)
        else:
            # Try old pattern
            old_pattern_dir = agents_dir / f"agent_{agent_name}"
            if (
                old_pattern_dir.exists()
                and old_pattern_dir.is_dir()
                and (old_pattern_dir / "__init__.py").exists()
            ):
                self._discover_agent_old_pattern(agent_name, old_pattern_dir)
            else:
                logger.error(f"Agent directory not found for: {agent_name}")

    def get_agent(self, agent_name: str) -> BaseAgent | None:
        """Get an agent by name."""
        return self._agents.get(agent_name)

    def list_agents(self) -> dict[str, dict[str, str]]:
        """List all registered agents with their information."""
        return {
            name: {
                "class": agent.__class__.__name__,
                "llm_provider": agent.llm.config.provider.value,
                "model": agent.llm.config.model_name,
                "status": "registered",
            }
            for name, agent in self._agents.items()
        }

    def get_agent_count(self) -> int:
        """Get the number of registered agents."""
        return len(self._agents)

    def health_check_all(self) -> dict[str, dict]:
        """Run health checks on all agents."""
        results = {}
        for name, agent in self._agents.items():
            try:
                results[name] = agent.health_check()
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        return results


# Global agent registry instance
agent_registry = AgentRegistry()
