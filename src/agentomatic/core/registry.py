"""Agent registry — auto-discovery and management."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from loguru import logger

from .manifest import AgentManifest, RegisteredAgent


class AgentRegistry:
    """Central registry that auto-discovers and manages agent plugins.

    Discovery scans a directory for Python packages that export:
      - ``manifest: AgentManifest`` in ``__init__.py`` (required)
      - ``node_fn: async (state) -> dict`` in ``__init__.py`` (recommended)
      - ``get_graph()`` in ``graph.py`` (optional)
      - Custom router in ``api.py`` (optional)
      - Config class in ``config.py`` (optional)
      - Prompt versions in ``prompts.json`` (optional)
    """

    def __init__(self) -> None:
        self._agents: dict[str, RegisteredAgent] = {}

    @property
    def count(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)

    def discover(self, agents_dir: Path, package_prefix: str = "") -> None:
        """Auto-discover agents from a directory.

        Args:
            agents_dir: Path to the agents directory.
            package_prefix: Python package prefix for imports.
        """
        agents_dir = Path(agents_dir).resolve()
        if not agents_dir.exists():
            logger.warning(f"Agents directory not found: {agents_dir}")
            return

        logger.info(f"🔍 Discovering agents in {agents_dir}")

        for entry in sorted(agents_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue
            if not (entry / "__init__.py").exists():
                continue

            try:
                self._discover_single(entry, package_prefix)
            except Exception as exc:
                logger.error(f"  ❌ Failed to discover {entry.name}: {exc}")

        logger.info(f"📦 Discovery complete — {self.count} agents registered")

    def _discover_single(self, agent_dir: Path, package_prefix: str) -> None:
        """Discover a single agent from its directory."""
        agent_name = agent_dir.name
        module_path = f"{package_prefix}.{agent_name}" if package_prefix else agent_name

        # Import the agent's __init__.py
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning(f"  ⚠️ Could not import {module_path}: {exc}")
            return

        # Look for manifest
        manifest = getattr(mod, "manifest", None)
        if not isinstance(manifest, AgentManifest):
            logger.debug(f"  ⏭️ Skipping {agent_name} — no AgentManifest")
            return

        # Look for node function
        node_fn = getattr(mod, "node_fn", None)

        # Create registered agent
        agent = RegisteredAgent(
            manifest=manifest,
            node_fn=node_fn,
            module_path=module_path,
        )

        # Discover optional enhancements
        enhancements: list[str] = []

        # graph.py → get_graph()
        agent.graph_fn = self._discover_graph(module_path)
        if agent.graph_fn:
            enhancements.append("+graph")

        # api.py → router (custom or auto-generated)
        agent.router = self._discover_router(module_path, agent_name)
        if agent.router:
            enhancements.append("+router")

        # config.py → agent config
        agent.config = self._discover_config(module_path, agent_name)
        if agent.config:
            enhancements.append("+config")

        # prompts.json → PromptManager
        agent.prompt_manager = self._discover_prompts(agent_dir, agent_name)
        if agent.prompt_manager:
            enhancements.append("+prompts")

        # Register
        self._agents[agent_name] = agent
        extras = " ".join(enhancements) if enhancements else "(minimal)"
        logger.info(f"  ✅ Registered: {agent_name} ({manifest.slug}) {extras}")

    def _discover_graph(self, module_path: str) -> Any:
        """Try to import ``get_graph()`` from agent's ``graph.py``."""
        try:
            graph_mod = importlib.import_module(f"{module_path}.graph")
            return getattr(graph_mod, "get_graph", None)
        except ImportError:
            return None

    def _discover_router(self, module_path: str, agent_name: str) -> Any:
        """Try to import custom router from agent's ``api.py``."""
        try:
            api_mod = importlib.import_module(f"{module_path}.api")
            router = getattr(api_mod, "router", None)
            if router:
                logger.debug(f"  📌 Custom router found for {agent_name}")
            return router
        except ImportError:
            return None

    def _discover_config(self, module_path: str, agent_name: str) -> Any:
        """Try to import config from agent's ``config.py``."""
        try:
            config_mod = importlib.import_module(f"{module_path}.config")
            # Look for a class named {Name}Config
            config_cls_name = f"{agent_name.title().replace('_', '')}Config"
            config_cls = getattr(config_mod, config_cls_name, None)
            if config_cls:
                return config_cls()
            # Fallback: look for 'config' attribute
            return getattr(config_mod, "config", None)
        except ImportError:
            return None

    def _discover_prompts(self, agent_dir: Path, agent_name: str) -> Any:
        """Try to load ``prompts.json`` from agent directory."""
        prompts_file = agent_dir / "prompts.json"
        if prompts_file.exists():
            try:
                from agentomatic.prompts.manager import PromptManager

                pm = PromptManager(agent_name)
                pm.load_from_file(prompts_file)
                return pm
            except Exception as exc:
                logger.warning(
                    f"  ⚠️ Failed to load prompts for {agent_name}: {exc}"
                )
        return None

    # --- Accessors ---

    def get(self, name: str) -> RegisteredAgent | None:
        """Get an agent by name."""
        return self._agents.get(name)

    def all(self) -> dict[str, RegisteredAgent]:
        """Get all registered agents."""
        return dict(self._agents)

    def get_subagents(self) -> dict[str, RegisteredAgent]:
        """Get only subagents (routable by orchestrator)."""
        return {
            name: agent
            for name, agent in self._agents.items()
            if agent.manifest.is_subagent
        }

    def list_names(self) -> list[str]:
        """List all agent names."""
        return list(self._agents.keys())

    def get_agent_routers(self) -> list[tuple[str, Any]]:
        """Get (name, router) pairs for agents with routers."""
        return [
            (name, agent.router)
            for name, agent in self._agents.items()
            if agent.router is not None
        ]

    def get_intent_keywords(self) -> dict[str, list[str]]:
        """Get intent keywords for all agents."""
        return {
            name: list(agent.manifest.intent_keywords)
            for name, agent in self._agents.items()
            if agent.manifest.intent_keywords
        }
