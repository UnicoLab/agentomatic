"""Agent registry — auto-discovery and management."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from .manifest import AgentManifest, RegisteredAgent

#: Environment variable used to scope which agents are discovered. When set to a
#: comma-separated list of agent names/slugs, only those agents load — used by
#: ``agentomatic deploy --with-agent-stubs`` to give each replica a single agent.
AGENT_ALLOWLIST_ENV = "AGENTOMATIC_AGENTS"


def _agent_allowlist() -> set[str] | None:
    """Return the normalised ``AGENTOMATIC_AGENTS`` allow-list, or ``None``.

    Returns:
        A set of lower-cased agent names/slugs to load, or ``None`` when the
        environment variable is unset/empty (meaning "load everything").
    """
    raw = os.environ.get(AGENT_ALLOWLIST_ENV, "").strip()
    if not raw:
        return None
    return {token.strip().lower() for token in raw.split(",") if token.strip()}


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
        self._slug_index: dict[str, str] = {}
        self.before_node_hooks: list[Callable[[str, dict[str, Any]], None]] = []
        self.after_node_hooks: list[Callable[[str, dict[str, Any]], None]] = []
        self.stack_manager: Any = None

    @property
    def count(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)

    def discover(self, agents_dir: Path, package_prefix: str = "") -> None:
        """Auto-discover agents from a directory.

        Supports both folder-based agents (with ``__init__.py`` +
        ``AgentManifest``) and class-based agents (with
        ``agent.py`` containing a ``BaseGraphAgent`` subclass).

        Args:
            agents_dir: Path to the agents directory.
            package_prefix: Python package prefix for imports.
        """
        agents_dir = Path(agents_dir).resolve()
        if not agents_dir.exists():
            logger.warning(f"Agents directory not found: {agents_dir}")
            return

        logger.info(f"🔍 Discovering agents in {agents_dir}")

        allowlist = _agent_allowlist()
        if allowlist is not None:
            logger.info(f"🔒 {AGENT_ALLOWLIST_ENV} allow-list active: {sorted(allowlist)}")

        for entry in sorted(agents_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue

            # Scope discovery to the allow-list (folder name == agent name).
            # Skipping here also avoids importing non-selected agents' code.
            if allowlist is not None and entry.name.lower() not in allowlist:
                logger.debug(f"  ⏭️ Skipping {entry.name} — not in {AGENT_ALLOWLIST_ENV}")
                continue

            has_init = (entry / "__init__.py").exists()
            has_agent_py = (entry / "agent.py").exists()

            if not has_init and not has_agent_py:
                continue

            try:
                if has_init:
                    # Standard discovery: __init__.py with manifest
                    self._discover_single(entry, package_prefix)
                elif has_agent_py:
                    # Class-agent discovery: agent.py with BaseGraphAgent
                    self._discover_class_agent(entry, package_prefix)
            except Exception as exc:
                logger.error(f"  ❌ Failed to discover {entry.name}: {exc}")

        logger.info(f"📦 Discovery complete — {self.count} agents registered")

    def _discover_single(self, agent_dir: Path, package_prefix: str) -> None:
        """Discover a single agent from its directory.

        Prefer class-based ``agent.py`` (executable graph) when present; the
        ``AgentManifest`` in ``__init__.py`` still provides the agent card
        when no class agent is found (legacy / deepagent / custom templates).
        """
        agent_name = agent_dir.name
        module_path = f"{package_prefix}.{agent_name}" if package_prefix else agent_name

        # Class agents take precedence — they own the graph + studio adapter
        if (agent_dir / "agent.py").exists():
            if self._discover_class_agent(agent_dir, package_prefix):
                # Overlay manifesto card + enhancements from the package
                registered = self._agents.get(agent_name)
                if registered is not None:
                    self._enrich_registered_agent(registered, agent_dir, module_path, agent_name)
                    # Prefer explicit AgentManifest from __init__.py when present
                    try:
                        mod = importlib.import_module(module_path)
                        manifest = getattr(mod, "manifest", None)
                        if isinstance(manifest, AgentManifest):
                            registered.manifest = manifest
                            self._index_slug(agent_name, registered)
                    except ImportError:
                        pass
                return

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
            _studio_graph_fn=getattr(mod, "studio_graph_topology", None),
            _studio_state_fn=getattr(mod, "studio_state_provider", None),
            # Also look for a stream provider if it exists
            _studio_stream_fn=getattr(mod, "studio_stream_provider", None),
        )

        self._enrich_registered_agent(agent, agent_dir, module_path, agent_name)

        # Register
        self._agents[agent_name] = agent
        self._index_slug(agent_name, agent)
        extras = self._enhancement_summary(agent)
        logger.info(f"  ✅ Registered: {agent_name} ({manifest.slug}) {extras}")

    def _enhancement_summary(self, agent: RegisteredAgent) -> str:
        """Return a short log string of discovered enhancements."""
        flags: list[str] = []
        if agent.graph_fn:
            flags.append("+graph")
        if agent.router:
            flags.append("+router")
        if agent.config:
            flags.append("+config")
        if agent.prompt_manager:
            flags.append("+prompts")
        if agent.llm_config:
            flags.append("+llm")
        if agent.schema_validator:
            flags.append("+schemas")
        if agent.security_policy:
            flags.append("+security")
        if agent.delegation_config:
            flags.append("+delegation")
        if agent.connections:
            flags.append("+connections")
        return " ".join(flags) if flags else "(minimal)"

    def _enrich_registered_agent(
        self,
        agent: RegisteredAgent,
        agent_dir: Path,
        module_path: str,
        agent_name: str,
    ) -> None:
        """Attach optional package enhancements onto a RegisteredAgent."""
        if not agent.graph_fn:
            agent.graph_fn = self._discover_graph(module_path)
        if not agent.router:
            agent.router = self._discover_router(module_path, agent_name)
        if not agent.config:
            agent.config = self._discover_config(module_path, agent_name)
        if not agent.prompt_manager:
            agent.prompt_manager = self._discover_prompts(agent_dir, agent_name)
        if not agent.llm_config:
            agent.llm_config = self._discover_llm_config(module_path, agent_name)
        if not agent.schema_validator:
            agent.schema_validator = self._discover_schemas(module_path, agent_name)
        if not agent.security_policy:
            agent.security_policy = self._discover_security(module_path, agent_name)
        if not agent.delegation_config:
            agent.delegation_config = self._discover_delegation(module_path, agent_name)
        if not agent.connections:
            agent.connections = self._discover_connections(module_path, agent_name)

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
                logger.warning(f"  ⚠️ Failed to load prompts for {agent_name}: {exc}")
        return None

    def _discover_llm_config(self, module_path: str, agent_name: str) -> Any:
        """Try to import ``AgentLLMConfig`` from agent's ``llm.py``."""
        try:
            llm_mod = importlib.import_module(f"{module_path}.llm")
            config_cls = getattr(llm_mod, "AgentLLMConfig", None)
            if config_cls:
                return config_cls()
            return getattr(llm_mod, "llm_config", None)
        except ImportError:
            return None

    def _discover_schemas(self, module_path: str, agent_name: str) -> Any:
        """Try to import Request/Response schemas from agent's ``schemas.py``."""
        try:
            schemas_mod = importlib.import_module(f"{module_path}.schemas")
            # Convention: {Title}Request and {Title}Response
            title = agent_name.title().replace("_", "")
            request_model = getattr(schemas_mod, f"{title}Request", None)
            response_model = getattr(schemas_mod, f"{title}Response", None)
            if request_model or response_model:
                from agentomatic.core.schemas import SchemaValidator

                return SchemaValidator(
                    request_model=request_model,
                    response_model=response_model,
                )
            return None
        except ImportError:
            return None

    def _discover_security(self, module_path: str, agent_name: str) -> Any:
        """Try to import ``AgentSecurityPolicy`` from agent's ``security.py``."""
        try:
            sec_mod = importlib.import_module(f"{module_path}.security")
            return getattr(sec_mod, "policy", None)
        except ImportError:
            return None

    def _discover_connections(self, module_path: str, agent_name: str) -> Any:
        """Try to import connection configs from agent's ``connections.py``.

        Looks for a ``connections`` (or ``CONNECTIONS``) attribute containing
        a list of connection configuration objects.
        """
        try:
            conn_mod = importlib.import_module(f"{module_path}.connections")
        except ImportError:
            return None
        configs = getattr(conn_mod, "connections", None) or getattr(conn_mod, "CONNECTIONS", None)
        if configs:
            return list(configs)
        return None

    def _discover_delegation(self, module_path: str, agent_name: str) -> Any:
        """Try to import delegation config from agent's ``delegation.py``."""
        try:
            del_mod = importlib.import_module(f"{module_path}.delegation")
            # Look for DELEGATION_TARGETS list or get_handoff_tools function
            targets = getattr(del_mod, "DELEGATION_TARGETS", None)
            get_tools = getattr(del_mod, "get_handoff_tools", None)
            if targets or get_tools:
                return {
                    "targets": targets or [],
                    "get_handoff_tools": get_tools,
                }
            return None
        except ImportError:
            return None

    # --- Accessors ---

    def _index_slug(self, name: str, agent: RegisteredAgent) -> None:
        """Index an agent by its slug for name-or-slug lookups."""
        slug = getattr(agent.manifest, "slug", None) if agent.manifest else None
        if slug and slug != name:
            self._slug_index[slug] = name

    def get(self, name: str) -> RegisteredAgent | None:
        """Get an agent by folder name or slug.

        Studio and other clients often address agents by ``manifest.slug``,
        while registration keys them by folder ``name``.  This lookup accepts
        either so the Studio UI does not 404 when slug != name.
        """
        agent = self._agents.get(name)
        if agent is not None:
            return agent
        mapped = self._slug_index.get(name)
        if mapped is not None:
            return self._agents.get(mapped)
        # Last resort: scan by slug (covers agents registered before index)
        for agent in self._agents.values():
            if agent.manifest and agent.manifest.slug == name:
                return agent
        return None

    def all(self) -> dict[str, RegisteredAgent]:
        """Get all registered agents."""
        return dict(self._agents)

    def get_subagents(self) -> dict[str, RegisteredAgent]:
        """Get only subagents (routable by orchestrator)."""
        return {name: agent for name, agent in self._agents.items() if agent.manifest.is_subagent}

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

    # ------------------------------------------------------------------
    # Class-based agent registration (v0.7)
    # ------------------------------------------------------------------

    def register_class_agent(self, agent: Any) -> None:
        """Register a class-based agent (``BaseGraphAgent`` subclass).

        Converts the agent instance to a ``RegisteredAgent`` and
        registers it alongside folder-based agents.

        Args:
            agent: A ``BaseGraphAgent`` instance.

        Example::

            from agentomatic.agents import BaseGraphAgent
            registry.register_class_agent(my_agent)
        """
        try:
            registered = agent.as_registered_agent()
            name = registered.name
            self._agents[name] = registered
            self._index_slug(name, registered)
            logger.info(f"  ✅ Class-agent registered: {name}")
        except Exception as exc:
            logger.error(f"  ❌ Failed to register class-agent: {exc}")

    def _discover_class_agent(
        self,
        agent_dir: Path,
        package_prefix: str,
    ) -> bool:
        """Try to discover a class-based agent from ``agent.py``.

        Returns True if a class-agent was found and registered.
        """
        agent_file = agent_dir / "agent.py"
        if not agent_file.exists():
            return False

        agent_name = agent_dir.name
        module_path = (
            f"{package_prefix}.{agent_name}.agent" if package_prefix else f"{agent_name}.agent"
        )

        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            return False

        # Look for BaseGraphAgent subclasses
        from agentomatic.agents.base import BaseGraphAgent

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseGraphAgent)
                and obj is not BaseGraphAgent
            ):
                try:
                    llm = self._resolve_agent_llm(agent_name, agent_dir, package_prefix)
                    prompt_manager = self._discover_prompts(agent_dir, agent_name)
                    instance = self._instantiate_class_agent(
                        obj, llm=llm, prompt_manager=prompt_manager
                    )
                    self.register_class_agent(instance)
                    return True
                except Exception as exc:
                    logger.warning(f"  ⚠️ Could not instantiate {attr_name}: {exc}")
        return False

    def _resolve_agent_llm(
        self,
        agent_name: str,
        agent_dir: Path,
        package_prefix: str,
    ) -> Any:
        """Resolve an LLM for *agent_name* from llm.py + stack / global get_llm."""
        module_path = f"{package_prefix}.{agent_name}" if package_prefix else agent_name
        llm_config = self._discover_llm_config(module_path, agent_name)
        role = "default"
        if llm_config is not None:
            roles = getattr(llm_config, "roles", None)
            if isinstance(roles, dict) and roles:
                role = roles.get("default", next(iter(roles.values())))
        try:
            from agentomatic.providers.llm import get_llm_for_agent

            return get_llm_for_agent(
                agent_name,
                role=role,
                stack_manager=self.stack_manager,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not resolve LLM for {}: {}", agent_name, exc)
            return None

    @staticmethod
    def _instantiate_class_agent(
        cls: type,
        *,
        llm: Any = None,
        prompt_manager: Any = None,
    ) -> Any:
        """Instantiate a BaseGraphAgent, injecting llm/prompt_manager when accepted."""
        import inspect

        try:
            sig = inspect.signature(cls)
            params = dict(sig.parameters)
        except (TypeError, ValueError):
            params = {}

        kwargs: dict[str, Any] = {}
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if llm is not None and ("llm" in params or accepts_kwargs):
            kwargs["llm"] = llm
        if prompt_manager is not None and ("prompt_manager" in params or accepts_kwargs):
            kwargs["prompt_manager"] = prompt_manager

        try:
            return cls(**kwargs)
        except TypeError:
            # Older agents that only take llm=
            if "llm" in kwargs and "prompt_manager" in kwargs:
                try:
                    return cls(llm=kwargs["llm"])
                except TypeError:
                    pass
            return cls()
