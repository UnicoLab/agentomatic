"""AgentPlatform — the main entry point for agentomatic."""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .lifespan import configure_logging
from .manifest import AgentManifest, RegisteredAgent
from .registry import AgentRegistry
from .router_factory import create_default_router


class AgentPlatform:
    """Zero-code multi-agent API platform.

    Usage::

        from agentomatic import AgentPlatform

        platform = AgentPlatform.from_folder("agents/")
        app = platform.build()
        # uvicorn main:app --reload
    """

    def __init__(
        self,
        agents_dir: str | Path = "agents/",
        *,
        title: str = "Agentomatic Platform",
        description: str = "Multi-agent API platform powered by Agentomatic",
        version: str = "1.0.0",
        api_prefix: str = "/api/v1",
        package_prefix: str = "",
        cors_origins: list[str] | None = None,
        log_level: str = "INFO",
        settings: Any = None,
    ) -> None:
        """Initialise the platform.

        Args:
            agents_dir: Filesystem path containing agent packages.
            title: Application title shown in docs.
            description: Application description shown in docs.
            version: Semantic version string.
            api_prefix: URL prefix for all agent endpoints.
            package_prefix: Python import prefix for agents.
            cors_origins: Allowed CORS origins (default ``["*"]``).
            log_level: Root log level.
            settings: Optional settings object forwarded to hooks.
        """
        self.agents_dir = Path(agents_dir).resolve()
        self.title = title
        self.description = description
        self.version = version
        self.api_prefix = api_prefix
        self.package_prefix = package_prefix
        self.cors_origins = cors_origins or ["*"]
        self.log_level = log_level
        self.settings = settings

        self._registry = AgentRegistry()
        self._on_startup: list[Callable[..., Any]] = []
        self._on_shutdown: list[Callable[..., Any]] = []
        self._extra_routers: list[tuple[str, Any, dict[str, Any]]] = []
        self._app: FastAPI | None = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_folder(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> AgentPlatform:
        """Create a platform from an agents directory.

        Args:
            path: Path to the agents directory.
            **kwargs: Additional arguments forwarded to ``__init__``.

        Returns:
            A new :class:`AgentPlatform` instance.
        """
        return cls(agents_dir=path, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry."""
        return self._registry

    # ------------------------------------------------------------------
    # Programmatic registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        manifest: AgentManifest,
        node_fn: Callable[..., Awaitable[Any]] | None = None,
        graph_fn: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register an agent programmatically (no folder needed).

        Args:
            manifest: Agent identity manifest.
            node_fn: Async function to invoke the agent.
            graph_fn: Function returning a compiled graph.
            **kwargs: Extra keyword arguments forwarded to
                :class:`RegisteredAgent`.
        """
        agent = RegisteredAgent(
            manifest=manifest,
            node_fn=node_fn,
            graph_fn=graph_fn,
            **kwargs,
        )
        self._registry._agents[manifest.name] = agent  # noqa: SLF001
        logger.info(
            f"  ✅ Programmatically registered: {manifest.name} ({manifest.slug})"
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_startup(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a startup hook (decorator).

        Args:
            fn: Callable to run during application startup.

        Returns:
            The original callable (unchanged).
        """
        self._on_startup.append(fn)
        return fn

    def on_shutdown(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a shutdown hook (decorator).

        Args:
            fn: Callable to run during application shutdown.

        Returns:
            The original callable (unchanged).
        """
        self._on_shutdown.append(fn)
        return fn

    # ------------------------------------------------------------------
    # Custom routers
    # ------------------------------------------------------------------

    def include_router(self, router: Any, prefix: str = "", **kwargs: Any) -> None:
        """Add a custom router to the platform.

        Args:
            router: A :class:`~fastapi.APIRouter` instance.
            prefix: URL prefix.
            **kwargs: Extra arguments forwarded to
                :meth:`~fastapi.FastAPI.include_router`.
        """
        self._extra_routers.append((prefix, router, kwargs))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> FastAPI:
        """Build and return the FastAPI application.

        This is the main method.  It:
          1. Creates the FastAPI app with lifespan
          2. Adds middleware (CORS, etc.)
          3. Discovers agents from the folder
          4. Auto-generates endpoints per agent
          5. Mounts everything

        Returns:
            Configured :class:`~fastapi.FastAPI` application.
        """
        platform = self  # capture for closure

        # Track which agents are already pre-registered (programmatic)
        _pre_registered = set(platform._registry.list_names())

        @asynccontextmanager
        async def lifespan(app: FastAPI):  # noqa: ARG001
            """Manage startup / shutdown lifecycle."""
            # --- Startup ---
            configure_logging(platform.log_level)
            logger.info(f"🚀 {platform.title} starting...")
            logger.info(f"📂 Agents directory: {platform.agents_dir}")

            # Ensure agents directory is importable
            parent = str(platform.agents_dir.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)

            # Auto-discover agents from folder (skips already-registered)
            prefix = platform.package_prefix or platform.agents_dir.name
            platform._registry.discover(platform.agents_dir, prefix)

            # Auto-generate + mount routers for NEWLY discovered agents
            for name, agent in platform._registry.all().items():
                if name in _pre_registered:
                    continue  # already mounted at build-time
                if agent.router is None and agent.manifest.is_subagent:
                    agent.router = create_default_router(
                        agent_name=name,
                        registry=platform._registry,
                    )
                    logger.debug(f"  📌 Auto-generated router for {name}")
                if agent.router and agent.manifest.is_subagent:
                    app.include_router(
                        agent.router,
                        prefix=f"{platform.api_prefix}/{name}",
                        tags=[name.title()],
                    )
                    logger.info(f"  🔌 Mounted: {platform.api_prefix}/{name}")

            # Run startup hooks
            for hook in platform._on_startup:
                result = hook()
                if hasattr(result, "__await__"):
                    await result

            logger.info(f"✅ Platform ready — {platform._registry.count} agent(s)")
            yield

            # --- Shutdown ---
            for hook in platform._on_shutdown:
                result = hook()
                if hasattr(result, "__await__"):
                    await result
            logger.info("🛑 Platform stopped")

        # Create FastAPI app
        app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.version,
            lifespan=lifespan,
        )

        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ------------------------------------------------------------------
        # Mount routers for PRE-REGISTERED (programmatic) agents immediately
        # ------------------------------------------------------------------
        for name, agent in self._registry.all().items():
            if agent.router is None and agent.manifest.is_subagent:
                agent.router = create_default_router(
                    agent_name=name,
                    registry=self._registry,
                )
            if agent.router and agent.manifest.is_subagent:
                app.include_router(
                    agent.router,
                    prefix=f"{self.api_prefix}/{name}",
                    tags=[name.title()],
                )

        # ------------------------------------------------------------------
        # Platform-level endpoints
        # ------------------------------------------------------------------

        @app.get("/health")
        async def health() -> dict[str, Any]:
            """Aggregate health across all agents."""
            agents: dict[str, Any] = {}
            for name, agent in self._registry.all().items():
                try:
                    agents[name] = await agent.health_check()
                except Exception as exc:  # noqa: BLE001
                    agents[name] = {"status": "error", "error": str(exc)}
            overall = (
                "healthy"
                if all(a.get("status") == "healthy" for a in agents.values())
                else "degraded"
            )
            return {
                "status": overall,
                "agents": agents,
                "agent_count": len(agents),
            }

        @app.get("/readiness")
        async def readiness() -> dict[str, Any]:
            """Kubernetes-style readiness probe."""
            return {"status": "ready", "agents": self._registry.count}

        # A2A discovery
        @app.get("/.well-known/agent.json")
        async def a2a_discovery() -> dict[str, Any]:
            """Return A2A agent cards for all registered agents."""
            cards: dict[str, Any] = {}
            for name, agent in self._registry.all().items():
                m = agent.manifest
                cards[name] = {
                    "name": m.slug,
                    "description": m.description,
                    "version": m.version,
                    "endpoints": {
                        "invoke": f"{self.api_prefix}/{name}/invoke",
                        "chat": f"{self.api_prefix}/{name}/chat",
                    },
                }
            return {
                "platform": self.title,
                "version": self.version,
                "agents": cards,
            }

        # Agents list
        @app.get(f"{self.api_prefix}/agents")
        async def list_agents() -> dict[str, Any]:
            """List all registered agents."""
            return {
                "agents": {
                    name: {
                        "slug": a.slug,
                        "description": a.manifest.description,
                        "version": a.manifest.version,
                        "framework": a.manifest.framework,
                    }
                    for name, a in self._registry.all().items()
                }
            }

        # Root
        @app.get("/")
        async def root() -> dict[str, Any]:
            """Platform index."""
            return {
                "name": self.title,
                "version": self.version,
                "agents": self._registry.count,
                "docs": "/docs",
                "health": "/health",
                "a2a": "/.well-known/agent.json",
            }

        # Extra routers
        for prefix, router, kwargs in self._extra_routers:
            app.include_router(router, prefix=prefix, **kwargs)

        self._app = app
        return app

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 8000,
        reload: bool = False,
        workers: int = 1,
        **kwargs: Any,
    ) -> None:
        """Build and run the platform with uvicorn.

        Args:
            host: Bind address.
            port: Bind port.
            reload: Enable auto-reload (development).
            workers: Number of worker processes.
            **kwargs: Extra arguments forwarded to :func:`uvicorn.run`.
        """
        import uvicorn

        app = self.build()
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            **kwargs,
        )
