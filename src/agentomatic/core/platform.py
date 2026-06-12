"""AgentPlatform — the main entry point for agentomatic."""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .lifespan import configure_logging
from .manifest import AgentManifest, RegisteredAgent
from .registry import AgentRegistry
from .router_factory import create_default_router

if TYPE_CHECKING:
    from agentomatic.storage.base import BaseStore


class AgentPlatform:
    """Zero-code multi-agent API platform.

    Usage::

        from agentomatic import AgentPlatform

        platform = AgentPlatform.from_folder("agents/")
        app = platform.build()
        # uvicorn main:app --reload

    With storage + middleware::

        from agentomatic import AgentPlatform
        from agentomatic.storage import MemoryStore

        platform = AgentPlatform.from_folder(
            "agents/",
            store=MemoryStore(),
            enable_auth=True,
            auth_api_key="secret",
            enable_rate_limit=True,
            enable_metrics=True,
        )
        app = platform.build()
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
        # --- Storage ---
        store: BaseStore | None = None,
        # --- Middleware toggles ---
        enable_logging: bool = True,
        enable_auth: bool = False,
        auth_api_key: str = "",
        enable_rate_limit: bool = False,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
        enable_metrics: bool = False,
        enable_feedback: bool = True,
        enable_telemetry: bool = True,
        # --- Custom middleware ---
        middleware: list[tuple[type, dict[str, Any]]] | None = None,
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
            settings: Optional :class:`PlatformSettings` object.
            store: Pluggable storage backend (``BaseStore`` subclass).
            enable_logging: Add request-logging middleware.
            enable_auth: Add API-key auth middleware.
            auth_api_key: API key (required when ``enable_auth=True``).
            enable_rate_limit: Add rate-limiting middleware.
            rate_limit_requests: Max requests per window.
            rate_limit_window: Window duration in seconds.
            enable_metrics: Add Prometheus metrics middleware.
            enable_feedback: Auto-add feedback endpoints per agent (default ``True``).
            enable_telemetry: Auto-configure OpenTelemetry tracing (default ``True``).
            middleware: Custom middleware list ``[(MiddlewareClass, {kwargs}), ...]``.
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

        # Storage
        self._store = store

        # Middleware config
        self._enable_logging = enable_logging
        self._enable_auth = enable_auth
        self._auth_api_key = auth_api_key
        self._enable_rate_limit = enable_rate_limit
        self._rate_limit_requests = rate_limit_requests
        self._rate_limit_window = rate_limit_window
        self._enable_metrics = enable_metrics
        self._enable_feedback = enable_feedback
        self._enable_telemetry = enable_telemetry
        self._custom_middleware = middleware or []

        # Internal
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

    @property
    def store(self) -> BaseStore | None:
        """Access the storage backend."""
        return self._store

    @store.setter
    def store(self, value: BaseStore) -> None:
        """Set the storage backend."""
        self._store = value

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
        logger.info(f"  ✅ Programmatically registered: {manifest.name} ({manifest.slug})")

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_startup(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a startup hook (decorator)."""
        self._on_startup.append(fn)
        return fn

    def on_shutdown(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a shutdown hook (decorator)."""
        self._on_shutdown.append(fn)
        return fn

    # ------------------------------------------------------------------
    # Custom routers
    # ------------------------------------------------------------------

    def include_router(self, router: Any, prefix: str = "", **kwargs: Any) -> None:
        """Add a custom router to the platform."""
        self._extra_routers.append((prefix, router, kwargs))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> FastAPI:
        """Build and return the FastAPI application.

        This is the main method.  It:
          1. Creates the FastAPI app with lifespan
          2. Adds middleware (CORS, auth, rate-limit, metrics, logging)
          3. Discovers agents from the folder
          4. Auto-generates endpoints per agent
          5. Wires storage into routers
          6. Mounts everything

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

            # Initialize storage if configured
            if platform._store:
                await platform._store.initialize()
                logger.info("🗄️ Storage backend initialized")

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
                        thread_store=platform._store,
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
            # Close storage
            if platform._store:
                await platform._store.close()
                logger.info("🗄️ Storage backend closed")

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

        # ------------------------------------------------------------------
        # Middleware pipeline (added in reverse order — last added = first run)
        # ------------------------------------------------------------------

        # CORS (always)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Logging
        if self._enable_logging:
            from agentomatic.middleware.logging import LoggingMiddleware

            app.add_middleware(LoggingMiddleware)

        # Auth
        if self._enable_auth and self._auth_api_key:
            from agentomatic.middleware.auth import AuthMiddleware

            app.add_middleware(AuthMiddleware, api_key=self._auth_api_key)
            logger.info("🔒 Auth middleware enabled")

        # Rate limiting
        if self._enable_rate_limit:
            from agentomatic.middleware.rate_limit import RateLimitMiddleware

            app.add_middleware(
                RateLimitMiddleware,
                max_requests=self._rate_limit_requests,
                window_seconds=self._rate_limit_window,
            )
            logger.info(
                f"🚦 Rate limit: {self._rate_limit_requests} req/{self._rate_limit_window}s"
            )

        # Metrics
        if self._enable_metrics:
            from agentomatic.middleware.metrics import MetricsMiddleware

            app.add_middleware(MetricsMiddleware)
            logger.info("📊 Metrics middleware enabled")

        # Custom middleware
        for mw_cls, mw_kwargs in self._custom_middleware:
            app.add_middleware(mw_cls, **mw_kwargs)

        # Feedback collector
        if self._enable_feedback:
            from agentomatic.middleware.feedback import FeedbackCollector, set_collector

            collector = FeedbackCollector(store=self._store)
            set_collector(collector)
            logger.info("📝 Feedback collection enabled")

        # OpenTelemetry auto-instrumentation
        if self._enable_telemetry:
            try:
                from agentomatic.observability.telemetry import setup_telemetry

                setup_telemetry(app)
            except Exception as exc:
                logger.debug(f"Telemetry setup skipped: {exc}")

        # ------------------------------------------------------------------
        # Mount routers for PRE-REGISTERED (programmatic) agents immediately
        # ------------------------------------------------------------------
        for name, agent in self._registry.all().items():
            if agent.router is None and agent.manifest.is_subagent:
                agent.router = create_default_router(
                    agent_name=name,
                    registry=self._registry,
                    thread_store=self._store,
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
            """Aggregate health across all agents + storage."""
            agents: dict[str, Any] = {}
            for name, agent in self._registry.all().items():
                try:
                    agents[name] = await agent.health_check()
                except Exception as exc:  # noqa: BLE001
                    agents[name] = {"status": "error", "error": str(exc)}

            # Storage health
            storage_health: dict[str, Any] = {"status": "not_configured"}
            if self._store:
                try:
                    storage_health = await self._store.health_check()
                except Exception as exc:  # noqa: BLE001
                    storage_health = {"status": "unhealthy", "error": str(exc)}

            overall = (
                "healthy"
                if all(a.get("status") == "healthy" for a in agents.values())
                else "degraded"
            )
            return {
                "status": overall,
                "agents": agents,
                "agent_count": len(agents),
                "storage": storage_health,
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

        # Storage stats
        if self._store:

            @app.get(f"{self.api_prefix}/storage/stats")
            async def storage_stats() -> dict[str, Any]:
                """Storage backend statistics."""
                return await self._store.get_stats()

            # Feedback endpoint
            @app.post(f"{self.api_prefix}/feedback")
            async def submit_feedback(
                thread_id: str,
                user_id: str,
                agent_name: str,
                rating: int | None = None,
                comment: str | None = None,
            ) -> dict[str, Any]:
                """Submit feedback."""
                return await self._store.add_feedback(
                    thread_id,
                    user_id,
                    agent_name,
                    rating=rating,
                    comment=comment,
                )

            @app.get(f"{self.api_prefix}/feedback")
            async def list_feedback(
                agent_name: str | None = None,
                limit: int = 50,
            ) -> dict[str, Any]:
                """List collected feedback."""
                items = await self._store.get_feedback(
                    agent_name=agent_name,
                    limit=limit,
                )
                return {"feedback": items, "count": len(items)}

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
