"""AgentPlatform — the main entry point for agentomatic."""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from agentomatic.endpoints.registry import EndpointRegistry
from agentomatic.endpoints.router import create_endpoint_router
from agentomatic.plugins.registry import PluginRegistry
from agentomatic.plugins.router import create_plugin_router

from .lifespan import configure_logging
from .manifest import AgentManifest, RegisteredAgent
from .registry import AgentRegistry
from .router_factory import create_default_router

if TYPE_CHECKING:
    from fastapi.routing import APIRoute

    from agentomatic.security.jwt_auth import JWTConfig
    from agentomatic.stacks.manager import StackConfig, StackManager
    from agentomatic.storage.base import BaseStore


def _agent_tag(name: str) -> str:
    """Return a human-friendly OpenAPI tag for an agent name.

    Avoids ``str.title`` underscore mangling (``weather_bot`` → ``Weather Bot``).
    """
    return name.replace("_", " ").replace("-", " ").title()


def _generate_unique_id(route: APIRoute) -> str:
    """Generate a clean, stable OpenAPI ``operationId`` for a route.

    Combines the first tag (when present) with the route name to produce
    readable client-codegen identifiers instead of FastAPI's verbose,
    path-derived defaults.
    """
    tag = route.tags[0] if route.tags else ""
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(tag)).strip("_").lower()
    if slug:
        return f"{slug}_{route.name}"
    return route.name


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
        plugins_dir: str | Path = "plugins/",
        endpoints_dir: str | Path = "endpoints/",
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
        # --- Memory ---
        max_history_messages: int = 50,
        summarize_after: int = 30,
        # --- Studio ---
        enable_studio: bool = False,
        # --- v0.6: Stacks & Security ---
        stack: str | StackConfig | None = None,
        stacks_dir: str | Path = "stacks/",
        enable_jwt_auth: bool = False,
        jwt_config: JWTConfig | None = None,
        enable_zero_trust: bool = False,
        # --- v0.11: Control plane & connections ---
        enable_control_plane: bool = False,
        control_token: str = "",
        connections: list[Any] | None = None,
        enable_connections_context: bool = True,
    ) -> None:
        """Initialise the platform.

        Args:
            agents_dir: Filesystem path containing agent packages.
            plugins_dir: Filesystem path containing plugin packages.
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
            enable_studio: Mount the Studio debug API and UI (default ``False``).
            middleware: Custom middleware list ``[(MiddlewareClass, {kwargs}), ...]``.
            stack: Active stack name or :class:`StackConfig` (default: auto-detect).
            stacks_dir: Directory containing stack YAML files.
            enable_jwt_auth: Add JWT/OAuth2 auth middleware.
            jwt_config: JWT configuration (required when ``enable_jwt_auth=True``).
            enable_zero_trust: Enable zero-trust policy enforcement.
            enable_control_plane: Mount the production control plane API and
                request-gating middleware (default ``False``).
            control_token: Optional shared secret protecting mutating control
                plane operations (via the ``X-Control-Token`` header).
            connections: Optional list of platform-wide connection configs
                (databases / vector stores / HTTP services) registered under
                the shared scope.
            enable_connections_context: Attach the routed agent's connection
                manager to ``request.state.connections`` on every request
                (default ``True``).
        """
        self.agents_dir = Path(agents_dir).resolve()
        self.plugins_dir = Path(plugins_dir).resolve()
        self.endpoints_dir = Path(endpoints_dir).resolve()
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
        self._enable_studio = enable_studio
        self._custom_middleware = middleware or []

        # v0.6: Stacks & Security
        self._stack_manager: StackManager | None = None
        self._enable_jwt_auth = enable_jwt_auth
        self._jwt_config = jwt_config
        self._enable_zero_trust = enable_zero_trust
        self._init_stack(stack, stacks_dir)

        # v0.11: Control plane & connections
        self._enable_control_plane = enable_control_plane
        self._control_token = control_token
        self._platform_connections = list(connections or [])
        self._enable_connections_context = enable_connections_context

        # Memory config
        self._max_history_messages = max_history_messages
        self._summarize_after = summarize_after

        # Internal
        self._registry = AgentRegistry()
        self._plugin_registry = PluginRegistry()
        self._endpoint_registry = EndpointRegistry()
        self._pipelines: dict[str, Any] = {}
        self._on_startup: list[Callable[..., Any]] = []
        self._on_shutdown: list[Callable[..., Any]] = []
        self._extra_routers: list[tuple[str, Any, dict[str, Any]]] = []
        self._app: FastAPI | None = None
        self._discovered: bool = False  # guard against double discovery
        self._plugins_discovered: bool = False
        self._endpoints_discovered: bool = False

        # Control plane state (used only when enabled)
        from agentomatic.control.state import ControlPlaneState

        self._control_state = ControlPlaneState()

    def _init_stack(
        self,
        stack: str | StackConfig | None,
        stacks_dir: str | Path,
    ) -> None:
        """Initialise the stack manager if a stack is requested."""
        import os

        from agentomatic.config.settings import load_environment

        # Always try to load .env
        load_environment()

        if stack is None:
            # Auto-detect: AGENTOMATIC_STACK env > .agentomatic-stack file
            stack_name = os.environ.get("AGENTOMATIC_STACK", "")
            if not stack_name:
                stack_file = Path(".agentomatic-stack")
                if stack_file.exists():
                    stack_name = stack_file.read_text().strip()
            if stack_name:
                stack = stack_name

        if stack is not None:
            from agentomatic.stacks.manager import StackConfig, StackManager

            if isinstance(stack, str):
                self._stack_manager = StackManager(stacks_dir=stacks_dir)
                try:
                    self._stack_manager.load(stack)
                    logger.info(f"📦 Loaded stack: {stack}")
                except Exception as exc:
                    logger.warning(f"Failed to load stack '{stack}': {exc}")
                    self._stack_manager = None
            elif isinstance(stack, StackConfig):
                self._stack_manager = StackManager(stacks_dir=stacks_dir)
                self._stack_manager.apply_dotenv(stack.env_file)
                for key, value in stack.environment.items():
                    os.environ.setdefault(key, value)

                from agentomatic.config.settings import reset_settings

                reset_settings()

                self._stack_manager._active_stack = stack
                logger.info(f"📦 Loaded explicit StackConfig: {stack.name}")

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
    def endpoint_registry(self) -> EndpointRegistry:
        """Access the custom endpoint registry."""
        return self._endpoint_registry

    @property
    def pipelines(self) -> dict[str, Any]:
        """Access the discovered pipelines (populated during ``build``)."""
        return self._pipelines

    def register_endpoint(self, endpoint: Any) -> None:
        """Register a custom endpoint programmatically."""
        self._endpoint_registry.register(endpoint)

    @property
    def store(self) -> BaseStore | None:
        """Access the storage backend."""
        return self._store

    @store.setter
    def store(self, value: BaseStore) -> None:
        """Set the storage backend."""
        self._store = value

    def register_before_node_hook(self, hook: Callable[[str, dict[str, Any]], None]) -> None:
        """Register a hook to execute before any agent node/graph runs."""
        self._registry.before_node_hooks.append(hook)

    def register_after_node_hook(self, hook: Callable[[str, dict[str, Any]], None]) -> None:
        """Register a hook to execute after any agent node/graph runs."""
        self._registry.after_node_hooks.append(hook)

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

    def discover_agents(self) -> None:
        """Manually trigger agent discovery from the configured folder.

        This is useful for testing scenarios, embedding, or any context
        where the async lifespan startup may not have run yet.  Safe to
        call multiple times — subsequent calls are no-ops.

        Example::

            platform = AgentPlatform.from_folder("agents/")
            platform.discover_agents()  # synchronous discovery
            app = platform.build()
        """
        if self._discovered:
            return

        # Ensure agents directory is importable
        parent = str(self.agents_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        prefix = self.package_prefix or self.agents_dir.name
        self._registry.discover(self.agents_dir, prefix)
        self._discovered = True

    def _build_openapi_tags(self) -> list[dict[str, Any]]:
        """Build OpenAPI tag metadata (descriptions + ordering) for Swagger."""
        tags: list[dict[str, Any]] = [
            {"name": "Platform", "description": "Platform health, readiness and discovery."},
        ]
        for name, agent in sorted(self._registry.all().items()):
            if agent.manifest.is_subagent:
                tags.append(
                    {
                        "name": _agent_tag(name),
                        "description": agent.manifest.description or f"Agent '{name}'.",
                    }
                )
        tags.append({"name": "Endpoints", "description": "Custom HTTP endpoints."})
        tags.append({"name": "Pipelines", "description": "Multi-step agent pipelines."})
        if self._plugin_registry.count:
            tags.append({"name": "Plugins", "description": "Classical ML model plugins."})
        if self._enable_control_plane:
            tags.append(
                {"name": "Control Plane", "description": "Runtime operations and observability."}
            )
        if self._enable_studio:
            tags.append({"name": "Studio Debug API", "description": "Studio debugging surface."})
        return tags

    async def _init_connections(self) -> None:
        """Register and initialise per-agent and platform-wide connections."""
        from agentomatic.connections.manager import (
            PLATFORM_SCOPE,
            register_connections,
        )

        # Platform-wide connections
        if self._platform_connections:
            register_connections(PLATFORM_SCOPE, self._platform_connections)

        # Per-agent connections discovered from ``connections.py``
        for name, agent in self._registry.all().items():
            configs = getattr(agent, "connections", None)
            if configs:
                register_connections(name, list(configs))

        from agentomatic.connections.manager import all_managers

        managers = all_managers()
        if managers:
            logger.info(f"🔗 Initializing {len(managers)} connection scope(s)...")
            for scope, manager in managers.items():
                await manager.initialize()
                logger.info(f"  ✅ Connections ready for scope '{scope}' ({manager.count})")

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

        # Ensure agents directory is importable BEFORE build so
        # programmatic and synchronous access to agents works.
        parent = str(platform.agents_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        # Eagerly discover agents so they are available immediately
        # after build() returns — no need to wait for lifespan startup.
        # This also fixes TestClient scenarios where lifespan may not
        # fully trigger.
        if not platform._discovered:
            prefix = platform.package_prefix or platform.agents_dir.name
            platform._registry.discover(platform.agents_dir, prefix)
            platform._discovered = True

        if not platform._plugins_discovered:
            plugin_prefix = platform.package_prefix or platform.plugins_dir.name
            platform._plugin_registry.discover(platform.plugins_dir, plugin_prefix)
            platform._plugins_discovered = True

        if not platform._endpoints_discovered:
            endpoint_prefix = platform.package_prefix or platform.endpoints_dir.name
            platform._endpoint_registry.discover(platform.endpoints_dir, endpoint_prefix)
            platform._endpoints_discovered = True

        try:
            from agentomatic.observability.metrics import REGISTERED_ENDPOINTS

            REGISTERED_ENDPOINTS.set(platform._endpoint_registry.count)
        except Exception:  # noqa: BLE001 - metrics are optional
            pass

        # Track which agents are already registered (programmatic + discovered)
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

            # Re-discover in lifespan only if not already done (hot-reload)
            if not platform._discovered:
                prefix = platform.package_prefix or platform.agents_dir.name
                platform._registry.discover(platform.agents_dir, prefix)
                platform._discovered = True

            if not platform._plugins_discovered:
                plugin_prefix = platform.package_prefix or platform.plugins_dir.name
                platform._plugin_registry.discover(platform.plugins_dir, plugin_prefix)
                platform._plugins_discovered = True

            # --- Load ML Plugins ---
            logger.info("🧠 Loading ML Model Plugins...")
            for name, plugin in platform._plugin_registry.list_plugins().items():
                try:
                    await plugin.load_model()
                    logger.info(f"  ✅ Plugin '{name}' loaded successfully")
                except Exception as e:
                    logger.error(f"  ❌ Failed to load plugin '{name}': {e}")

            # --- Start custom endpoints ---
            if platform._endpoint_registry.count:
                logger.info("🌐 Starting custom endpoints...")
                for name, endpoint in platform._endpoint_registry.list_endpoints().items():
                    try:
                        await endpoint.startup()
                        logger.info(f"  ✅ Endpoint '{name}' ready")
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"  ❌ Failed to start endpoint '{name}': {e}")

            # --- Register + initialize connections (per-agent + platform) ---
            await platform._init_connections()

            # Auto-generate + mount routers for NEWLY discovered agents
            for name, agent in platform._registry.all().items():
                if name in _pre_registered:
                    continue  # already mounted at build-time
                if agent.router is None and agent.manifest.is_subagent:
                    agent.router = create_default_router(
                        agent_name=name,
                        registry=platform._registry,
                        thread_store=platform._store,
                        max_history_messages=platform._max_history_messages,
                        summarize_after=platform._summarize_after,
                    )
                    logger.debug(f"  📌 Auto-generated router for {name}")
                if agent.router and agent.manifest.is_subagent:
                    app.include_router(
                        agent.router,
                        prefix=f"{platform.api_prefix}/{name}",
                        tags=[_agent_tag(name)],
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
            # Stop custom endpoints
            for name, endpoint in platform._endpoint_registry.list_endpoints().items():
                try:
                    await endpoint.shutdown()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Endpoint '{name}' shutdown error: {e}")

            # Close all connections
            try:
                from agentomatic.connections.manager import all_managers

                for manager in all_managers().values():
                    await manager.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Connection shutdown error: {e}")

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
            openapi_tags=self._build_openapi_tags(),
            generate_unique_id_function=_generate_unique_id,
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
            expose_headers=["X-Studio-Run-Id"],
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

        # Zero Trust Enforcer (v0.6) — added BEFORE JWT so that (thanks to
        # Starlette's reverse middleware ordering) the JWT middleware runs
        # first and populates ``request.state.jwt_claims`` before enforcement.
        # Per-agent enforcement is opt-in via each agent's ``security.py``
        # policy (``require_auth`` / ``allowed_roles`` / ``allowed_scopes``).
        if self._enable_zero_trust:
            try:
                from agentomatic.middleware.zero_trust import ZeroTrustMiddleware
                from agentomatic.security.zero_trust import ZeroTrustEnforcer

                enforcer = ZeroTrustEnforcer(require_auth_globally=False)
                for name, agent in self._registry.all().items():
                    if agent.security_policy is not None:
                        enforcer.register_policy(name, agent.security_policy)
                app.state.zero_trust_enforcer = enforcer
                app.add_middleware(
                    ZeroTrustMiddleware,
                    enforcer=enforcer,
                    registry=self._registry,
                    api_prefix=self.api_prefix,
                )
                logger.info("🛡️ Zero-trust enforcement enabled")
            except Exception as exc:
                logger.warning(f"Zero-trust setup failed: {exc}")

        # JWT Auth (v0.6)
        if self._enable_jwt_auth:
            try:
                from agentomatic.security.jwt_auth import JWTAuthMiddleware, JWTConfig

                jwt_cfg = self._jwt_config or JWTConfig(enabled=True)
                app.add_middleware(JWTAuthMiddleware, config=jwt_cfg)
                logger.info("🔐 JWT auth middleware enabled")
            except Exception as exc:
                logger.warning(f"JWT auth setup failed: {exc}")

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

        # Control plane request gating (maintenance mode + agent drain)
        if self._enable_control_plane:
            from agentomatic.control.middleware import MaintenanceMiddleware

            app.add_middleware(
                MaintenanceMiddleware,
                state=self._control_state,
                api_prefix=self.api_prefix,
            )

        # Per-request connection context (request.state.connections)
        if self._enable_connections_context:
            from agentomatic.middleware.connections import ConnectionsMiddleware

            app.add_middleware(
                ConnectionsMiddleware,
                registry=self._registry,
                api_prefix=self.api_prefix,
            )

        # Custom middleware
        for mw_cls, mw_kwargs in self._custom_middleware:
            app.add_middleware(mw_cls, **mw_kwargs)  # type: ignore[arg-type]

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
                    max_history_messages=self._max_history_messages,
                    summarize_after=self._summarize_after,
                )
            if agent.router and agent.manifest.is_subagent:
                app.include_router(
                    agent.router,
                    prefix=f"{self.api_prefix}/{name}",
                    tags=[_agent_tag(name)],
                )

        # ------------------------------------------------------------------
        # Mount Plugin API Routes
        # ------------------------------------------------------------------
        plugins_router = APIRouter(prefix=self.api_prefix + "/plugins")

        @plugins_router.get("", response_model=list[dict[str, Any]], tags=["Plugins"])
        async def list_plugins() -> list[dict[str, Any]]:
            """List all registered plugins."""
            return [
                {
                    "name": p.plugin_name,
                    "description": p.plugin_description,
                    "version": p.plugin_version,
                    "is_loaded": p.is_loaded,
                }
                for p in self._plugin_registry.list_plugins().values()
            ]

        for plugin_name, plugin in self._plugin_registry.list_plugins().items():
            plugin_router = create_plugin_router(plugin)
            plugins_router.include_router(
                plugin_router,
                prefix=f"/{plugin_name}",
            )
            logger.info(f"🔌 Mounted plugin endpoints for: {plugin_name}")
        app.include_router(plugins_router)

        # ------------------------------------------------------------------
        # Mount Custom Endpoint API Routes
        # ------------------------------------------------------------------
        endpoints_router = APIRouter(prefix=self.api_prefix + "/endpoints")

        @endpoints_router.get("", response_model=list[dict[str, Any]], tags=["Endpoints"])
        async def list_endpoints() -> list[dict[str, Any]]:
            """List all registered custom endpoints."""
            return [ep.info() for ep in self._endpoint_registry.list_endpoints().values()]

        for endpoint_name, endpoint in self._endpoint_registry.list_endpoints().items():
            endpoint_router = create_endpoint_router(endpoint)
            endpoints_router.include_router(
                endpoint_router,
                prefix=f"/{endpoint_name}",
            )
            logger.info(f"🌐 Mounted custom endpoint: {endpoint_name}")
        app.include_router(endpoints_router)

        # ------------------------------------------------------------------
        # Pipeline auto-discovery and mounting
        # ------------------------------------------------------------------
        try:
            from agentomatic.pipelines.loader import PipelineLoader
            from agentomatic.pipelines.router import create_pipeline_router

            pipelines: dict[str, Any] = {}

            # Discover from pipelines/ directory
            pipelines_dir = self.agents_dir.parent / "pipelines"
            if pipelines_dir.exists():
                pipelines.update(PipelineLoader.discover_pipelines(pipelines_dir))

            # Discover from agents/*/pipeline.yaml
            agents_pipelines = PipelineLoader.discover_pipelines(self.agents_dir)
            pipelines.update(agents_pipelines)

            self._pipelines = pipelines

            if pipelines:
                pipeline_router = create_pipeline_router(
                    pipelines,
                    self._registry,
                    endpoints=self._endpoint_registry,
                )
                app.include_router(
                    pipeline_router,
                    prefix=self.api_prefix,
                    tags=["Pipelines"],
                )
                logger.info(
                    f"🔁 Mounted {len(pipelines)} pipeline(s): {', '.join(pipelines.keys())}"
                )
        except ImportError:
            logger.debug("Pipeline module not available — skipping")
        except Exception as exc:
            logger.warning(f"Pipeline discovery failed: {exc}")

        # ------------------------------------------------------------------
        # Platform-level endpoints
        # ------------------------------------------------------------------

        @app.get("/health", tags=["Platform"])
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

        @app.get("/readiness", tags=["Platform"])
        async def readiness() -> dict[str, Any]:
            """Kubernetes-style readiness probe."""
            return {"status": "ready", "agents": self._registry.count}

        # A2A discovery
        @app.get("/.well-known/agent.json", tags=["Platform"])
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
        @app.get(f"{self.api_prefix}/agents", tags=["Platform"])
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
            store = self._store  # local var for mypy narrowing across closures

            @app.get(f"{self.api_prefix}/storage/stats", tags=["Platform"])
            async def storage_stats() -> dict[str, Any]:
                """Storage backend statistics."""
                return await store.get_stats()

            # Feedback endpoint
            @app.post(f"{self.api_prefix}/feedback", tags=["Platform"])
            async def submit_feedback(
                thread_id: str,
                user_id: str,
                agent_name: str,
                rating: int | None = None,
                comment: str | None = None,
            ) -> dict[str, Any]:
                """Submit feedback."""
                return await store.add_feedback(
                    thread_id,
                    user_id,
                    agent_name,
                    rating=rating,
                    comment=comment,
                )

            @app.get(f"{self.api_prefix}/feedback", tags=["Platform"])
            async def list_feedback(
                agent_name: str | None = None,
                limit: int = 50,
            ) -> dict[str, Any]:
                """List collected feedback."""
                items = await store.get_feedback(
                    agent_name=agent_name,
                    limit=limit,
                )
                return {"feedback": items, "count": len(items)}

        # Root
        @app.get("/", tags=["Platform"])
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

        # ------------------------------------------------------------------
        # Control Plane API
        # ------------------------------------------------------------------
        if self._enable_control_plane:
            try:
                from agentomatic.control.router import create_control_router

                control_router = create_control_router(
                    self,
                    self._control_state,
                    control_token=self._control_token,
                )
                app.include_router(control_router, prefix=self.api_prefix + "/control")
                logger.info(f"🎛️ Control plane mounted at {self.api_prefix}/control")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Control plane setup failed: {exc}")

        # Extra routers
        for prefix, router, kwargs in self._extra_routers:
            app.include_router(router, prefix=prefix, **kwargs)

        # ------------------------------------------------------------------
        # Studio Debug API + UI
        # ------------------------------------------------------------------
        if self._enable_studio:
            try:
                from agentomatic.studio.router import create_studio_router

                studio_router = create_studio_router(
                    registry=self._registry,
                    store=self._store,
                    platform_title=self.title,
                    platform_version=self.version,
                )
                app.include_router(studio_router)
                logger.info("🎨 Studio API mounted at /studio/")

                # Mount the built Studio UI (always — shows helpful
                # error page if assets are missing)
                from agentomatic.studio.serve import mount_studio_ui

                mount_studio_ui(app)

                # Convenience redirects: /studio → /studio/ui/
                from fastapi.responses import RedirectResponse

                @app.get("/studio", include_in_schema=False)
                @app.get("/studio/", include_in_schema=False)
                async def _studio_redirect() -> RedirectResponse:
                    return RedirectResponse(url="/studio/ui/")

            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Studio setup failed: {exc}")
        else:
            # Studio disabled — mount informative error pages so users
            # hitting /studio/ui/ get guidance instead of a bare 404.
            try:
                from agentomatic.studio.serve import mount_studio_disabled_page

                mount_studio_disabled_page(app)
            except Exception:  # noqa: BLE001
                pass  # Non-critical — don't log noise

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
