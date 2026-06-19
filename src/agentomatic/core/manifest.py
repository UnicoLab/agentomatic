"""Agent manifest and registered agent types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import APIRouter

    from agentomatic.prompts.manager import PromptManager


@dataclass(frozen=True, slots=True)
class AgentManifest:
    """Identity card for an agent plugin.

    Every agent must export a ``manifest`` instance in its ``__init__.py``.
    The registry discovers this automatically.

    Args:
        name: Short machine name (must match folder name).
        slug: Full unique identifier (e.g. 'my-platform-agent-holidays').
        description: Human-readable description.
        intent_keywords: Keywords for orchestrator intent routing.
        version: SemVer version string.
        is_subagent: Whether this agent is routable by an orchestrator.
        framework: Agent framework type ('langgraph', 'langchain', 'custom').
        metadata: Arbitrary metadata (used in A2A agent cards).
    """

    name: str
    slug: str
    description: str = ""
    intent_keywords: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    is_subagent: bool = True
    framework: str = "langgraph"  # 'langgraph' | 'langchain' | 'custom'
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RegisteredAgent:
    """An agent that has been discovered and registered by the platform.

    Contains the manifest, callable functions, and optional enhancements
    (router, config, prompt manager) discovered from the agent's folder.
    """

    manifest: AgentManifest
    node_fn: Callable[..., Awaitable[Any]] | None = None
    graph_fn: Callable[[], Any] | None = None
    module_path: str = ""

    # Optional enhancements (populated during discovery)
    router: APIRouter | None = None
    config: Any = None
    prompt_manager: PromptManager | None = None

    # Studio hooks (set programmatically or via decorators)
    _studio_graph_fn: Callable[..., Any] | None = field(default=None, repr=False)
    _studio_state_fn: Callable[..., Any] | None = field(default=None, repr=False)
    _studio_stream_fn: Callable[..., Any] | None = field(default=None, repr=False)
    _studio_adapter: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        """Return the agent's short machine name."""
        return self.manifest.name

    @property
    def slug(self) -> str:
        """Return the agent's full unique identifier."""
        return self.manifest.slug

    async def health_check(self) -> dict[str, Any]:
        """Check agent health."""
        result: dict[str, Any] = {
            "agent": self.name,
            "slug": self.slug,
            "version": self.manifest.version,
            "framework": self.manifest.framework,
        }
        # Check node function
        result["node_fn_ready"] = self.node_fn is not None
        # Check graph
        if self.graph_fn:
            try:
                graph = self.graph_fn()
                result["graph_ready"] = graph is not None
            except Exception as exc:
                result["graph_ready"] = False
                result["graph_error"] = str(exc)
        else:
            result["graph_ready"] = False
        # Check prompts
        if self.prompt_manager:
            result["prompt_versions"] = self.prompt_manager.list_versions()
        # Check config
        result["has_config"] = self.config is not None
        # Overall status
        result["status"] = (
            "healthy" if result.get("node_fn_ready") or result.get("graph_ready") else "degraded"
        )
        return result
