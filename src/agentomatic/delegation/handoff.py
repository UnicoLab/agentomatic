"""Handoff tools for agent-to-agent delegation.

Provides both langgraph-swarm native handoffs and HTTP-based fallback
delegation via the platform API.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger


def create_agent_handoff(
    target_agent: str,
    *,
    description: str = "",
    platform_url: str = "http://localhost:8000",
    use_swarm: bool = True,
) -> Any:
    """Create a LangChain-compatible handoff tool for agent delegation.

    Resolution order:
    1. If use_swarm=True and langgraph-swarm is installed, use create_handoff_tool
    2. Otherwise, create an HTTP-based delegation tool via platform API
    3. If langchain_core is also missing, return a simple callable wrapper

    Args:
        target_agent: Name of the agent to delegate to.
        description: Human-readable description of when to use this handoff.
        platform_url: Base URL of the platform API (for HTTP delegation).
        use_swarm: Whether to try langgraph-swarm first.

    Returns:
        A tool compatible with LangGraph's create_react_agent.
    """
    if use_swarm:
        try:
            from langgraph_swarm import create_handoff_tool  # type: ignore[import-untyped]

            logger.debug("Using langgraph-swarm handoff for '{}'", target_agent)
            return create_handoff_tool(
                agent_name=target_agent,
                description=description or f"Delegate to {target_agent}",
            )
        except ImportError:
            logger.debug(
                "langgraph-swarm not installed, falling back to HTTP handoff for '{}'",
                target_agent,
            )

    return _create_http_handoff(target_agent, description, platform_url)


def _create_http_handoff(
    target_agent: str,
    description: str,
    platform_url: str,
) -> Any:
    """Create HTTP-based delegation tool.

    Attempts to use langchain_core's @tool decorator for framework
    compatibility. Falls back to a plain callable if langchain_core is
    not installed.

    Args:
        target_agent: Name of the target agent.
        description: Human-readable tool description.
        platform_url: Base URL of the platform API.

    Returns:
        A callable tool that delegates via HTTP POST.
    """
    tool_description = description or f"Delegate task to {target_agent} agent"

    try:
        from langchain_core.tools import tool  # type: ignore[import-untyped]

        @tool(f"delegate_to_{target_agent}", description=tool_description)
        def delegate(query: str) -> str:
            """Delegate the given query to another agent via HTTP API."""
            return _invoke_http_delegation(target_agent, query, platform_url)

        logger.debug("Created langchain_core HTTP handoff tool for '{}'", target_agent)
        return delegate

    except ImportError:
        logger.debug(
            "langchain_core not installed, creating plain callable handoff for '{}'",
            target_agent,
        )

        def delegate_plain(query: str) -> str:
            """Delegate the given query to another agent via HTTP API."""
            return _invoke_http_delegation(target_agent, query, platform_url)

        delegate_plain.__name__ = f"delegate_to_{target_agent}"
        delegate_plain.__doc__ = tool_description
        return delegate_plain


def _invoke_http_delegation(
    target_agent: str,
    query: str,
    platform_url: str,
) -> str:
    """Execute the HTTP POST call to delegate a query.

    Args:
        target_agent: Name of the agent to call.
        query: The query string to send.
        platform_url: Base URL of the platform API.

    Returns:
        The response string from the target agent.

    Raises:
        httpx.HTTPStatusError: If the API returns a non-2xx status code.
    """
    url = f"{platform_url}/api/v1/{target_agent}/invoke"
    logger.info("Delegating to '{}' via HTTP: {}", target_agent, url)

    response = httpx.post(
        url,
        json={"query": query},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", str(data))


class AgentDelegator:
    """High-level delegation manager for creating handoff tools.

    Usage in agent's graph.py::

        delegator = AgentDelegator()
        handoff_tools = delegator.create_handoffs(["agent_a", "agent_b"])
        agent = create_react_agent(model, tools + handoff_tools)

    Args:
        platform_url: Base URL of the platform API.
        use_swarm: Whether to prefer langgraph-swarm handoffs.
    """

    def __init__(
        self,
        *,
        platform_url: str = "http://localhost:8000",
        use_swarm: bool = True,
    ) -> None:
        self._platform_url = platform_url
        self._use_swarm = use_swarm

    def create_handoffs(
        self,
        targets: list[str],
        descriptions: dict[str, str] | None = None,
    ) -> list[Any]:
        """Create handoff tools for multiple target agents.

        Args:
            targets: List of agent names to create handoff tools for.
            descriptions: Optional mapping of agent name to description.

        Returns:
            List of handoff tools, one per target agent.
        """
        descriptions = descriptions or {}
        return [
            create_agent_handoff(
                target,
                description=descriptions.get(target, ""),
                platform_url=self._platform_url,
                use_swarm=self._use_swarm,
            )
            for target in targets
        ]
