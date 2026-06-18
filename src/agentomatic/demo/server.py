"""Demo platform server — spins up Agentomatic with the built-in demo agent.

Creates an :class:`~agentomatic.core.platform.AgentPlatform` programmatically,
registers the demo agent, enables Studio, and returns the platform ready to
``.run()``.

Usage::

    from agentomatic.demo.server import create_demo_platform

    platform = create_demo_platform()
    platform.run(host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from loguru import logger

from agentomatic.core.platform import AgentPlatform
from agentomatic.demo.agent import (
    demo_graph_topology,
    demo_state_provider,
    manifest,
    node_fn,
)


def create_demo_platform(
    *,
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 8000,
    enable_studio: bool = True,
) -> AgentPlatform:
    """Build a ready-to-run platform with the built-in demo agent.

    The function creates a temporary (empty) agents directory so that
    ``AgentPlatform`` can initialise normally, then programmatically
    registers the demo agent and wires up its Studio hooks.

    Args:
        host: Bind address (informational, not used here).
        port: Bind port (informational, not used here).
        enable_studio: Whether to mount the Studio debug UI.

    Returns:
        A configured :class:`AgentPlatform` ready to call ``.run()``.
    """
    # Create a minimal temporary agents dir (platform expects one)
    tmp_agents = Path(tempfile.mkdtemp(prefix="agentomatic_demo_"))
    (tmp_agents / "__init__.py").write_text("")

    logger.info("🎪 Creating demo platform …")

    platform = AgentPlatform(
        agents_dir=tmp_agents,
        title="Agentomatic Demo Platform",
        description="Built-in demo platform for E2E testing with Studio",
        enable_studio=enable_studio,
        enable_telemetry=False,  # keep the demo lightweight
    )

    # --- Register the demo agent programmatically ---
    platform.register_agent(
        manifest=manifest,
        node_fn=node_fn,
    )

    # --- Attach Studio hooks to the registered agent ---
    agent = platform.registry.get(manifest.name)
    if agent is not None:
        agent._studio_graph_fn = demo_graph_topology  # noqa: SLF001
        agent._studio_state_fn = demo_state_provider  # noqa: SLF001
        logger.info("  🎨 Studio hooks attached to demo_assistant")

    logger.info(f"  📡 API will be at http://{host}:{port}")
    if enable_studio:
        logger.info(f"  🎨 Studio UI at  http://{host}:{port}/studio/ui/")

    return platform
