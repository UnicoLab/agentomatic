"""Tests for the ``AGENTOMATIC_AGENTS`` discovery allow-list (P1-3).

``agentomatic deploy --with-agent-stubs`` sets ``AGENTOMATIC_AGENTS={name}`` per
replica so each container serves a single agent. These tests verify the env var
actually scopes which agents load during discovery.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from agentomatic.core.registry import AgentRegistry, _agent_allowlist


def _make_agent(agents_dir: Path, name: str) -> None:
    """Create a minimal folder-based agent package under *agents_dir*."""
    pkg = agents_dir / name
    pkg.mkdir(parents=True)
    pkg.joinpath("__init__.py").write_text(
        textwrap.dedent(
            f'''
            """Test agent {name}."""
            from __future__ import annotations

            from agentomatic import AgentManifest

            manifest = AgentManifest(name="{name}", slug="{name}")


            async def node_fn(state: dict) -> dict:
                return {{"response": "{name}"}}
            '''
        )
    )


def test_allowlist_parsing(monkeypatch: Any) -> None:
    monkeypatch.delenv("AGENTOMATIC_AGENTS", raising=False)
    assert _agent_allowlist() is None

    monkeypatch.setenv("AGENTOMATIC_AGENTS", " Alpha , beta ,")
    assert _agent_allowlist() == {"alpha", "beta"}


def test_allowlist_scopes_discovery(tmp_path: Path, monkeypatch: Any) -> None:
    """Only agents named in AGENTOMATIC_AGENTS should be registered."""
    prefix = "allowlist_pkg_a"
    agents = tmp_path / prefix
    agents.mkdir()
    _make_agent(agents, "alpha")
    _make_agent(agents, "beta")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("AGENTOMATIC_AGENTS", "alpha")

    registry = AgentRegistry()
    registry.discover(agents, prefix)

    assert set(registry.all().keys()) == {"alpha"}


def test_no_allowlist_loads_all(tmp_path: Path, monkeypatch: Any) -> None:
    """Without the env var, every discoverable agent loads."""
    prefix = "allowlist_pkg_b"
    agents = tmp_path / prefix
    agents.mkdir()
    _make_agent(agents, "alpha")
    _make_agent(agents, "beta")

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("AGENTOMATIC_AGENTS", raising=False)

    registry = AgentRegistry()
    registry.discover(agents, prefix)

    assert set(registry.all().keys()) == {"alpha", "beta"}
