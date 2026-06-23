"""Tests for the delegation & swarms module.

Tests cover:
  - create_agent_handoff() with HTTP fallback
  - create_agent_handoff() swarm fallback when langgraph-swarm missing
  - AgentDelegator.create_handoffs()
  - SwarmOrchestrator registration and agent listing
  - SwarmOrchestrator.create_swarm() with unknown pattern
  - SwarmPlaceholder behaviour for unimplemented patterns
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# =====================================================================
# Module Imports
# =====================================================================


class TestDelegationImports:
    """Verify the delegation module exports are correct."""

    def test_import_create_agent_handoff(self):
        from agentomatic.delegation import create_agent_handoff

        assert create_agent_handoff is not None

    def test_import_agent_delegator(self):
        from agentomatic.delegation import AgentDelegator

        assert AgentDelegator is not None

    def test_import_swarm_orchestrator(self):
        from agentomatic.delegation import SwarmOrchestrator

        assert SwarmOrchestrator is not None


# =====================================================================
# create_agent_handoff – HTTP fallback
# =====================================================================


class TestCreateAgentHandoffHTTP:
    """Test create_agent_handoff with use_swarm=False (HTTP mode)."""

    def test_creates_callable_tool(self):
        """HTTP handoff should return a callable tool."""
        from agentomatic.delegation.handoff import create_agent_handoff

        tool = create_agent_handoff(
            "my_agent",
            use_swarm=False,
            platform_url="http://test:9000",
        )
        assert hasattr(tool, "invoke")

    def test_tool_name_contains_target(self):
        """Tool name should contain the target agent name."""
        from agentomatic.delegation.handoff import create_agent_handoff

        tool = create_agent_handoff("research_bot", use_swarm=False)
        # Whether langchain_core is present or not, the name should match
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
        assert "research_bot" in name

    @patch("agentomatic.delegation.handoff._invoke_http_delegation")
    def test_tool_calls_http_delegation(self, mock_invoke):
        """HTTP tool should delegate via _invoke_http_delegation."""
        from agentomatic.delegation.handoff import create_agent_handoff

        mock_invoke.return_value = "agent response"

        tool = create_agent_handoff(
            "writer",
            use_swarm=False,
            platform_url="http://test:8080",
        )

        # Call the underlying function (unwrap langchain tool if needed)
        func = getattr(tool, "func", tool)
        result = func("write me a poem")

        mock_invoke.assert_called_once_with("writer", "write me a poem", "http://test:8080")
        assert result == "agent response"

    @patch("agentomatic.delegation.handoff.httpx")
    def test_invoke_http_delegation_success(self, mock_httpx):
        """_invoke_http_delegation should POST and return response."""
        from agentomatic.delegation.handoff import _invoke_http_delegation

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "hello from agent"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        result = _invoke_http_delegation("helper", "do stuff", "http://localhost:8000")

        mock_httpx.post.assert_called_once_with(
            "http://localhost:8000/api/v1/helper/invoke",
            json={"query": "do stuff"},
            timeout=60.0,
        )
        assert result == "hello from agent"

    @patch("agentomatic.delegation.handoff.httpx")
    def test_invoke_http_delegation_fallback_str(self, mock_httpx):
        """When 'response' key is missing, fall back to str(data)."""
        from agentomatic.delegation.handoff import _invoke_http_delegation

        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "raw data"}
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        result = _invoke_http_delegation("helper", "query", "http://localhost:8000")
        assert result == str({"output": "raw data"})


# =====================================================================
# create_agent_handoff – swarm fallback
# =====================================================================


class TestCreateAgentHandoffSwarmFallback:
    """Test graceful fallback when langgraph-swarm is not installed."""

    @patch.dict("sys.modules", {"langgraph_swarm": None})
    def test_falls_back_to_http_when_swarm_unavailable(self):
        """Should fall back to HTTP when langgraph-swarm import fails."""
        from agentomatic.delegation.handoff import create_agent_handoff

        tool = create_agent_handoff(
            "fallback_agent",
            use_swarm=True,
            platform_url="http://test:9999",
        )
        assert hasattr(tool, "invoke")
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
        assert "fallback_agent" in name

    def test_uses_swarm_when_available(self):
        """Should use langgraph-swarm create_handoff_tool when available."""
        mock_swarm_module = MagicMock()
        mock_tool = MagicMock()
        mock_swarm_module.create_handoff_tool.return_value = mock_tool

        with patch.dict("sys.modules", {"langgraph_swarm": mock_swarm_module}):
            from importlib import reload

            import agentomatic.delegation.handoff as handoff_mod

            reload(handoff_mod)

            result = handoff_mod.create_agent_handoff("smart_agent", use_swarm=True)

        assert result is mock_tool
        mock_swarm_module.create_handoff_tool.assert_called_once_with(
            agent_name="smart_agent",
            description="Delegate to smart_agent",
        )


# =====================================================================
# AgentDelegator
# =====================================================================


class TestAgentDelegator:
    """Test AgentDelegator.create_handoffs()."""

    def test_creates_correct_number_of_handoffs(self):
        from agentomatic.delegation.handoff import AgentDelegator

        delegator = AgentDelegator(use_swarm=False)
        tools = delegator.create_handoffs(["agent_a", "agent_b", "agent_c"])

        assert len(tools) == 3
        for t in tools:
            assert hasattr(t, "invoke")

    def test_passes_descriptions(self):
        from agentomatic.delegation.handoff import AgentDelegator

        descriptions = {
            "agent_a": "Handles research tasks",
            "agent_b": "Handles writing tasks",
        }
        delegator = AgentDelegator(use_swarm=False)
        tools = delegator.create_handoffs(
            ["agent_a", "agent_b"],
            descriptions=descriptions,
        )
        assert len(tools) == 2

    def test_empty_targets_returns_empty_list(self):
        from agentomatic.delegation.handoff import AgentDelegator

        delegator = AgentDelegator(use_swarm=False)
        tools = delegator.create_handoffs([])
        assert tools == []


# =====================================================================
# SwarmOrchestrator – registration
# =====================================================================


class TestSwarmOrchestratorRegistration:
    """Test agent registration and listing."""

    def test_register_and_list_agents(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("alpha", MagicMock())
        orch.register_agent("beta", MagicMock())

        assert sorted(orch.registered_agents) == ["alpha", "beta"]

    def test_register_duplicate_raises_value_error(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("alpha", MagicMock())

        with pytest.raises(ValueError, match="already registered"):
            orch.register_agent("alpha", MagicMock())

    def test_unregister_agent(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("temp", MagicMock())
        assert "temp" in orch.registered_agents

        orch.unregister_agent("temp")
        assert "temp" not in orch.registered_agents

    def test_unregister_missing_raises_key_error(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        with pytest.raises(KeyError, match="not registered"):
            orch.unregister_agent("ghost")

    def test_no_agents_initially(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        assert orch.registered_agents == []


# =====================================================================
# SwarmOrchestrator – create_swarm
# =====================================================================


class TestSwarmOrchestratorCreateSwarm:
    """Test create_swarm() with various patterns."""

    def test_unknown_pattern_raises_value_error(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("a", MagicMock())

        with pytest.raises(ValueError, match="Unknown swarm pattern"):
            orch.create_swarm(pattern="unknown_pattern")

    def test_no_agents_raises_value_error(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        with pytest.raises(ValueError, match="No agents registered"):
            orch.create_swarm(pattern="handoff")

    def test_missing_agent_name_raises_value_error(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("real_agent", MagicMock())

        with pytest.raises(ValueError, match="not registered"):
            orch.create_swarm(agents=["nonexistent"], pattern="handoff")

    @patch.dict("sys.modules", {"langgraph_swarm": None})
    def test_handoff_pattern_raises_import_error_without_swarm(self):
        """Handoff pattern requires langgraph-swarm."""
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("a", MagicMock())

        with pytest.raises(ImportError, match="langgraph-swarm"):
            orch.create_swarm(pattern="handoff")

    def test_supervisor_pattern_returns_placeholder(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("a", MagicMock())
        orch.register_agent("b", MagicMock())

        result = orch.create_swarm(pattern="supervisor")

        assert result.pattern == "supervisor"
        assert sorted(result.agents) == ["a", "b"]
        with pytest.raises(NotImplementedError):
            result.invoke({"input": "test"})

    def test_round_robin_pattern_returns_placeholder(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("x", MagicMock())

        result = orch.create_swarm(pattern="round_robin")

        assert result.pattern == "round_robin"
        with pytest.raises(NotImplementedError):
            result("test")

    def test_supervisor_placeholder_repr(self):
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("agent1", MagicMock())

        result = orch.create_swarm(pattern="supervisor")
        repr_str = repr(result)

        assert "supervisor" in repr_str
        assert "agent1" in repr_str

    def test_create_swarm_selects_specific_agents(self):
        """Should only include explicitly listed agents."""
        from agentomatic.delegation.swarm import SwarmOrchestrator

        orch = SwarmOrchestrator()
        orch.register_agent("a", MagicMock())
        orch.register_agent("b", MagicMock())
        orch.register_agent("c", MagicMock())

        result = orch.create_swarm(agents=["a", "c"], pattern="supervisor")
        assert sorted(result.agents) == ["a", "c"]

    def test_handoff_pattern_with_swarm_available(self):
        """Should use langgraph-swarm when available."""
        mock_compiled = MagicMock()
        mock_swarm_graph = MagicMock()
        mock_swarm_graph.compile.return_value = mock_compiled

        mock_create_swarm = MagicMock(return_value=mock_swarm_graph)
        mock_swarm_module = MagicMock()
        mock_swarm_module.create_swarm = mock_create_swarm

        with patch.dict("sys.modules", {"langgraph_swarm": mock_swarm_module}):
            from importlib import reload

            import agentomatic.delegation.swarm as swarm_mod

            reload(swarm_mod)

            from agentomatic.delegation.swarm import SwarmOrchestrator

            orch = SwarmOrchestrator()
            agent_a = MagicMock()
            agent_b = MagicMock()
            orch.register_agent("a", agent_a)
            orch.register_agent("b", agent_b)

            result = orch.create_swarm(pattern="handoff")

        assert result is mock_compiled
        mock_create_swarm.assert_called_once()
        call_args = mock_create_swarm.call_args[0][0]
        assert set(call_args) == {agent_a, agent_b}
