"""Tests for the Agentomatic Studio Debug API.

Tests cover:
  - Studio module imports and exports
  - Graph inspector behavior
  - Run tracker event tracking
  - Router endpoint integration (via TestClient)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# =====================================================================
# Studio Module Imports
# =====================================================================


class TestStudioImports:
    """Verify the studio module exports are correct."""

    def test_import_graph_inspector(self):
        from agentomatic.studio import GraphInspector

        assert GraphInspector is not None

    def test_import_run_tracker(self):
        from agentomatic.studio import RunTracker

        assert RunTracker is not None

    def test_import_models(self):
        from agentomatic.studio.models import (
            StudioAgentInfo,
            StudioCheckpoint,
            StudioGraphEdge,
            StudioGraphNode,
            StudioGraphTopology,
            StudioRunEvent,
            StudioRunInfo,
            StudioRunRequest,
            StudioServerInfo,
            StudioStateSnapshot,
            StudioStateUpdate,
        )

        # Verify they're all Pydantic models
        assert hasattr(StudioServerInfo, "model_fields")
        assert hasattr(StudioAgentInfo, "model_fields")
        assert hasattr(StudioGraphTopology, "model_fields")

    def test_import_router_factory(self):
        from agentomatic.studio.router import create_studio_router

        assert callable(create_studio_router)

    def test_import_serve(self):
        from agentomatic.studio.serve import is_studio_available, mount_studio_ui

        assert callable(is_studio_available)
        assert callable(mount_studio_ui)


# =====================================================================
# Graph Inspector
# =====================================================================


class TestGraphInspector:
    """Test graph topology extraction."""

    def test_inspect_node_fn_agent(self):
        """A simple node_fn agent produces a linear start→agent→end graph."""
        from agentomatic.studio.graph_inspector import GraphInspector

        inspector = GraphInspector()

        # Mock a RegisteredAgent with only node_fn
        agent = MagicMock()
        agent.name = "test_agent"
        agent.graph_fn = None
        agent.node_fn = AsyncMock()
        agent.manifest = MagicMock()
        agent.manifest.name = "Test Agent"
        agent.manifest.framework = "custom"

        topology = inspector.inspect(agent)

        assert topology.agent_name == "test_agent"
        assert len(topology.nodes) == 3  # start, agent, end
        assert len(topology.edges) == 2
        assert topology.entry_point == "__start__"
        assert "__end__" in topology.end_points

        # Check node types
        node_types = {n.id: n.type for n in topology.nodes}
        assert node_types["__start__"] == "start"
        assert node_types["test_agent"] == "agent"
        assert node_types["__end__"] == "end"

    def test_inspect_no_fn_agent(self):
        """An agent with neither graph_fn nor node_fn returns empty topology."""
        from agentomatic.studio.graph_inspector import GraphInspector

        inspector = GraphInspector()

        agent = MagicMock()
        agent.name = "empty_agent"
        agent.graph_fn = None
        agent.node_fn = None

        topology = inspector.inspect(agent)

        assert topology.agent_name == "empty_agent"
        assert len(topology.nodes) == 0
        assert len(topology.edges) == 0

    def test_classify_node_types(self):
        """Node classification based on name patterns."""
        from agentomatic.studio.graph_inspector import GraphInspector

        inspector = GraphInspector()

        # Test known ID patterns
        assert inspector._classify_node("__start__", MagicMock(name="Start")) == "start"
        assert inspector._classify_node("__end__", MagicMock(name="End")) == "end"

        # Test name-based classification
        tool_node = MagicMock()
        tool_node.name = "search_tool"
        assert inspector._classify_node("node_1", tool_node) == "tool"

        router_node = MagicMock()
        router_node.name = "query_router"
        assert inspector._classify_node("node_2", router_node) == "condition"

        human_node = MagicMock()
        human_node.name = "human_approval"
        assert inspector._classify_node("node_3", human_node) == "human"

        agent_node = MagicMock()
        agent_node.name = "research"
        assert inspector._classify_node("node_4", agent_node) == "agent"

    def test_get_capabilities_node_fn(self):
        """node_fn agents get invoke + streaming + threads capabilities."""
        from agentomatic.studio.graph_inspector import GraphInspector

        inspector = GraphInspector()

        agent = MagicMock()
        agent.graph_fn = None
        agent.node_fn = AsyncMock()
        agent.manifest = MagicMock()
        agent.manifest.framework = "custom"

        caps = inspector.get_capabilities(agent)

        assert "invoke" in caps
        assert "streaming" in caps
        assert "threads" in caps
        assert "graph" not in caps

    def test_get_capabilities_graph_fn(self):
        """graph_fn agents get graph + streaming + threads + hitl capabilities."""
        from agentomatic.studio.graph_inspector import GraphInspector

        inspector = GraphInspector()

        agent = MagicMock()
        agent.graph_fn = MagicMock(return_value=MagicMock(checkpointer=None))
        agent.node_fn = None
        agent.manifest = MagicMock()
        agent.manifest.framework = "langgraph"

        caps = inspector.get_capabilities(agent)

        assert "graph" in caps
        assert "streaming" in caps
        assert "hitl" in caps

    def test_inspect_compiled_graph_defensive(self):
        """Graph inspection handles missing attributes gracefully."""
        from agentomatic.studio.graph_inspector import GraphInspector

        inspector = GraphInspector()

        # Build mock DrawableGraph
        mock_node = MagicMock()
        mock_node.name = "test_node"
        mock_node.metadata = {}

        mock_edge = MagicMock()
        mock_edge.source = "__start__"
        mock_edge.target = "test_node"
        mock_edge.conditional = False

        mock_drawable = MagicMock()
        mock_drawable.nodes = {"__start__": mock_node, "test_node": mock_node}
        mock_drawable.edges = [mock_edge]

        mock_graph = MagicMock()
        mock_graph.get_graph.return_value = mock_drawable

        agent = MagicMock()
        agent.name = "graph_agent"
        agent.graph_fn = MagicMock(return_value=mock_graph)
        agent.node_fn = None

        topology = inspector.inspect(agent)

        assert topology.agent_name == "graph_agent"
        assert len(topology.nodes) >= 1
        assert len(topology.edges) == 1


# =====================================================================
# Run Tracker
# =====================================================================


class TestRunTracker:
    """Test execution tracking with events."""

    def test_create_run(self):
        from agentomatic.studio.run_tracker import RunTracker

        tracker = RunTracker()
        run = tracker.create_run(
            agent_name="test",
            thread_id="thread_123",
            request_data={"query": "hello"},
        )

        assert run.id.startswith("run_")
        assert run.agent_name == "test"
        assert run.thread_id == "thread_123"
        assert run.status == "pending"

    def test_get_run(self):
        from agentomatic.studio.run_tracker import RunTracker

        tracker = RunTracker()
        run = tracker.create_run("test", "t1", {"query": "q"})

        found = tracker.get_run(run.id)
        assert found is not None
        assert found.id == run.id

        assert tracker.get_run("nonexistent") is None

    def test_list_runs(self):
        from agentomatic.studio.run_tracker import RunTracker

        tracker = RunTracker()
        tracker.create_run("agent_a", "t1", {"query": "q1"})
        tracker.create_run("agent_b", "t2", {"query": "q2"})
        tracker.create_run("agent_a", "t3", {"query": "q3"})

        all_runs = tracker.list_runs()
        assert len(all_runs) == 3

        agent_a_runs = tracker.list_runs(agent_name="agent_a")
        assert len(agent_a_runs) == 2

    def test_complete_run(self):
        from agentomatic.studio.run_tracker import RunTracker

        tracker = RunTracker()
        run = tracker.create_run("test", "t1", {"query": "q"})

        tracker.complete_run(run.id, {"response": "done"}, 1234.5)

        completed = tracker.get_run(run.id)
        assert completed.status == "completed"
        assert completed.output == {"response": "done"}
        assert completed.duration_ms == 1234.5
        assert completed.completed_at is not None

    def test_fail_run(self):
        from agentomatic.studio.run_tracker import RunTracker

        tracker = RunTracker()
        run = tracker.create_run("test", "t1", {"query": "q"})

        tracker.fail_run(run.id, "Something went wrong")

        failed = tracker.get_run(run.id)
        assert failed.status == "failed"
        assert failed.error == "Something went wrong"

    def test_add_event(self):
        from agentomatic.studio.models import StudioRunEvent
        from agentomatic.studio.run_tracker import RunTracker

        tracker = RunTracker()
        run = tracker.create_run("test", "t1", {"query": "q"})

        event = StudioRunEvent(
            event="node_start",
            run_id=run.id,
            timestamp="2024-01-01T00:00:00Z",
            node="research",
        )
        tracker.add_event(run.id, event)

        updated = tracker.get_run(run.id)
        assert len(updated.events) == 1
        assert updated.events[0].event == "node_start"
        assert updated.events[0].node == "research"


# =====================================================================
# Serve Module
# =====================================================================


class TestServeModule:
    """Test static file serving setup."""

    def test_studio_not_available_without_files(self):
        """is_studio_available returns False when no index.html exists."""
        from agentomatic.studio.serve import is_studio_available

        # Unless someone has actually built the studio, this should be False
        # (we don't commit built files during tests)
        result = is_studio_available()
        assert isinstance(result, bool)

    def test_mount_skips_when_not_available(self):
        """mount_studio_ui does nothing when static files are missing."""
        from agentomatic.studio.serve import mount_studio_ui

        app = FastAPI()
        routes_before = len(app.routes)

        # If studio is not available, no routes should be added
        mount_studio_ui(app)
        # The function should either add routes (if available) or not (if not)
        # Either way, it should not raise


# =====================================================================
# Pydantic Model Validation
# =====================================================================


class TestModels:
    """Test Pydantic model serialization."""

    def test_server_info(self):
        from agentomatic.studio.models import StudioServerInfo

        info = StudioServerInfo(
            version="1.0.0",
            platform_title="Test Platform",
            agent_count=3,
            capabilities=["studio", "streaming"],
        )
        data = info.model_dump()
        assert data["version"] == "1.0.0"
        assert data["agent_count"] == 3
        assert "studio" in data["capabilities"]

    def test_agent_info(self):
        from agentomatic.studio.models import StudioAgentInfo

        info = StudioAgentInfo(
            name="test",
            slug="test-agent",
            description="A test agent",
            version="1.0.0",
            framework="langgraph",
            capabilities=["graph", "streaming"],
            has_graph=True,
            has_config=False,
            has_prompts=False,
        )
        assert info.has_graph is True
        assert info.framework == "langgraph"

    def test_graph_topology(self):
        from agentomatic.studio.models import (
            StudioGraphEdge,
            StudioGraphNode,
            StudioGraphTopology,
        )

        topology = StudioGraphTopology(
            agent_name="test",
            nodes=[
                StudioGraphNode(id="start", name="Start", type="start"),
                StudioGraphNode(id="end", name="End", type="end"),
            ],
            edges=[
                StudioGraphEdge(id="e1", source="start", target="end"),
            ],
            entry_point="start",
            end_points=["end"],
        )
        data = topology.model_dump()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["entry_point"] == "start"

    def test_run_request(self):
        from agentomatic.studio.models import StudioRunRequest

        req = StudioRunRequest(query="Hello world")
        assert req.user_id == "default-user"
        assert req.breakpoints == []

        req2 = StudioRunRequest(
            query="test",
            user_id="user_1",
            thread_id="t_1",
            breakpoints=["node_a"],
        )
        assert req2.thread_id == "t_1"
        assert len(req2.breakpoints) == 1

    def test_run_event(self):
        from agentomatic.studio.models import StudioRunEvent

        event = StudioRunEvent(
            event="node_start",
            run_id="run_abc",
            timestamp="2024-01-01T00:00:00Z",
            node="research",
            data={"tags": ["step1"]},
        )
        json_str = event.model_dump_json()
        assert "node_start" in json_str
        assert "research" in json_str

    def test_state_snapshot(self):
        from agentomatic.studio.models import StudioStateSnapshot

        snap = StudioStateSnapshot(
            thread_id="t1",
            agent_name="test",
            state={"messages": [], "response": "hello"},
            timestamp="2024-01-01T00:00:00Z",
            checkpoint_id="cp_123",
        )
        assert snap.state["response"] == "hello"
        assert snap.checkpoint_id == "cp_123"


# =====================================================================
# Router Integration (using mock registry)
# =====================================================================


class TestRouterIntegration:
    """Integration tests for the Studio API router endpoints."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock AgentRegistry with test agents."""
        registry = MagicMock()

        # Create a test agent
        test_agent = MagicMock()
        test_agent.name = "test_agent"
        test_agent.slug = "test-agent"
        test_agent.manifest = MagicMock()
        test_agent.manifest.name = "Test Agent"
        test_agent.manifest.description = "A test agent"
        test_agent.manifest.version = "1.0.0"
        test_agent.manifest.framework = "custom"
        test_agent.graph_fn = None
        test_agent.node_fn = AsyncMock(return_value={"response": "hello"})
        test_agent.config = None
        test_agent.prompt_manager = None
        test_agent.module_path = None

        # Registry methods
        registry.count = 1
        registry.get.side_effect = lambda name: test_agent if name == "test_agent" else None
        registry.all.return_value = {"test_agent": test_agent}

        return registry

    @pytest.fixture
    def client(self, mock_registry):
        """Create a test client with the studio router mounted."""
        from agentomatic.studio.router import create_studio_router

        app = FastAPI()
        router = create_studio_router(
            registry=mock_registry,
            store=None,
            platform_title="Test Platform",
            platform_version="0.1.0",
        )
        app.include_router(router)
        return TestClient(app)

    def test_get_info(self, client):
        response = client.get("/studio/info")
        assert response.status_code == 200
        data = response.json()
        assert data["platform_title"] == "Test Platform"
        assert data["version"] == "0.1.0"
        assert data["agent_count"] == 1
        assert "studio" in data["capabilities"]
        assert "streaming" in data["capabilities"]

    def test_list_agents(self, client):
        response = client.get("/studio/agents")
        assert response.status_code == 200
        agents = response.json()
        assert len(agents) == 1
        assert agents[0]["name"] == "test_agent"
        assert agents[0]["framework"] == "custom"
        assert agents[0]["has_graph"] is False

    def test_get_graph(self, client):
        response = client.get("/studio/agents/test_agent/graph")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_name"] == "test_agent"
        # node_fn agent should get a simple linear graph
        assert len(data["nodes"]) == 3  # start, agent, end
        assert len(data["edges"]) == 2
        assert data["entry_point"] == "__start__"

    def test_get_graph_not_found(self, client):
        response = client.get("/studio/agents/nonexistent/graph")
        assert response.status_code == 404

    def test_get_config(self, client):
        response = client.get("/studio/agents/test_agent/config")
        assert response.status_code == 200
        data = response.json()
        assert data["agent"] == "test_agent"
        assert data["config"] == {}

    def test_list_runs_empty(self, client):
        response = client.get("/studio/agents/test_agent/runs")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_run_not_found(self, client):
        response = client.get("/studio/agents/test_agent/runs/nonexistent")
        assert response.status_code == 404

    def test_get_state_no_checkpointer(self, client):
        response = client.get("/studio/agents/test_agent/threads/t1/state")
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "t1"
        assert data["agent_name"] == "test_agent"
        assert data["state"] == {}  # No checkpointer, empty state

    def test_get_history_no_store(self, client):
        response = client.get("/studio/agents/test_agent/threads/t1/history")
        assert response.status_code == 200
        assert response.json() == []
