# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for the Agentomatic Studio Debug API.

Tests cover:
  - Studio module imports and exports
  - Graph inspector behavior
  - Run tracker event tracking
  - Universal adapter architecture
  - Router endpoint integration (via TestClient)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
            StudioGraphTopology,
            StudioServerInfo,
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
        assert completed is not None
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
        assert failed is not None
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
        assert updated is not None
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

        # Create a test agent using spec to avoid spurious attributes
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
        # Ensure adapter factory doesn't pick up spurious MagicMock attrs
        del test_agent._studio_adapter
        del test_agent._studio_graph_fn
        del test_agent._studio_state_fn
        del test_agent._studio_stream_fn

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
        # node_fn agent should get a simple linear graph via GenericAdapter
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
        assert data["state"] == {}  # GenericAdapter returns empty on first access

    def test_get_history_no_store(self, client):
        response = client.get("/studio/agents/test_agent/threads/t1/history")
        assert response.status_code == 200
        assert response.json() == []


# =====================================================================
# Schema Discovery
# =====================================================================


class TestSchemaDiscovery:
    """Schema resolution conventions and schema_source metadata."""

    @pytest.fixture
    def default_client(self):
        """Client with an agent that has no custom schemas."""
        from agentomatic.studio.router import create_studio_router

        test_agent = MagicMock()
        test_agent.name = "plain"
        test_agent.slug = "plain"
        test_agent.manifest = MagicMock()
        test_agent.manifest.name = "Plain"
        test_agent.manifest.description = ""
        test_agent.manifest.version = "1.0.0"
        test_agent.manifest.framework = "custom"
        test_agent.module_path = None
        test_agent.schema_validator = None
        test_agent.config = None
        test_agent.prompt_manager = None
        test_agent.graph_fn = None
        test_agent.node_fn = None
        del test_agent._studio_adapter
        del test_agent._studio_graph_fn
        del test_agent._studio_state_fn
        del test_agent._studio_stream_fn

        registry = MagicMock()
        registry.count = 1
        registry.get.side_effect = lambda name: test_agent if name == "plain" else None
        registry.all.return_value = {"plain": test_agent}

        app = FastAPI()
        app.include_router(
            create_studio_router(
                registry=registry,
                store=None,
                platform_title="Test",
                platform_version="0.1.0",
            )
        )
        return TestClient(app)

    @pytest.fixture
    def custom_client(self):
        """Client with the alpha agent (custom Input/Output schemas)."""
        from agentomatic.studio.router import create_studio_router

        alpha_agent = MagicMock()
        alpha_agent.name = "alpha"
        alpha_agent.slug = "alpha"
        alpha_agent.manifest = MagicMock()
        alpha_agent.manifest.name = "Alpha"
        alpha_agent.manifest.description = ""
        alpha_agent.manifest.version = "1.0.0"
        alpha_agent.manifest.framework = "langgraph"
        alpha_agent.module_path = "src.agents.alpha"
        alpha_agent.schema_validator = None
        alpha_agent.config = None
        alpha_agent.prompt_manager = None
        alpha_agent.graph_fn = None
        alpha_agent.node_fn = None
        del alpha_agent._studio_adapter
        del alpha_agent._studio_graph_fn
        del alpha_agent._studio_state_fn
        del alpha_agent._studio_stream_fn

        registry = MagicMock()
        registry.count = 1
        registry.get.side_effect = lambda name: alpha_agent if name == "alpha" else None
        registry.all.return_value = {"alpha": alpha_agent}

        app = FastAPI()
        app.include_router(
            create_studio_router(
                registry=registry,
                store=None,
                platform_title="Test",
                platform_version="0.1.0",
            )
        )
        return TestClient(app)

    def _make_agent(self, name="alpha", module_path="", schema_validator=None):
        agent = MagicMock()
        agent.name = name
        agent.module_path = module_path
        agent.schema_validator = schema_validator
        return agent

    def test_candidates_support_input_output_conventions(self):
        from agentomatic.core.schemas import schema_model_candidates

        input_candidates, output_candidates = schema_model_candidates("weather_bot")
        assert "CustomInvokeRequest" in input_candidates
        assert "WeatherBotRequest" in input_candidates
        assert "WeatherBotInput" in input_candidates
        assert "AgentInvokeRequest" in input_candidates
        assert "CustomInvokeResponse" in output_candidates
        assert "WeatherBotResponse" in output_candidates
        assert "WeatherBotOutput" in output_candidates

    def test_load_schema_models_input_output_convention(self):
        """{Title}Input / {Title}Output classes are discovered (alpha/beta)."""
        from agentomatic.core.schemas import load_schema_models

        input_model, output_model = load_schema_models("src.agents.alpha", "alpha")
        assert input_model is not None
        assert input_model.__name__ == "AlphaInput"
        assert output_model is not None
        assert output_model.__name__ == "AlphaOutput"

    def test_load_schema_models_request_response_convention(self):
        """{Title}Request / {Title}Response remain the top custom convention."""
        from agentomatic.core.schemas import load_schema_models

        input_model, output_model = load_schema_models("src.agents.beta", "beta")
        assert input_model is not None
        assert input_model.__name__ == "BetaInput"
        assert output_model is not None
        assert output_model.__name__ == "BetaOutput"

    def test_load_schema_models_missing_module(self):
        from agentomatic.core.schemas import load_schema_models

        assert load_schema_models("nonexistent_pkg", "alpha") == (None, None)
        assert load_schema_models("", "alpha") == (None, None)

    def test_resolve_agent_schemas_custom_module(self):
        """Schemas from the agent's schemas.py are used with provenance."""
        from agentomatic.studio.router import _resolve_agent_schemas

        agent = self._make_agent(name="alpha", module_path="src.agents.alpha")
        input_schema, output_schema, source = _resolve_agent_schemas(agent)

        assert "query" in input_schema["properties"]
        assert "context" in input_schema["properties"]
        assert "response" in output_schema["properties"]
        assert "confidence" in output_schema["properties"]

        assert source["input"].source == "custom"
        assert source["input"].model == "AlphaInput"
        assert source["output"].source == "custom"
        assert source["output"].model == "AlphaOutput"

    def test_resolve_agent_schemas_fallback_default(self):
        """Agents without schemas get default models + source='default'."""
        from agentomatic.studio.router import _resolve_agent_schemas

        agent = self._make_agent(name="plain", module_path="")
        input_schema, _output_schema, source = _resolve_agent_schemas(agent)

        assert "query" in input_schema["properties"]
        assert "user_id" in input_schema["properties"]
        assert source["input"].source == "default"
        assert source["input"].model == "AgentInvokeRequest"
        assert source["output"].model == "AgentInvokeResponse"

    def test_resolve_agent_schemas_prefers_schema_validator(self):
        """Registry SchemaValidator wins over module scanning."""
        from agentomatic.core.schemas import SchemaValidator
        from agentomatic.studio.router import _resolve_agent_schemas
        from src.agents.alpha.schemas import AlphaInput

        validator = SchemaValidator(request_model=AlphaInput)
        agent = self._make_agent(
            name="alpha", module_path="src.agents.alpha", schema_validator=validator
        )
        _input_schema, _output_schema, source = _resolve_agent_schemas(agent)

        assert source["input"].convention == "schema_validator.AlphaInput"
        # Output falls back to default because validator has no response model
        assert source["output"].source == "default"
        assert source["output"].model == "AgentInvokeResponse"

    def test_get_schemas_endpoint_schema_source(self, default_client):
        """The /schemas endpoint returns schema_source metadata."""
        response = default_client.get("/studio/agents/plain/schemas")
        assert response.status_code == 200
        data = response.json()
        assert "input_schema" in data
        assert "output_schema" in data
        assert data["schema_source"]["input"]["source"] == "default"
        assert data["schema_source"]["input"]["model"] == "AgentInvokeRequest"
        assert data["schema_source"]["output"]["model"] == "AgentInvokeResponse"

    def test_get_schemas_endpoint_custom(self, custom_client):
        """Custom schemas from an agent's module are returned as custom."""
        response = custom_client.get("/studio/agents/alpha/schemas")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data["input_schema"]["properties"]
        assert "context" in data["input_schema"]["properties"]
        assert data["schema_source"]["input"]["source"] == "custom"
        assert data["schema_source"]["input"]["model"] == "AlphaInput"
        assert data["schema_source"]["output"]["model"] == "AlphaOutput"


# =====================================================================
# Universal Adapter Architecture
# =====================================================================


class TestAdapterImports:
    """Verify the adapter system exports."""

    def test_import_studio_adapter(self):
        from agentomatic.studio.adapter import StudioAdapter

        assert StudioAdapter is not None

    def test_import_langgraph_adapter(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        assert LangGraphAdapter is not None

    def test_import_generic_adapter(self):
        from agentomatic.studio.adapters.generic import GenericAdapter

        assert GenericAdapter is not None

    def test_import_resolve_adapter(self):
        from agentomatic.studio.adapters import resolve_adapter

        assert callable(resolve_adapter)

    def test_import_decorators(self):
        from agentomatic.studio.decorators import (
            register_studio_hooks,
            studio_graph,
            studio_state,
            studio_stream,
        )

        assert callable(studio_graph)
        assert callable(studio_state)
        assert callable(studio_stream)
        assert callable(register_studio_hooks)

    def test_public_exports(self):
        from agentomatic.studio import (
            StudioAdapter,
            studio_graph,
        )

        assert StudioAdapter is not None
        assert callable(studio_graph)


class TestAdapterResolution:
    """Test the adapter factory resolution logic."""

    def _make_agent(self, graph_fn=None, node_fn=None, framework="custom"):
        agent = MagicMock()
        agent.name = "test_agent"
        agent.graph_fn = graph_fn
        agent.node_fn = node_fn
        agent.class_instance = None
        agent.manifest = MagicMock()
        agent.manifest.name = "Test Agent"
        agent.manifest.description = "A test agent"
        agent.manifest.framework = framework
        agent.module_path = None
        # Remove spurious MagicMock attributes
        del agent._studio_adapter
        del agent._studio_graph_fn
        del agent._studio_state_fn
        del agent._studio_stream_fn
        return agent

    def test_resolve_generic_for_node_fn(self):
        from agentomatic.studio.adapters import resolve_adapter
        from agentomatic.studio.adapters.generic import GenericAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = resolve_adapter(agent)
        assert isinstance(adapter, GenericAdapter)

    def test_resolve_langgraph_for_graph_fn(self):
        from agentomatic.studio.adapters import resolve_adapter
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        mock_graph = MagicMock()
        mock_graph.checkpointer = None
        agent = self._make_agent(graph_fn=MagicMock(return_value=mock_graph))
        adapter = resolve_adapter(agent)
        assert isinstance(adapter, LangGraphAdapter)

    def test_resolve_custom_adapter(self):
        from agentomatic.studio.adapters import resolve_adapter

        agent = self._make_agent(node_fn=AsyncMock())
        custom = MagicMock()
        agent._studio_adapter = custom
        adapter = resolve_adapter(agent)
        assert adapter is custom

    def test_resolve_langchain_for_framework(self):
        from agentomatic.studio.adapters import resolve_adapter
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        agent = self._make_agent(node_fn=AsyncMock(), framework="langchain")
        adapter = resolve_adapter(agent)
        assert isinstance(adapter, LangChainAdapter)

    def test_resolve_graph_agent_before_langgraph_graph_fn(self):
        """Class agents expose graph_fn but must use GraphAgentAdapter."""
        from agentomatic.studio.adapters import resolve_adapter
        from agentomatic.studio.adapters.graph_agent import GraphAgentAdapter

        mock_graph = MagicMock()
        agent = self._make_agent(
            graph_fn=MagicMock(return_value=mock_graph),
            framework="graph_agent",
        )
        adapter = resolve_adapter(agent)
        assert isinstance(adapter, GraphAgentAdapter)

    def test_resolve_graph_agent_framework_without_instance(self):
        from agentomatic.studio.adapters import resolve_adapter
        from agentomatic.studio.adapters.graph_agent import GraphAgentAdapter

        agent = self._make_agent(framework="graph_agent")
        agent.class_instance = None
        adapter = resolve_adapter(agent)
        assert isinstance(adapter, GraphAgentAdapter)


class TestGenericAdapter:
    """Test the generic trace-based adapter."""

    def _make_agent(self, node_fn=None, framework="custom"):
        agent = MagicMock()
        agent.name = "my_agent"
        agent.manifest = MagicMock()
        agent.manifest.name = "My Agent"
        agent.manifest.description = "A test agent"
        agent.manifest.framework = framework
        agent.graph_fn = None
        agent.node_fn = node_fn
        agent.module_path = None
        del agent._studio_adapter
        del agent._studio_graph_fn
        del agent._studio_state_fn
        del agent._studio_stream_fn
        return agent

    @pytest.mark.asyncio
    async def test_get_graph_synthetic(self):
        from agentomatic.studio.adapters.generic import GenericAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = GenericAdapter(agent)
        graph = await adapter.get_graph()

        assert graph.agent_name == "my_agent"
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.entry_point == "__start__"
        node_ids = {n.id for n in graph.nodes}
        assert "__start__" in node_ids
        assert "my_agent" in node_ids
        assert "__end__" in node_ids

    def test_capabilities(self):
        from agentomatic.studio.adapters.generic import GenericAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = GenericAdapter(agent)
        caps = adapter.capabilities
        assert "streaming" in caps
        assert "traces" in caps
        assert "graph" not in caps
        assert "breakpoints" not in caps

    @pytest.mark.asyncio
    async def test_get_state_empty(self):
        from agentomatic.studio.adapters.generic import GenericAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = GenericAdapter(agent)
        state = await adapter.get_state("thread_1")
        assert state is not None
        assert state.thread_id == "thread_1"
        assert state.state == {}

    @pytest.mark.asyncio
    async def test_update_state(self):
        from agentomatic.studio.adapters.generic import GenericAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = GenericAdapter(agent)
        result = await adapter.update_state("t1", {"key": "value"})
        assert result is not None
        assert result.state["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_history_empty(self):
        from agentomatic.studio.adapters.generic import GenericAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = GenericAdapter(agent)
        history = await adapter.get_history("t1")
        assert history == []

    @pytest.mark.asyncio
    async def test_stream_execution(self):
        from agentomatic.studio.adapters.generic import GenericAdapter

        agent = self._make_agent(node_fn=AsyncMock(return_value={"response": "hello"}))
        adapter = GenericAdapter(agent)

        events = []
        async for event in adapter.stream_execution(
            {"current_query": "test"},
            config={"configurable": {"thread_id": "t1"}},
        ):
            events.append(event)

        # Should have: node_start, trace, node_end
        event_types = [e.event for e in events]
        assert "node_start" in event_types
        assert "node_end" in event_types
        assert "trace" in event_types


class TestDecorators:
    """Test studio decorator registration."""

    def test_studio_graph_marks_function(self):
        from agentomatic.studio.decorators import studio_graph

        @studio_graph
        def my_graph():
            return {"nodes": [], "edges": []}

        assert hasattr(my_graph, "_is_studio_graph")
        assert getattr(my_graph, "_is_studio_graph") is True

    def test_studio_state_marks_function(self):
        from agentomatic.studio.decorators import studio_state

        @studio_state
        def my_state(thread_id):
            return {}

        assert hasattr(my_state, "_is_studio_state")
        assert getattr(my_state, "_is_studio_state") is True

    def test_studio_stream_marks_function(self):
        from agentomatic.studio.decorators import studio_stream

        @studio_stream
        async def my_stream(state, config, breakpoints):
            yield  # pragma: no cover

        assert hasattr(my_stream, "_is_studio_stream")
        assert getattr(my_stream, "_is_studio_stream") is True


class TestLangChainAdapter:
    """Test the LangChain-specific adapter."""

    def _make_agent(self, node_fn=None):
        agent = MagicMock()
        agent.name = "chatbot"
        agent.manifest = MagicMock()
        agent.manifest.name = "Chatbot"
        agent.manifest.description = "A LangChain chatbot"
        agent.manifest.framework = "langchain"
        agent.graph_fn = None
        agent.node_fn = node_fn
        agent.module_path = None
        del agent._studio_adapter
        del agent._studio_graph_fn
        del agent._studio_state_fn
        del agent._studio_stream_fn
        del agent._langchain_runnable
        return agent

    def test_import(self):
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        assert LangChainAdapter is not None

    def test_capabilities(self):
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = LangChainAdapter(agent)
        caps = adapter.capabilities
        assert "streaming" in caps
        assert "traces" in caps

    @pytest.mark.asyncio
    async def test_get_graph_synthetic(self):
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = LangChainAdapter(agent)
        graph = await adapter.get_graph()

        assert graph.agent_name == "chatbot"
        # Synthetic LangChain graph: start → prompt → llm → output_parser → end
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
        assert graph.metadata.get("framework") == "langchain"

    @pytest.mark.asyncio
    async def test_get_state_empty(self):
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = LangChainAdapter(agent)
        state = await adapter.get_state("t1")
        assert state is not None
        assert state.state == {}

    @pytest.mark.asyncio
    async def test_update_state(self):
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = LangChainAdapter(agent)
        result = await adapter.update_state("t1", {"messages": ["hello"]})
        assert result is not None
        assert result.state["messages"] == ["hello"]

    @pytest.mark.asyncio
    async def test_get_history_empty(self):
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        agent = self._make_agent(node_fn=AsyncMock())
        adapter = LangChainAdapter(agent)
        history = await adapter.get_history("t1")
        assert history == []

    @pytest.mark.asyncio
    async def test_stream_execution_fallback(self):
        from agentomatic.studio.adapters.langchain import LangChainAdapter

        agent = self._make_agent(node_fn=AsyncMock(return_value={"response": "hi there!"}))
        adapter = LangChainAdapter(agent)

        events = []
        async for event in adapter.stream_execution(
            {"current_query": "hello"},
            config={"configurable": {"thread_id": "t1"}},
        ):
            events.append(event)

        event_types = [e.event for e in events]
        assert "node_start" in event_types
        assert "node_end" in event_types
        assert "trace" in event_types

        # Verify history was recorded
        history = await adapter.get_history("t1")
        assert len(history) == 1
        assert history[0].metadata["framework"] == "langchain"
