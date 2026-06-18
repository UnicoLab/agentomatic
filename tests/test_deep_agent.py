"""Tests for deep_agent integration with Agentomatic Studio.

Tests cover:
  - LangGraphAdapter deep_agent event mapping (subagents, planning, interrupts)
  - Node classification for deep_agent-specific node types
  - Adapter resolution for deep_agent framework hint
  - Deep agent template generation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

# =====================================================================
# Node Classification for Deep Agent Nodes
# =====================================================================


class TestDeepAgentNodeClassification:
    """Test that LangGraphAdapter classifies deep_agent nodes correctly."""

    def test_classify_subagent_node(self):
        from types import SimpleNamespace

        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        node = SimpleNamespace(name="task_dispatcher")
        assert LangGraphAdapter._classify_node("task_agent", node) == "subagent"

    def test_classify_delegate_node(self):
        from types import SimpleNamespace

        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        node = SimpleNamespace(name="delegate_work")
        assert LangGraphAdapter._classify_node("delegate_work", node) == "subagent"

    def test_classify_planning_node(self):
        from types import SimpleNamespace

        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        node = SimpleNamespace(name="write_todos")
        assert LangGraphAdapter._classify_node("write_todos", node) == "planning"

    def test_classify_todo_node(self):
        from types import SimpleNamespace

        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        node = SimpleNamespace(name="plan_step")
        assert LangGraphAdapter._classify_node("plan_step", node) == "planning"

    def test_classify_filesystem_node(self):
        from types import SimpleNamespace

        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        node = SimpleNamespace(name="read_file_op")
        assert LangGraphAdapter._classify_node("read_file", node) == "tool"

    def test_classify_execute_node(self):
        from types import SimpleNamespace

        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        node = SimpleNamespace(name="execute_cmd")
        assert LangGraphAdapter._classify_node("execute_cmd", node) == "tool"

    def test_classify_standard_tool_still_works(self):
        from types import SimpleNamespace

        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        node = SimpleNamespace(name="tool_call")
        assert LangGraphAdapter._classify_node("tool_call", node) == "tool"

    def test_classify_agent_unchanged(self):
        from types import SimpleNamespace

        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        node = SimpleNamespace(name="my_agent")
        assert LangGraphAdapter._classify_node("my_agent", node) == "agent"


# =====================================================================
# Deep Agent Event Mapping
# =====================================================================


class TestDeepAgentEventMapping:
    """Test enhanced event mapping for deep_agent-specific events."""

    def test_map_tool_start(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_tool_start",
                "name": "internet_search",
                "data": {"input": {"query": "AI news"}},
            }
        )
        assert event is not None
        assert event.event == "node_start"
        assert event.node == "tool:internet_search"

    def test_map_tool_end(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_tool_end",
                "name": "internet_search",
                "data": {"output": "search results here"},
            }
        )
        assert event is not None
        assert event.event == "node_end"
        assert event.node == "tool:internet_search"

    def test_map_chat_model_stream(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        chunk = MagicMock()
        chunk.content = "Hello"

        event = LangGraphAdapter._map_event(
            {
                "event": "on_chat_model_stream",
                "name": "ChatModel",
                "data": {"chunk": chunk},
            }
        )
        assert event is not None
        assert event.event == "message_chunk"
        assert event.data["content"] == "Hello"

    def test_map_planning_tool_write_todos(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_tool_start",
                "name": "write_todos",
                "data": {"input": {"todos": ["Research AI", "Write report"]}},
            }
        )
        assert event is not None
        assert event.event == "task_update"
        assert event.node == "planning:write_todos"

    def test_map_subagent_start(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_chain_start",
                "name": "researcher",
                "data": {},
                "tags": [],
                "metadata": {"langgraph_checkpoint_ns": "subagent:researcher"},
            }
        )
        assert event is not None
        assert event.event == "subagent_start"
        assert event.data["namespace"] == "subagent:researcher"

    def test_map_subagent_end(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_chain_end",
                "name": "researcher",
                "data": {"output": "research complete"},
                "metadata": {"langgraph_checkpoint_ns": "subagent:researcher"},
            }
        )
        assert event is not None
        assert event.event == "subagent_end"
        assert event.data["output"] == "research complete"

    def test_map_regular_chain_start_no_namespace(self):
        """Regular chain starts without deep_agent namespace should map as node_start."""
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_chain_start",
                "name": "process",
                "data": {},
                "tags": [],
                "metadata": {},
            }
        )
        assert event is not None
        assert event.event == "node_start"

    def test_map_retriever_start(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_retriever_start",
                "name": "VectorStoreRetriever",
                "data": {"input": {"query": "What is AI?"}},
            }
        )
        assert event is not None
        assert event.event == "node_start"
        assert event.node == "retriever:VectorStoreRetriever"
        assert event.data["query"] == "What is AI?"

    def test_map_retriever_end(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_retriever_end",
                "name": "VectorStoreRetriever",
                "data": {"output": [{"page_content": "doc1"}, {"page_content": "doc2"}]},
            }
        )
        assert event is not None
        assert event.event == "node_end"
        assert event.node == "retriever:VectorStoreRetriever"
        assert event.data["document_count"] == 2

    def test_map_llm_start(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_llm_start",
                "name": "OpenAI",
                "data": {"prompts": ["Hello, world!"]},
            }
        )
        assert event is not None
        assert event.event == "node_start"
        assert event.node == "llm:OpenAI"

    def test_map_llm_end(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_llm_end",
                "name": "OpenAI",
                "data": {"output": {"text": "Generated response"}},
            }
        )
        assert event is not None
        assert event.event == "node_end"
        assert event.node == "llm:OpenAI"

    def test_map_unknown_event_returns_none(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        event = LangGraphAdapter._map_event(
            {
                "event": "on_something_else",
                "name": "unknown",
                "data": {},
            }
        )
        assert event is None


# =====================================================================
# Adapter Resolution for Deep Agent
# =====================================================================


class TestDeepAgentAdapterResolution:
    """Test adapter factory resolves deep_agent agents correctly."""

    def _make_agent(self, graph_fn=None, node_fn=None, framework="custom"):
        agent = MagicMock()
        agent.name = "deep_test"
        agent.graph_fn = graph_fn
        agent.node_fn = node_fn
        agent.manifest = MagicMock()
        agent.manifest.name = "Deep Test"
        agent.manifest.description = "A deep agent test"
        agent.manifest.framework = framework
        agent.module_path = None
        # Remove spurious MagicMock attributes
        del agent._studio_adapter
        del agent._studio_graph_fn
        del agent._studio_state_fn
        del agent._studio_stream_fn
        return agent

    def test_deep_agent_with_graph_fn_uses_langgraph(self):
        """Deep agent with graph_fn should resolve to LangGraphAdapter."""
        from agentomatic.studio.adapters import resolve_adapter
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        mock_graph = MagicMock()
        mock_graph.checkpointer = None
        agent = self._make_agent(
            graph_fn=MagicMock(return_value=mock_graph),
            framework="langgraph",
        )
        adapter = resolve_adapter(agent)
        assert isinstance(adapter, LangGraphAdapter)

    def test_deep_agent_framework_hint_uses_langgraph(self):
        """Agent with framework='deepagent' should resolve to LangGraphAdapter."""
        from agentomatic.studio.adapters import resolve_adapter
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        mock_graph = MagicMock()
        mock_graph.checkpointer = None
        agent = self._make_agent(
            graph_fn=MagicMock(return_value=mock_graph),
            framework="deepagent",
        )
        adapter = resolve_adapter(agent)
        assert isinstance(adapter, LangGraphAdapter)

    def test_deep_agent_without_graph_fn_and_deepagent_framework(self):
        """Agent with framework='deepagent' but no graph_fn should still get LangGraph adapter."""
        from agentomatic.studio.adapters import resolve_adapter
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        agent = self._make_agent(
            node_fn=AsyncMock(),
            framework="deepagent",
        )
        adapter = resolve_adapter(agent)
        assert isinstance(adapter, LangGraphAdapter)


# =====================================================================
# LangGraphAdapter Capabilities for Deep Agent
# =====================================================================


class TestDeepAgentCapabilities:
    """Test that LangGraphAdapter reports deep_agent capabilities."""

    def test_capabilities_with_deep_agent_graph(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        # Mock a graph with deep_agent nodes
        mock_graph = MagicMock()
        mock_graph.checkpointer = None

        # Create drawable with deep_agent nodes
        write_todos_node = MagicMock()
        write_todos_node.name = "write_todos"
        task_node = MagicMock()
        task_node.name = "task"
        agent_node = MagicMock()
        agent_node.name = "agent"

        mock_drawable = MagicMock()
        mock_drawable.nodes = {
            "write_todos": write_todos_node,
            "task": task_node,
            "agent": agent_node,
        }
        mock_graph.get_graph.return_value = mock_drawable

        agent = MagicMock()
        agent.name = "deep_agent_test"
        agent.graph_fn = MagicMock(return_value=mock_graph)
        agent.manifest = MagicMock()
        agent.manifest.framework = "langgraph"

        adapter = LangGraphAdapter(agent=agent)
        caps = adapter.capabilities
        assert "graph" in caps
        assert "streaming" in caps
        assert "deep_agent" in caps
        assert "subagents" in caps
        assert "planning" in caps

    def test_capabilities_without_deep_agent_nodes(self):
        from agentomatic.studio.adapters.langgraph import LangGraphAdapter

        mock_graph = MagicMock()
        mock_graph.checkpointer = None

        regular_node = MagicMock()
        regular_node.name = "process"
        mock_drawable = MagicMock()
        mock_drawable.nodes = {"process": regular_node}
        mock_graph.get_graph.return_value = mock_drawable

        agent = MagicMock()
        agent.name = "regular_agent"
        agent.graph_fn = MagicMock(return_value=mock_graph)
        agent.manifest = MagicMock()
        agent.manifest.framework = "langgraph"

        adapter = LangGraphAdapter(agent=agent)
        caps = adapter.capabilities
        assert "graph" in caps
        assert "streaming" in caps
        assert "deep_agent" not in caps


# =====================================================================
# Deep Agent Template
# =====================================================================


class TestDeepAgentTemplate:
    """Test the deepagent scaffold template."""

    def test_template_exists(self):
        from agentomatic.cli.templates import TEMPLATES

        assert "deepagent" in TEMPLATES

    def test_template_generates_files(self):
        from agentomatic.cli.templates import get_template_files

        files = get_template_files("deepagent", "my_research_agent")
        assert "__init__.py" in files
        assert "agent.py" in files
        assert "config.py" in files
        assert "README.md" in files

    def test_template_init_has_manifest(self):
        from agentomatic.cli.templates import get_template_files

        files = get_template_files("deepagent", "researcher")
        init_content = files["__init__.py"]
        assert "AgentManifest" in init_content
        assert "researcher" in init_content
        assert "graph_fn" in init_content

    def test_template_agent_has_create_deep_agent(self):
        from agentomatic.cli.templates import get_template_files

        files = get_template_files("deepagent", "researcher")
        agent_content = files["agent.py"]
        assert "create_deep_agent" in agent_content
        assert "deepagents" in agent_content

    def test_template_has_tools(self):
        from agentomatic.cli.templates import get_template_files

        files = get_template_files("deepagent", "researcher")
        agent_content = files["agent.py"]
        assert "internet_search" in agent_content


# =====================================================================
# Studio Models — New Event Types
# =====================================================================


class TestDeepAgentEventModels:
    """Test that Studio models support deep_agent event types."""

    def test_subagent_start_event(self):
        from agentomatic.studio.models import StudioRunEvent

        event = StudioRunEvent(
            event="subagent_start",
            run_id="r1",
            timestamp="2025-01-01T00:00:00Z",
            node="researcher",
            data={"namespace": "subagent:researcher"},
        )
        assert event.event == "subagent_start"

    def test_subagent_end_event(self):
        from agentomatic.studio.models import StudioRunEvent

        event = StudioRunEvent(
            event="subagent_end",
            run_id="r1",
            timestamp="2025-01-01T00:00:00Z",
            node="researcher",
            data={"output": "done"},
        )
        assert event.event == "subagent_end"

    def test_task_update_event(self):
        from agentomatic.studio.models import StudioRunEvent

        event = StudioRunEvent(
            event="task_update",
            run_id="r1",
            timestamp="2025-01-01T00:00:00Z",
            node="planning:write_todos",
            data={"todos": ["Step 1", "Step 2"]},
        )
        assert event.event == "task_update"
        assert event.data["todos"] == ["Step 1", "Step 2"]

    def test_breakpoint_hit_with_resumable(self):
        from agentomatic.studio.models import StudioRunEvent

        event = StudioRunEvent(
            event="breakpoint_hit",
            run_id="r1",
            timestamp="2025-01-01T00:00:00Z",
            node="approval_gate",
            data={"reason": "Human approval required", "resumable": True},
        )
        assert event.event == "breakpoint_hit"
        assert event.data["resumable"] is True

    def test_node_type_subagent(self):
        from agentomatic.studio.models import StudioGraphNode

        node = StudioGraphNode(id="task", name="Task Delegator", type="subagent")
        assert node.type == "subagent"

    def test_node_type_planning(self):
        from agentomatic.studio.models import StudioGraphNode

        node = StudioGraphNode(id="planner", name="Write Todos", type="planning")
        assert node.type == "planning"

    def test_framework_deepagent(self):
        from agentomatic.studio.models import StudioAgentInfo

        info = StudioAgentInfo(
            name="research_agent",
            slug="agent-research",
            framework="deepagent",
            capabilities=["graph", "streaming", "deep_agent", "subagents"],
        )
        assert info.framework == "deepagent"
