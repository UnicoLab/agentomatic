"""Full-featured LangGraph adapter for Agentomatic Studio.

Wraps a LangGraph ``CompiledGraph`` to provide the complete Studio
debugging experience: graph topology extraction, SSE event streaming,
checkpoint-based time-travel, live state mutation, and conditional
breakpoints.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.studio.adapter import StudioAdapter
from agentomatic.studio.models import (
    StudioCheckpoint,
    StudioGraphEdge,
    StudioGraphNode,
    StudioGraphTopology,
    StudioRunEvent,
    StudioStateSnapshot,
)

if TYPE_CHECKING:
    from agentomatic.core.manifest import RegisteredAgent
    from agentomatic.storage.base import BaseStore


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class LangGraphAdapter(StudioAdapter):
    """Full-featured adapter for LangGraph ``CompiledGraph`` agents.

    Provides the complete Studio debugging experience including:

    - **Graph topology** extraction via ``graph.get_graph()``
    - **SSE streaming** via ``graph.astream_events()``
    - **Checkpoint time-travel** via the graph's checkpointer
    - **Live state mutation** via ``graph.aupdate_state()``
    - **Conditional breakpoints** via ``interrupt_before_nodes``

    Args:
        agent: The registered agent with a ``graph_fn``.
        store: Optional storage backend for checkpoint fallback.
    """

    def __init__(
        self,
        agent: RegisteredAgent,
        store: BaseStore | None = None,
    ) -> None:
        super().__init__(agent.name)
        self._agent = agent
        self._store = store

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> list[str]:
        caps = ["graph", "streaming"]
        try:
            graph = self._agent.graph_fn()
            if getattr(graph, "checkpointer", None) is not None:
                caps.extend(["checkpoints", "state", "breakpoints"])
        except Exception:
            pass
        if self._agent.manifest.framework == "langgraph":
            caps.append("hitl")

        # Detect deep_agent capabilities from graph node names
        try:
            drawable = graph.get_graph()
            node_names = [
                getattr(n, "name", str(k)).lower()
                for k, n in getattr(drawable, "nodes", {}).items()
            ]
            if any("write_todo" in n or "task" in n for n in node_names):
                caps.append("deep_agent")
                caps.append("subagents")
                caps.append("planning")
        except Exception:
            pass

        return caps

    # ------------------------------------------------------------------
    # Graph topology
    # ------------------------------------------------------------------

    async def get_graph(self) -> StudioGraphTopology:
        graph = self._agent.graph_fn()
        drawable = graph.get_graph()

        nodes: list[StudioGraphNode] = []
        for node_id, node_data in getattr(drawable, "nodes", {}).items():
            node_type = self._classify_node(node_id, node_data)
            node_name = getattr(node_data, "name", None) or str(node_id)
            node_meta = getattr(node_data, "metadata", None) or {}
            if not isinstance(node_meta, dict):
                node_meta = {}
            nodes.append(
                StudioGraphNode(
                    id=str(node_id),
                    name=node_name,
                    type=node_type,
                    metadata=node_meta,
                )
            )

        edges: list[StudioGraphEdge] = []
        for idx, edge in enumerate(getattr(drawable, "edges", [])):
            source = getattr(edge, "source", None)
            target = getattr(edge, "target", None)
            if source is None or target is None:
                continue
            condition = getattr(edge, "conditional", None) or getattr(edge, "data", None)
            if condition is True:
                condition = "conditional"
            elif condition and not isinstance(condition, str):
                condition = str(condition)
            edges.append(
                StudioGraphEdge(
                    id=f"edge_{idx}",
                    source=str(source),
                    target=str(target),
                    condition=condition if condition else None,
                )
            )

        entry_point: str | None = None
        end_points: list[str] = []
        for node in nodes:
            if node.type == "start" or node.id == "__start__":
                entry_point = node.id
            elif node.type == "end" or node.id == "__end__":
                end_points.append(node.id)

        return StudioGraphTopology(
            agent_name=self._agent.name,
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
            end_points=end_points,
        )

    # ------------------------------------------------------------------
    # SSE streaming
    # ------------------------------------------------------------------

    async def stream_execution(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
        breakpoints: list[str] | None = None,
        checkpoint_id: str | None = None,
    ) -> AsyncGenerator[StudioRunEvent, None]:
        graph = self._agent.graph_fn()
        config = dict(config or {})

        if breakpoints:
            try:
                graph.interrupt_before_nodes = frozenset(breakpoints)
            except Exception:
                pass

        if checkpoint_id and "configurable" in config:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        try:
            async for lg_event in graph.astream_events(state, config=config, version="v2"):
                studio_event = self._map_event(lg_event)
                if studio_event:
                    yield studio_event
        except Exception as exc:
            exc_name = type(exc).__name__
            if exc_name in ("NodeInterrupt", "GraphInterrupt"):
                yield StudioRunEvent(
                    event="breakpoint_hit",
                    run_id="",
                    timestamp=_now_iso(),
                    node=getattr(exc, "node", "unknown"),
                    data={
                        "reason": str(exc),
                        "resumable": True,
                        "interrupt_type": exc_name,
                    },
                )
            else:
                raise

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    async def get_state(self, thread_id: str) -> StudioStateSnapshot | None:
        state_data: dict[str, Any] = {}
        checkpoint_id: str | None = None

        try:
            graph = self._agent.graph_fn()
            checkpointer = getattr(graph, "checkpointer", None)
            if checkpointer is not None:
                cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                if hasattr(checkpointer, "aget_tuple"):
                    cp_tuple = await checkpointer.aget_tuple(cfg)
                elif hasattr(checkpointer, "get_tuple"):
                    cp_tuple = checkpointer.get_tuple(cfg)
                else:
                    cp_tuple = None

                if cp_tuple:
                    state_data = cp_tuple.checkpoint or {}
                    checkpoint_id = (
                        cp_tuple.config.get("configurable", {}).get("checkpoint_id")
                        if cp_tuple.config
                        else None
                    )
        except Exception as exc:
            logger.warning(f"LangGraph get_state failed: {exc}")

        # Fallback to store
        if not state_data and self._store is not None:
            try:
                cps = await self._store.list_checkpoints(thread_id, "", limit=1)
                if cps:
                    latest = cps[0]
                    state_data = latest.get("checkpoint", {})
                    checkpoint_id = latest.get("checkpoint_id")
            except Exception as exc:
                logger.warning(f"Store fallback get_state failed: {exc}")

        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=self.agent_name,
            state=state_data,
            timestamp=_now_iso(),
            checkpoint_id=checkpoint_id,
        )

    async def update_state(
        self,
        thread_id: str,
        updates: dict[str, Any],
    ) -> StudioStateSnapshot | None:
        current = await self.get_state(thread_id)
        merged = {**(current.state if current else {}), **updates}

        try:
            graph = self._agent.graph_fn()
            if hasattr(graph, "update_state"):
                cfg = {"configurable": {"thread_id": thread_id}}
                await graph.aupdate_state(cfg, updates)
                logger.debug(f"State updated via LangGraph checkpointer for {thread_id}")
        except Exception as exc:
            logger.warning(f"LangGraph update_state failed: {exc}")

        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=self.agent_name,
            state=merged,
            timestamp=_now_iso(),
            checkpoint_id=current.checkpoint_id if current else None,
        )

    # ------------------------------------------------------------------
    # Checkpoint history
    # ------------------------------------------------------------------

    async def get_history(self, thread_id: str) -> list[StudioCheckpoint]:
        checkpoints: list[StudioCheckpoint] = []

        if self._store is not None:
            try:
                raw = await self._store.list_checkpoints(thread_id, "")
                for idx, cp in enumerate(raw):
                    checkpoints.append(
                        StudioCheckpoint(
                            id=cp.get("checkpoint_id", f"cp_{idx}"),
                            thread_id=thread_id,
                            step=idx,
                            state=cp.get("checkpoint", {}),
                            metadata=cp.get("metadata", {}),
                            parent_id=cp.get("parent_checkpoint_id"),
                            timestamp=cp.get("timestamp", _now_iso()),
                        )
                    )
            except Exception as exc:
                logger.warning(f"Failed to list checkpoints: {exc}")

        return checkpoints

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_node(node_id: str, node_data: Any) -> str:
        if node_id == "__start__":
            return "start"
        if node_id == "__end__":
            return "end"
        name = (getattr(node_data, "name", None) or str(node_id)).lower()
        if "tool" in name:
            return "tool"
        if any(kw in name for kw in ("condition", "router", "route", "branch")):
            return "condition"
        if any(kw in name for kw in ("human", "approval", "review")):
            return "human"
        # Deep-agent-specific node classifications
        if any(kw in name for kw in ("task", "delegate", "subagent", "spawn")):
            return "subagent"
        if any(kw in name for kw in ("plan", "todo", "write_todo")):
            return "planning"
        if any(kw in name for kw in ("file", "filesystem", "read_file", "write_file")):
            return "tool"
        if any(kw in name for kw in ("execute", "shell", "command")):
            return "tool"
        return "agent"

    @staticmethod
    def _map_event(lg_event: dict[str, Any]) -> StudioRunEvent | None:
        event_type = lg_event.get("event", "")
        name = lg_event.get("name", "")
        data = lg_event.get("data", {})

        # ----- Subagent delegation events (must come before generic chain) -----
        if event_type == "on_chain_start" and name != "LangGraph":
            metadata = lg_event.get("metadata", {})
            parent_ns = metadata.get("langgraph_checkpoint_ns", "")
            if parent_ns and ":" in parent_ns:
                tags = lg_event.get("tags", [])
                return StudioRunEvent(
                    event="subagent_start",
                    run_id="",
                    timestamp=_now_iso(),
                    node=name,
                    data={"namespace": parent_ns, "tags": tags},
                )
            return StudioRunEvent(
                event="node_start",
                run_id="",
                timestamp=_now_iso(),
                node=name,
                data={"tags": lg_event.get("tags", [])},
            )

        if event_type == "on_chain_end" and name != "LangGraph":
            metadata = lg_event.get("metadata", {})
            parent_ns = metadata.get("langgraph_checkpoint_ns", "")
            output = data.get("output", {})
            if not isinstance(output, (dict, list, str, int, float, bool, type(None))):
                output = str(output)
            if parent_ns and ":" in parent_ns:
                return StudioRunEvent(
                    event="subagent_end",
                    run_id="",
                    timestamp=_now_iso(),
                    node=name,
                    data={"namespace": parent_ns, "output": output},
                )
            return StudioRunEvent(
                event="node_end",
                run_id="",
                timestamp=_now_iso(),
                node=name,
                data={"output": output},
            )

        if event_type == "on_chat_model_stream":
            chunk = data.get("chunk", {})
            content = ""
            if hasattr(chunk, "content"):
                content = chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get("content", "")
            if content:
                return StudioRunEvent(
                    event="message_chunk",
                    run_id="",
                    timestamp=_now_iso(),
                    node=name,
                    data={"content": content},
                )

        if event_type == "on_tool_start":
            tool_input = data.get("input", {})
            if not isinstance(tool_input, (dict, list, str, int, float, bool, type(None))):
                tool_input = str(tool_input)
            # Deep-agent planning tool detection
            if name in ("write_todos", "write_todo"):
                return StudioRunEvent(
                    event="task_update",
                    run_id="",
                    timestamp=_now_iso(),
                    node=f"planning:{name}",
                    data={"todos": tool_input},
                )
            return StudioRunEvent(
                event="node_start",
                run_id="",
                timestamp=_now_iso(),
                node=f"tool:{name}",
                data={"tool_input": tool_input},
            )

        if event_type == "on_tool_end":
            tool_output = data.get("output", "")
            if not isinstance(tool_output, (dict, list, str, int, float, bool, type(None))):
                tool_output = str(tool_output)
            return StudioRunEvent(
                event="node_end",
                run_id="",
                timestamp=_now_iso(),
                node=f"tool:{name}",
                data={"tool_output": tool_output},
            )

        # ----- Retriever events (RAG agents) -----
        if event_type == "on_retriever_start":
            query = data.get("input", {})
            if isinstance(query, dict):
                query = query.get("query", str(query))
            return StudioRunEvent(
                event="node_start",
                run_id="",
                timestamp=_now_iso(),
                node=f"retriever:{name}",
                data={"query": str(query)},
            )

        if event_type == "on_retriever_end":
            docs = data.get("output", [])
            doc_count = len(docs) if isinstance(docs, list) else 0
            return StudioRunEvent(
                event="node_end",
                run_id="",
                timestamp=_now_iso(),
                node=f"retriever:{name}",
                data={"document_count": doc_count},
            )

        # ----- LLM events (non-chat models) -----
        if event_type == "on_llm_start":
            return StudioRunEvent(
                event="node_start",
                run_id="",
                timestamp=_now_iso(),
                node=f"llm:{name}",
                data={"prompts": data.get("prompts", [])},
            )

        if event_type == "on_llm_end":
            generations = data.get("output", {})
            if hasattr(generations, "generations"):
                text = str(generations.generations[0][0].text) if generations.generations else ""
            elif isinstance(generations, dict):
                text = str(generations.get("text", ""))
            else:
                text = str(generations)
            return StudioRunEvent(
                event="node_end",
                run_id="",
                timestamp=_now_iso(),
                node=f"llm:{name}",
                data={"output": text},
            )

        return None
