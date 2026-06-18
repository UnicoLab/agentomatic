"""LangChain adapter for Agentomatic Studio.

Provides a rich Studio experience for LangChain-based agents (chains,
LCEL runnables, chatbots, etc.) by wrapping the LangChain streaming
API with trace capture, synthetic graph generation, and state tracking.

This adapter bridges the gap between LangGraph's full-featured APIs
and the generic fallback, giving LangChain users a first-class
Studio experience with:

- **Automatic graph generation** from LCEL chain structure
- **Rich SSE streaming** via LangChain's ``astream_events`` (v2)
- **Message history** tracking for chatbot-style agents
- **Trace capture** with timing, token counts, and model metadata
"""

from __future__ import annotations

import time
from collections import defaultdict
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


class LangChainAdapter(StudioAdapter):
    """Studio adapter for LangChain agents and LCEL runnables.

    Provides an enhanced debugging experience for LangChain-based
    agents including:

    - **LCEL graph extraction** via ``runnable.get_graph()`` (when available)
    - **Rich SSE streaming** via ``runnable.astream_events()``
    - **Message state tracking** for chatbot-style agents
    - **Execution history** for the History tab

    This adapter is selected automatically when an agent's manifest
    declares ``framework='langchain'`` and it has a ``node_fn``.

    You can also manually assign it::

        from agentomatic.studio.adapters.langchain import LangChainAdapter

        agent._studio_adapter = LangChainAdapter(agent)

    Args:
        agent: The registered agent.
        store: Optional storage backend.
    """

    def __init__(
        self,
        agent: RegisteredAgent,
        store: BaseStore | None = None,
    ) -> None:
        super().__init__(agent.name)
        self._agent = agent
        self._store = store
        self._state_store: dict[str, dict[str, Any]] = {}
        self._history_store: dict[str, list[StudioCheckpoint]] = defaultdict(list)
        self._execution_counter: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> list[str]:
        caps = ["streaming", "traces"]
        # LangChain runnables often expose .get_graph()
        runnable = self._get_runnable()
        if runnable and hasattr(runnable, "get_graph"):
            caps.append("graph")
        return caps

    # ------------------------------------------------------------------
    # Graph topology
    # ------------------------------------------------------------------

    async def get_graph(self) -> StudioGraphTopology:
        """Extract graph from LangChain runnable if available.

        Many LCEL runnables support ``.get_graph()`` which returns a
        ``Graph`` object with ``.nodes`` and ``.edges``. We serialize
        that into the Studio format.
        """
        runnable = self._get_runnable()

        if runnable and hasattr(runnable, "get_graph"):
            try:
                return self._extract_lcel_graph(runnable)
            except Exception as exc:
                logger.warning(f"LCEL graph extraction failed: {exc}")

        # Fallback: synthesize a chatbot-style graph
        return StudioGraphTopology(
            agent_name=self.agent_name,
            nodes=[
                StudioGraphNode(id="__start__", name="Input", type="start"),
                StudioGraphNode(
                    id="prompt",
                    name="Prompt Template",
                    type="agent",
                    metadata={"description": "Format the input prompt"},
                ),
                StudioGraphNode(
                    id="llm",
                    name="LLM",
                    type="agent",
                    metadata={"description": "Language model inference"},
                ),
                StudioGraphNode(
                    id="output_parser",
                    name="Output Parser",
                    type="agent",
                    metadata={"description": "Parse model response"},
                ),
                StudioGraphNode(id="__end__", name="Output", type="end"),
            ],
            edges=[
                StudioGraphEdge(id="e0", source="__start__", target="prompt"),
                StudioGraphEdge(id="e1", source="prompt", target="llm"),
                StudioGraphEdge(id="e2", source="llm", target="output_parser"),
                StudioGraphEdge(id="e3", source="output_parser", target="__end__"),
            ],
            entry_point="__start__",
            end_points=["__end__"],
            metadata={"mode": "langchain", "framework": "langchain"},
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
        """Stream execution using LangChain's astream_events API.

        Falls back to standard node_fn execution if astream_events
        is not available on the underlying runnable.
        """
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        runnable = self._get_runnable()
        start_time = time.monotonic()

        # Try LangChain's astream_events first
        if runnable and hasattr(runnable, "astream_events"):
            try:
                async for event in self._stream_lc_events(runnable, state, thread_id):
                    yield event
                return
            except Exception as exc:
                logger.warning(f"LangChain astream_events failed: {exc}")

        # Fallback: wrap node_fn with trace events
        yield StudioRunEvent(
            event="node_start",
            run_id="",
            timestamp=_now_iso(),
            node=self.agent_name,
            data={"input": state, "framework": "langchain"},
        )

        try:
            result = await self._agent.node_fn(state) if self._agent.node_fn else {}
            output = result if isinstance(result, dict) else {"response": str(result)}
        except Exception as exc:
            yield StudioRunEvent(
                event="node_end",
                run_id="",
                timestamp=_now_iso(),
                node=self.agent_name,
                data={"error": str(exc)},
                duration_ms=round((time.monotonic() - start_time) * 1000, 2),
            )
            return

        duration = round((time.monotonic() - start_time) * 1000, 2)

        # Record state
        self._state_store[thread_id] = {
            "last_input": state,
            "last_output": output,
            "messages": state.get("messages", []),
            "updated_at": _now_iso(),
        }

        self._record_history(thread_id, state, output, duration)

        yield StudioRunEvent(
            event="trace",
            run_id="",
            timestamp=_now_iso(),
            node=self.agent_name,
            data={
                "level": "info",
                "message": f"LangChain execution completed in {duration}ms",
                "duration_ms": duration,
            },
        )

        yield StudioRunEvent(
            event="node_end",
            run_id="",
            timestamp=_now_iso(),
            node=self.agent_name,
            data={"output": output},
            duration_ms=duration,
        )

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    async def get_state(self, thread_id: str) -> StudioStateSnapshot | None:
        stored = self._state_store.get(thread_id, {})
        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=self.agent_name,
            state=stored,
            timestamp=_now_iso(),
        )

    async def update_state(
        self,
        thread_id: str,
        updates: dict[str, Any],
    ) -> StudioStateSnapshot | None:
        current = self._state_store.get(thread_id, {})
        merged = {**current, **updates}
        self._state_store[thread_id] = merged
        return StudioStateSnapshot(
            thread_id=thread_id,
            agent_name=self.agent_name,
            state=merged,
            timestamp=_now_iso(),
        )

    # ------------------------------------------------------------------
    # Checkpoint history
    # ------------------------------------------------------------------

    async def get_history(self, thread_id: str) -> list[StudioCheckpoint]:
        return list(reversed(self._history_store.get(thread_id, [])))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_runnable(self) -> Any | None:
        """Try to get the underlying LangChain runnable from the agent."""
        # Check if the agent has a runnable attribute
        runnable = getattr(self._agent, "_langchain_runnable", None)
        if runnable:
            return runnable
        # Try to extract from node_fn closure or module
        if self._agent.module_path:
            try:
                import importlib

                mod = importlib.import_module(self._agent.module_path)
                for attr_name in ["chain", "runnable", "llm", "chatbot", "agent"]:
                    obj = getattr(mod, attr_name, None)
                    if obj and hasattr(obj, "invoke"):
                        return obj
            except ImportError:
                pass
        return None

    def _extract_lcel_graph(self, runnable: Any) -> StudioGraphTopology:
        """Extract graph topology from an LCEL runnable's get_graph()."""
        lc_graph = runnable.get_graph()
        nodes: list[StudioGraphNode] = []
        edges: list[StudioGraphEdge] = []

        for node_id, node_data in getattr(lc_graph, "nodes", {}).items():
            name = getattr(node_data, "name", None) or str(node_id)
            node_type = "agent"
            name_lower = name.lower()
            if "start" in name_lower or node_id == "__start__":
                node_type = "start"
            elif "end" in name_lower or node_id == "__end__":
                node_type = "end"
            elif "tool" in name_lower:
                node_type = "tool"
            elif "prompt" in name_lower or "template" in name_lower:
                node_type = "agent"

            nodes.append(
                StudioGraphNode(
                    id=str(node_id),
                    name=name,
                    type=node_type,
                    metadata=getattr(node_data, "metadata", {}) or {},
                )
            )

        for idx, edge in enumerate(getattr(lc_graph, "edges", [])):
            source = getattr(edge, "source", None)
            target = getattr(edge, "target", None)
            if source and target:
                edges.append(
                    StudioGraphEdge(
                        id=f"edge_{idx}",
                        source=str(source),
                        target=str(target),
                    )
                )

        entry_point = None
        end_points = []
        for n in nodes:
            if n.type == "start":
                entry_point = n.id
            elif n.type == "end":
                end_points.append(n.id)

        return StudioGraphTopology(
            agent_name=self.agent_name,
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
            end_points=end_points,
            metadata={"mode": "lcel", "framework": "langchain"},
        )

    async def _stream_lc_events(
        self,
        runnable: Any,
        state: dict[str, Any],
        thread_id: str,
    ) -> AsyncGenerator[StudioRunEvent, None]:
        """Stream events from LangChain's astream_events API."""
        start_time = time.monotonic()
        lc_input = state.get("current_query", state)

        async for lc_event in runnable.astream_events(lc_input, version="v2"):
            event_type = lc_event.get("event", "")
            name = lc_event.get("name", "")
            data = lc_event.get("data", {})

            if event_type == "on_chain_start":
                yield StudioRunEvent(
                    event="node_start",
                    run_id="",
                    timestamp=_now_iso(),
                    node=name,
                    data={"tags": lc_event.get("tags", [])},
                )

            elif event_type == "on_chain_end":
                output = data.get("output", {})
                if not isinstance(output, (dict, list, str, int, float, bool, type(None))):
                    output = str(output)
                yield StudioRunEvent(
                    event="node_end",
                    run_id="",
                    timestamp=_now_iso(),
                    node=name,
                    data={"output": output},
                )

            elif event_type == "on_chat_model_stream":
                chunk = data.get("chunk", {})
                content = ""
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict):
                    content = chunk.get("content", "")
                if content:
                    yield StudioRunEvent(
                        event="message_chunk",
                        run_id="",
                        timestamp=_now_iso(),
                        node=name,
                        data={"content": content},
                    )

            elif event_type == "on_llm_start":
                yield StudioRunEvent(
                    event="node_start",
                    run_id="",
                    timestamp=_now_iso(),
                    node=f"llm:{name}",
                    data={"model": lc_event.get("tags", [])},
                )

            elif event_type == "on_llm_end":
                yield StudioRunEvent(
                    event="node_end",
                    run_id="",
                    timestamp=_now_iso(),
                    node=f"llm:{name}",
                    data={
                        "token_usage": data.get("output", {})
                        .get("llm_output", {})
                        .get("token_usage", {}),
                    },
                )

            elif event_type == "on_tool_start":
                yield StudioRunEvent(
                    event="node_start",
                    run_id="",
                    timestamp=_now_iso(),
                    node=f"tool:{name}",
                    data={"tool_input": data.get("input", {})},
                )

            elif event_type == "on_tool_end":
                yield StudioRunEvent(
                    event="node_end",
                    run_id="",
                    timestamp=_now_iso(),
                    node=f"tool:{name}",
                    data={"tool_output": str(data.get("output", ""))},
                )

        duration = round((time.monotonic() - start_time) * 1000, 2)
        self._record_history(thread_id, state, {}, duration)

    def _record_history(
        self,
        thread_id: str,
        state: dict[str, Any],
        output: dict[str, Any],
        duration: float,
    ) -> None:
        """Record an execution in the history store."""
        self._execution_counter[thread_id] += 1
        step = self._execution_counter[thread_id]
        self._history_store[thread_id].append(
            StudioCheckpoint(
                id=f"lc_trace_{thread_id}_{step}",
                thread_id=thread_id,
                step=step,
                state={"input": state, "output": output},
                metadata={"duration_ms": duration, "framework": "langchain"},
                parent_id=f"lc_trace_{thread_id}_{step - 1}" if step > 1 else None,
                timestamp=_now_iso(),
            )
        )
