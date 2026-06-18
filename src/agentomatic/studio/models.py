"""Pydantic models for the Agentomatic Studio debug API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Server / Discovery
# ---------------------------------------------------------------------------


class StudioServerInfo(BaseModel):
    """Platform-level information returned by ``GET /studio/info``."""

    version: str = Field(..., description="Platform version string")
    platform_title: str = Field(..., description="Human-readable platform title")
    agent_count: int = Field(..., description="Number of registered agents")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Platform capabilities (e.g. 'studio', 'streaming', 'storage')",
    )


class StudioAgentInfo(BaseModel):
    """Agent summary with debugging capabilities."""

    name: str = Field(..., description="Agent machine name")
    slug: str = Field(..., description="Unique agent slug")
    description: str = Field("", description="Human-readable description")
    version: str = Field("1.0.0", description="Agent version")
    framework: str = Field(
        "langgraph", description="Framework: 'langgraph' | 'langchain' | 'custom'"
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Debug capabilities: 'graph', 'streaming', 'checkpoints', 'hitl', etc.",
    )
    has_graph: bool = Field(False, description="Whether the agent has a compiled graph")
    has_config: bool = Field(False, description="Whether the agent has a config object")
    has_prompts: bool = Field(False, description="Whether the agent has prompt versions")


# ---------------------------------------------------------------------------
# Graph Topology
# ---------------------------------------------------------------------------


class StudioGraphNode(BaseModel):
    """A single node in the agent's execution graph."""

    id: str = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Human-readable node name")
    type: str = Field(
        "default",
        description="Node type: 'start', 'end', 'agent', 'tool', 'condition', 'human', 'default'",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudioGraphEdge(BaseModel):
    """A directed edge between two graph nodes."""

    id: str = Field(..., description="Unique edge identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    condition: str | None = Field(None, description="Condition label for conditional edges")
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudioGraphTopology(BaseModel):
    """Full graph topology of an agent's execution flow."""

    agent_name: str = Field(..., description="Agent this topology belongs to")
    nodes: list[StudioGraphNode] = Field(default_factory=list)
    edges: list[StudioGraphEdge] = Field(default_factory=list)
    entry_point: str | None = Field(None, description="ID of the entry node")
    end_points: list[str] = Field(default_factory=list, description="IDs of terminal nodes")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent Schemas
# ---------------------------------------------------------------------------


class StudioAgentSchemas(BaseModel):
    """JSON Schema definitions for agent input and output models."""

    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for input")
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for output"
    )


# ---------------------------------------------------------------------------
# Run Models
# ---------------------------------------------------------------------------


class StudioRunRequest(BaseModel):
    """Request to execute an agent run with optional debugging features."""

    query: str = Field(..., description="User query or input text")
    user_id: str = Field("default-user", description="User identifier")
    thread_id: str | None = Field(None, description="Thread ID for conversation continuity")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")
    prompt_version: str = Field("v1", description="Prompt version to use")
    breakpoints: list[str] = Field(
        default_factory=list,
        description="Node names to break at",
    )
    checkpoint_id: str | None = Field(
        None,
        description="Optional checkpoint ID to resume/replay from",
    )


class StudioRunEvent(BaseModel):
    """A single event emitted during agent execution.

    Streamed as SSE frames via the ``/runs/stream`` endpoint.
    """

    event: str = Field(
        ...,
        description=(
            "Event type: 'run_start', 'node_start', 'node_end', "
            "'state_update', 'message_chunk', 'run_complete', 'run_error', 'breakpoint_hit'"
        ),
    )
    run_id: str = Field(..., description="Run this event belongs to")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    node: str | None = Field(None, description="Node name (if event is node-scoped)")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload")
    duration_ms: float | None = Field(None, description="Duration in milliseconds")


class StudioRunInfo(BaseModel):
    """Full run record including all emitted events."""

    id: str = Field(..., description="Unique run identifier")
    agent_name: str = Field(..., description="Agent that was executed")
    thread_id: str | None = Field(None, description="Conversation thread ID")
    status: str = Field(
        "pending",
        description="Run status: 'pending', 'running', 'completed', 'failed', 'cancelled', 'paused'",
    )
    created_at: str = Field(..., description="ISO-8601 creation timestamp")
    completed_at: str | None = Field(None, description="ISO-8601 completion timestamp")
    duration_ms: float | None = Field(None, description="Total duration in milliseconds")
    events: list[StudioRunEvent] = Field(default_factory=list)
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(None, description="Error message if status is 'failed'")


# ---------------------------------------------------------------------------
# Breakpoints (reserved for future HITL debugging)
# ---------------------------------------------------------------------------


class StudioBreakpoint(BaseModel):
    """A breakpoint definition on a graph node."""

    node: str = Field(..., description="Node name to break at")
    condition: str | None = Field(None, description="Optional condition expression")
    enabled: bool = Field(True, description="Whether the breakpoint is active")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class StudioStateSnapshot(BaseModel):
    """Point-in-time snapshot of an agent's thread state."""

    thread_id: str = Field(..., description="Thread this state belongs to")
    agent_name: str = Field(..., description="Agent name")
    state: dict[str, Any] = Field(default_factory=dict, description="Full state dict")
    timestamp: str = Field(..., description="ISO-8601 snapshot timestamp")
    checkpoint_id: str | None = Field(None, description="Checkpoint ID if backed by storage")


class StudioStateUpdate(BaseModel):
    """Partial state patch to apply to a thread."""

    updates: dict[str, Any] = Field(..., description="Key-value pairs to merge into state")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class StudioCheckpoint(BaseModel):
    """A checkpoint in the agent's execution history."""

    id: str = Field(..., description="Checkpoint identifier")
    thread_id: str = Field(..., description="Thread this checkpoint belongs to")
    step: int = Field(0, description="Step number in the execution")
    state: dict[str, Any] = Field(default_factory=dict, description="Checkpoint state")
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = Field(None, description="Parent checkpoint ID")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
