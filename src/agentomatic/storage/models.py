"""SQLAlchemy ORM models for persistent thread/message storage."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class ThreadModel(Base):
    """Conversation thread."""

    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: f"thread_{uuid.uuid4().hex[:12]}"
    )
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_name: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    parent_thread_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("threads.id", ondelete="SET NULL"), nullable=True, index=True
    )
    fork_message_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    messages: Mapped[list[MessageModel]] = relationship(
        "MessageModel",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="MessageModel.created_at",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "title": self.title,
            "metadata": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.message_count,
            "parent_thread_id": self.parent_thread_id,
            "fork_message_index": self.fork_message_index,
        }


class MessageModel(Base):
    """Chat message within a thread."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("threads.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(32))  # user, assistant, system, tool
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    thread: Mapped[ThreadModel] = relationship("ThreadModel", back_populates="messages")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata_json or {},
            "timestamp": self.created_at.isoformat() if self.created_at else None,
        }


class FeedbackModel(Base):
    """User feedback on agent responses."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("threads.id", ondelete="CASCADE"), index=True
    )
    message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_name: Mapped[str] = mapped_column(String(128))
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_type: Mapped[str] = mapped_column(
        String(32), default="thumbs"
    )  # thumbs, rating, text
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "message_id": self.message_id,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "rating": self.rating,
            "comment": self.comment,
            "feedback_type": self.feedback_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SuspendedStateModel(Base):
    """Execution state suspended for Human-in-the-Loop approval."""

    __tablename__ = "suspended_states"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # approval_id
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("threads.id", ondelete="CASCADE"), index=True
    )
    agent_name: Mapped[str] = mapped_column(String(128))
    node_name: Mapped[str] = mapped_column(String(128))
    state_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        default=lambda: datetime.now(UTC) + timedelta(days=7),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "agent_name": self.agent_name,
            "node_name": self.node_name,
            "state_snapshot": self.state_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class CheckpointModel(Base):
    """LangGraph execution checkpoint for thread state persistence."""

    __tablename__ = "checkpoints"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(128), primary_key=True, default="")
    checkpoint_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_json: Mapped[dict] = mapped_column(JSON)
    metadata_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "checkpoint_ns": self.checkpoint_ns,
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "checkpoint": self.checkpoint_json,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AgentInvocationLogModel(Base):
    """Invocation call history for agents/plugins/pipelines/etc."""

    __tablename__ = "agent_invocation_logs"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"invlog_{uuid.uuid4().hex[:16]}",
    )
    # resource_name lives in agent_name for backward compatibility.
    agent_name: Mapped[str] = mapped_column(String(128), index=True)
    resource_type: Mapped[str] = mapped_column(
        String(32),
        default="agent",
        index=True,
    )  # agent|plugin|pipeline|ingestion|endpoint
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )
    endpoint: Mapped[str] = mapped_column(
        String(64), default="invoke"
    )  # invoke|chat|stream|predict|run|pipeline_step|…
    input_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="ok", index=True)  # ok|error|suspended

    def to_dict(self) -> dict[str, Any]:
        rtype = self.resource_type or "agent"
        return {
            "id": self.id,
            "resource_type": rtype,
            "resource_name": self.agent_name,
            "agent_name": self.agent_name,  # BC alias
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "endpoint": self.endpoint,
            "input": self.input_json or {},
            "output": self.output_json or {},
            "metadata": self.metadata_json or {},
            "error": self.error,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }


class LogAnalysisModel(Base):
    """LLM-produced analysis of recent invocation logs."""

    __tablename__ = "log_analyses"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"logan_{uuid.uuid4().hex[:16]}",
    )
    agent_name: Mapped[str] = mapped_column(String(128), index=True)
    resource_type: Mapped[str] = mapped_column(
        String(32),
        default="agent",
        index=True,
    )
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(64), default="unknown")
    recommendations: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )

    def to_dict(self) -> dict[str, Any]:
        rtype = self.resource_type or "agent"
        return {
            "id": self.id,
            "resource_type": rtype,
            "resource_name": self.agent_name,
            "agent_name": self.agent_name,
            "score": self.score,
            "summary": self.summary,
            "status": self.status,
            "recommendations": self.recommendations or [],
            "metadata": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class OptimizationRunModel(Base):
    """Auditable prompt/fit retrain history (experiment artefacts)."""

    __tablename__ = "optimization_runs"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"optrun_{uuid.uuid4().hex[:16]}",
    )
    experiment_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_name: Mapped[str] = mapped_column(String(128), index=True)
    baseline_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_versions: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    score_history: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    learnings: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)
    artefacts: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "agent_name": self.agent_name,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "prompt_versions": self.prompt_versions or {},
            "score_history": self.score_history or [],
            "learnings": self.learnings or [],
            "artefacts": self.artefacts or {},
            "metadata": self.metadata_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
