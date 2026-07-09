"""Configuration models for per-agent connections.

A *connection* is a reusable, authenticated resource an agent (or the
platform) needs at runtime — most commonly a database or an external HTTP
service.  Connections are declared once and resolved lazily so credentials
never live in code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentomatic.endpoints.models import UpstreamAuthConfig


class ConnectionKind(StrEnum):
    """Supported connection kinds."""

    DATABASE = "database"
    HTTP = "http"
    VECTOR = "vector"
    CUSTOM = "custom"


class ConnectionPurpose(StrEnum):
    """The intended usage of a connection.

    Purpose is orthogonal to :class:`ConnectionKind`: the same kind of
    backend (e.g. a Postgres database) can serve different purposes
    (conversation ``MEMORY``, ``RAG`` document storage, ``ANALYTICS``…).
    Agent code can look connections up by purpose via
    :meth:`~agentomatic.connections.manager.ConnectionManager.for_purpose`.
    """

    GENERAL = "general"
    MEMORY = "memory"
    RAG = "rag"
    VECTOR = "vector"
    CACHE = "cache"
    DOCUMENTS = "documents"
    ANALYTICS = "analytics"


class DatabaseConnectionConfig(BaseModel):
    """Configuration for a database connection.

    The ``url`` is a SQLAlchemy async URL (e.g.
    ``postgresql+asyncpg://user:pass@host/db``).  Any string field supports
    ``${ENV}`` interpolation so secrets can be injected from the
    environment.  Alternatively provide ``username``/``password`` and they
    will be spliced into the URL at connect time.
    """

    kind: ConnectionKind = ConnectionKind.DATABASE
    name: str = Field(..., min_length=1, description="Logical connection name.")
    purpose: ConnectionPurpose = Field(
        ConnectionPurpose.GENERAL,
        description="What the connection is used for (memory, rag, analytics…).",
    )
    url: str = Field(..., description="SQLAlchemy async URL (supports ${ENV}).")
    username: str = Field("", description="Optional username (supports ${ENV}).")
    password: str = Field("", description="Optional password (supports ${ENV}).")
    pool_size: int = Field(5, ge=1)
    max_overflow: int = Field(10, ge=0)
    pool_timeout: int = Field(30, ge=1)
    pool_pre_ping: bool = Field(True, description="Check connections before use.")
    echo: bool = Field(False, description="Echo SQL statements to the log.")
    connect_args: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HttpConnectionConfig(BaseModel):
    """Configuration for an authenticated HTTP service connection."""

    kind: ConnectionKind = ConnectionKind.HTTP
    name: str = Field(..., min_length=1, description="Logical connection name.")
    purpose: ConnectionPurpose = Field(
        ConnectionPurpose.GENERAL,
        description="What the connection is used for (rag, general…).",
    )
    base_url: str = Field(..., description="Base URL of the service (supports ${ENV}).")
    headers: dict[str, str] = Field(default_factory=dict)
    auth: UpstreamAuthConfig = Field(default_factory=UpstreamAuthConfig)
    timeout: float = Field(30.0, gt=0)
    max_retries: int = Field(2, ge=0, le=10)
    verify_ssl: bool = Field(True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorConnectionConfig(BaseModel):
    """Configuration for a vector-store connection (RAG / vector search).

    Provider-agnostic: pick a ``provider`` (``qdrant``, ``chroma``,
    ``weaviate``, ``pinecone``, ``milvus`` or a custom name registered with
    :func:`~agentomatic.connections.vector.register_vector_provider`) and the
    matching client is built lazily.  Every string field supports ``${ENV}``
    interpolation so endpoints and API keys never live in code.

    Note:
        For ``pgvector`` use a :class:`DatabaseConnectionConfig` with
        ``purpose=ConnectionPurpose.VECTOR`` instead — the vector index lives
        inside your Postgres database.
    """

    kind: ConnectionKind = ConnectionKind.VECTOR
    name: str = Field(..., min_length=1, description="Logical connection name.")
    purpose: ConnectionPurpose = Field(
        ConnectionPurpose.VECTOR,
        description="Usage (defaults to vector; may be rag/documents).",
    )
    provider: str = Field(
        ...,
        min_length=1,
        description="Vector backend: qdrant | chroma | weaviate | pinecone | milvus | <custom>.",
    )
    url: str = Field("", description="Server URL / host (supports ${ENV}).")
    api_key: str = Field("", description="Optional API key (supports ${ENV}).")
    collection: str = Field("", description="Default collection / index / class name.")
    dimension: int | None = Field(None, ge=1, description="Embedding dimension, if fixed.")
    distance: str = Field("cosine", description="Distance metric (cosine, euclid, dot…).")
    namespace: str = Field("", description="Optional namespace / tenant.")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to the provider client.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class CustomConnectionConfig(BaseModel):
    """Configuration for *any* backend via a factory — no classes required.

    This is the zero-boilerplate escape hatch: point ``factory`` at any
    callable (or dotted import path) that returns a client, and Agentomatic
    manages its lifecycle generically.  Connect to redis, mongo, elasticsearch,
    neo4j, dynamodb, an in-house SDK — anything — with a single declaration::

        CustomConnectionConfig(
            name="cache",
            factory="redis.asyncio.from_url",   # dotted path or callable
            args=["${REDIS_URL}"],
            purpose=ConnectionPurpose.CACHE,
        )

    All strings in ``args`` / ``kwargs`` (and ``factory`` when it is a path)
    support ``${ENV}`` interpolation.  Lifecycle methods are auto-detected
    (``aclose`` / ``close`` / ``disconnect`` for shutdown) unless overridden.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: ConnectionKind = ConnectionKind.CUSTOM
    name: str = Field(..., min_length=1, description="Logical connection name.")
    purpose: ConnectionPurpose = Field(
        ConnectionPurpose.GENERAL,
        description="What the connection is used for.",
    )
    factory: Any = Field(
        ...,
        description="Callable or dotted path (e.g. 'pkg.mod:func') returning a client.",
    )
    args: list[Any] = Field(
        default_factory=list,
        description="Positional args passed to the factory (strings support ${ENV}).",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword args passed to the factory (strings support ${ENV}).",
    )
    close_method: str = Field(
        "",
        description="Client method to call on shutdown (auto-detected when empty).",
    )
    health_method: str = Field(
        "",
        description="Optional client method to call for health checks.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


ConnectionConfig = (
    DatabaseConnectionConfig
    | HttpConnectionConfig
    | VectorConnectionConfig
    | CustomConnectionConfig
)
