"""Per-agent connections for Agentomatic.

Declare databases and authenticated HTTP services once (per agent or
platform-wide) and access live, initialised connections anywhere with
:func:`get_connections`.
"""

from __future__ import annotations

from agentomatic.connections.custom import CustomConnection
from agentomatic.connections.database import DatabaseConnection
from agentomatic.connections.http import HttpConnection
from agentomatic.connections.manager import (
    PLATFORM_SCOPE,
    ConnectionManager,
    all_managers,
    get_connections,
    initialize_connections,
    register_connection_type,
    register_connections,
    reset_connections,
)
from agentomatic.connections.models import (
    ConnectionConfig,
    ConnectionKind,
    ConnectionPurpose,
    CustomConnectionConfig,
    DatabaseConnectionConfig,
    HttpConnectionConfig,
    VectorConnectionConfig,
)
from agentomatic.connections.stores import (
    create_store_from_connection,
    register_store_provider,
    registered_store_providers,
)
from agentomatic.connections.vector import (
    VectorConnection,
    VectorStore,
    register_vector_provider,
    register_vector_store_adapter,
    registered_vector_providers,
)

__all__ = [
    "PLATFORM_SCOPE",
    "ConnectionConfig",
    "ConnectionKind",
    "ConnectionManager",
    "ConnectionPurpose",
    "CustomConnection",
    "CustomConnectionConfig",
    "DatabaseConnection",
    "DatabaseConnectionConfig",
    "HttpConnection",
    "HttpConnectionConfig",
    "VectorConnection",
    "VectorConnectionConfig",
    "VectorStore",
    "all_managers",
    "create_store_from_connection",
    "get_connections",
    "initialize_connections",
    "register_connection_type",
    "register_connections",
    "register_store_provider",
    "register_vector_provider",
    "register_vector_store_adapter",
    "registered_store_providers",
    "registered_vector_providers",
    "reset_connections",
]
