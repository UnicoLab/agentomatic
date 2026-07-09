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
from agentomatic.connections.vector import (
    VectorConnection,
    register_vector_provider,
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
    "all_managers",
    "get_connections",
    "register_connection_type",
    "register_connections",
    "register_vector_provider",
    "registered_vector_providers",
    "reset_connections",
]
