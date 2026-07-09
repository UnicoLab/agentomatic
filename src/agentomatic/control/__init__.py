"""Production control plane for Agentomatic.

Provides an admin REST API (``{api_prefix}/control``) and request-gating
middleware to observe and operate the platform at runtime.
"""

from __future__ import annotations

from agentomatic.control.middleware import MaintenanceMiddleware
from agentomatic.control.router import create_control_router
from agentomatic.control.state import ControlPlaneState

__all__ = [
    "ControlPlaneState",
    "MaintenanceMiddleware",
    "create_control_router",
]
