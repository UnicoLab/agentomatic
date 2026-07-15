"""Module-level ASGI app factory for uvicorn ``--reload`` / multi-worker runs.

Modern uvicorn requires an *import string* (not a pre-built app instance) when
``reload`` or ``workers > 1`` are used, because it re-imports the application
inside each worker subprocess (passing an instance makes uvicorn ``sys.exit(1)``).

:meth:`agentomatic.core.platform.AgentPlatform.run` serialises the platform's
reconstructable configuration into the ``AGENTOMATIC_FACTORY_CONFIG`` environment
variable and points uvicorn at :func:`create_app` here (``factory=True``). The
child process reads that config and rebuilds an equivalent folder-based platform.

Programmatically-configured platforms (custom stores, middleware, lifecycle
hooks, or agents registered in code) cannot be reconstructed this way; ``run``
detects that and degrades to a single in-process instance instead of using the
factory.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

FACTORY_CONFIG_ENV = "AGENTOMATIC_FACTORY_CONFIG"


def create_app() -> FastAPI:
    """Build a FastAPI app from the serialised platform config in the env.

    Returns:
        The FastAPI application produced by ``AgentPlatform.build()``.

    Raises:
        RuntimeError: If ``AGENTOMATIC_FACTORY_CONFIG`` is not set (i.e. the
            factory was invoked outside of ``AgentPlatform.run``).
    """
    from agentomatic.core.platform import AgentPlatform

    raw = os.environ.get(FACTORY_CONFIG_ENV)
    if not raw:
        raise RuntimeError(
            f"{FACTORY_CONFIG_ENV} is not set; agentomatic._runtime:create_app "
            "is only usable via AgentPlatform.run(reload=... / workers>1), "
            "which populates it before starting uvicorn."
        )
    config: dict[str, Any] = json.loads(raw)
    agents_dir = config.pop("agents_dir")
    platform = AgentPlatform.from_folder(agents_dir, **config)
    return platform.build()
