"""Built-in debug UI powered by Chainlit.

Automatically mounts at ``/chat`` when Chainlit is installed.
Install: ``pip install agentomatic[ui]``
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from fastapi import FastAPI

_CHAT_MODULE = str(Path(__file__).parent / "chat.py")


def is_available() -> bool:
    """Check if Chainlit is installed."""
    try:
        import chainlit  # noqa: F401
        return True
    except ImportError:
        return False


def mount(app: FastAPI, path: str = "/chat") -> None:
    """Mount Chainlit debug UI into a FastAPI app.

    Args:
        app: The FastAPI application.
        path: URL path to mount at (default ``/chat``).
    """
    if not is_available():
        logger.warning(
            "⚠️  Chainlit not installed — debug UI disabled. "
            "Install with: pip install agentomatic[ui]"
        )
        return

    try:
        from chainlit.utils import mount_chainlit
        mount_chainlit(app=app, target=_CHAT_MODULE, path=path)
        logger.info(f"🎨 Debug UI mounted at {path}")
    except Exception as exc:
        logger.warning(f"⚠️  Failed to mount debug UI: {exc}")
