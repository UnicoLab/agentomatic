"""Serve the Agentomatic Studio frontend as static files.

When the studio frontend is built (``npm run build`` in agentomatic-studio),
the resulting static files are copied into ``studio/static/`` alongside this
module.  This module mounts them on a FastAPI app so the Studio UI is served
directly from the agentomatic server.

Usage::

    from agentomatic.studio.serve import mount_studio_ui

    # Inside platform.build():
    mount_studio_ui(app)
    # Studio UI available at http://host:port/studio/ui/
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from fastapi import FastAPI

# Directory containing the built React app (index.html, static/js, static/css)
STATIC_DIR = Path(__file__).parent / "static"


def is_studio_available() -> bool:
    """Check if the built Studio UI assets are available."""
    return (STATIC_DIR / "index.html").exists()


def mount_studio_ui(app: FastAPI, path_prefix: str = "/studio/ui") -> None:
    """Mount the Studio UI static files on a FastAPI application.

    The Studio is a React single-page application (SPA).  We serve:
      - ``/studio/ui/``  → ``index.html``
      - ``/studio/ui/static/...`` → JS / CSS / media bundles
      - Any non-file path under ``/studio/ui/`` → ``index.html`` (SPA fallback)

    Args:
        app: The FastAPI application instance.
        path_prefix: URL path prefix for the Studio UI.
    """
    if not is_studio_available():
        logger.warning(
            "Studio UI assets not found. "
            "Build the frontend first: cd agentomatic-studio && npm run build, "
            "then copy build/ contents to src/agentomatic/studio/static/"
        )
        return

    from fastapi import Request
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    # Normalise prefix
    prefix = path_prefix.rstrip("/")

    # Mount the static assets (JS, CSS, media) — must come first so
    # /studio/ui/static/js/main.xxx.js is served as a file, not routed
    # to the SPA fallback.
    static_subdir = STATIC_DIR / "static"
    if static_subdir.exists():
        app.mount(
            f"{prefix}/static",
            StaticFiles(directory=str(static_subdir)),
            name="studio-static-assets",
        )

    # Serve root-level assets (favicon, manifest, robots.txt, etc.)
    @app.get(f"{prefix}/{{filename:path}}")
    async def studio_spa(request: Request, filename: str) -> FileResponse | HTMLResponse:
        """Serve Studio UI files or fallback to index.html for SPA routing."""
        # Try to serve the exact file
        file_path = STATIC_DIR / filename
        if filename and file_path.is_file() and file_path.resolve().is_relative_to(STATIC_DIR):
            return FileResponse(str(file_path))
        # SPA fallback: serve index.html for any non-file path
        return FileResponse(str(STATIC_DIR / "index.html"))

    # Redirect /studio/ui to /studio/ui/ for consistency
    @app.get(prefix, include_in_schema=False)
    async def studio_root_redirect() -> HTMLResponse:
        """Redirect bare path to trailing-slash version."""
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url=f"{prefix}/")  # type: ignore[return-value]

    logger.info(f"🎨 Studio UI mounted at {prefix}/")
