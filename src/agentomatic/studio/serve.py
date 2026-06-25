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

# ── Runtime FastAPI imports ──────────────────────────────────────────
# These MUST live at module level so that ``from __future__ import
# annotations`` (which turns every annotation into a lazy string) can
# still be resolved by FastAPI's dependency-injection machinery.
# When they were imported *locally* inside ``mount_studio_ui()``,
# FastAPI could not find ``Request`` in the module globals and fell
# back to treating it as a required query parameter → 422.
from fastapi import Request  # noqa: E402
from fastapi.responses import (  # noqa: E402
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles  # noqa: E402

# Directory containing the built React app (index.html, static/js, static/css)
STATIC_DIR = Path(__file__).parent / "static"

# ── Informative error page ───────────────────────────────────────────
_ASSETS_MISSING_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Agentomatic Studio — Assets Not Found</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f0a1a; color: #e2e8f0;
      display: flex; justify-content: center; align-items: center;
      min-height: 100vh; margin: 0;
    }
    .card {
      background: rgba(124,58,237,.08); border: 1px solid rgba(124,58,237,.3);
      border-radius: 16px; padding: 48px; max-width: 560px; text-align: center;
    }
    h1 { color: #7c3aed; font-size: 1.6rem; margin: 0 0 12px; }
    p  { line-height: 1.6; color: #94a3b8; }
    code {
      background: rgba(124,58,237,.15); padding: 2px 8px;
      border-radius: 4px; font-size: .9em; color: #c4b5fd;
    }
    pre {
      background: rgba(0,0,0,.4); padding: 16px; border-radius: 8px;
      text-align: left; overflow-x: auto; font-size: .85em; color: #a5b4fc;
    }
    .hint { margin-top: 24px; font-size: .85em; color: #64748b; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎨 Studio UI Assets Not Found</h1>
    <p>The Agentomatic Studio API is running, but the built
    React UI assets are missing from the package.</p>
    <p><strong>To fix:</strong></p>
    <pre>cd agentomatic-studio && npm ci && npm run build
# Then copy the build output:
./scripts/build_studio.sh</pre>
    <p>Or reinstall with the latest version:</p>
    <pre>pip install --upgrade agentomatic</pre>
    <p class="hint">The Studio API endpoints at
    <code>/studio/info</code> and <code>/studio/agents</code>
    are still available.</p>
  </div>
</body>
</html>
"""

_STUDIO_DISABLED_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Agentomatic Studio — Disabled</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f0a1a; color: #e2e8f0;
      display: flex; justify-content: center; align-items: center;
      min-height: 100vh; margin: 0;
    }
    .card {
      background: rgba(124,58,237,.08); border: 1px solid rgba(124,58,237,.3);
      border-radius: 16px; padding: 48px; max-width: 560px; text-align: center;
    }
    h1 { color: #7c3aed; font-size: 1.6rem; margin: 0 0 12px; }
    p  { line-height: 1.6; color: #94a3b8; }
    code {
      background: rgba(124,58,237,.15); padding: 2px 8px;
      border-radius: 4px; font-size: .9em; color: #c4b5fd;
    }
    pre {
      background: rgba(0,0,0,.4); padding: 16px; border-radius: 8px;
      text-align: left; overflow-x: auto; font-size: .85em; color: #a5b4fc;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎨 Studio Is Disabled</h1>
    <p>The Agentomatic Studio debug UI is not enabled on
    this platform instance.</p>
    <p><strong>To enable:</strong></p>
    <pre># CLI:
agentomatic run --studio

# Python:
platform = AgentPlatform.from_folder(
    "agents/", enable_studio=True
)</pre>
  </div>
</body>
</html>
"""


def is_studio_available() -> bool:
    """Check if the built Studio UI assets are available."""
    return (STATIC_DIR / "index.html").exists()


def mount_studio_ui(app: FastAPI, path_prefix: str = "/studio/ui") -> None:
    """Mount the Studio UI static files on a FastAPI application.

    The Studio is a React single-page application (SPA).  We serve:
      - ``/studio/ui/``  → ``index.html``
      - ``/studio/ui/static/...`` → JS / CSS / media bundles
      - Any non-file path under ``/studio/ui/`` → ``index.html`` (SPA fallback)

    If the built assets are not found, a helpful error page is served
    instead, guiding the user to build or reinstall the package.

    Args:
        app: The FastAPI application instance.
        path_prefix: URL path prefix for the Studio UI.
    """
    # Normalise prefix
    prefix = path_prefix.rstrip("/")

    if not is_studio_available():
        logger.warning(
            "Studio UI assets not found at %s. "
            "Build the frontend first: cd agentomatic-studio && npm run build, "
            "then copy build/ contents to src/agentomatic/studio/static/",
            STATIC_DIR,
        )

        # Still mount a helpful error page so the user doesn't get
        # an opaque 404.
        @app.get(f"{prefix}/{{filename:path}}", response_model=None)
        async def studio_assets_missing(
            request: Request,
            filename: str,
        ) -> HTMLResponse:
            """Serve a helpful error page when UI assets are missing."""
            return HTMLResponse(
                content=_ASSETS_MISSING_HTML,
                status_code=503,
            )

        @app.get(prefix, include_in_schema=False)
        async def studio_missing_redirect() -> RedirectResponse:
            """Redirect bare path to trailing-slash version."""
            return RedirectResponse(url=f"{prefix}/")

        logger.info("🎨 Studio error page mounted at %s/ (assets missing)", prefix)
        return

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
    @app.get(f"{prefix}/{{filename:path}}", response_model=None)
    async def studio_spa(
        request: Request,
        filename: str,
    ) -> FileResponse | HTMLResponse:
        """Serve Studio UI files or fallback to index.html for SPA routing."""
        # Try to serve the exact file
        file_path = STATIC_DIR / filename
        if filename and file_path.is_file() and file_path.resolve().is_relative_to(STATIC_DIR):
            return FileResponse(str(file_path))
        # SPA fallback: serve index.html for any non-file path
        return FileResponse(str(STATIC_DIR / "index.html"))

    # Redirect /studio/ui to /studio/ui/ for consistency
    @app.get(prefix, include_in_schema=False)
    async def studio_root_redirect() -> RedirectResponse:
        """Redirect bare path to trailing-slash version."""
        return RedirectResponse(url=f"{prefix}/")

    logger.info("🎨 Studio UI mounted at %s/", prefix)


def mount_studio_disabled_page(
    app: FastAPI,
    path_prefix: str = "/studio/ui",
) -> None:
    """Mount informative error pages when Studio is explicitly disabled.

    Called by the platform when ``enable_studio=False`` to ensure
    users hitting ``/studio/ui/`` get a clear message instead of a
    generic 404.

    Args:
        app: The FastAPI application instance.
        path_prefix: URL path prefix for the Studio UI.
    """
    prefix = path_prefix.rstrip("/")

    @app.get(f"{prefix}/{{filename:path}}", response_model=None)
    async def studio_disabled(
        request: Request,
        filename: str,
    ) -> HTMLResponse:
        """Serve a helpful page when Studio is disabled."""
        return HTMLResponse(
            content=_STUDIO_DISABLED_HTML,
            status_code=503,
        )

    @app.get(prefix, include_in_schema=False)
    async def studio_disabled_redirect() -> RedirectResponse:
        """Redirect bare path to trailing-slash version."""
        return RedirectResponse(url=f"{prefix}/")

    # Also add a JSON response at the API path for programmatic callers
    @app.get("/studio/info", include_in_schema=False)
    async def studio_info_disabled() -> JSONResponse:
        """Return a JSON error when Studio API is disabled."""
        return JSONResponse(
            status_code=503,
            content={
                "error": "studio_disabled",
                "message": (
                    "Studio is disabled. Run with --studio flag or set enable_studio=True."
                ),
            },
        )

    logger.debug("Studio disabled pages mounted at %s/", prefix)
