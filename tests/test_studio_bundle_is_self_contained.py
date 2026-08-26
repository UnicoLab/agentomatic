# pyright: reportMissingParameterType=none
"""The bundled Studio UI must not reach outside the deployment at page load.

Studio is a self-hosted admin interface. Anything it fetches from a third
party fails in air-gapped or egress-restricted deployments — where the request
hangs or resets before the page paints — and leaks every viewer's IP and
User-Agent to that third party, which is a compliance question for enterprise
operators.

Regression: the stylesheet opened with two Google Fonts ``@import`` rules, and
no icon was declared at all, so every browser also requested ``/favicon.ico``
from the origin root — a path the platform does not serve.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "src" / "agentomatic" / "studio" / "static"
ASSETS = STATIC / "assets"

#: Hosts a self-hosted UI must never contact just to render.
FORBIDDEN_HOSTS = (
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdn.jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "ajax.googleapis.com",
    "google-analytics.com",
    "googletagmanager.com",
)


def _served_assets() -> list[Path]:
    """Return the CSS and HTML the browser parses on load."""
    return [
        *sorted(ASSETS.glob("*.css")),
        STATIC / "index.html",
    ]


def test_the_bundle_is_present() -> None:
    """Guard against the tests silently passing on a missing bundle."""
    assert (STATIC / "index.html").is_file()
    assert list(ASSETS.glob("*.css"))
    assert list(ASSETS.glob("*.js"))


@pytest.mark.parametrize("host", FORBIDDEN_HOSTS)
def test_no_render_blocking_third_party_host(host: str) -> None:
    """CSS and HTML must not reference an external host."""
    for asset in _served_assets():
        content = asset.read_text(encoding="utf-8", errors="ignore")

        assert host not in content, f"{asset.name} references {host}"


def test_stylesheets_have_no_external_imports() -> None:
    """``@import`` is render-blocking, so an external one is the worst case."""
    for css in sorted(ASSETS.glob("*.css")):
        content = css.read_text(encoding="utf-8", errors="ignore")

        assert "@import url(http" not in content.replace(" ", ""), (
            f"{css.name} still imports a stylesheet over the network"
        )


def test_an_icon_is_declared() -> None:
    """Otherwise every browser requests /favicon.ico, which 404s."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'rel="icon"' in html or "rel='icon'" in html


def test_the_icon_does_not_cost_a_request() -> None:
    """An inline icon keeps the page to a single round trip."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    start = html.index('rel="icon"')
    href = html[start : start + 400]

    assert "data:image/" in href, "the declared icon should be inline"


def test_embedded_asset_urls_stay_beneath_the_studio_mount() -> None:
    """Auth protects root paths, so packaged assets must stay under /studio/ui/.

    The standalone Studio image is intentionally built with a root base, while
    the Python package is built with ``VITE_BASE_URL=/studio/ui/``.  An
    accidental root-relative ``/assets/...`` reference loads the HTML but
    leaves an authenticated embedded deployment with a blank React root.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    asset_urls = re.findall(r'(?:src|href)="([^"]*assets/[^"]+)"', html)

    assert asset_urls, "the packaged page should reference at least one JS/CSS asset"
    for url in asset_urls:
        assert url.startswith("/studio/ui/assets/"), (
            f"embedded Studio asset escapes its mounted path: {url}"
        )
        relative_path = url.removeprefix("/studio/ui/")
        assert (STATIC / relative_path).is_file(), f"missing packaged asset: {url}"


def test_lazy_chunk_imports_are_packaged_and_self_contained() -> None:
    """Every dynamic Vite chunk must exist beside the packaged entrypoint.

    Studio loads Graph, Chat, Builder, and the operational pages on demand.
    Checking only the HTML entrypoint misses an easy-to-ship failure mode: an
    otherwise healthy shell whose first navigation raises a module-load error
    because one lazy chunk was omitted from the wheel.
    """
    asset_root = ASSETS.resolve()
    imports = re.compile(r"(?:import\(|from\s*)[\"']([^\"']+\.(?:js|css))[\"']")

    for chunk in sorted(ASSETS.glob("*.js")):
        content = chunk.read_text(encoding="utf-8", errors="ignore")
        for reference in imports.findall(content):
            assert not reference.startswith(("http:", "https:", "//")), (
                f"{chunk.name} lazily imports an external module: {reference}"
            )
            resolved = (chunk.parent / reference).resolve()
            assert resolved.is_relative_to(asset_root), (
                f"{chunk.name} lazy import escapes packaged assets: {reference}"
            )
            assert resolved.is_file(), f"{chunk.name} references missing lazy chunk: {reference}"


def test_embedded_mount_caches_hashed_chunks_but_revalidates_the_spa_shell() -> None:
    """A production browser should not re-download immutable lazy chunks."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agentomatic.studio.serve import mount_studio_ui

    app = FastAPI()
    mount_studio_ui(app)
    client = TestClient(app)
    chunk = next(ASSETS.glob("index-*.js"))

    shell = client.get("/studio/ui/")
    asset = client.get(f"/studio/ui/assets/{chunk.name}")
    asset_head = client.head(f"/studio/ui/assets/{chunk.name}")

    assert shell.status_code == asset.status_code == 200
    assert asset_head.status_code == 200
    assert shell.headers["cache-control"] == "no-cache"
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert asset_head.headers["cache-control"] == asset.headers["cache-control"]


def test_embedded_mount_serves_assets_from_a_linked_package_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linked wheel installs must not mistake safe chunks for SPA routes.

    ``uv`` and some deployment tools link installed files to a shared cache.
    The asset path then resolves outside the lexical ``site-packages`` path;
    containment must compare resolved roots or the browser receives HTML where
    it expects a JavaScript module.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import agentomatic.studio.serve as serve

    linked_static = tmp_path / "linked-static"
    linked_static.symlink_to(STATIC, target_is_directory=True)
    monkeypatch.setattr(serve, "STATIC_DIR", linked_static)
    app = FastAPI()
    serve.mount_studio_ui(app)
    client = TestClient(app)
    chunk = next(ASSETS.glob("index-*.js"))

    response = client.get(f"/studio/ui/assets/{chunk.name}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_bundle_has_no_stale_entrypoints_or_embedded_distribution_archives() -> None:
    """A package release must contain one coherent Studio build, not leftovers.

    Old Vite entrypoints are dead weight at best and can keep vulnerable code in
    a production wheel. A previous manual copy also placed wheel/tar artifacts
    inside ``static/``, bloating every package release. The build script cleans
    the target directory; preserve that guarantee here.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script_urls = re.findall(r'<script[^>]+src="([^"?]+\.js)"', html)
    stylesheet_urls = re.findall(r'<link[^>]+href="([^"?]+\.css)"', html)

    entrypoints = {Path(url).name for url in script_urls if "/assets/" in url}
    stylesheets = {Path(url).name for url in stylesheet_urls if "/assets/" in url}
    assert entrypoints == {path.name for path in ASSETS.glob("index-*.js")}
    assert stylesheets == {path.name for path in ASSETS.glob("index-*.css")}
    assert not [path for path in STATIC.rglob("*") if path.suffix in {".whl", ".gz", ".zip"}]


def test_fonts_still_have_local_fallbacks() -> None:
    """Removing the imports is only safe while the stacks name real fallbacks."""
    for css in sorted(ASSETS.glob("*.css")):
        content = css.read_text(encoding="utf-8", errors="ignore")
        if "font-family:Inter" not in content:
            continue

        assert "sans-serif" in content
        assert "monospace" in content
