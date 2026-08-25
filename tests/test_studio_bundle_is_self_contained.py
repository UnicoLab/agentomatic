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

from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "src" / "agentomatic" / "studio" / "static"

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
        *sorted((STATIC / "static" / "css").glob("*.css")),
        STATIC / "index.html",
    ]


def test_the_bundle_is_present() -> None:
    """Guard against the tests silently passing on a missing bundle."""
    assert (STATIC / "index.html").is_file()
    assert list((STATIC / "static" / "css").glob("*.css"))
    assert list((STATIC / "static" / "js").glob("*.js"))


@pytest.mark.parametrize("host", FORBIDDEN_HOSTS)
def test_no_render_blocking_third_party_host(host: str) -> None:
    """CSS and HTML must not reference an external host."""
    for asset in _served_assets():
        content = asset.read_text(encoding="utf-8", errors="ignore")

        assert host not in content, f"{asset.name} references {host}"


def test_stylesheets_have_no_external_imports() -> None:
    """``@import`` is render-blocking, so an external one is the worst case."""
    for css in sorted((STATIC / "static" / "css").glob("*.css")):
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


def test_fonts_still_have_local_fallbacks() -> None:
    """Removing the imports is only safe while the stacks name real fallbacks."""
    for css in sorted((STATIC / "static" / "css").glob("*.css")):
        content = css.read_text(encoding="utf-8", errors="ignore")
        if "font-family:Inter" not in content:
            continue

        assert "sans-serif" in content
        assert "monospace" in content
