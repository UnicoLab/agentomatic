# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Backend ↔ Studio-frontend API alignment.

The Studio UI ships as a pre-built React bundle synced from a separate repo,
so nothing in this repository's own test suite would otherwise notice if a
backend route were renamed, moved, or removed out from under it — the UI would
simply start 404-ing at runtime.

This module parses the API paths the shipped bundle actually calls out of the
JavaScript, then asserts every one of them resolves to a real route on a fully
featured platform. It is a drift alarm for exactly the failure mode that
"the checked-in bundle is stale relative to the backend" produces.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform

_BUNDLE_DIR = Path(__file__).resolve().parents[1] / ("src/agentomatic/studio/static/static/js")

# Paths the bundle builds dynamically in ways the static extractor cannot see
# (or that are intentionally probed with a fallback). Keep this list tiny and
# justified — every entry is a hole in the drift alarm.
_EXTRACTOR_BLIND_SPOTS: frozenset[str] = frozenset()


# =====================================================================
# Bundle parsing
# =====================================================================


def _find_bundle() -> Path | None:
    """Return the shipped Studio JS bundle, if the UI assets are present."""
    if not _BUNDLE_DIR.is_dir():
        return None
    bundles = sorted(_BUNDLE_DIR.glob("main.*.js"))
    return bundles[0] if bundles else None


def _parse_concat_chain(text: str, quote_index: int) -> tuple[str, int]:
    """Parse a JS string-concat chain into a path template.

    ``"/studio/agents/".concat(e,"/graph")`` → ``/studio/agents/*/graph``

    Args:
        text: Full bundle source.
        quote_index: Index of the opening double quote of the chain.

    Returns:
        ``(path_template, index_just_past_the_chain)``.
    """
    end_quote = text.index('"', quote_index + 1)
    parts: list[str] = [text[quote_index + 1 : end_quote]]
    cursor = end_quote + 1

    while text.startswith(".concat(", cursor):
        cursor += len(".concat(")
        depth, arg_start = 1, cursor
        while depth:
            char = text[cursor]
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            cursor += 1
        inner = text[arg_start : cursor - 1]

        # Split the concat() arguments on top-level commas only.
        args: list[str] = []
        depth, current = 0, ""
        for char in inner:
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            if char == "," and depth == 0:
                args.append(current)
                current = ""
            else:
                current += char
        args.append(current)

        for arg in args:
            arg = arg.strip()
            if len(arg) >= 2 and arg.startswith('"') and arg.endswith('"'):
                parts.append(arg[1:-1])
            else:
                # A runtime expression — a path parameter.
                parts.append("*")

    return "".join(parts), cursor


def extract_frontend_api_paths(bundle_source: str) -> set[str]:
    """Extract the ``/api/v1`` and ``/studio`` paths the bundle requests."""
    paths: set[str] = set()
    for match in re.finditer(r'(?:request|fetch)\(\s*"(?=/(?:api/v1|studio)/)', bundle_source):
        quote_index = bundle_source.index('"', match.end() - 1)
        try:
            template, _ = _parse_concat_chain(bundle_source, quote_index)
        except (ValueError, IndexError):  # pragma: no cover - defensive
            continue
        # Drop any query string; routing only cares about the path.
        paths.add(template.split("?", 1)[0].rstrip("/"))
    return {p for p in paths if p}


# =====================================================================
# Backend route collection
# =====================================================================


async def _echo(state: dict[str, Any]) -> dict[str, Any]:
    return {"response": "ok", "agent_type": "echo"}


_DEMO_PLUGIN_SOURCE = '''
"""Minimal plugin so per-plugin routes (predict/model_card) are mounted."""
from __future__ import annotations

from pydantic import BaseModel

from agentomatic.plugins import BaseMLPlugin


class DemoInput(BaseModel):
    text: str = ""


class DemoOutput(BaseModel):
    label: str = ""


class DemoPlugin(BaseMLPlugin[DemoInput, DemoOutput]):
    plugin_name = "demo"
    plugin_description = "Alignment-test plugin"

    async def predict(self, inputs: DemoInput) -> DemoOutput:
        return DemoOutput(label="ok")
'''


@pytest.fixture(scope="module")
def backend_route_templates(tmp_path_factory) -> set[str]:
    """Every route path a fully featured platform exposes.

    A plugin is discovered from disk so the per-plugin routes the Studio UI
    calls (``/predict``, ``/model_card``) are actually mounted — without one,
    the plugin section of the UI has nothing to align against.
    """
    import importlib
    import sys

    tmp_path = tmp_path_factory.mktemp("alignment")
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    # The platform discovers plugins under the package prefix
    # ``plugins_dir.name`` (i.e. ``plugins.demo_plugin``), so this must be a
    # real package with its PARENT on sys.path.
    (plugins_dir / "__init__.py").write_text("", encoding="utf-8")
    (plugins_dir / "demo_plugin.py").write_text(_DEMO_PLUGIN_SOURCE, encoding="utf-8")

    # The repository has its own top-level ``plugins`` package which would
    # otherwise shadow this one, so drop any cached import of it and put the
    # temp parent first on sys.path. invalidate_caches() is required because
    # these files were created after interpreter start.
    saved_modules = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "plugins" or name.startswith("plugins.")
    }
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        platform = AgentPlatform(
            agents_dir=tmp_path / "agents",
            plugins_dir=plugins_dir,
            endpoints_dir=tmp_path / "endpoints",
            title="Alignment",
            enable_studio=True,
            enable_control_plane=True,
            control_token="t",
        )
        platform.register_agent(
            manifest=AgentManifest(name="echo_agent", slug="echo", description="Echo"),
            node_fn=_echo,
        )
        app = platform.build()
        with TestClient(app):
            routes = {r.path for r in app.routes if hasattr(r, "path")}
        # Fail loudly if the demo plugin was not discovered — otherwise the
        # per-plugin alignment assertions would silently have nothing to check.
        assert any("/plugins/demo" in r for r in routes), (
            "demo plugin was not discovered, so per-plugin routes are missing; "
            f"the alignment check would be vacuous. plugin routes seen: "
            f"{sorted(r for r in routes if '/plugins' in r)}"
        )
        return routes
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n == "plugins" or n.startswith("plugins.")]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


def _segments(path: str) -> list[str]:
    return [s for s in path.split("/") if s]


def _matches(frontend_path: str, backend_path: str) -> bool:
    """Whether a frontend path template matches a backend route template.

    ``*`` (frontend runtime expression) and ``{param}`` (FastAPI path param)
    both match any single segment.
    """
    fe, be = _segments(frontend_path), _segments(backend_path)
    if len(fe) != len(be):
        return False
    for fe_seg, be_seg in zip(fe, be, strict=True):
        if fe_seg == "*" or (be_seg.startswith("{") and be_seg.endswith("}")):
            continue
        if fe_seg != be_seg:
            return False
    return True


# =====================================================================
# Tests
# =====================================================================


def test_studio_bundle_is_present() -> None:
    """The packaged Studio UI assets must ship with the wheel."""
    assert _find_bundle() is not None, (
        f"No Studio JS bundle found under {_BUNDLE_DIR}. The Studio UI is "
        "expected to be synced into the package."
    )


def test_extractor_finds_a_meaningful_number_of_paths() -> None:
    """Guard the parser itself — a silent regex break would void this suite."""
    bundle = _find_bundle()
    assert bundle is not None
    paths = extract_frontend_api_paths(bundle.read_text(encoding="utf-8", errors="replace"))
    assert len(paths) >= 25, f"extractor found only {len(paths)} paths — parser likely broke"
    # Spot-check a few well-known calls the UI certainly makes.
    assert "/studio/agents" in paths
    assert "/api/v1/control/agents" in paths


def test_every_frontend_api_path_exists_on_the_backend(backend_route_templates) -> None:
    """Every endpoint the shipped Studio UI calls must exist on the backend.

    A failure here means the checked-in UI bundle and the Python backend have
    drifted: the UI will 404 at runtime against this version of the platform.
    """
    bundle = _find_bundle()
    assert bundle is not None
    frontend_paths = extract_frontend_api_paths(
        bundle.read_text(encoding="utf-8", errors="replace")
    )

    missing = sorted(
        fe
        for fe in frontend_paths
        if fe not in _EXTRACTOR_BLIND_SPOTS
        and not any(_matches(fe, be) for be in backend_route_templates)
    )
    assert not missing, (
        "The Studio UI calls endpoints that do not exist on the backend "
        f"(bundle/backend drift): {missing}"
    )


def test_studio_debug_api_paths_are_all_under_the_studio_prefix(
    backend_route_templates,
) -> None:
    """Sanity: the Studio calls we protect with auth really are /studio/*."""
    bundle = _find_bundle()
    assert bundle is not None
    paths = extract_frontend_api_paths(bundle.read_text(encoding="utf-8", errors="replace"))
    studio_paths = {p for p in paths if p.startswith("/studio")}
    assert studio_paths, "expected the UI to call the Studio debug API"
    # None of them are the public UI shell — those are asset requests, not
    # API calls, so the debug API is entirely inside the authenticated set.
    assert all(not p.startswith("/studio/ui") for p in studio_paths)
