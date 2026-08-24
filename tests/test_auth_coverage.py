# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Exhaustive auth coverage over every mounted route.

Two auth bypasses shipped in this codebase (the Studio debug API riding a
``/studio`` skip-prefix, and the control-plane drain being escapable via an
agent's slug alias). Both were found by inspection, which cannot prove the
absence of a third.

These tests instead enumerate *every* route the platform mounts and probe it
with auth enabled and no credentials. The set that answers anything other than
401 must exactly equal an explicit, reviewed allowlist — so a newly added route
that is accidentally public fails here rather than in production.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform

_API_KEY = "unit-test-api-key"
_CONTROL_TOKEN = "unit-test-control-token"

#: Routes that are public *by design*, each with the reason it must stay so.
#: Adding to this set is a security decision and should be reviewed as one.
_INTENTIONALLY_PUBLIC: dict[str, str] = {
    "/": "root landing page — no data",
    "/health": "liveness probe: orchestrators cannot authenticate",
    "/readiness": "readiness probe: same",
    "/status": "human status page — verified to carry no secrets",
    "/docs": "Swagger UI shell (the API it documents is still gated)",
    "/docs/oauth2-redirect": "Swagger OAuth redirect target",
    "/redoc": "ReDoc shell",
    "/openapi.json": "API schema — deliberately always available",
    "/studio/ui": "Studio SPA shell",
    "/studio/ui/static": "Studio static assets",
    "/studio/ui/{filename:path}": "Studio SPA fallback route (serves index.html)",
}


def _build_platform(tmp_path, **overrides: Any) -> AgentPlatform:
    async def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": "ok", "agent_type": "echo"}

    kwargs: dict[str, Any] = {
        "agents_dir": tmp_path / "agents",
        "plugins_dir": tmp_path / "plugins",
        "endpoints_dir": tmp_path / "endpoints",
        "enable_studio": True,
        "enable_control_plane": True,
        "control_token": _CONTROL_TOKEN,
        "enable_auth": True,
        "auth_api_key": _API_KEY,
    }
    kwargs.update(overrides)
    platform = AgentPlatform(**kwargs)
    platform.register_agent(
        manifest=AgentManifest(name="echo_agent", slug="echo", description="echo"),
        node_fn=echo,
    )
    return platform


def _concrete(path: str) -> str | None:
    """Substitute path params, or return None when the route can't be probed."""
    filled = path
    for placeholder in (
        "{name}",
        "{agent_name}",
        "{thread_id}",
        "{run_id}",
        "{task_id}",
        "{log_id}",
        "{tid}",
        "{filename:path}",
    ):
        replacement = "echo_agent" if "name" in placeholder else "probe"
        filled = filled.replace(placeholder, replacement)
    return None if "{" in filled else filled


def _probe_unauthenticated(app) -> list[tuple[str, str, int]]:
    """Return ``(method, path, status)`` for routes reachable with no credentials."""
    reachable: list[tuple[str, str, int]] = []
    with TestClient(app, raise_server_exceptions=False) as client:
        for route in app.routes:
            template = getattr(route, "path", None)
            if not template:
                continue
            target = _concrete(template)
            if target is None:
                continue
            methods = getattr(route, "methods", None) or {"GET"}
            for method in sorted(m for m in methods if m not in {"HEAD", "OPTIONS"}):
                body = {} if method in {"POST", "PUT", "PATCH"} else None
                response = client.request(method, target, json=body)
                if response.status_code != 401:
                    reachable.append((method, template, response.status_code))
    return reachable


def test_api_key_auth_covers_every_route_except_the_reviewed_allowlist(tmp_path) -> None:
    platform = _build_platform(tmp_path)
    reachable = _probe_unauthenticated(platform.build())

    unexpected = sorted({path for _, path, _ in reachable} - set(_INTENTIONALLY_PUBLIC))
    assert not unexpected, (
        "These routes answered without credentials while API-key auth was "
        f"enabled: {unexpected}. If a route is genuinely meant to be public, "
        "add it to _INTENTIONALLY_PUBLIC with a justification — that is a "
        "security decision and should be reviewed as one."
    )


def test_jwt_auth_covers_every_route_except_the_reviewed_allowlist(tmp_path) -> None:
    """The JWT middleware keeps its own skip list, so it needs its own sweep."""
    platform = _build_platform(
        tmp_path,
        enable_auth=False,
        auth_api_key="",
        enable_jwt_auth=True,
    )
    reachable = _probe_unauthenticated(platform.build())

    unexpected = sorted({path for _, path, _ in reachable} - set(_INTENTIONALLY_PUBLIC))
    assert not unexpected, (
        f"Routes reachable without a JWT while JWT auth was enabled: {unexpected}"
    )


def test_the_sweep_actually_probes_a_meaningful_number_of_routes(tmp_path) -> None:
    """Guard the harness: a broken substitution would skip everything and
    make the assertions above vacuously true.
    """
    platform = _build_platform(tmp_path)
    app = platform.build()

    probed = sum(
        1
        for route in app.routes
        if getattr(route, "path", None) and _concrete(route.path) is not None
        for m in (getattr(route, "methods", None) or {"GET"})
        if m not in {"HEAD", "OPTIONS"}
    )
    assert probed >= 100, f"only {probed} routes probed — substitution likely broke"


def test_agent_data_routes_are_gated(tmp_path) -> None:
    """Spot-check the routes that actually carry data, under both aliases."""
    platform = _build_platform(tmp_path)
    with TestClient(platform.build(), raise_server_exceptions=False) as client:
        for path in (
            "/api/v1/echo_agent/invoke",
            "/api/v1/echo/invoke",  # slug alias
            "/api/v1/echo_agent/chat",
            "/studio/agents",
            "/studio/agents/echo_agent/config",
            "/api/v1/control/agents",
        ):
            method = "POST" if path.endswith(("invoke", "chat")) else "GET"
            response = client.request(method, path, json={"query": "x"})
            assert response.status_code == 401, f"{path} reachable without credentials"


@pytest.mark.parametrize(
    "public_path",
    # The templated SPA-fallback route is excluded here rather than skipped
    # inside the test: its id contains brackets, which pytest plugins that
    # render skip reasons through rich try to parse as markup.
    sorted(p for p in _INTENTIONALLY_PUBLIC if "{" not in p),
)
def test_public_routes_do_not_leak_configured_secrets(public_path, tmp_path) -> None:
    """Whatever is public must not carry the API key or control token."""
    platform = _build_platform(tmp_path)
    with TestClient(platform.build(), raise_server_exceptions=False) as client:
        body = client.get(public_path).text

    assert _API_KEY not in body
    assert _CONTROL_TOKEN not in body


# =====================================================================
# The skip-prefix must not be escapable
# =====================================================================


async def _raw_get(app, path: str) -> tuple[int, bytes]:
    """Send a raw, un-normalised path straight into the ASGI app.

    An HTTP client normally collapses ``..`` before sending, which would mask a
    traversal bug. Real attackers do not (``curl --path-as-is``, raw sockets),
    so the scope is constructed by hand here.
    """
    chunks: list[bytes] = []
    status: int | None = None
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("1.2.3.4", 1),
        "server": ("testserver", 80),
        "scheme": "http",
        "root_path": "",
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    return status or 0, b"".join(chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/studio/ui/../agents",
        "/studio/ui/..%2fagents",
        "/studio/ui/../../../../etc/passwd",
        "//studio/agents",
        "/studio/uiadmin",
        "/studio/ui-admin",
    ],
)
async def test_public_studio_prefix_cannot_be_escaped(path, tmp_path) -> None:
    """``/studio/ui`` is public; the debug API next to it is not.

    A traversal that starts inside the public prefix must not reach the
    protected routes — neither by skipping auth and then normalising into
    ``/studio/agents``, nor by reading files off disk through the SPA fallback.
    """
    platform = _build_platform(tmp_path)
    app = platform.build()

    async with app.router.lifespan_context(app):
        status, body = await _raw_get(app, path)

    # Either the request is rejected, or it lands on the SPA shell — never on
    # agent data and never on a file from outside the static directory.
    assert b"root:x:" not in body, f"{path} read a file off disk"
    if status == 200:
        assert body.lstrip()[:9].lower() == b"<!doctype", (
            f"{path} returned a 200 that was not the SPA shell: {body[:120]!r}"
        )
    else:
        assert status in {401, 404, 307}, f"{path} returned unexpected status {status}"
