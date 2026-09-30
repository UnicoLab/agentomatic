# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Every route must answer, not crash — swept exhaustively.

The unit suite passed 2092 tests while a stock deployment returned a bare
``500 Internal Server Error`` on ``/api/v1/{agent}/optimization-runs``: the
route guarded a lazy store proxy with ``is None`` (never true for a proxy), so
a ``RuntimeError`` escaped the handler. Two more turned up the same way —
``/studio/agents/{name}/graph`` for an agent with no ``graph_fn``, and the
thread-summary route.

None of those were caught by testing individual features, because each was
only reachable in a configuration no individual test happened to build. This
module instead walks *every* mounted route and asserts none of them 5xx, both in
the default posture a fresh `agentomatic run` uses (no store configured, so the
platform falls back to a bounded in-memory store) and with that fallback
disabled (``AGENTOMATIC_EPHEMERAL_THREADS=0``, genuinely no store).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform
from agentomatic.storage import MemoryStore

#: Substitutions that turn a route template into a concrete probe URL. The
#: values are deliberately non-existent ids — a missing resource is a 404, not
#: a 500, and asserting that is the point.
_PATH_PARAMS = {
    "{name}": "a1",
    "{agent_name}": "a1",
    "{thread_id}": "no-such-thread",
    "{run_id}": "no-such-run",
    "{task_id}": "no-such-task",
    "{log_id}": "no-such-log",
    "{tid}": "no-such-thread",
    "{filename:path}": "index.html",
}

#: A superset body — routes ignore the keys they don't declare, and the ones
#: they do declare get a plausible value so we exercise the handler rather than
#: bouncing off request validation.
_PROBE_BODY: dict[str, Any] = {
    "query": "hi",
    "content": "hi",
    "value": "x",
    "enabled": False,
    "updates": {},
    "message": {"content": "hi"},
    "input": {"query": "hi"},
    "message_index": 0,
    "text": "hi",
}

_CONTROL_TOKEN = "sweep-control-token"


def _build_swept_app():
    """A fully-featured platform with no store configured."""
    import tempfile
    from pathlib import Path

    async def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": "ok", "agent_type": "echo"}

    tmp = Path(tempfile.mkdtemp())
    platform = AgentPlatform(
        agents_dir=tmp / "agents",
        plugins_dir=tmp / "plugins",
        endpoints_dir=tmp / "endpoints",
        enable_studio=True,
        enable_control_plane=True,
        control_token=_CONTROL_TOKEN,
    )
    platform.register_agent(
        manifest=AgentManifest(name="a1", slug="a1", description="echo"),
        node_fn=echo,
    )
    return platform.build()


@pytest.fixture(scope="module", params=["ephemeral-store", "no-store"])
def swept_app(request):
    """Both store postures: the default fallback, and the fallback disabled.

    The flag is read at lifespan start-up, which runs each time a test enters
    ``TestClient``, so it stays set for the lifetime of the module-scoped app.
    """
    with pytest.MonkeyPatch.context() as mp:
        if request.param == "no-store":
            mp.setenv("AGENTOMATIC_EPHEMERAL_THREADS", "0")
        else:
            mp.delenv("AGENTOMATIC_EPHEMERAL_THREADS", raising=False)
        yield _build_swept_app()


def _probe_targets(app) -> list[tuple[str, str, str]]:
    """Return ``(method, concrete_path, route_template)`` for every route."""
    targets: list[tuple[str, str, str]] = []
    for route in app.routes:
        template = getattr(route, "path", None)
        if not template:
            continue
        concrete = template
        for placeholder, value in _PATH_PARAMS.items():
            concrete = concrete.replace(placeholder, value)
        if "{" in concrete:  # an unknown param we can't fill safely
            continue
        methods = getattr(route, "methods", None) or {"GET"}
        for method in sorted(m for m in methods if m not in {"HEAD", "OPTIONS"}):
            targets.append((method, concrete, template))
    return targets


def test_no_route_returns_an_unhandled_server_error(swept_app) -> None:
    """No route may 5xx in the default configuration.

    A 4xx is fine everywhere — missing resource, unconfigured backend, bad
    input. A 5xx means an exception escaped a handler.
    """
    failures: list[str] = []

    with TestClient(swept_app, raise_server_exceptions=False) as client:
        for method, path, template in _probe_targets(swept_app):
            body = _PROBE_BODY if method in {"POST", "PUT", "PATCH"} else None
            try:
                response = client.request(
                    method, path, json=body, headers={"X-Control-Token": _CONTROL_TOKEN}
                )
            except Exception as exc:  # noqa: BLE001 - a raised error is a failure too
                failures.append(f"{method} {template} raised {type(exc).__name__}: {exc}")
                continue
            if response.status_code >= 500:
                failures.append(
                    f"{method} {template} -> {response.status_code}: {response.text[:160]}"
                )

    assert not failures, "Routes returned a server error:\n" + "\n".join(failures)


def test_the_sweep_covers_a_meaningful_number_of_routes(swept_app) -> None:
    """Guard the harness — a broken substitution would skip everything and make
    the assertion above vacuously true.
    """
    targets = _probe_targets(swept_app)
    assert len(targets) >= 80, f"only {len(targets)} route/method pairs probed"


def test_store_dependent_routes_answer_4xx_rather_than_crashing(monkeypatch) -> None:
    """The specific regression: these need a store, and none is configured."""
    monkeypatch.setenv("AGENTOMATIC_EPHEMERAL_THREADS", "0")
    with TestClient(_build_swept_app(), raise_server_exceptions=False) as client:
        for path in (
            "/api/v1/a1/optimization-runs",
            "/api/v1/a1/logs",
            "/api/v1/a1/threads/no-such-thread/summary",
        ):
            response = client.get(path)
            assert 400 <= response.status_code < 500, (
                f"{path} -> {response.status_code} (expected a 4xx): {response.text[:160]}"
            )


def test_default_posture_serves_thread_routes(monkeypatch) -> None:
    """A fresh project configures no store — threads must still work.

    Studio's "New Chat" creates a thread before the first message; without a
    store that answered 400 "Thread storage not configured", so no chat could
    ever start on a freshly scaffolded project.
    """
    monkeypatch.delenv("AGENTOMATIC_EPHEMERAL_THREADS", raising=False)
    with TestClient(_build_swept_app(), raise_server_exceptions=False) as client:
        created = client.post("/api/v1/a1/threads", json={"user_id": "u", "title": "New Chat"})
        assert created.status_code == 200, created.text
        thread_id = created.json()["id"]
        assert client.get(f"/api/v1/a1/threads/{thread_id}").status_code == 200
        assert client.get("/api/v1/a1/threads").json()["count"] == 1
        assert client.get("/api/v1/a1/optimization-runs").status_code == 200


def test_studio_graph_degrades_for_an_agent_without_a_graph(swept_app) -> None:
    """The Studio UI calls this for *every* agent; a node_fn-only agent has no
    graph to draw, which must not be a 500.
    """
    with TestClient(swept_app, raise_server_exceptions=False) as client:
        response = client.get("/studio/agents/a1/graph")

    assert response.status_code == 200, response.text
    assert response.json()["agent_name"] == "a1"


@pytest.fixture(scope="module")
def swept_app_with_store():
    """The *other* posture: a store, invocation history, and the task API on.

    The no-store sweep above cannot reach the code paths that only run once a
    store exists (history reads, checkpoint lookups, thread summaries). A
    container sweep in that configuration is what surfaced the A2A and
    template defects, so the same surface is covered here.
    """
    import tempfile
    from pathlib import Path

    async def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": "ok", "agent_type": "echo"}

    tmp = Path(tempfile.mkdtemp())
    platform = AgentPlatform(
        agents_dir=tmp / "agents",
        plugins_dir=tmp / "plugins",
        endpoints_dir=tmp / "endpoints",
        enable_studio=True,
        enable_control_plane=True,
        control_token=_CONTROL_TOKEN,
        store=MemoryStore(),
        logs_history=True,
    )
    platform.register_agent(
        manifest=AgentManifest(name="a1", slug="a1", description="echo"),
        node_fn=echo,
    )
    return platform.build()


def test_no_route_returns_a_server_error_with_a_store_configured(swept_app_with_store) -> None:
    """Same sweep, store-backed posture — the one a real deployment runs."""
    failures: list[str] = []

    with TestClient(swept_app_with_store, raise_server_exceptions=False) as client:
        for method, path, template in _probe_targets(swept_app_with_store):
            body = _PROBE_BODY if method in {"POST", "PUT", "PATCH"} else None
            try:
                response = client.request(
                    method, path, json=body, headers={"X-Control-Token": _CONTROL_TOKEN}
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{method} {template} raised {type(exc).__name__}: {exc}")
                continue
            if response.status_code >= 500:
                failures.append(
                    f"{method} {template} -> {response.status_code}: {response.text[:160]}"
                )

    assert not failures, "Routes returned a server error:\n" + "\n".join(failures)
