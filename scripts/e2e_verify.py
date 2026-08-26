#!/usr/bin/env python
"""End-to-end verification harness for a running Agentomatic platform.

Exercises every public surface the platform advertises — platform routes,
Studio (the exact calls the bundled React UI makes), agents, plugins,
endpoints, ingestion, pipelines, tasks, control plane, metrics and auth —
against a live server and reports a pass/fail table.

The harness is deployment-agnostic: point it at ``agentomatic run``, at
``uvicorn main:app``, or at a container published by ``agentomatic deploy``.

Usage::

    uv run python scripts/e2e_verify.py --base-url http://localhost:8000 \
        --agent my_agent --plugin scorer --pipeline basic_flow \
        --api-key secret --control-token tok --json report.json

Exit code is ``0`` only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

#: How many times to wait out a 429 before treating it as a failure.
_RATE_LIMIT_RETRIES = 3
#: Cap on a single Retry-After wait, so a long window cannot stall the run.
_RATE_LIMIT_MAX_WAIT = 65.0
#: ``Retry-After`` is an integer number of seconds.  The in-memory limiter
#: truncates a fractional remaining window, so a half-second cushion can wake
#: the verifier *before* the oldest request has expired and produce a false
#: second 429 under concurrent load.
_RATE_LIMIT_RETRY_PAD = 1.1

#: Task statuses that mean the work finished successfully.
_TERMINAL_OK = frozenset({"completed", "succeeded", "success"})
#: Task statuses that mean the work finished unsuccessfully.
_TERMINAL_BAD = frozenset({"failed", "error", "cancelled", "canceled"})

#: Every step type the pipeline engine implements. Used to report which
#: ones a deployment's published pipelines actually exercise.
_ALL_STEP_TYPES = (
    "agent",
    "plugin",
    "endpoint",
    "ingestion",
    "parallel",
    "map",
    "transform",
    "loop",
    "sub_pipeline",
)


def discover_agent(base_url: str, api_key: str, timeout: float) -> str | None:
    """Return one registered agent for a no-argument deployment probe.

    The verifier used to default to a fixture-only ``ag_basic`` name.  That
    made a perfectly healthy deployment look broken when its agent directory
    used any other name.  Discover from the public registry instead; callers
    can still pass ``--agent`` to choose the exact contract to exercise.
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["X-Api-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/api/v1/agents",
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:  # noqa: BLE001 - main reports an actionable CLI error
        return None

    agents = payload.get("agents", payload) if isinstance(payload, dict) else payload
    if isinstance(agents, dict):
        names = [name for name in agents if isinstance(name, str) and name]
    elif isinstance(agents, list):
        names = [item.get("name") or item.get("slug") for item in agents if isinstance(item, dict)]
    else:
        names = []
    return next((name for name in names if isinstance(name, str) and name), None)


def _collect_step_types(steps: Any, into: set[str]) -> None:
    """Record every ``step_type`` in a pipeline config, nesting included."""
    if isinstance(steps, dict):
        steps = [steps]
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("step_type"):
            into.add(str(step["step_type"]))
        for nested in ("steps", "body", "step", "branches"):
            if nested in step:
                _collect_step_types(step[nested], into)


# ---------------------------------------------------------------------------
# Result plumbing
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One verified assertion about the running platform."""

    group: str
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class Report:
    """Accumulates checks and renders the final verdict."""

    checks: list[Check] = field(default_factory=list)

    def add(self, group: str, name: str, ok: bool, detail: str = "") -> bool:
        """Record a check outcome and return ``ok`` for chaining."""
        self.checks.append(Check(group, name, ok, detail))
        return ok

    def skip(self, group: str, name: str, why: str) -> None:
        """Record a check that did not apply to this deployment."""
        self.checks.append(Check(group, name, True, why, skipped=True))

    @property
    def failures(self) -> list[Check]:
        """Return every failed check."""
        return [c for c in self.checks if not c.ok]

    def render(self) -> str:
        """Render a human-readable summary table."""
        lines: list[str] = []
        groups: dict[str, list[Check]] = {}
        for c in self.checks:
            groups.setdefault(c.group, []).append(c)
        for group, items in groups.items():
            passed = sum(1 for i in items if i.ok and not i.skipped)
            skipped = sum(1 for i in items if i.skipped)
            failed = sum(1 for i in items if not i.ok)
            status = "FAIL" if failed else "PASS"
            lines.append(
                f"[{status}] {group:<22} {passed:>3} passed"
                + (f", {skipped} skipped" if skipped else "")
                + (f", {failed} FAILED" if failed else "")
            )
            for i in items:
                if not i.ok:
                    lines.append(f"         ✗ {i.name}: {i.detail}")
        total = len(self.checks)
        failed = len(self.failures)
        lines.append("")
        lines.append(f"TOTAL: {total} checks, {total - failed} passed, {failed} failed")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class Verifier:
    """Drives every check against one base URL."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        control_token: str = "",
        agent: str = "",
        plugin: str = "",
        pipeline: str = "",
        endpoint: str = "",
        read_endpoint: str = "",
        ingestor: str = "",
        builder_smoke_name: str = "",
        timeout: float = 30.0,
        expect_auth: bool = False,
        expect_studio: bool = True,
    ) -> None:
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.control_token = control_token
        self.agent = agent
        self.plugin = plugin
        self.pipeline = pipeline
        self.endpoint = endpoint
        self.read_endpoint = read_endpoint
        self.ingestor = ingestor
        self.builder_smoke_name = builder_smoke_name
        self.expect_auth = expect_auth
        self.expect_studio = expect_studio
        self.report = Report()
        #: Set by ``verify_agent_rest``'s thread probe. A deployment with
        #: no store is a legitimate posture, so store-dependent checks
        #: report as skipped rather than failing.
        self.thread_store_available = True
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            # Mirror exactly what the Studio bundle sends.
            headers["X-Api-Key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base, headers=headers, timeout=timeout)

    # -- helpers ---------------------------------------------------------

    def _req(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        honour_retry_after: bool = True,
    ) -> httpx.Response | None:
        """Issue a request, returning ``None`` on transport failure.

        When the deployment enables rate limiting, this harness is itself a
        burst of traffic from one IP. A ``429`` is the limiter behaving
        correctly, so wait out ``Retry-After`` and try again rather than
        reporting a false failure. Set ``honour_retry_after=False`` when the
        ``429`` is the thing under test.
        """
        for attempt in range(_RATE_LIMIT_RETRIES):
            try:
                resp = self.client.request(method, path, json=json_body, headers=headers)
            except Exception:  # noqa: BLE001 - reported as a failed check
                return None
            if resp.status_code != 429 or not honour_retry_after:
                return resp
            if attempt == _RATE_LIMIT_RETRIES - 1:
                return resp
            delay = float(resp.headers.get("Retry-After") or 1)
            time.sleep(min(delay, _RATE_LIMIT_MAX_WAIT) + _RATE_LIMIT_RETRY_PAD)
        return None

    def check(
        self,
        group: str,
        name: str,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        expect: tuple[int, ...] = (200,),
        headers: dict[str, str] | None = None,
        validate: Any = None,
    ) -> Any:
        """Call one route, assert its status, and optionally validate the body.

        Args:
            group: Report grouping label.
            name: Human-readable check name.
            method: HTTP method.
            path: Path relative to the base URL.
            json_body: Optional JSON request body.
            expect: Acceptable status codes.
            headers: Extra request headers.
            validate: Optional ``callable(payload) -> str``; a non-empty
                return value marks the check failed with that message.

        Returns:
            The decoded JSON payload when available, else ``None``.
        """
        resp = self._req(method, path, json_body=json_body, headers=headers)
        if resp is None:
            self.report.add(group, name, False, f"{method} {path} — transport error")
            return None
        if resp.status_code not in expect:
            body = resp.text[:300].replace("\n", " ")
            self.report.add(
                group,
                name,
                False,
                f"{method} {path} → {resp.status_code} (want {expect}): {body}",
            )
            return None
        payload: Any = None
        if resp.content:
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001 - non-JSON bodies are fine
                payload = None
        if validate is not None and payload is not None:
            problem = validate(payload)
            if problem:
                self.report.add(group, name, False, f"{method} {path} — {problem}")
                return payload
        self.report.add(group, name, True)
        return payload

    def sse(
        self,
        group: str,
        name: str,
        path: str,
        body: dict[str, Any],
        *,
        want_events: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Stream a POST SSE endpoint exactly as the Studio bundle does."""
        events: list[dict[str, Any]] = []
        raw_lines: list[str] = []
        for attempt in range(_RATE_LIMIT_RETRIES):
            events, raw_lines, retry_after = self._stream_once(group, name, path, body)
            if retry_after is None:
                break
            if attempt == _RATE_LIMIT_RETRIES - 1:
                self.report.add(group, name, False, f"{path} — still rate limited")
                return events
            time.sleep(min(retry_after, _RATE_LIMIT_MAX_WAIT) + _RATE_LIMIT_RETRY_PAD)
        else:  # pragma: no cover - loop always breaks or returns
            return events
        if not events:
            self.report.add(group, name, False, f"{path} — no SSE data frames ({raw_lines[:4]})")
            return events
        seen = {str(e.get("event") or e.get("type") or "") for e in events}
        missing = [w for w in want_events if w not in seen]
        if missing:
            self.report.add(
                group, name, False, f"{path} — missing events {missing}; got {sorted(seen)}"
            )
            return events
        self.report.add(group, name, True)
        return events

    def _stream_once(
        self, group: str, name: str, path: str, body: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[str], float | None]:
        """Read one SSE attempt.

        Returns:
            ``(events, raw_lines, retry_after)``. ``retry_after`` is set only
            when the deployment rate-limited this attempt, in which case the
            caller should wait and retry rather than report a failure.
        """
        events: list[dict[str, Any]] = []
        raw_lines: list[str] = []
        try:
            with self.client.stream(
                "POST",
                path,
                json=body,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code == 429:
                    return events, raw_lines, float(resp.headers.get("Retry-After") or 1)
                if resp.status_code != 200:
                    text = resp.read().decode("utf-8", "replace")[:300]
                    self.report.add(group, name, False, f"{path} → {resp.status_code}: {text}")
                    return events, raw_lines, None
                ctype = resp.headers.get("content-type", "")
                if "text/event-stream" not in ctype:
                    self.report.add(group, name, False, f"{path} — content-type {ctype!r}")
                    return events, raw_lines, None
                for line in resp.iter_lines():
                    raw_lines.append(line)
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        events.append(json.loads(data))
                    except Exception:  # noqa: BLE001 - captured below
                        pass
        except Exception as exc:  # noqa: BLE001
            self.report.add(group, name, False, f"{path} — {type(exc).__name__}: {exc}")
        return events, raw_lines, None

    def await_task(
        self,
        group: str,
        name: str,
        submitted: Any,
        *,
        timeout: float = 30.0,
        expected_terminal_statuses: frozenset[str] | None = None,
    ) -> Any:
        """Poll a 202-submitted task to a terminal state and return its record.

        Args:
            group: Report grouping label.
            name: Human-readable check name.
            submitted: Decoded body of the 202 response.
            timeout: Seconds to wait for a terminal status.
            expected_terminal_statuses: Terminal statuses that constitute a
                successful verification. Ordinary work defaults to succeeded;
                a deliberate cancellation expects cancelled instead.

        Returns:
            The final task record, or ``None`` when it never completed.
        """
        expected = expected_terminal_statuses or _TERMINAL_OK
        task_id = None
        if isinstance(submitted, dict):
            task_id = submitted.get("id") or submitted.get("task_id")
        if not task_id:
            self.report.add(group, name, False, f"no task id in {submitted}")
            return None
        deadline = time.time() + timeout
        record: Any = None
        while time.time() < deadline:
            resp = self._req("GET", f"/api/v1/tasks/{task_id}")
            if resp is not None and resp.status_code == 200:
                try:
                    record = resp.json()
                except Exception:  # noqa: BLE001
                    record = None
                status = (record or {}).get("status")
                if status in expected or status in _TERMINAL_BAD:
                    break
            time.sleep(0.4)
        status = (record or {}).get("status")
        if status not in expected:
            err = (record or {}).get("error")
            self.report.add(group, name, False, f"status={status!r} error={err!r}")
            return record
        self.report.add(group, name, True)
        return record

    @staticmethod
    def task_id_from(payload: Any) -> str | None:
        """Return a unified task id from one of the supported response shapes."""
        if not isinstance(payload, dict):
            return None
        task_id = payload.get("id") or payload.get("task_id")
        return task_id if isinstance(task_id, str) and task_id else None

    def delete_created_task(self, group: str, name: str, submitted: Any) -> None:
        """Remove only a terminal task record created by this verifier."""
        task_id = self.task_id_from(submitted)
        if task_id:
            self.check(group, name, "DELETE", f"/api/v1/tasks/{task_id}", expect=(200, 204))

    # -- groups ----------------------------------------------------------

    def verify_platform(self) -> None:
        """Health, readiness, status, OpenAPI and docs."""
        g = "platform"
        self.check(
            g,
            "health",
            "GET",
            "/health",
            validate=lambda p: "" if p.get("status") else "no status field",
        )
        self.check(g, "ready", "GET", "/ready")
        self.check(g, "readiness", "GET", "/readiness")
        self.check(
            g,
            "status",
            "GET",
            "/status",
            validate=lambda p: "" if isinstance(p, dict) else "not an object",
        )
        self.check(
            g,
            "api status",
            "GET",
            "/api/v1/status",
            validate=lambda p: "" if isinstance(p, dict) else "not an object",
        )

        def _openapi(p: Any) -> str:
            if not isinstance(p, dict) or "paths" not in p:
                return "no paths"
            if not p["paths"]:
                return "empty paths"
            return ""

        self.check(g, "openapi.json", "GET", "/openapi.json", validate=_openapi)
        for doc in ("/docs", "/redoc"):
            resp = self._req("GET", doc)
            ok = resp is not None and resp.status_code == 200
            self.report.add(
                g,
                f"docs {doc}",
                ok,
                "" if ok else f"{doc} → {getattr(resp, 'status_code', 'transport error')}",
            )

        def _agents(p: Any) -> str:
            items = p.get("agents") if isinstance(p, dict) else p
            if not items:
                return "no agents listed"
            return ""

        self.check(g, "agents registry", "GET", "/api/v1/agents", validate=_agents)

    def verify_studio(self) -> None:
        """Every call the bundled Studio React client makes."""
        g = "studio"
        if not self.expect_studio:
            self.report.skip(g, "studio disabled", "AGENTOMATIC_ENABLE_STUDIO=0")
            return
        a = self.agent

        def _info(p: Any) -> str:
            # Field names the Studio bundle reads verbatim (ConnectionSetup,
            # ControlPlaneView): info.version, info.platform_title, info.agent_count.
            missing = [k for k in ("version", "platform_title", "agent_count") if k not in p]
            return f"missing {missing}" if missing else ""

        self.check(g, "GET /studio/info", "GET", "/studio/info", validate=_info)

        def _agents(p: Any) -> str:
            items = p if isinstance(p, list) else p.get("agents", [])
            if not items:
                return "no agents"
            first = items[0] if isinstance(items, list) else None
            if isinstance(first, dict):
                # The bundle reads these fields directly off each entry.
                missing = [k for k in ("name", "slug", "framework") if k not in first]
                if missing:
                    return f"agent entry missing {missing}"
            return ""

        self.check(g, "GET /studio/agents", "GET", "/studio/agents", validate=_agents)

        def _graph(p: Any) -> str:
            if not isinstance(p, dict):
                return "not an object"
            if "nodes" not in p or "edges" not in p:
                return f"missing nodes/edges, got {sorted(p)[:8]}"
            return ""

        self.check(g, "GET graph", "GET", f"/studio/agents/{a}/graph", validate=_graph)
        self.check(g, "GET schemas", "GET", f"/studio/agents/{a}/schemas")
        self.check(g, "GET config", "GET", f"/studio/agents/{a}/config")

        def _run(p: Any) -> str:
            # useStudioStore matches runs on `run.id`; RunInfo also carries
            # agent_name/status/created_at, which the runs list renders.
            if not isinstance(p, dict):
                return "not an object"
            missing = [k for k in ("id", "agent_name", "status", "created_at") if k not in p]
            return f"missing {missing}" if missing else ""

        run = self.check(
            g,
            "POST runs",
            "POST",
            f"/studio/agents/{a}/runs",
            json_body={"query": "e2e studio run", "user_id": "e2e"},
            validate=_run,
        )
        self.check(g, "GET runs list", "GET", f"/studio/agents/{a}/runs?limit=50")
        if isinstance(run, dict) and run.get("id"):
            self.check(g, "GET run by id", "GET", f"/studio/agents/{a}/runs/{run['id']}")

        thread_id = run.get("thread_id") if isinstance(run, dict) else None
        if thread_id:
            self.check(
                g, "GET thread state", "GET", f"/studio/agents/{a}/threads/{thread_id}/state"
            )
            self.check(
                g,
                "POST thread state",
                "POST",
                f"/studio/agents/{a}/threads/{thread_id}/state",
                json_body={"updates": {"e2e": True}},
            )
            self.check(
                g, "GET thread history", "GET", f"/studio/agents/{a}/threads/{thread_id}/history"
            )
        else:
            self.report.skip(g, "thread state/history", "run returned no thread_id")

        self.sse(
            g,
            "POST runs/stream (SSE)",
            f"/studio/agents/{a}/runs/stream",
            {"query": "e2e stream", "user_id": "e2e"},
            want_events=("run_start", "run_complete"),
        )

        resp = self._req("GET", "/studio/ui")
        ok = resp is not None and resp.status_code in (200, 307, 308)
        self.report.add(
            g,
            "GET /studio/ui",
            ok,
            "" if ok else f"→ {getattr(resp, 'status_code', 'transport error')}",
        )
        # The SPA bundle must actually be served, not just the route exist.
        resp = self._req("GET", "/studio/ui/index.html")
        ok = resp is not None and resp.status_code == 200 and b"<div id=" in resp.content
        self.report.add(
            g,
            "Studio SPA bundle",
            ok,
            "" if ok else f"index.html → {getattr(resp, 'status_code', 'transport error')}",
        )

    def verify_live_schema_contracts(self) -> None:
        """Verify every deployed resource publishes Studio's live contracts.

        The React views intentionally derive their request forms from the
        platform rather than hard-code fixture fields.  A successful call to a
        named fixture is not enough: this catches a plugin, endpoint method,
        or agent added to a real deployment without the OpenAPI/schema contract
        its independent Studio tester needs.
        """
        g = "schema-contracts"
        document = self.check(g, "GET OpenAPI schema source", "GET", "/openapi.json")
        paths = document.get("paths", {}) if isinstance(document, dict) else {}

        def operation(path: str, method: str, *, request: bool) -> None:
            item = paths.get(path) if isinstance(paths, dict) else None
            spec = item.get(method.lower()) if isinstance(item, dict) else None
            problems: list[str] = []
            if not isinstance(spec, dict):
                problems.append("operation missing")
            else:
                if request and not spec.get("requestBody"):
                    problems.append("JSON request schema missing")
                responses = spec.get("responses", {})
                success = (
                    next(
                        (
                            response
                            for code, response in responses.items()
                            if str(code).startswith("2")
                        ),
                        None,
                    )
                    if isinstance(responses, dict)
                    else None
                )
                content = success.get("content", {}) if isinstance(success, dict) else {}
                if not isinstance(content, dict) or "application/json" not in content:
                    problems.append("JSON success schema missing")
            self.report.add(
                g,
                f"OpenAPI {method.upper()} {path}",
                not problems,
                "; ".join(problems),
            )

        def list_items(path: str, key: str) -> list[dict[str, Any]]:
            payload = self.check(g, f"GET {path}", "GET", path)
            raw = (
                payload
                if isinstance(payload, list)
                else payload.get(key, [])
                if isinstance(payload, dict)
                else []
            )
            return (
                [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
            )

        for agent in list_items("/studio/agents", "agents"):
            name = agent.get("name") or agent.get("slug")
            if not isinstance(name, str) or not name:
                self.report.add(g, "Studio agent has name", False, repr(agent))
                continue
            schemas = self.check(
                g,
                f"GET agent schemas {name}",
                "GET",
                f"/studio/agents/{quote(name, safe='')}/schemas",
            )
            valid = (
                isinstance(schemas, dict)
                and isinstance(schemas.get("input_schema"), dict)
                and isinstance(schemas.get("output_schema"), dict)
            )
            self.report.add(
                g, f"Agent schema payload {name}", valid, "input_schema/output_schema required"
            )
            operation(f"/api/v1/{quote(name, safe='')}/invoke", "POST", request=True)

        for plugin in list_items("/api/v1/plugins", "plugins"):
            name = plugin.get("name")
            if isinstance(name, str) and name:
                operation(f"/api/v1/plugins/{quote(name, safe='')}/predict", "POST", request=True)

        for endpoint in list_items("/api/v1/endpoints", "endpoints"):
            name = endpoint.get("name")
            endpoint_path = endpoint.get("path")
            methods = endpoint.get("methods")
            if (
                not isinstance(name, str)
                or not isinstance(endpoint_path, str)
                or not isinstance(methods, list)
            ):
                self.report.add(g, "Endpoint contract metadata", False, repr(endpoint))
                continue
            full_path = f"/api/v1/endpoints/{quote(name, safe='')}{endpoint_path}"
            for method in methods:
                if isinstance(method, str) and method:
                    # GET/HEAD inputs correctly live in OpenAPI query
                    # parameters, while all other interactive endpoint methods
                    # must publish a JSON request body for SchemaForm.
                    operation(full_path, method, request=method.upper() not in {"GET", "HEAD"})

        for ingestor in list_items("/api/v1/ingestors", "ingestors"):
            name = ingestor.get("name")
            if isinstance(name, str) and name:
                operation(f"/api/v1/ingestion/{quote(name, safe='')}/run", "POST", request=True)

        for pipeline in list_items("/api/v1/pipelines", "pipelines"):
            name = pipeline.get("name")
            if not isinstance(name, str) or not name:
                continue
            config = self.check(
                g,
                f"GET pipeline schema {name}",
                "GET",
                f"/api/v1/pipelines/{quote(name, safe='')}/config",
            )
            valid = (
                isinstance(config, dict)
                and isinstance(config.get("input_schema"), (dict, type(None)))
                and isinstance(config.get("output_schema"), (dict, type(None)))
            )
            self.report.add(
                g,
                f"Pipeline schema payload {name}",
                valid,
                "input_schema/output_schema must be object or null",
            )
            # Pipeline execution uses a parameterised FastAPI route, unlike
            # dynamically registered agent/plugin/endpoint names above.
            operation("/api/v1/pipelines/{name}/run", "POST", request=True)

        # The Connections view shares one parameterised, redacted contract for
        # every database/vector/service probe.
        operation("/api/v1/control/connections/{scope}/{name}", "GET", request=False)

    def verify_agent_rest(self) -> None:
        """The documented agent REST contract."""
        g = "agent-rest"
        a = self.agent
        base = f"/api/v1/{a}"

        def _invoke(p: Any) -> str:
            if not isinstance(p, dict):
                return "not an object"
            if "result" not in p and "output" not in p and "response" not in p:
                return f"no result/output/response key: {sorted(p)[:8]}"
            return ""

        self.check(
            g,
            "POST invoke",
            "POST",
            f"{base}/invoke",
            json_body={"query": "hello e2e"},
            validate=_invoke,
        )
        # The wire field is `query`; posting `current_query` must be rejected.
        self.check(
            g,
            "invoke rejects current_query",
            "POST",
            f"{base}/invoke",
            json_body={"current_query": "hello"},
            expect=(422,),
        )
        self.check(
            g,
            "POST chat",
            "POST",
            f"{base}/chat",
            json_body={"content": "hi there"},
        )
        submitted = self.check(
            g,
            "POST invoke/batch (202)",
            "POST",
            f"{base}/invoke/batch",
            json_body={"inputs": [{"query": "one"}, {"query": "two"}]},
            expect=(202,),
        )
        record = self.await_task(g, "invoke/batch completes", submitted)
        results = (record or {}).get("result")
        ok = isinstance(results, list) and len(results) == 2
        self.report.add(
            g,
            "invoke/batch returns both items",
            ok,
            "" if ok else f"batch result was {results!r}",
        )
        self.delete_created_task(g, "DELETE invoke/batch task", submitted)
        # A batch body that names the item list wrongly, or carries no items,
        # must be rejected — never accepted as a zero-item "succeeded" batch.
        self.check(
            g,
            "batch rejects unknown item key",
            "POST",
            f"{base}/invoke/batch",
            json_body={"items": [{"query": "one"}]},
            expect=(422,),
        )
        self.check(
            g,
            "batch rejects empty inputs",
            "POST",
            f"{base}/invoke/batch",
            json_body={"inputs": []},
            expect=(422,),
        )
        self.check(g, "GET health", "GET", f"{base}/health")
        self.check(g, "GET card", "GET", f"{base}/card")
        self.check(g, "GET config", "GET", f"{base}/config")
        self.check(g, "GET prompts", "GET", f"{base}/prompts")

        self.sse(
            g,
            "POST invoke/stream (SSE)",
            f"{base}/invoke/stream",
            {"query": "stream me"},
        )

        # Threads: the Studio client drives this whole lifecycle. A deployment
        # with no store configured is a legitimate posture (the platform says
        # so with a 400), not a failure — report the whole group as skipped
        # rather than as broken, and say why.
        probe = self._req(
            "POST",
            f"{base}/threads",
            json_body={"user_id": "e2e-user", "title": "e2e thread"},
        )
        if probe is not None and probe.status_code == 400 and "storage" in probe.text.lower():
            self.thread_store_available = False
            self.report.skip(
                g,
                "thread lifecycle",
                "no store configured (set DATABASE_URL / AGENTOMATIC_LOGS_HISTORY)",
            )
            self.report.skip(g, "optimization-runs", "no store configured")
            return
        thread = self.check(
            g,
            "POST threads",
            "POST",
            f"{base}/threads",
            json_body={"user_id": "e2e-user", "title": "e2e thread"},
        )
        tid = None
        if isinstance(thread, dict):
            payload = thread.get("thread") or thread
            tid = payload.get("id") or payload.get("thread_id")
        if not tid:
            self.report.add(g, "thread id present", False, f"no id in {thread}")
            return
        self.report.add(g, "thread id present", True)
        self.check(g, "GET optimization-runs", "GET", f"{base}/optimization-runs")
        self.check(g, "GET threads", "GET", f"{base}/threads")
        self.check(g, "GET thread", "GET", f"{base}/threads/{tid}")
        self.check(
            g,
            "PATCH thread",
            "PATCH",
            f"{base}/threads/{tid}",
            json_body={"title": "renamed"},
        )
        # Post a message through chat so the thread has history to read back.
        self.check(
            g,
            "chat into thread",
            "POST",
            f"{base}/chat",
            json_body={"content": "remember this", "thread_id": tid, "user_id": "e2e-user"},
        )
        self.check(g, "GET messages", "GET", f"{base}/threads/{tid}/messages")
        self.check(g, "GET summary", "GET", f"{base}/threads/{tid}/summary")
        self.check(g, "GET lineage", "GET", f"{base}/threads/{tid}/lineage")
        self.check(g, "GET pending approvals", "GET", f"{base}/threads/{tid}/pending")
        self.check(
            g,
            "POST fork",
            "POST",
            f"{base}/threads/{tid}/fork",
            json_body={"message_index": 0},
        )
        self.check(
            g,
            "POST feedback",
            "POST",
            f"{base}/feedback",
            json_body={"thread_id": tid, "rating": 5, "comment": "e2e"},
            expect=(200, 201),
        )
        self.check(g, "GET feedback", "GET", f"{base}/feedback")
        self.check(g, "GET feedback/export", "GET", f"{base}/feedback/export")
        self.check(g, "DELETE messages", "DELETE", f"{base}/threads/{tid}/messages")
        self.check(g, "DELETE thread", "DELETE", f"{base}/threads/{tid}")

    def verify_a2a(self) -> None:
        """Agent-to-Agent discovery card and task protocol."""
        g = "a2a"
        # The well-known card is how a peer agent discovers this platform.
        self.check(
            g,
            "GET /.well-known/agent.json",
            "GET",
            "/.well-known/agent.json",
            validate=lambda p: "" if isinstance(p, dict) and p else "empty card",
        )
        base = f"/api/v1/{self.agent}/a2a"
        task = self.check(
            g,
            "POST a2a/tasks",
            "POST",
            f"{base}/tasks",
            json_body={
                "message": {"role": "user", "parts": [{"type": "text", "text": "a2a hello"}]}
            },
            expect=(200, 201, 202),
        )
        tid = self.task_id_from(task)
        if not tid and isinstance(task, dict):
            nested = task.get("task")
            tid = nested.get("id") if isinstance(nested, dict) else None
        if tid:
            self.report.add(g, "a2a task id", True)
            self.check(g, "GET a2a task", "GET", f"{base}/tasks/{tid}")
            # A fast local agent can complete between task creation and this
            # cancellation request.  In that race the API correctly returns
            # 409 rather than changing an already-succeeded task into a
            # cancelled one.  Verify that terminal state instead of treating
            # the expected race as an end-to-end failure.
            cancel_response = self._req("POST", f"{base}/tasks/{tid}/cancel")
            if cancel_response is None:
                self.report.add(g, "POST a2a cancel", False, "request failed")
                expected_statuses = frozenset({"cancelled"})
            elif cancel_response.status_code in (200, 202):
                self.report.add(g, "POST a2a cancel", True)
                expected_statuses = frozenset({"cancelled"})
            elif cancel_response.status_code == 409:
                self.report.add(g, "POST a2a cancel", True, "task already terminal")
                expected_statuses = _TERMINAL_OK | frozenset({"cancelled"})
            else:
                body = cancel_response.text[:300]
                self.report.add(
                    g,
                    "POST a2a cancel",
                    False,
                    f"POST {base}/tasks/{tid}/cancel → {cancel_response.status_code}: {body}",
                )
                expected_statuses = frozenset({"cancelled"})
            self.await_task(
                g,
                "a2a cancellation preserves a terminal state",
                task,
                expected_terminal_statuses=expected_statuses,
            )
            self.delete_created_task(g, "DELETE a2a task", {"id": tid})
        else:
            self.report.add(g, "a2a task id", False, f"no task id in {task}")

    def verify_plugins(self) -> None:
        """Plugin registry, model card and inference routes."""
        g = "plugins"
        p = self.plugin

        if not p:
            # An empty registry is valid for an agents-only deployment. The
            # component-specific contract below is deliberately opt-in so a
            # fresh project does not fail by probing invented placeholder
            # names such as ``scorer``.
            self.check(g, "GET /api/v1/plugins", "GET", "/api/v1/plugins")
            self.report.skip(g, "plugin routes", "no plugin configured")
            return

        def _list(payload: Any) -> str:
            items = payload if isinstance(payload, list) else payload.get("plugins", [])
            if not items:
                return "no plugins listed"
            return ""

        self.check(g, "GET /api/v1/plugins", "GET", "/api/v1/plugins", validate=_list)
        self.check(g, "GET model_card", "GET", f"/api/v1/plugins/{p}/model_card")
        self.check(g, "GET plugin health", "GET", f"/api/v1/plugins/{p}/health")
        self.check(
            g,
            "POST predict",
            "POST",
            f"/api/v1/plugins/{p}/predict",
            json_body={"text": "a reasonably long sentence"},
        )
        submitted = self.check(
            g,
            "POST predict/batch (202)",
            "POST",
            f"/api/v1/plugins/{p}/predict/batch",
            json_body={"inputs": [{"text": "one"}, {"text": "two"}]},
            expect=(202,),
        )
        self.await_task(g, "predict/batch completes", submitted)
        self.delete_created_task(g, "DELETE predict/batch task", submitted)
        self.check(
            g,
            "POST plugin reload",
            "POST",
            f"/api/v1/plugins/{p}/reload",
            expect=(200, 202),
        )

    def verify_endpoints(self) -> None:
        """Custom endpoint mounting and invocation."""
        g = "endpoints"
        self.check(g, "GET /api/v1/endpoints", "GET", "/api/v1/endpoints")
        e = self.endpoint
        if e:
            self.check(g, "GET endpoint info", "GET", f"/api/v1/endpoints/{e}/info")
            self.check(g, "GET endpoint health", "GET", f"/api/v1/endpoints/{e}/health")
            self.check(
                g,
                "POST endpoint call",
                "POST",
                f"/api/v1/endpoints/{e}/call",
                json_body={"text": "shout"},
            )
        else:
            self.report.skip(g, "endpoint routes", "no endpoint configured")
        read_name = self.read_endpoint
        if not read_name:
            return
        info = self.check(
            g,
            "GET-only endpoint info",
            "GET",
            f"/api/v1/endpoints/{read_name}/info",
        )
        path = info.get("path") if isinstance(info, dict) else None
        methods = (
            {str(method).upper() for method in info.get("methods", [])}
            if isinstance(info, dict)
            else set()
        )
        if not isinstance(path, str) or "GET" not in methods:
            self.report.add(
                g, "GET-only endpoint contract", False, "endpoint must publish a GET path"
            )
            return
        self.check(
            g,
            "GET-only endpoint query call",
            "GET",
            f"/api/v1/endpoints/{read_name}{path}?text=browser-schema-contract",
            validate=lambda body: (
                "response is not a JSON object" if not isinstance(body, dict) else ""
            ),
        )

    def verify_ingestion(self) -> None:
        """Ingestion registry and run routes."""
        g = "ingestion"
        # The Studio bundle uses this registry alias even without an ingestor.
        self.check(g, "GET /api/v1/ingestors", "GET", "/api/v1/ingestors")
        i = self.ingestor
        if not i:
            self.report.skip(g, "ingestor routes", "no ingestor configured")
            return
        self.check(g, "GET /api/v1/ingestion", "GET", "/api/v1/ingestion")
        self.check(g, "GET ingestor info", "GET", f"/api/v1/ingestion/{i}/info")
        self.check(g, "GET ingestor health", "GET", f"/api/v1/ingestion/{i}/health")

        def _run(p: Any) -> str:
            if not isinstance(p, dict):
                return "not an object"
            if p.get("status") not in ("succeeded", "success", "completed", "partial"):
                return f"status={p.get('status')!r} errors={p.get('errors')!r}"
            return ""

        self.check(
            g,
            "POST ingestor run",
            "POST",
            f"/api/v1/ingestion/{i}/run",
            json_body={"source": "inline://e2e verification document"},
            validate=_run,
        )

    def verify_pipelines(self) -> None:
        """Pipeline discovery, validation, visualisation and execution."""
        g = "pipelines"
        p = self.pipeline
        if not p:
            self.check(g, "GET /api/v1/pipelines", "GET", "/api/v1/pipelines")
            self.report.skip(g, "pipeline routes", "no pipeline configured")
            return

        def _list(payload: Any) -> str:
            items = payload if isinstance(payload, list) else payload.get("pipelines", [])
            if not items:
                return "no pipelines listed"
            return ""

        self.check(g, "GET /api/v1/pipelines", "GET", "/api/v1/pipelines", validate=_list)
        self.check(g, "GET pipeline config", "GET", f"/api/v1/pipelines/{p}/config")
        self.check(g, "GET pipeline validate", "GET", f"/api/v1/pipelines/{p}/validate")

        def _viz(payload: Any) -> str:
            if not isinstance(payload, dict) or not payload.get("mermaid"):
                return "no mermaid field"
            return ""

        self.check(
            g, "GET pipeline visualize", "GET", f"/api/v1/pipelines/{p}/visualize", validate=_viz
        )
        self.check(
            g,
            "POST pipeline run",
            "POST",
            f"/api/v1/pipelines/{p}/run",
            json_body={"input": {"query": "pipeline e2e"}},
        )
        self.check(
            g,
            "POST validate-draft",
            "POST",
            "/api/v1/pipelines/validate-draft",
            json_body={"yaml": "name: draft_check\nsteps:\n  - agent: " + self.agent + "\n"},
        )

    def verify_builder_lifecycle(self) -> None:
        """Prove a Studio-authored draft can persist, reload, execute and delete.

        This is deliberately opt-in: saving an arbitrary name in a production
        pipeline directory would be inappropriate.  The Docker fixture passes
        a dedicated ``--builder-smoke-name`` and supplies the endpoint/plugin
        resources used by the visual field-link draft.
        """
        g = "builder"
        name = self.builder_smoke_name
        if not name:
            self.report.skip(g, "builder lifecycle", "no --builder-smoke-name supplied")
            return
        if not self.endpoint or not self.plugin:
            self.report.skip(
                g,
                "builder lifecycle",
                "requires --endpoint and --plugin for the visual-link draft",
            )
            return

        draft = {
            "name": name,
            "description": "Disposable deployment verification pipeline.",
            "steps": [
                {
                    "name": "enrich",
                    "endpoint": self.endpoint,
                    "input": {"text": "builder field-link source"},
                },
                {
                    "name": "score",
                    "plugin": self.plugin,
                    "input": {"text": "$.steps.enrich.text"},
                },
            ],
        }
        encoded_name = quote(name, safe="")
        saved = self.check(
            g,
            "POST save visual-link draft",
            "POST",
            f"/api/v1/pipelines/{encoded_name}",
            json_body={"pipeline": draft},
            validate=lambda payload: (
                "save response was not valid"
                if not isinstance(payload, dict) or payload.get("valid") is not True
                else ""
            ),
        )
        if not isinstance(saved, dict) or saved.get("valid") is not True:
            return

        try:
            config = self.check(
                g, "GET saved Builder draft", "GET", f"/api/v1/pipelines/{encoded_name}/config"
            )
            raw_input = (
                ((config or {}).get("steps") or [{}, {}])[1].get("input")
                if isinstance(config, dict)
                else None
            )
            mappings = (
                raw_input.get("mappings", raw_input) if isinstance(raw_input, dict) else None
            )
            self.report.add(
                g,
                "visual field link persisted",
                isinstance(mappings, dict) and mappings.get("text") == "$.steps.enrich.text",
                ""
                if isinstance(mappings, dict) and mappings.get("text") == "$.steps.enrich.text"
                else f"mapping={mappings!r}",
            )
            self.check(
                g,
                "POST run saved Builder draft",
                "POST",
                f"/api/v1/pipelines/{encoded_name}/run",
                json_body={"input": {}},
                validate=lambda payload: (
                    "Builder draft did not succeed"
                    if not isinstance(payload, dict) or payload.get("status") != "success"
                    else ""
                ),
            )
        finally:
            self.check(
                g,
                "DELETE saved Builder draft",
                "DELETE",
                f"/api/v1/pipelines/{encoded_name}",
                expect=(200, 204),
            )

    def verify_every_pipeline(self) -> None:
        """Run *every* published pipeline, not just the sampled one.

        ``verify_pipelines`` proves the routes work against one pipeline. A
        deployment usually publishes several, each built from different step
        types, and a step type that only ever validates is not a step type
        that runs. This executes them all and reports which step types were
        actually exercised.
        """
        g = "pipelines-all"
        listing = self.check(g, "GET /api/v1/pipelines", "GET", "/api/v1/pipelines")
        if listing is None:
            return
        items = listing if isinstance(listing, list) else listing.get("pipelines", [])
        names = [i["name"] if isinstance(i, dict) else str(i) for i in items]
        if not names:
            self.report.skip(g, "run every pipeline", "no pipelines published")
            return

        seen: set[str] = set()
        for name in sorted(names):
            cfg = self.check(g, f"config: {name}", "GET", f"/api/v1/pipelines/{name}/config")
            if isinstance(cfg, dict):
                _collect_step_types(cfg.get("steps") or [], seen)

            def _ran(payload: Any) -> str:
                if not isinstance(payload, dict):
                    return "non-dict result"
                if payload.get("status") != "success":
                    return f"status={payload.get('status')} {str(payload.get('error'))[:120]}"
                bad = {
                    n: st.get("status")
                    for n, st in (payload.get("steps") or {}).items()
                    if st.get("status") not in ("success", "skipped")
                }
                return f"steps not ok: {bad}" if bad else ""

            self.check(
                g,
                f"run: {name}",
                "POST",
                f"/api/v1/pipelines/{name}/run",
                json_body={"input": {"query": "one two three four five"}},
                validate=_ran,
            )

        for step_type in sorted(seen):
            self.report.add(g, f"step type executed: {step_type}", True)
        missing = sorted(set(_ALL_STEP_TYPES) - seen)
        if missing:
            self.report.skip(
                g, "full step-type coverage", f"not published by this deployment: {missing}"
            )

    def verify_isolation(self) -> None:
        """Concurrent callers must never see each other's data.

        Agents are singletons: every request gets the same instance. A class
        agent that parks per-run data on ``self`` instead of in its state
        dataclass serves caller A's answer to caller B — a failure that is
        invisible to sequential testing and is the worst kind to ship.

        Each request carries a marker unique to it; a response containing
        somebody else's marker is a leak.
        """
        g = "isolation"
        # The persistent-store branch submits two concurrent batches.  Do not
        # let a verifier that has already exercised other real resources turn
        # a legitimate rate limit into a fake isolation failure.  The normal
        # deployment still uses 24 callers; a lower advertised remaining
        # budget scales the fan-out while retaining a real concurrent test.
        fanout = 24
        remaining = self._remaining_budget()
        if remaining is not None:
            fanout = min(fanout, max(2, (remaining - 4) // 2))
        self.report.add(
            g,
            "concurrency budget",
            fanout >= 2,
            f"{fanout} parallel callers" if remaining is not None else "rate limit not advertised",
        )
        run = f"{int(time.time()):x}"

        def marker(i: int) -> str:
            return f"MK{i:04d}{run}ZQ"

        def one(i: int) -> tuple[int, int, str]:
            resp = self._req(
                "POST",
                f"/api/v1/{self.agent}/invoke",
                json_body={"query": f"Reply with exactly {marker(i)} and nothing else."},
            )
            if resp is None:
                return i, 0, ""
            return i, resp.status_code, resp.text

        with ThreadPoolExecutor(max_workers=fanout) as pool:
            results = list(pool.map(one, range(fanout)))

        bad = [(i, code) for i, code, _ in results if code != 200]
        self.report.add(
            g, f"{fanout} concurrent invokes all succeeded", not bad, f"non-200: {bad[:5]}"
        )

        leaked = [
            (marker(i), [marker(j) for j in range(fanout) if j != i and marker(j) in body][:3])
            for i, code, body in results
            if code == 200 and any(marker(j) in body for j in range(fanout) if j != i)
        ]
        self.report.add(
            g, "no response carried another caller's marker", not leaked, str(leaked[:3])
        )

        if not self.thread_store_available:
            self.report.skip(g, "concurrent threads stay separate", "no thread store")
            return

        def one_chat(i: int) -> tuple[int, int, Any]:
            resp = self._req(
                "POST",
                f"/api/v1/{self.agent}/chat",
                json_body={"content": f"Token {marker(i)}", "thread_id": f"iso-{marker(i)}"},
            )
            if resp is None:
                return i, 0, None
            try:
                return i, resp.status_code, resp.json()
            except Exception:  # noqa: BLE001 - non-JSON body is a failure below
                return i, resp.status_code, None

        with ThreadPoolExecutor(max_workers=fanout) as pool:
            chats = list(pool.map(one_chat, range(fanout)))

        wrong = [
            (marker(i), (body or {}).get("thread_id"))
            for i, code, body in chats
            if code == 200 and (body or {}).get("thread_id") != f"iso-{marker(i)}"
        ]
        self.report.add(g, "each reply came back on its own thread", not wrong, str(wrong[:3]))

        borrowed = [
            (marker(i), (body or {}).get("history_loaded"))
            for i, code, body in chats
            if code == 200 and ((body or {}).get("history_loaded") or 0) != 0
        ]
        self.report.add(
            g, "no fresh thread picked up another's history", not borrowed, str(borrowed[:3])
        )

    def verify_tasks(self) -> None:
        """Async task manager routes for every deployed resource type.

        The specialised ``/invoke/async`` route below proves the legacy
        agent sugar works.  Studio's Task Board uses the generic ``/tasks``
        route instead, so explicit component flags also exercise that route
        for plugins, endpoints, ingestors, and pipelines.  These inputs
        intentionally mirror the fixture payloads used by their direct route
        checks above; a deployment that does not opt into a resource name is
        never asked to run an unknown workload.
        """
        g = "tasks"
        self.check(g, "GET /api/v1/tasks", "GET", "/api/v1/tasks")

        # The Task Board submits through this generic route.  When an agent
        # publishes required inputs, it must reject an incomplete payload
        # *before* persisting or running a task, just like its /invoke route.
        agent_schemas = self._req(
            "GET", f"/studio/agents/{quote(self.agent, safe='')}/schemas"
        )
        input_schema: Any = None
        if agent_schemas is not None and agent_schemas.status_code == 200:
            try:
                input_schema = agent_schemas.json().get("input_schema")
            except Exception:  # noqa: BLE001
                input_schema = None
        required_fields = (
            input_schema.get("required", []) if isinstance(input_schema, dict) else []
        )
        if isinstance(required_fields, list) and required_fields:
            invalid_task = self._req(
                "POST",
                "/api/v1/tasks",
                json_body={"target_type": "agent", "target": self.agent, "input": {}},
            )
            self.report.add(
                g,
                "generic task enforces required agent input",
                invalid_task is not None and invalid_task.status_code == 422,
                ""
                if invalid_task is not None and invalid_task.status_code == 422
                else f"→ {getattr(invalid_task, 'status_code', 'transport error')}",
            )
        else:
            self.report.skip(
                g,
                "generic task required-input rejection",
                "selected agent publishes no required input fields",
            )

        created = self.check(
            g,
            "POST agent invoke/async",
            "POST",
            f"/api/v1/{self.agent}/invoke/async",
            json_body={"query": "async e2e"},
            expect=(200, 201, 202),
        )
        task_id = None
        if isinstance(created, dict):
            task_id = created.get("task_id") or created.get("id")
        if not task_id:
            self.report.add(g, "async task id", False, f"no task id in {created}")
            return
        self.report.add(g, "async task id", True)
        self.check(g, "GET task", "GET", f"/api/v1/tasks/{task_id}")
        self.await_task(g, "async task completes", created)
        self.check(g, "GET task result", "GET", f"/api/v1/tasks/{task_id}/result", expect=(200,))
        self.check(g, "DELETE task", "DELETE", f"/api/v1/tasks/{task_id}", expect=(200, 204))

        def verify_generic_task(
            label: str,
            target_type: str,
            target: str,
            input_payload: dict[str, Any],
        ) -> None:
            """Submit, inspect and remove one real Task Board workload."""
            submitted = self.check(
                g,
                f"POST generic {label} task (sync)",
                "POST",
                "/api/v1/tasks",
                json_body={
                    "target_type": target_type,
                    "target": target,
                    "input": input_payload,
                    "mode": "sync",
                    "wait": True,
                },
                expect=(200,),
                validate=lambda payload: (
                    "task did not succeed"
                    if not isinstance(payload, dict) or payload.get("status") not in _TERMINAL_OK
                    else ""
                ),
            )
            task_id = (submitted or {}).get("id") or (submitted or {}).get("task_id")
            if not task_id:
                self.report.add(g, f"generic {label} task id", False, f"no task id in {submitted}")
                return
            self.check(
                g, f"GET generic {label} task result", "GET", f"/api/v1/tasks/{task_id}/result"
            )
            self.check(
                g,
                f"DELETE generic {label} task",
                "DELETE",
                f"/api/v1/tasks/{task_id}",
                expect=(200, 204),
            )

        if self.plugin:
            verify_generic_task(
                "plugin",
                "plugin",
                self.plugin,
                {"text": "generic task plugin e2e"},
            )
        if self.endpoint:
            verify_generic_task(
                "endpoint",
                "endpoint",
                self.endpoint,
                {"text": "generic task endpoint e2e"},
            )
        if self.ingestor:
            verify_generic_task(
                "ingestor",
                "ingestion",
                self.ingestor,
                {"source": "inline://generic task ingestor e2e"},
            )
        if self.pipeline:
            verify_generic_task(
                "pipeline",
                "pipeline",
                self.pipeline,
                {"query": "generic task pipeline e2e"},
            )

    def verify_control_plane(self) -> None:
        """Control-plane read and mutate routes."""
        g = "control-plane"
        info = self._req("GET", "/api/v1/control")
        if info is None or info.status_code == 404:
            self.report.skip(g, "control plane", "not enabled on this deployment")
            return
        self.check(g, "GET /api/v1/control", "GET", "/api/v1/control")
        self.check(g, "GET control/agents", "GET", "/api/v1/control/agents")
        self.check(g, "GET control/agent", "GET", f"/api/v1/control/agents/{self.agent}")
        self.check(g, "GET control/endpoints", "GET", "/api/v1/control/endpoints")

        def _connection_probe_contract(document: Any) -> str:
            """Keep the schema-driven Connections page tied to real OpenAPI."""
            try:
                schema = document["paths"]["/api/v1/control/connections/{scope}/{name}"]["get"][
                    "responses"
                ]["200"]["content"]["application/json"]["schema"]
            except (KeyError, TypeError):
                return "missing GET connection probe response schema"
            reference = schema.get("$ref") if isinstance(schema, dict) else ""
            return (
                ""
                if isinstance(reference, str) and reference.endswith("/ControlConnectionProbe")
                else (f"unexpected connection probe schema: {schema!r}")
            )

        self.check(
            g,
            "OpenAPI connection probe contract",
            "GET",
            "/openapi.json",
            validate=_connection_probe_contract,
        )
        connections = self.check(
            g, "GET control/connections", "GET", "/api/v1/control/connections"
        )
        if isinstance(connections, list):
            first = next(
                (
                    (str(scope.get("scope")), str(name))
                    for scope in connections
                    if isinstance(scope, dict)
                    for name in (scope.get("connections") or {})
                ),
                None,
            )
            if first is None:
                self.report.skip(g, "GET control/connection probe", "no configured connections")
            else:
                scope, name = first
                self.check(
                    g,
                    "GET control/connection probe",
                    "GET",
                    f"/api/v1/control/connections/{quote(scope, safe='')}/{quote(name, safe='')}",
                    validate=lambda payload: (
                        "missing connection result"
                        if not isinstance(payload, dict) or payload.get("connection") != name
                        else ""
                    ),
                )
        self.check(g, "GET control/health", "GET", "/api/v1/control/health")
        self.check(g, "GET control/config", "GET", "/api/v1/control/config")
        self.check(g, "GET control/metrics/summary", "GET", "/api/v1/control/metrics/summary")

        ctl = {"X-Control-Token": self.control_token} if self.control_token else None
        self.check(
            g,
            "POST agent disable",
            "POST",
            f"/api/v1/control/agents/{self.agent}/disable",
            headers=ctl,
        )
        # A disabled agent must actually stop serving traffic.
        resp = self._req("POST", f"/api/v1/{self.agent}/invoke", json_body={"query": "x"})
        ok = resp is not None and resp.status_code in (403, 404, 503)
        self.report.add(
            g,
            "disabled agent refuses traffic",
            ok,
            "" if ok else f"invoke → {getattr(resp, 'status_code', 'transport error')}",
        )
        self.check(
            g,
            "POST agent enable",
            "POST",
            f"/api/v1/control/agents/{self.agent}/enable",
            headers=ctl,
        )
        resp = self._req("POST", f"/api/v1/{self.agent}/invoke", json_body={"query": "x"})
        ok = resp is not None and resp.status_code == 200
        self.report.add(
            g,
            "re-enabled agent serves traffic",
            ok,
            "" if ok else f"invoke → {getattr(resp, 'status_code', 'transport error')}",
        )
        self.check(
            g,
            "POST maintenance on",
            "POST",
            "/api/v1/control/maintenance",
            json_body={"enabled": True},
            headers=ctl,
        )
        self.check(
            g,
            "POST maintenance off",
            "POST",
            "/api/v1/control/maintenance",
            json_body={"enabled": False},
            headers=ctl,
        )

    def verify_metrics(self) -> None:
        """Prometheus exposition."""
        g = "metrics"
        resp = self._req("GET", "/metrics")
        if resp is None:
            self.report.add(g, "GET /metrics", False, "transport error")
            return
        if resp.status_code == 404:
            self.report.skip(g, "GET /metrics", "metrics disabled on this deployment")
            return
        ok = resp.status_code == 200 and b"# HELP" in resp.content
        self.report.add(
            g,
            "GET /metrics",
            ok,
            "" if ok else f"→ {resp.status_code}, body starts {resp.content[:80]!r}",
        )
        ok = b"agentomatic" in resp.content.lower()
        self.report.add(
            g, "metrics include agentomatic series", ok, "" if ok else "no agentomatic_* series"
        )

    def verify_auth(self) -> None:
        """Auth enforcement, when the deployment is configured for it."""
        g = "auth"
        if not self.expect_auth:
            self.report.skip(g, "auth enforcement", "deployment runs without auth")
            return
        anon = httpx.Client(base_url=self.base, timeout=15.0)

        def _anon(headers: dict[str, str] | None = None) -> httpx.Response:
            """POST as an anonymous caller, waiting out the shared rate limit.

            The anonymous client shares this harness's source IP, so it shares
            the limiter's per-IP budget — a 429 here says nothing about auth.
            """
            for attempt in range(_RATE_LIMIT_RETRIES):
                got = anon.post(
                    f"/api/v1/{self.agent}/invoke", json={"query": "x"}, headers=headers
                )
                if got.status_code != 429 or attempt == _RATE_LIMIT_RETRIES - 1:
                    return got
                time.sleep(
                    min(float(got.headers.get("Retry-After") or 1), _RATE_LIMIT_MAX_WAIT)
                    + _RATE_LIMIT_RETRY_PAD
                )
            return got

        try:
            # Protected route must reject an anonymous caller.
            resp = _anon()
            ok = resp.status_code in (401, 403)
            self.report.add(
                g, "anonymous invoke rejected", ok, "" if ok else f"→ {resp.status_code}"
            )
            # A wrong key must also be rejected.
            resp = _anon({"X-Api-Key": "definitely-wrong"})
            ok = resp.status_code in (401, 403)
            self.report.add(g, "bad key rejected", ok, "" if ok else f"→ {resp.status_code}")
            # Every probe route must stay open: an orchestrator carries no
            # credentials, and a readiness probe that 401s keeps a pod out of
            # service for good while the platform looks healthy in its logs.
            for path in ("/health", "/ready", "/readiness"):
                resp = anon.get(path)
                ok = resp.status_code == 200
                self.report.add(
                    g, f"{path} stays public", ok, "" if ok else f"→ {resp.status_code}"
                )
        except Exception as exc:  # noqa: BLE001
            self.report.add(g, "auth checks", False, f"{type(exc).__name__}: {exc}")
        finally:
            anon.close()
        # And the correct key must work.
        self.check(
            g,
            "valid key accepted",
            "POST",
            f"/api/v1/{self.agent}/invoke",
            json_body={"query": "authed"},
        )

    def verify_error_contract(self) -> None:
        """Unknown resources must 404 cleanly, never 500."""
        g = "errors"
        for name, method, path, body in (
            ("unknown agent", "POST", "/api/v1/definitely_not_an_agent/invoke", {"query": "x"}),
            (
                "unknown generic agent task",
                "POST",
                "/api/v1/tasks",
                {"target_type": "agent", "target": "definitely_not_an_agent", "input": {"query": "x"}},
            ),
            ("unknown plugin", "GET", "/api/v1/plugins/nope/model_card", None),
            ("unknown pipeline", "GET", "/api/v1/pipelines/nope/config", None),
            ("unknown endpoint", "GET", "/api/v1/endpoints/nope/info", None),
            ("unknown task", "GET", "/api/v1/tasks/00000000-0000-0000-0000-000000000000", None),
        ):
            resp = self._req(method, path, json_body=body)
            if resp is None:
                self.report.add(g, name, False, "transport error")
                continue
            ok = resp.status_code in (400, 401, 403, 404, 422)
            self.report.add(
                g,
                name,
                ok,
                "" if ok else f"{method} {path} → {resp.status_code} (expected 4xx)",
            )
        # Malformed body must be a 422, not a 500.
        resp = self._req("POST", f"/api/v1/{self.agent}/invoke", json_body={"query": 12345})
        ok = resp is not None and resp.status_code in (200, 422)
        self.report.add(
            g,
            "typed body validation",
            ok,
            "" if ok else f"→ {getattr(resp, 'status_code', 'transport error')}",
        )

    def verify_logs_history(self) -> None:
        """Per-agent invocation history, when the deployment records it."""
        g = "logs-history"
        base = f"/api/v1/{self.agent}"
        resp = self._req("GET", f"{base}/logs?limit=5")
        if resp is None:
            self.report.add(g, "GET logs", False, "transport error")
            return
        if resp.status_code == 400:
            self.report.skip(g, "logs history", "disabled (AGENTOMATIC_LOGS_HISTORY=0)")
            return

        def _logs(p: Any) -> str:
            entries = p.get("logs") if isinstance(p, dict) else p
            return "" if isinstance(entries, list) else f"no logs list: {p!r}"

        payload = self.check(g, "GET logs", "GET", f"{base}/logs?limit=5", validate=_logs)
        entries = (payload or {}).get("logs") if isinstance(payload, dict) else None
        # This deployment has served traffic already, so history must be non-empty.
        ok = bool(entries)
        self.report.add(
            g, "history records invocations", ok, "" if ok else "no entries after traffic"
        )
        if entries:
            log_id = entries[0].get("id")
            if log_id:
                self.check(g, "GET log by id", "GET", f"{base}/logs/{log_id}")

        # LLM analysis over those logs is opt-in and must say so when off.
        resp = self._req("GET", f"{base}/logs/analysis")
        if resp is not None and resp.status_code == 400:
            body = resp.text.lower()
            ok = "allow_logsllm_analysis" in body or "disabled" in body
            self.report.add(
                g,
                "log analysis refuses clearly when disabled",
                ok,
                "" if ok else f"unhelpful 400 body: {resp.text[:160]}",
            )
        else:
            self.check(g, "GET log analysis", "GET", f"{base}/logs/analysis")

    def verify_optimize(self) -> None:
        """The optimize-aware invoke path."""
        g = "optimize"
        self.check(
            g,
            "POST optimize/invoke",
            "POST",
            f"/api/v1/{self.agent}/optimize/invoke",
            json_body={"query": "optimize e2e"},
        )
        self.check(
            g,
            "GET optimization-runs",
            "GET",
            f"/api/v1/{self.agent}/optimization-runs",
            expect=(200, 400),
        )

    def verify_rate_limit(self) -> None:
        """Rate limiting, when the deployment enables it.

        Two properties matter in production and they pull in opposite
        directions: user traffic must be limited, and the endpoints that keep
        a pod alive — liveness/readiness probes and the metrics scrape — must
        never be. A kubelet and a scraper share one source IP with real
        traffic behind a NAT or ingress, so a limiter that counts them will
        restart healthy pods under load.
        """
        g = "rate-limit"
        resp = self._req("GET", "/api/v1/agents")
        if resp is None:
            self.report.add(g, "rate limit probe", False, "transport error")
            return
        if "X-RateLimit-Limit" not in resp.headers:
            self.report.skip(g, "rate limiting", "not enabled on this deployment")
            return
        limit = int(resp.headers.get("X-RateLimit-Limit") or 0)
        self.report.add(g, "advertises X-RateLimit-Limit", limit > 0, f"limit={limit}")

        # Probes and the scrape must survive well past the budget.
        for path in ("/health", "/ready", "/readiness"):
            codes = set()
            for _ in range(limit + 20):
                got = self._req("GET", path, honour_retry_after=False)
                codes.add(getattr(got, "status_code", "transport error"))
            ok = 429 not in codes
            self.report.add(g, f"{path} never throttled", ok, "" if ok else f"codes={codes}")
        codes = set()
        for _ in range(limit + 20):
            got = self._req("GET", "/metrics", honour_retry_after=False)
            codes.add(getattr(got, "status_code", "transport error"))
        ok = 429 not in codes
        self.report.add(g, "/metrics never throttled", ok, "" if ok else f"codes={codes}")

        # …and that flood must not have spent the caller's budget. Compare the
        # advertised remaining count either side of it rather than just calling
        # a route: by this point the harness has legitimately used budget of its
        # own, so "a normal call still works" would be measuring the wrong thing.
        before = self._remaining_budget()
        for path in ("/health", "/ready", "/readiness", "/metrics"):
            for _ in range(10):
                self._req("GET", path, honour_retry_after=False)
        after = self._remaining_budget()
        if before is None or after is None:
            self.report.skip(g, "probes do not consume the user budget", "no budget header")
        else:
            # `after` is read with one billed request, so allow that single unit.
            ok = after >= before - 1
            self.report.add(
                g,
                "probes do not consume the user budget",
                ok,
                "" if ok else f"remaining fell {before} → {after} across 40 probes",
            )

    def _remaining_budget(self) -> int | None:
        """Return the limiter's advertised remaining requests, if it advertises one."""
        resp = self._req("GET", "/api/v1/agents")
        if resp is None or "X-RateLimit-Remaining" not in resp.headers:
            return None
        try:
            return int(resp.headers["X-RateLimit-Remaining"])
        except ValueError:
            return None

    def run_all(self) -> Report:
        """Run every group and return the report."""
        self.verify_platform()
        self.verify_studio()
        self.verify_agent_rest()
        # Keep the concurrent isolation proof before the rest of the
        # resource matrix.  It still follows the thread-store probe in
        # verify_agent_rest, while a rate-limited deployment has enough
        # capacity for meaningful parallel callers.
        self.verify_isolation()
        self.verify_a2a()
        self.verify_plugins()
        self.verify_endpoints()
        self.verify_ingestion()
        self.verify_pipelines()
        self.verify_builder_lifecycle()
        self.verify_every_pipeline()
        self.verify_tasks()
        self.verify_logs_history()
        self.verify_optimize()
        self.verify_metrics()
        self.verify_rate_limit()
        # This enumerates every registered resource and can legitimately make
        # dozens of requests. Keep it after the fixed-window isolation and
        # rate-limit checks so their concurrent probes do not inherit the
        # matrix's already-spent request budget.
        self.verify_live_schema_contracts()
        self.verify_auth()
        self.verify_error_contract()
        # Control plane last: it toggles agent availability.
        self.verify_control_plane()
        return self.report

    def close(self) -> None:
        """Release the HTTP client."""
        self.client.close()


def main() -> int:
    """Parse arguments, run the suite, print the report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--control-token", default="")
    ap.add_argument(
        "--agent",
        default="",
        help="Agent to exercise (defaults to the first registered agent)",
    )
    ap.add_argument("--plugin", default="", help="Deployed plugin name to verify")
    ap.add_argument("--pipeline", default="", help="Deployed pipeline name to verify")
    ap.add_argument("--endpoint", default="", help="Deployed endpoint name to verify")
    ap.add_argument(
        "--read-endpoint",
        default="",
        help="GET-only endpoint to verify with a browser-safe query contract",
    )
    ap.add_argument("--ingestor", default="")
    ap.add_argument(
        "--builder-smoke-name",
        default="",
        help="Disposable pipeline name for the opt-in Builder save/reload/run/delete check",
    )
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--expect-auth", action="store_true")
    ap.add_argument("--no-studio", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="Seconds to wait for the server to answer /health before starting.",
    )
    args = ap.parse_args()

    if args.wait:
        deadline = time.time() + args.wait
        while time.time() < deadline:
            try:
                r = httpx.get(f"{args.base_url.rstrip('/')}/health", timeout=5.0)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)

    agent = args.agent or discover_agent(args.base_url, args.api_key, args.timeout)
    if not agent:
        ap.error(
            "Could not discover an agent from /api/v1/agents. "
            "Pass --agent NAME and ensure the server is reachable/authenticated."
        )

    v = Verifier(
        base_url=args.base_url,
        api_key=args.api_key,
        control_token=args.control_token,
        agent=agent,
        plugin=args.plugin,
        pipeline=args.pipeline,
        endpoint=args.endpoint,
        read_endpoint=args.read_endpoint,
        ingestor=args.ingestor,
        builder_smoke_name=args.builder_smoke_name,
        timeout=args.timeout,
        expect_auth=args.expect_auth,
        expect_studio=not args.no_studio,
    )
    try:
        report = v.run_all()
    finally:
        v.close()

    print(report.render())
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                [
                    {
                        "group": c.group,
                        "name": c.name,
                        "ok": c.ok,
                        "skipped": c.skipped,
                        "detail": c.detail,
                    }
                    for c in report.checks
                ],
                fh,
                indent=2,
            )
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
