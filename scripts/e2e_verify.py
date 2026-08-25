#!/usr/bin/env python
"""End-to-end verification harness for a running Agentomatic platform.

Exercises every public surface the platform advertises — platform routes,
Studio (the exact calls the bundled React UI makes), agents, plugins,
endpoints, ingestion, pipelines, tasks, control plane, metrics and auth —
against a live server and reports a pass/fail table.

The harness is deployment-agnostic: point it at ``agentomatic run``, at
``uvicorn main:app``, or at a container published by ``agentomatic deploy``.

Usage::

    python scripts/e2e_verify.py --base-url http://localhost:8000 \
        --agent ag_basic --plugin scorer --pipeline basic_flow \
        --api-key secret --control-token tok --json report.json

Exit code is ``0`` only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

#: How many times to wait out a 429 before treating it as a failure.
_RATE_LIMIT_RETRIES = 3
#: Cap on a single Retry-After wait, so a long window cannot stall the run.
_RATE_LIMIT_MAX_WAIT = 65.0

#: Task statuses that mean the work finished successfully.
_TERMINAL_OK = frozenset({"completed", "succeeded", "success"})
#: Task statuses that mean the work finished unsuccessfully.
_TERMINAL_BAD = frozenset({"failed", "error", "cancelled", "canceled"})


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
        agent: str = "ag_basic",
        plugin: str = "scorer",
        pipeline: str = "basic_flow",
        endpoint: str = "echo",
        ingestor: str = "",
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
        self.ingestor = ingestor
        self.expect_auth = expect_auth
        self.expect_studio = expect_studio
        self.report = Report()
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
            time.sleep(min(delay, _RATE_LIMIT_MAX_WAIT) + 0.5)
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
            time.sleep(min(retry_after, _RATE_LIMIT_MAX_WAIT) + 0.5)
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

    def await_task(self, group: str, name: str, submitted: Any, *, timeout: float = 30.0) -> Any:
        """Poll a 202-submitted task to a terminal state and return its record.

        Args:
            group: Report grouping label.
            name: Human-readable check name.
            submitted: Decoded body of the 202 response.
            timeout: Seconds to wait for a terminal status.

        Returns:
            The final task record, or ``None`` when it never completed.
        """
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
                if status in _TERMINAL_OK or status in _TERMINAL_BAD:
                    break
            time.sleep(0.4)
        status = (record or {}).get("status")
        if status not in _TERMINAL_OK:
            err = (record or {}).get("error")
            self.report.add(group, name, False, f"status={status!r} error={err!r}")
            return record
        self.report.add(group, name, True)
        return record

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
        tid = None
        if isinstance(task, dict):
            tid = task.get("id") or task.get("task_id") or (task.get("task") or {}).get("id")
        if tid:
            self.report.add(g, "a2a task id", True)
            self.check(g, "GET a2a task", "GET", f"{base}/tasks/{tid}")
            self.check(
                g,
                "POST a2a cancel",
                "POST",
                f"{base}/tasks/{tid}/cancel",
                expect=(200, 202, 409),
            )
        else:
            self.report.add(g, "a2a task id", False, f"no task id in {task}")

    def verify_plugins(self) -> None:
        """Plugin registry, model card and inference routes."""
        g = "plugins"
        p = self.plugin

        def _list(payload: Any) -> str:
            items = payload if isinstance(payload, list) else payload.get("plugins", [])
            if not items:
                return "no plugins listed"
            return ""

        self.check(g, "GET /api/v1/plugins", "GET", "/api/v1/plugins", validate=_list)
        if not p:
            self.report.skip(g, "plugin routes", "no plugin configured")
            return
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
        if not e:
            self.report.skip(g, "endpoint routes", "no endpoint configured")
            return
        self.check(g, "GET endpoint info", "GET", f"/api/v1/endpoints/{e}/info")
        self.check(g, "GET endpoint health", "GET", f"/api/v1/endpoints/{e}/health")
        self.check(
            g,
            "POST endpoint call",
            "POST",
            f"/api/v1/endpoints/{e}/call",
            json_body={"payload": {"text": "shout"}},
        )

    def verify_ingestion(self) -> None:
        """Ingestion registry and run routes."""
        g = "ingestion"
        self.check(g, "GET /api/v1/ingestion", "GET", "/api/v1/ingestion")
        # The Studio bundle uses the /ingestors alias — both must exist.
        self.check(g, "GET /api/v1/ingestors", "GET", "/api/v1/ingestors")
        i = self.ingestor
        if not i:
            self.report.skip(g, "ingestor routes", "no ingestor configured")
            return
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

        def _list(payload: Any) -> str:
            items = payload if isinstance(payload, list) else payload.get("pipelines", [])
            if not items:
                return "no pipelines listed"
            return ""

        self.check(g, "GET /api/v1/pipelines", "GET", "/api/v1/pipelines", validate=_list)
        p = self.pipeline
        if not p:
            self.report.skip(g, "pipeline routes", "no pipeline configured")
            return
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

    def verify_tasks(self) -> None:
        """Async task manager routes."""
        g = "tasks"
        self.check(g, "GET /api/v1/tasks", "GET", "/api/v1/tasks")
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
        self.check(g, "GET control/connections", "GET", "/api/v1/control/connections")
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
                    min(float(got.headers.get("Retry-After") or 1), _RATE_LIMIT_MAX_WAIT) + 0.5
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
        self.verify_a2a()
        self.verify_plugins()
        self.verify_endpoints()
        self.verify_ingestion()
        self.verify_pipelines()
        self.verify_tasks()
        self.verify_logs_history()
        self.verify_optimize()
        self.verify_metrics()
        self.verify_rate_limit()
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
    ap.add_argument("--agent", default="ag_basic")
    ap.add_argument("--plugin", default="scorer")
    ap.add_argument("--pipeline", default="basic_flow")
    ap.add_argument("--endpoint", default="echo")
    ap.add_argument("--ingestor", default="")
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

    v = Verifier(
        base_url=args.base_url,
        api_key=args.api_key,
        control_token=args.control_token,
        agent=args.agent,
        plugin=args.plugin,
        pipeline=args.pipeline,
        endpoint=args.endpoint,
        ingestor=args.ingestor,
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
