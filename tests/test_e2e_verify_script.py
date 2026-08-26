"""Regression coverage for optional-component handling in the live verifier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_verifier_module() -> Any:
    """Load the standalone deployment verifier without making ``scripts`` a package."""
    path = Path(__file__).parents[1] / "scripts" / "e2e_verify.py"
    spec = importlib.util.spec_from_file_location("_agentomatic_e2e_verify", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verifier_skips_unconfigured_optional_components() -> None:
    """Agents-only deployments must not probe placeholder component names."""
    verifier_module = _load_verifier_module()
    verifier = verifier_module.Verifier(
        "http://example.invalid",
        agent="greeter",
        plugin="",
        endpoint="",
        ingestor="",
        pipeline="",
    )
    calls: list[str] = []

    def check(_group: str, _name: str, _method: str, path: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(path)
        return {}

    verifier.check = check
    try:
        verifier.verify_plugins()
        verifier.verify_endpoints()
        verifier.verify_ingestion()
        verifier.verify_pipelines()
    finally:
        verifier.close()

    assert "/api/v1/plugins/scorer/model_card" not in calls
    assert "/api/v1/endpoints/echo/info" not in calls
    assert "/api/v1/ingestion" not in calls
    assert "/api/v1/pipelines/basic_flow/config" not in calls
    skipped = {check.name for check in verifier.report.checks if check.skipped}
    assert {"plugin routes", "endpoint routes", "ingestor routes", "pipeline routes"} <= skipped


def test_verifier_usage_uses_the_project_python_environment() -> None:
    """The documented command must have the verifier's HTTP dependency available."""
    module = _load_verifier_module()

    assert "uv run python scripts/e2e_verify.py" in module.__doc__


def test_verifier_waits_past_integer_retry_after_rounding() -> None:
    """A retry cushion must exceed a truncated fractional rate-limit window."""
    module = _load_verifier_module()

    assert module._RATE_LIMIT_RETRY_PAD > 1.0


def test_verifier_requires_the_connections_page_probe_openapi_contract() -> None:
    """The live verifier must guard Studio's schema source, not only its GET."""
    module = _load_verifier_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "OpenAPI connection probe contract" in source
    assert "/api/v1/control/connections/{scope}/{name}" in source
    assert "ControlConnectionProbe" in source


def test_verifier_audits_discovered_resource_schema_contracts() -> None:
    """Dynamic Studio forms need every registered resource, not fixture names."""
    module = _load_verifier_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "def verify_live_schema_contracts" in source
    assert 'g = "schema-contracts"' in source
    assert "/studio/agents" in source
    assert "/api/v1/plugins" in source
    assert "/api/v1/endpoints" in source
    assert "/api/v1/ingestors" in source
    assert "/api/v1/pipelines" in source
    assert "self.verify_live_schema_contracts()" in source
    assert source.index("self.verify_rate_limit()") < source.index(
        "self.verify_live_schema_contracts()"
    )


def test_verifier_removes_only_task_records_it_created() -> None:
    """A post-deploy test run must not clutter an operator's Task Board."""
    module = _load_verifier_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "def delete_created_task" in source
    assert '"DELETE invoke/batch task"' in source
    assert '"DELETE predict/batch task"' in source
    assert '"DELETE a2a task"' in source


def test_verifier_handles_the_a2a_cancellation_completion_race() -> None:
    """A task that already completed before cancellation remains a valid outcome."""
    module = _load_verifier_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert '"a2a cancellation preserves a terminal state"' in source
    assert 'expected_statuses = _TERMINAL_OK | frozenset({"cancelled"})' in source
    assert "task already terminal" in source


def test_verifier_paces_isolation_against_the_advertised_rate_budget() -> None:
    """Concurrent isolation must not fail merely because earlier checks spent the window."""
    module = _load_verifier_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert '"concurrency budget"' in source
    assert "fanout = min(fanout, max(2, (remaining - 4) // 2))" in source
    assert source.index("self.verify_agent_rest()") < source.index("self.verify_isolation()")
    assert source.index("self.verify_isolation()") < source.index("self.verify_a2a()")


def test_verifier_exercises_a_get_only_endpoint_with_query_input() -> None:
    """Read-only endpoint coverage must not accidentally fall back to POST."""
    verifier_module = _load_verifier_module()
    verifier = verifier_module.Verifier(
        "http://example.invalid",
        agent="greeter",
        read_endpoint="lookup",
    )
    calls: list[tuple[str, str]] = []

    def check(_group: str, _name: str, method: str, path: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append((method, path))
        if path.endswith("/lookup/info"):
            return {"path": "/search", "methods": ["GET"]}
        return {"echoed": "ok"}

    verifier.check = check
    try:
        verifier.verify_endpoints()
    finally:
        verifier.close()

    assert ("GET", "/api/v1/endpoints/lookup/search?text=browser-schema-contract") in calls
    assert not any(method == "POST" and "/lookup/" in path for method, path in calls)


def test_verifier_exercises_generic_task_dispatchers_for_explicit_resources() -> None:
    """Task Board coverage must exercise every resource type supplied to the verifier."""
    verifier_module = _load_verifier_module()
    verifier = verifier_module.Verifier(
        "http://example.invalid",
        agent="greeter",
        plugin="scorer",
        endpoint="enricher",
        ingestor="documents",
        pipeline="flow",
    )
    requests: list[dict[str, Any]] = []

    def check(_group: str, name: str, _method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if name == "POST agent invoke/async":
            return {"id": "agent"}
        if path == "/api/v1/tasks" and kwargs.get("json_body"):
            body = kwargs["json_body"]
            requests.append(body)
            task_id = body.get("target_type", "agent")
            return {"id": task_id, "status": "succeeded"}
        return {}

    verifier.check = check
    verifier.await_task = lambda *_args, **_kwargs: {"status": "succeeded"}
    try:
        verifier.verify_tasks()
    finally:
        verifier.close()

    generic = [request for request in requests if request.get("wait") is True]
    assert [(request["target_type"], request["target"]) for request in generic] == [
        ("plugin", "scorer"),
        ("endpoint", "enricher"),
        ("ingestion", "documents"),
        ("pipeline", "flow"),
    ]
    assert all(request["mode"] == "sync" for request in generic)


def test_verifier_exercises_opt_in_builder_persistence_lifecycle() -> None:
    """The deployment verifier must retain the Builder's save/reload/run/delete proof."""
    verifier_module = _load_verifier_module()
    verifier = verifier_module.Verifier(
        "http://example.invalid",
        plugin="scorer",
        endpoint="enricher",
        builder_smoke_name="builder_verify",
    )
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def check(_group: str, _name: str, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((method, path, kwargs.get("json_body")))
        if method == "POST" and path.endswith("/builder_verify"):
            return {"name": "builder_verify", "valid": True}
        if method == "GET" and path.endswith("/builder_verify/config"):
            return {
                "steps": [
                    {"name": "enrich", "endpoint": "enricher"},
                    {"name": "score", "input": {"mappings": {"text": "$.steps.enrich.text"}}},
                ]
            }
        if method == "POST" and path.endswith("/builder_verify/run"):
            return {"status": "success"}
        return {}

    verifier.check = check
    try:
        verifier.verify_builder_lifecycle()
    finally:
        verifier.close()

    assert calls[0] == (
        "POST",
        "/api/v1/pipelines/builder_verify",
        {
            "pipeline": {
                "name": "builder_verify",
                "description": "Disposable deployment verification pipeline.",
                "steps": [
                    {
                        "name": "enrich",
                        "endpoint": "enricher",
                        "input": {"text": "builder field-link source"},
                    },
                    {
                        "name": "score",
                        "plugin": "scorer",
                        "input": {"text": "$.steps.enrich.text"},
                    },
                ],
            }
        },
    )
    assert ("DELETE", "/api/v1/pipelines/builder_verify", None) in calls
    assert not verifier.report.failures


def test_discover_agent_accepts_registry_object_and_list(monkeypatch: Any) -> None:
    """No-argument verification must target a deployed agent, not a fixture name."""
    verifier_module = _load_verifier_module()

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"agents": {"coordinator": {"slug": "coordinator"}}}

    monkeypatch.setattr(verifier_module.httpx, "get", lambda *_args, **_kwargs: _Response())
    assert verifier_module.discover_agent("http://example.invalid", "key", 1.0) == "coordinator"

    class _ListResponse(_Response):
        def json(self) -> list[dict[str, str]]:
            return [{"slug": "writer"}]

    monkeypatch.setattr(verifier_module.httpx, "get", lambda *_args, **_kwargs: _ListResponse())
    assert verifier_module.discover_agent("http://example.invalid", "key", 1.0) == "writer"
