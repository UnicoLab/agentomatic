# pyright: reportMissingParameterType=none
"""Tests for local-model auto-detection in scaffolded stacks.

A generated ``stacks/local.yaml`` used to hard-code one model name, so a local
server holding anything else rejected the very first invoke with HTTP 400
(unknown model). The scaffold now asks the server what it serves.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentomatic.stacks.defaults import (
    DEFAULT_LOCAL_MODEL,
    default_local_model,
    probe_local_models,
)

SERVED_MODEL = "Llama-3.2-3B-Instruct-4bit"


def _make_handler(payload: bytes | None, status: int = 200) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if payload is None:
                self.send_response(500)
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            """Silence the default stderr access log."""

    return Handler


@pytest.fixture
def model_server() -> Iterator[str]:
    """Serve an OpenAI-compatible ``/v1/models`` response; yield its base URL."""
    body = json.dumps({"object": "list", "data": [{"id": SERVED_MODEL}]}).encode()
    server = HTTPServer(("127.0.0.1", 0), _make_handler(body))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


class TestProbeLocalModels:
    def test_reports_served_models(self, model_server: str) -> None:
        assert probe_local_models(model_server) == [SERVED_MODEL]

    def test_no_server_yields_empty_list(self) -> None:
        # Port 1 is reserved and never listening.
        assert probe_local_models("http://127.0.0.1:1/v1", timeout=0.5) == []

    def test_non_json_body_yields_empty_list(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _make_handler(b"<html>not json</html>"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert probe_local_models(f"http://127.0.0.1:{server.server_port}/v1") == []
        finally:
            server.shutdown()
            server.server_close()

    def test_server_error_yields_empty_list(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _make_handler(None))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert probe_local_models(f"http://127.0.0.1:{server.server_port}/v1") == []
        finally:
            server.shutdown()
            server.server_close()


class TestDefaultLocalModel:
    def test_env_var_wins_over_probe(
        self, model_server: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMLX_BASE_URL", model_server)
        monkeypatch.setenv("AGENTOMATIC_LOCAL_MODEL", "explicit-choice")
        assert default_local_model() == "explicit-choice"

    def test_probe_fills_in_when_no_env_var(
        self, model_server: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AGENTOMATIC_LOCAL_MODEL", raising=False)
        monkeypatch.delenv("OMLX_MODEL", raising=False)
        monkeypatch.setenv("OMLX_BASE_URL", model_server)
        assert default_local_model() == SERVED_MODEL

    def test_falls_back_when_nothing_is_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AGENTOMATIC_LOCAL_MODEL", raising=False)
        monkeypatch.delenv("OMLX_MODEL", raising=False)
        monkeypatch.setenv("OMLX_BASE_URL", "http://127.0.0.1:1/v1")
        assert default_local_model() == DEFAULT_LOCAL_MODEL

    def test_probe_can_be_disabled(
        self, model_server: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AGENTOMATIC_LOCAL_MODEL", raising=False)
        monkeypatch.delenv("OMLX_MODEL", raising=False)
        monkeypatch.setenv("OMLX_BASE_URL", model_server)
        assert default_local_model(probe=False) == DEFAULT_LOCAL_MODEL


def test_scaffolded_stack_uses_the_served_model(
    model_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the YAML written to disk names the model that is loaded."""
    monkeypatch.delenv("AGENTOMATIC_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("OMLX_MODEL", raising=False)
    monkeypatch.setenv("OMLX_BASE_URL", model_server)

    from agentomatic.stacks.defaults import get_default_stack_yaml

    assert SERVED_MODEL in get_default_stack_yaml("local")
