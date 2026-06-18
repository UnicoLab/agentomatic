"""Tests for middleware, decorators, config defaults, providers, and storage.

Targets low-coverage modules to boost overall test coverage above 55%.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────
# Config Defaults
# ─────────────────────────────────────────────────────────────────────
class TestConfigDefaults:
    """Verify config defaults are importable and sensible."""

    def test_defaults_exist(self):
        from agentomatic.config.defaults import (
            DEFAULT_API_PREFIX,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_PROVIDER,
            DEFAULT_LOG_LEVEL,
            DEFAULT_MAX_TOKENS,
            DEFAULT_TEMPERATURE,
        )

        assert DEFAULT_API_PREFIX == "/api/v1"
        assert DEFAULT_LOG_LEVEL == "INFO"
        assert DEFAULT_LLM_PROVIDER == "ollama"
        assert isinstance(DEFAULT_LLM_MODEL, str)
        assert 0.0 <= DEFAULT_TEMPERATURE <= 1.0
        assert DEFAULT_MAX_TOKENS > 0


# ─────────────────────────────────────────────────────────────────────
# Storage __init__ lazy imports
# ─────────────────────────────────────────────────────────────────────
class TestStorageLazyImport:
    """Test storage __init__ __getattr__ for lazy imports."""

    def test_import_sqlalchemy_store(self):
        from agentomatic.storage import SQLAlchemyStore

        assert SQLAlchemyStore is not None

    def test_import_checkpointer(self):
        from agentomatic.storage import AgentomaticCheckpointer

        assert AgentomaticCheckpointer is not None

    def test_import_unknown_raises(self):
        import agentomatic.storage as storage_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = storage_mod.NonExistentClass


# ─────────────────────────────────────────────────────────────────────
# Embeddings provider
# ─────────────────────────────────────────────────────────────────────
class TestEmbeddingsProvider:
    """Test the embeddings factory."""

    def setup_method(self):
        from agentomatic.providers.embeddings import reset_embeddings

        reset_embeddings()

    def teardown_method(self):
        from agentomatic.providers.embeddings import reset_embeddings

        reset_embeddings()

    def test_default_dummy_embeddings(self):
        from agentomatic.providers.embeddings import get_embeddings

        emb = get_embeddings()
        assert emb is not None

    def test_dummy_with_dimension(self):
        from agentomatic.providers.embeddings import get_embeddings

        emb = get_embeddings("dummy", dimension=384)
        assert emb is not None

    def test_singleton_returns_same(self):
        from agentomatic.providers.embeddings import get_embeddings

        emb1 = get_embeddings()
        emb2 = get_embeddings()
        assert emb1 is emb2

    def test_reset_clears_singleton(self):
        from agentomatic.providers.embeddings import get_embeddings, reset_embeddings

        get_embeddings()
        reset_embeddings()
        emb2 = get_embeddings()
        # After reset, new instance should be created
        assert emb2 is not None

    def test_ollama_provider_fallback(self):
        """If ollama not installed, should fallback to dummy."""
        from agentomatic.providers.embeddings import get_embeddings

        with patch.dict("sys.modules", {"langchain_ollama": None}):
            # This will try ollama, fail, and fallback
            emb = get_embeddings("ollama")
            assert emb is not None


# ─────────────────────────────────────────────────────────────────────
# Protocol Decorators
# ─────────────────────────────────────────────────────────────────────
class TestHandleApiErrors:
    """Test the handle_api_errors decorator."""

    @pytest.mark.asyncio
    async def test_success_passthrough(self):
        from agentomatic.protocols.decorators import handle_api_errors

        @handle_api_errors
        async def good_fn():
            return {"ok": True}

        result = await good_fn()
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_http_exception_reraise(self):
        from fastapi import HTTPException

        from agentomatic.protocols.decorators import handle_api_errors

        @handle_api_errors
        async def raises_http():
            raise HTTPException(status_code=404, detail="not found")

        with pytest.raises(HTTPException) as exc_info:
            await raises_http()
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_generic_exception_wrapped(self):
        from fastapi import HTTPException

        from agentomatic.protocols.decorators import handle_api_errors

        @handle_api_errors
        async def raises_generic():
            raise ValueError("boom")

        with pytest.raises(HTTPException) as exc_info:
            await raises_generic()
        assert exc_info.value.status_code == 500
        assert "boom" in exc_info.value.detail


class TestLogApiCall:
    """Test the log_api_call decorator."""

    @pytest.mark.asyncio
    async def test_success_logged(self):
        from agentomatic.protocols.decorators import log_api_call

        @log_api_call
        async def slow_fn():
            return 42

        result = await slow_fn()
        assert result == 42

    @pytest.mark.asyncio
    async def test_error_logged(self):
        from agentomatic.protocols.decorators import log_api_call

        @log_api_call
        async def fails():
            raise RuntimeError("err")

        with pytest.raises(RuntimeError):
            await fails()


class TestCreateStreamingResponse:
    """Test streaming response factory."""

    def test_creates_response(self):
        from agentomatic.protocols.decorators import create_streaming_response

        async def gen():
            yield "data: test\n\n"

        resp = create_streaming_response(gen(), agent_name="test-agent")
        assert resp.media_type == "text/event-stream"
        assert resp.headers.get("X-Agent") == "test-agent"
        assert resp.headers.get("Cache-Control") == "no-cache"


# ─────────────────────────────────────────────────────────────────────
# Auth Middleware
# ─────────────────────────────────────────────────────────────────────
class TestAuthMiddleware:
    """Test API key authentication middleware."""

    def _make_app(self, api_key="test-key"):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        from agentomatic.middleware.auth import AuthMiddleware

        async def home(request):
            return JSONResponse({"ok": True})

        async def health(request):
            return JSONResponse({"status": "healthy"})

        app = Starlette(
            routes=[
                Route("/", home),
                Route("/health", health),
                Route("/api/data", home),
            ],
        )
        app.add_middleware(AuthMiddleware, api_key=api_key)
        return app

    def test_skip_paths(self):
        client = TestClient(self._make_app())
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_missing_key_rejected(self):
        client = TestClient(self._make_app())
        resp = client.get("/api/data")
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    def test_wrong_key_rejected(self):
        client = TestClient(self._make_app())
        resp = client.get("/api/data", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_correct_header_key(self):
        client = TestClient(self._make_app())
        resp = client.get("/api/data", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_correct_query_param(self):
        client = TestClient(self._make_app())
        resp = client.get("/api/data?api_key=test-key")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────
# Rate Limit Middleware
# ─────────────────────────────────────────────────────────────────────
class TestRateLimitMiddleware:
    """Test sliding-window rate limiter."""

    def _make_app(self, max_requests=3, window_seconds=60):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        from agentomatic.middleware.rate_limit import RateLimitMiddleware

        async def home(request):
            return JSONResponse({"ok": True})

        async def health(request):
            return JSONResponse({"healthy": True})

        app = Starlette(
            routes=[
                Route("/api/test", home),
                Route("/health", health),
            ],
        )
        app.add_middleware(
            RateLimitMiddleware,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )
        return app

    def test_skip_paths(self):
        client = TestClient(self._make_app())
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_allowed_within_limit(self):
        client = TestClient(self._make_app(max_requests=5))
        for _ in range(5):
            resp = client.get("/api/test")
            assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    def test_rate_limited_after_max(self):
        client = TestClient(self._make_app(max_requests=2))
        client.get("/api/test")
        client.get("/api/test")
        resp = client.get("/api/test")
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]
        assert "Retry-After" in resp.headers

    def test_rate_limit_headers(self):
        client = TestClient(self._make_app(max_requests=10))
        resp = client.get("/api/test")
        assert resp.headers["X-RateLimit-Limit"] == "10"
        assert resp.headers["X-RateLimit-Remaining"] == "9"


# ─────────────────────────────────────────────────────────────────────
# Metrics Middleware (without prometheus)
# ─────────────────────────────────────────────────────────────────────
class TestMetricsMiddleware:
    """Test metrics middleware with prometheus."""

    _counter = 0

    def _make_app(self):
        TestMetricsMiddleware._counter += 1
        prefix = f"test_metrics_{TestMetricsMiddleware._counter}"

        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        from agentomatic.middleware.metrics import MetricsMiddleware

        async def home(request):
            return JSONResponse({"ok": True})

        app = Starlette(
            routes=[
                Route("/api/test", home),
                Route("/health", home),
            ],
        )
        app.add_middleware(MetricsMiddleware, prefix=prefix)
        return app

    def test_normal_request(self):
        client = TestClient(self._make_app())
        resp = client.get("/api/test")
        assert resp.status_code == 200

    def test_skip_health_path(self):
        """Health path is in skip list, should pass through."""
        client = TestClient(self._make_app())
        resp = client.get("/health")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────
# Studio Decorators
# ─────────────────────────────────────────────────────────────────────
class TestStudioDecorators:
    """Test studio graph/state/stream decorators."""

    def test_studio_graph_marks_function(self):
        from agentomatic.studio.decorators import studio_graph

        @studio_graph
        def my_graph():
            return {"nodes": [], "edges": []}

        assert my_graph._is_studio_graph is True
        result = my_graph()
        assert "nodes" in result

    def test_studio_state_marks_function(self):
        from agentomatic.studio.decorators import studio_state

        @studio_state
        def my_state(thread_id):
            return {"messages": []}

        assert my_state._is_studio_state is True
        result = my_state("t1")
        assert "messages" in result

    def test_studio_stream_marks_function(self):
        from agentomatic.studio.decorators import studio_stream

        @studio_stream
        async def my_stream(state, config, breakpoints):
            yield {"event": "start"}

        assert my_stream._is_studio_stream is True


class TestRegisterStudioHooks:
    """Test register_studio_hooks discovery."""

    def test_no_module_path_returns_early(self):
        from agentomatic.studio.decorators import register_studio_hooks

        agent = SimpleNamespace(module_path=None)
        register_studio_hooks(agent)
        # Should not raise and not set any attributes

    def test_import_error_returns_early(self):
        from agentomatic.studio.decorators import register_studio_hooks

        agent = SimpleNamespace(module_path="totally.nonexistent.module.xyz")
        register_studio_hooks(agent)
        # Should not raise

    def test_discovers_decorated_functions(self):
        """Create a mock module with decorated functions and verify discovery."""

        from agentomatic.studio.decorators import register_studio_hooks

        # Create a fake module
        fake_module = MagicMock()

        graph_fn = MagicMock()
        graph_fn._is_studio_graph = True
        graph_fn._is_studio_state = False
        graph_fn._is_studio_stream = False

        state_fn = MagicMock()
        state_fn._is_studio_graph = False
        state_fn._is_studio_state = True
        state_fn._is_studio_stream = False

        stream_fn = MagicMock()
        stream_fn._is_studio_graph = False
        stream_fn._is_studio_state = False
        stream_fn._is_studio_stream = True

        fake_module.graph_fn = graph_fn
        fake_module.state_fn = state_fn
        fake_module.stream_fn = stream_fn

        # dir() returns attribute names
        fake_module.__dir__ = lambda self: ["graph_fn", "state_fn", "stream_fn"]

        agent = SimpleNamespace(module_path="fake.module")

        with patch("importlib.import_module", return_value=fake_module):
            register_studio_hooks(agent)

        assert agent._studio_graph_fn is graph_fn
        assert agent._studio_state_fn is state_fn
        assert agent._studio_stream_fn is stream_fn


# ─────────────────────────────────────────────────────────────────────
# APIResponse model
# ─────────────────────────────────────────────────────────────────────
class TestAPIResponseModel:
    """Test the APIResponse pydantic model."""

    def test_default_values(self):
        from agentomatic.protocols.decorators import APIResponse

        resp = APIResponse()
        assert resp.success is True
        assert resp.data is None
        assert resp.message == ""
        assert resp.error is None
        assert resp.timestamp  # not empty

    def test_error_response(self):
        from agentomatic.protocols.decorators import APIResponse

        resp = APIResponse(success=False, error="something broke", data={"code": 500})
        assert resp.success is False
        assert resp.error == "something broke"
        assert resp.data["code"] == 500

    def test_serialization(self):
        from agentomatic.protocols.decorators import APIResponse

        resp = APIResponse(success=True, data={"users": [1, 2]}, message="ok")
        d = resp.model_dump()
        assert d["success"] is True
        assert d["data"] == {"users": [1, 2]}
        assert d["message"] == "ok"
