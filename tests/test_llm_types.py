"""Tests for the pluggable LLM type system (LLMSpec / LLMCallable).

Verifies that:
- String model specs are dispatched to LLMCaller
- Async callables work as LLMSpec
- Sync callables work as LLMSpec
- LangChain-protocol objects (ainvoke/invoke) work
- call_llm_json parses JSON from callables
- LLMCaller.call and call_with_json accept non-string models
- Backward compatibility: all string-based usage unchanged
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentomatic.optimize.llm_types import (
    call_llm,
    call_llm_json,
)

# =====================================================================
# Helpers — custom callables for testing
# =====================================================================


async def _async_llm(
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> str:
    """Async callable LLM."""
    prefix = f"[{system_prompt}] " if system_prompt else ""
    return f"{prefix}Async response to: {prompt}"


def _sync_llm(
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> str:
    """Sync callable LLM."""
    prefix = f"[{system_prompt}] " if system_prompt else ""
    return f"{prefix}Sync response to: {prompt}"


async def _json_llm(
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> str:
    """Callable that returns JSON as a string."""
    return json.dumps({"result": "ok", "prompt_len": len(prompt)})


class _LangChainLikeAsync:
    """Mock LangChain-compatible model with ainvoke."""

    async def ainvoke(self, messages: list) -> Any:
        content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        return SimpleNamespace(content=f"LangChain async: {content}")


class _LangChainLikeSync:
    """Mock LangChain-compatible model with invoke (no ainvoke)."""

    def invoke(self, messages: list) -> Any:
        content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        return SimpleNamespace(content=f"LangChain sync: {content}")


# =====================================================================
# Protocol conformance
# =====================================================================


class TestLLMCallableProtocol:
    """Test that LLMCallable protocol matching works at runtime."""

    def test_async_function_is_callable(self):
        """Async functions should match LLMCallable protocol."""
        assert callable(_async_llm)

    def test_sync_function_is_callable(self):
        """Sync functions should be callable."""
        assert callable(_sync_llm)

    def test_string_is_not_callable(self):
        """Strings are not callables."""
        assert not callable("ollama/mistral:7b")


# =====================================================================
# call_llm — dispatcher tests
# =====================================================================


class TestCallLlm:
    """Test the unified call_llm dispatcher."""

    @pytest.mark.asyncio
    async def test_string_model_delegates_to_llmcaller(self):
        """String specs should be routed to LLMCaller.call."""
        with patch(
            "agentomatic.optimize.llm_caller.LLMCaller.call",
            new_callable=AsyncMock,
            return_value="mocked response",
        ) as mock_call:
            result = await call_llm("ollama/mistral:7b", "Hello")
            assert result == "mocked response"
            mock_call.assert_called_once()
            # First arg should be the model string
            assert mock_call.call_args[0][0] == "ollama/mistral:7b"
            assert mock_call.call_args[0][1] == "Hello"

    @pytest.mark.asyncio
    async def test_async_callable(self):
        """Async callables should be called directly."""
        result = await call_llm(_async_llm, "Hello world")
        assert "Async response to: Hello world" in result

    @pytest.mark.asyncio
    async def test_async_callable_with_system_prompt(self):
        """System prompt should be passed to async callables."""
        result = await call_llm(_async_llm, "Hello", system_prompt="Be helpful")
        assert "[Be helpful]" in result
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_sync_callable(self):
        """Sync callables should be wrapped in to_thread."""
        result = await call_llm(_sync_llm, "Hello world")
        assert "Sync response to: Hello world" in result

    @pytest.mark.asyncio
    async def test_langchain_ainvoke(self):
        """Objects with ainvoke should use LangChain protocol."""
        model = _LangChainLikeAsync()
        # We need langchain_core for this test
        try:
            from langchain_core.messages import HumanMessage  # noqa: F401

            result = await call_llm(model, "Test prompt")
            assert "LangChain async" in result
        except ImportError:
            # Without langchain-core, it falls back to dict messages
            result = await call_llm(model, "Test prompt")
            assert "LangChain async" in result

    @pytest.mark.asyncio
    async def test_langchain_invoke_sync(self):
        """Objects with invoke (no ainvoke) should use sync LangChain protocol."""
        model = _LangChainLikeSync()
        result = await call_llm(model, "Test prompt")
        assert "LangChain sync" in result

    @pytest.mark.asyncio
    async def test_invalid_model_raises(self):
        """Non-string, non-callable should raise TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            await call_llm(42, "Hello")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_none_model_raises(self):
        """None should raise TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            await call_llm(None, "Hello")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_async_callable_exception_returns_empty(self):
        """Async callable that raises should return '' gracefully."""

        async def failing_llm(
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> str:
            raise ConnectionError("LLM server down")

        result = await call_llm(failing_llm, "Hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_sync_callable_exception_returns_empty(self):
        """Sync callable that raises should return '' gracefully."""

        def failing_sync_llm(
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> str:
            raise RuntimeError("Inference failed")

        result = await call_llm(failing_sync_llm, "Hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_langchain_ainvoke_exception_returns_empty(self):
        """LangChain model that raises in ainvoke should return ''."""

        class FailingLangChain:
            async def ainvoke(self, messages):
                raise ValueError("Model overloaded")

        result = await call_llm(FailingLangChain(), "Hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_callable_returns_none_gracefully(self):
        """Callable returning None should be str-ified to 'None'."""

        async def none_llm(
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> str:
            return None  # type: ignore[return-value]

        result = await call_llm(none_llm, "Hello")
        assert result == "None"  # str(None)


# =====================================================================
# call_llm_json — JSON extraction tests
# =====================================================================


class TestCallLlmJson:
    """Test the JSON-extracting LLM call."""

    @pytest.mark.asyncio
    async def test_string_model_delegates_to_llmcaller(self):
        """String models delegate to LLMCaller.call_with_json."""
        with patch(
            "agentomatic.optimize.llm_caller.LLMCaller.call_with_json",
            new_callable=AsyncMock,
            return_value={"ok": True},
        ) as mock_call:
            result = await call_llm_json("ollama/mistral:7b", "Return JSON")
            assert result == {"ok": True}
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_callable_json_response(self):
        """Callable returning valid JSON should be parsed."""
        result = await call_llm_json(_json_llm, "Test")
        assert result["result"] == "ok"
        assert "prompt_len" in result

    @pytest.mark.asyncio
    async def test_callable_json_with_code_fences(self):
        """JSON wrapped in code fences should be extracted."""

        async def fenced_json_llm(
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> str:
            return '```json\n{"status": "success"}\n```'

        result = await call_llm_json(fenced_json_llm, "Test")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_callable_invalid_json_retries(self):
        """Invalid JSON should retry and return {} on failure."""
        call_count = 0

        async def bad_json_llm(
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> str:
            nonlocal call_count
            call_count += 1
            return "This is not JSON at all"

        result = await call_llm_json(bad_json_llm, "Test", max_retries=1)
        assert result == {}
        # 1 initial + 1 retry = 2 calls
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_callable_empty_response_retries(self):
        """Empty responses should trigger retries."""
        call_count = 0

        async def empty_llm(
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> str:
            nonlocal call_count
            call_count += 1
            return ""

        result = await call_llm_json(empty_llm, "Test", max_retries=1)
        assert result == {}
        assert call_count == 2


# =====================================================================
# LLMCaller integration — non-string dispatch
# =====================================================================


class TestLLMCallerCallableDispatch:
    """Test that LLMCaller.call accepts non-string models."""

    @pytest.mark.asyncio
    async def test_llmcaller_call_async_callable(self):
        """LLMCaller.call should dispatch async callables."""
        from agentomatic.optimize.llm_caller import LLMCaller

        result = await LLMCaller.call(_async_llm, "Hello")
        assert "Async response" in result

    @pytest.mark.asyncio
    async def test_llmcaller_call_sync_callable(self):
        """LLMCaller.call should dispatch sync callables."""
        from agentomatic.optimize.llm_caller import LLMCaller

        result = await LLMCaller.call(_sync_llm, "Hello")
        assert "Sync response" in result

    @pytest.mark.asyncio
    async def test_llmcaller_call_with_json_callable(self):
        """LLMCaller.call_with_json should handle callables."""
        from agentomatic.optimize.llm_caller import LLMCaller

        result = await LLMCaller.call_with_json(_json_llm, "Get JSON")
        assert result["result"] == "ok"

    @pytest.mark.asyncio
    async def test_llmcaller_string_still_works(self):
        """String model specs should still work through LLMCaller."""
        from agentomatic.optimize.llm_caller import LLMCaller

        with patch(
            "agentomatic.optimize.llm_caller._call_ollama",
            new_callable=AsyncMock,
            return_value="ollama response",
        ):
            result = await LLMCaller.call("ollama/mistral:7b", "Hello")
            assert result == "ollama response"


# =====================================================================
# LLMSpec as type annotation — consumer verification
# =====================================================================


class TestLLMSpecConsumers:
    """Verify that key classes accept LLMSpec in their constructors."""

    def test_prompt_optimizer_accepts_callable(self):
        """PromptOptimizer should accept callable for llm params."""
        from agentomatic.optimize.optimizer import PromptOptimizer

        # Should not raise
        opt = PromptOptimizer(
            agent="test",
            llm=_async_llm,  # type: ignore[arg-type]
            rewrite_llm=_async_llm,  # type: ignore[arg-type]
            eval_llm=_async_llm,  # type: ignore[arg-type]
        )
        assert opt.llm is _async_llm
        assert opt.rewrite_llm is _async_llm

    def test_prompt_optimizer_still_accepts_string(self):
        """PromptOptimizer should still work with string models."""
        from agentomatic.optimize.optimizer import PromptOptimizer

        opt = PromptOptimizer(
            agent="test",
            llm="ollama/mistral:7b",
        )
        assert opt.llm == "ollama/mistral:7b"

    def test_data_synthesizer_accepts_callable(self):
        """DataSynthesizer should accept callable for model."""
        from agentomatic.optimize.synthesizer import DataSynthesizer

        synth = DataSynthesizer(model=_async_llm)  # type: ignore[arg-type]
        assert synth.model is _async_llm

    def test_iterative_rewrite_accepts_callable(self):
        """IterativeRewrite strategy should accept callable."""
        from agentomatic.optimize.strategies import IterativeRewrite

        strategy = IterativeRewrite(model=_async_llm)  # type: ignore[arg-type]
        assert strategy.model is _async_llm


# =====================================================================
# Studio serve tests
# =====================================================================


class TestStudioServe:
    """Test Studio UI serve improvements."""

    def test_mount_studio_ui_with_assets(self):
        """When assets exist, Studio should mount successfully."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agentomatic.studio.serve import (
            is_studio_available,
            mount_studio_ui,
        )

        if not is_studio_available():
            pytest.skip("Studio assets not available in test environment")

        app = FastAPI()
        mount_studio_ui(app)
        client = TestClient(app)

        resp = client.get("/studio/ui/")
        assert resp.status_code == 200
        assert "Agentomatic Studio" in resp.text

    def test_mount_studio_disabled_page(self):
        """Disabled page should return 503 with helpful message."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agentomatic.studio.serve import mount_studio_disabled_page

        app = FastAPI()
        mount_studio_disabled_page(app)
        client = TestClient(app)

        resp = client.get("/studio/ui/")
        assert resp.status_code == 503
        assert "Studio Is Disabled" in resp.text
        assert "enable_studio=True" in resp.text

    def test_disabled_studio_info_endpoint(self):
        """When disabled, /studio/info should return JSON error."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agentomatic.studio.serve import mount_studio_disabled_page

        app = FastAPI()
        mount_studio_disabled_page(app)
        client = TestClient(app)

        resp = client.get("/studio/info")
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "studio_disabled"


# =====================================================================
# Provider layer — set_llm / get_llm(instance=) / get_structured_llm
# =====================================================================


class TestProviderCustomLLM:
    """Test the provider layer accepts custom LLMs."""

    def setup_method(self):
        """Reset LLM singleton before each test."""
        from agentomatic.providers import reset_llm

        reset_llm()

    def teardown_method(self):
        """Reset LLM singleton after each test."""
        from agentomatic.providers import reset_llm

        reset_llm()

    def test_set_llm_stores_instance(self):
        """set_llm should store the custom instance."""
        from agentomatic.providers import get_llm, set_llm

        set_llm(_async_llm)
        assert get_llm() is _async_llm

    def test_get_llm_instance_kwarg(self):
        """get_llm(instance=...) should store and return it."""
        from agentomatic.providers import get_llm

        result = get_llm(instance=_async_llm)
        assert result is _async_llm
        # Should be cached
        assert get_llm() is _async_llm

    def test_get_named_llm_instance(self):
        """get_named_llm should store custom instance by name."""
        from agentomatic.providers import get_named_llm

        result = get_named_llm("judge", instance=_async_llm)
        assert result is _async_llm
        # Should retrieve from cache
        assert get_named_llm("judge") is _async_llm

    def test_get_structured_llm_with_instance(self):
        """get_structured_llm should accept a pre-built LLM."""
        from pydantic import BaseModel

        from agentomatic.providers import get_structured_llm

        class TestOutput(BaseModel):
            answer: str = ""

        # An LLM without with_structured_output should use fallback
        result = get_structured_llm(TestOutput, instance=_async_llm)
        # Should be wrapped in StructuredOutputFallbackWrapper
        assert hasattr(result, "llm")
        assert result.llm is _async_llm


# =====================================================================
# Studio path traversal safety
# =====================================================================


class TestStudioSecurity:
    """Test Studio serve path traversal protection."""

    def test_path_traversal_returns_index(self):
        """Path traversal attempts should NOT serve arbitrary files."""
        from agentomatic.studio.serve import is_studio_available

        if not is_studio_available():
            pytest.skip("Studio assets not available in test environment")

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agentomatic.studio.serve import mount_studio_ui

        app = FastAPI()
        mount_studio_ui(app)
        client = TestClient(app)

        # Path traversal attempt — should be rejected (404 or SPA fallback)
        resp = client.get("/studio/ui/../../etc/passwd")
        # Must NOT serve the actual file — 404 is the expected behavior
        # because the HTTP framework normalizes the path
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            # If 200, it's the SPA fallback (index.html), not /etc/passwd
            assert "passwd" not in resp.text
