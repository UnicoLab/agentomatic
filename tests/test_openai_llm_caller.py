"""Unit tests for hardened openai/ routing in LLMCaller."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentomatic.optimize.llm_caller import (
    LLMCaller,
    _looks_like_openai_cloud_model,
    _openai_supports_temperature,
    _openai_uses_max_completion_tokens,
    _resolve_openai_endpoint,
    parse_model_spec,
)


@pytest.fixture(autouse=True)
def _reset_llm_caller_defaults() -> Any:
    prev_url = LLMCaller._default_base_url
    prev_key = LLMCaller._default_api_key
    LLMCaller.configure()
    yield
    LLMCaller._default_base_url = prev_url
    LLMCaller._default_api_key = prev_key


def test_parse_openai_spec() -> None:
    assert parse_model_spec("openai/gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert parse_model_spec("openai/o3-mini") == ("openai", "o3-mini")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gpt-4o-mini", True),
        ("gpt-4.1", True),
        ("o3-mini", True),
        ("o1-preview", True),
        ("LFM2.5-8B-A1B-MLX-4bit", False),
        ("Qwen3.5-9B", False),
    ],
)
def test_looks_like_openai_cloud_model(name: str, expected: bool) -> None:
    assert _looks_like_openai_cloud_model(name) is expected


def test_openai_token_and_temperature_helpers() -> None:
    assert _openai_uses_max_completion_tokens("o3-mini") is True
    assert _openai_uses_max_completion_tokens("gpt-4o-mini") is False
    assert _openai_supports_temperature("gpt-4o-mini") is True
    assert _openai_supports_temperature("o1-preview") is False


def test_resolve_ignores_local_env_base_for_cloud_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-cloud-key")
    base, key = _resolve_openai_endpoint(
        "gpt-4o-mini",
        base_url=None,
        api_key=None,
        explicit_base=False,
    )
    assert base is None  # api.openai.com
    assert key == "sk-test-cloud-key"


def test_resolve_keeps_explicit_local_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    base, key = _resolve_openai_endpoint(
        "gpt-4o-mini",
        base_url="http://127.0.0.1:8000/v1",
        api_key="local-key",
        explicit_base=True,
    )
    assert base == "http://127.0.0.1:8000/v1"
    assert key == "local-key"


def test_resolve_requires_api_key_for_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _resolve_openai_endpoint(
            "gpt-4o-mini",
            base_url=None,
            api_key=None,
            explicit_base=False,
        )


def _fake_async_openai(captured: dict[str, Any], *, content: str = "OK") -> MagicMock:
    class _FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            captured["create"] = kwargs
            msg = MagicMock()
            msg.content = content
            msg.reasoning_content = None
            return MagicMock(choices=[MagicMock(message=msg)])

    class _FakeClient:
        chat = MagicMock(completions=_FakeCompletions())

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

    def _ctor(**kwargs: Any) -> _FakeClient:
        captured["client"] = kwargs
        return _FakeClient()

    return MagicMock(side_effect=_ctor)


@pytest.mark.asyncio
async def test_call_openai_cloud_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    captured: dict[str, Any] = {}

    with patch("openai.AsyncOpenAI", _fake_async_openai(captured)):
        text = await LLMCaller.call(
            "openai/gpt-4o-mini",
            "Say OK",
            temperature=0.2,
            max_tokens=32,
            json_mode=True,
        )

    assert text == "OK"
    create = captured["create"]
    assert create["model"] == "gpt-4o-mini"
    assert create["temperature"] == 0.2
    assert create["max_tokens"] == 32
    assert create["response_format"] == {"type": "json_object"}
    assert "max_completion_tokens" not in create
    assert captured["client"]["api_key"] == "sk-unit-test"
    assert "base_url" not in captured["client"]


@pytest.mark.asyncio
async def test_call_openai_reasoning_model_uses_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    captured: dict[str, Any] = {}

    with patch("openai.AsyncOpenAI", _fake_async_openai(captured, content="done")):
        await LLMCaller.call("openai/o3-mini", "hi", max_tokens=128)

    create = captured["create"]
    assert create["max_completion_tokens"] == 128
    assert "max_tokens" not in create
    assert "temperature" not in create


@pytest.mark.asyncio
async def test_local_env_does_not_hijack_cloud_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    captured: dict[str, Any] = {}

    with patch("openai.AsyncOpenAI", _fake_async_openai(captured, content="cloud")):
        await LLMCaller.call("openai/gpt-4o-mini", "hi")

    assert "base_url" not in captured["client"]
    assert captured["client"]["api_key"] == "sk-unit-test"


@pytest.mark.asyncio
async def test_configure_still_routes_openai_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    LLMCaller.configure(base_url="http://127.0.0.1:8000/v1", api_key="local-key")
    captured: dict[str, Any] = {}

    with patch("openai.AsyncOpenAI", _fake_async_openai(captured, content="local")):
        text = await LLMCaller.call("openai/gpt-4o-mini", "hi")

    assert text == "local"
    assert captured["client"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert captured["client"]["api_key"] == "local-key"
    assert "extra_body" in captured["create"]
