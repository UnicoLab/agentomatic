"""Tests for ordered LLM fallback chains."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from agentomatic.providers.fallback import (
    DEFAULT_FALLBACK_ON,
    EmptyLLMResponseError,
    FallbackLLM,
    is_empty_llm_response,
    model_label,
    normalize_fallback_on,
    should_fallback,
)
from agentomatic.providers.llm import (
    apply_stack_defaults,
    get_failover_count,
    get_llm,
    get_named_llm,
    reset_llm,
)
from agentomatic.stacks.manager import LLMFallbackSpec, LLMStackEntry, StackManager


class _FailThenSucceed:
    """Minimal chat model: fail N times, then return content."""

    def __init__(self, failures: int, content: str = "ok") -> None:
        self.failures_left = failures
        self.content = content
        self.calls = 0

    def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        if self.failures_left > 0:
            self.failures_left -= 1
            raise TimeoutError("request timed out")
        return SimpleNamespace(content=self.content)

    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self.invoke(*args, **kwargs)


class _RateLimited:
    def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("Error 429: Rate limit exceeded")

    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self.invoke(*args, **kwargs)


class _ValueErrorModel:
    def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise ValueError("bad prompt")

    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self.invoke(*args, **kwargs)


class _AlwaysEmpty:
    def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return SimpleNamespace(content="")

    async def ainvoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return SimpleNamespace(content="")


@pytest.fixture(autouse=True)
def _reset_llm_singleton() -> None:
    reset_llm()
    yield
    reset_llm()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_normalize_fallback_on_defaults() -> None:
    assert normalize_fallback_on(None) == frozenset(DEFAULT_FALLBACK_ON)


def test_model_label() -> None:
    assert model_label("openai", "gpt-4o") == "openai/gpt-4o"
    assert model_label("ollama") == "ollama"


def test_is_empty_llm_response() -> None:
    assert is_empty_llm_response(SimpleNamespace(content=""))
    assert is_empty_llm_response(SimpleNamespace(content="  \n"))
    assert not is_empty_llm_response(SimpleNamespace(content="hi"))


@pytest.mark.parametrize(
    ("exc", "triggers", "expected"),
    [
        (TimeoutError("x"), {"timeout"}, True),
        (ConnectionError("refused"), {"connection"}, True),
        (RuntimeError("429 rate limit"), {"rate_limit"}, True),
        (EmptyLLMResponseError("empty"), {"empty_response"}, True),
        (ValueError("nope"), {"timeout", "connection"}, False),
        (ValueError("nope"), {"any_error"}, True),
    ],
)
def test_should_fallback(
    exc: BaseException,
    triggers: set[str],
    expected: bool,
) -> None:
    assert should_fallback(exc, frozenset(triggers)) is expected


# ---------------------------------------------------------------------------
# FallbackLLM behaviour
# ---------------------------------------------------------------------------


def test_fallback_llm_retries_on_timeout() -> None:
    primary = _RateLimited()
    backup = _FailThenSucceed(failures=0, content="from-backup")
    chain = FallbackLLM(
        primary,
        [backup],
        labels=["primary/a", "backup/b"],
        fallback_on=["timeout", "rate_limit"],
    )
    result = chain.invoke([{"role": "user", "content": "hi"}])
    assert result.content == "from-backup"
    assert get_failover_count() == 1
    assert len(chain.fallbacks) == 1


@pytest.mark.asyncio
async def test_fallback_llm_ainvoke_empty_response() -> None:
    primary = _AlwaysEmpty()
    backup = _FailThenSucceed(failures=0, content="fallback-ok")
    chain = FallbackLLM(
        primary,
        [backup],
        labels=["p", "b"],
        fallback_on=["empty_response"],
    )
    result = await chain.ainvoke("hi")
    assert result.content == "fallback-ok"
    assert get_failover_count() >= 1


def test_fallback_llm_does_not_retry_unlisted_errors() -> None:
    chain = FallbackLLM(
        _ValueErrorModel(),
        [_FailThenSucceed(failures=0, content="unused")],
        labels=["p", "b"],
        fallback_on=["timeout", "connection", "rate_limit", "empty_response"],
    )
    with pytest.raises(ValueError, match="bad prompt"):
        chain.invoke("x")
    assert get_failover_count() == 0


def test_fallback_llm_any_error_retries() -> None:
    chain = FallbackLLM(
        _ValueErrorModel(),
        [_FailThenSucceed(failures=0, content="recovered")],
        labels=["p", "b"],
        fallback_on=["any_error"],
    )
    assert chain.invoke("x").content == "recovered"


def test_get_llm_builds_fallback_chain() -> None:
    llm = get_llm(provider="dummy", fallbacks=["dummy"])
    assert isinstance(llm, FallbackLLM)
    assert len(llm.fallbacks) == 1


def test_get_llm_without_fallbacks_is_plain_model() -> None:
    llm = get_llm(provider="dummy")
    assert not isinstance(llm, FallbackLLM)


def test_get_named_llm_with_dict_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, dict]] = []

    def fake_build(provider: str, **kwargs):  # noqa: ANN003
        built.append((provider, kwargs))
        return _FailThenSucceed(failures=0, content=f"{provider}-ok")

    monkeypatch.setattr("agentomatic.providers.llm._build_llm", fake_build)
    llm = get_named_llm(
        "named",
        provider="openai",
        model="gpt-4o",
        fallbacks=[{"provider": "ollama", "model": "mistral:7b"}],
    )
    assert isinstance(llm, FallbackLLM)
    assert built[0][0] == "openai"
    assert built[1][0] == "ollama"
    assert built[1][1]["model"] == "mistral:7b"


# ---------------------------------------------------------------------------
# Stack YAML wiring
# ---------------------------------------------------------------------------


def test_llm_stack_entry_fallbacks_roundtrip() -> None:
    entry = LLMStackEntry(
        provider="openai",
        model="gpt-4o",
        fallbacks=[
            "fast",
            LLMFallbackSpec(provider="ollama", model="mistral:7b"),
        ],
        fallback_on=["timeout", "rate_limit"],
    )
    data = entry.model_dump()
    restored = LLMStackEntry.model_validate(data)
    assert restored.fallbacks[0] == "fast"
    assert isinstance(restored.fallbacks[1], LLMFallbackSpec)
    assert restored.fallbacks[1].model == "mistral:7b"
    assert restored.fallback_on == ["timeout", "rate_limit"]


def test_stack_yaml_fallbacks_resolved(tmp_path) -> None:  # noqa: ANN001
    stack_yaml = {
        "name": "fb",
        "llm": {
            "default": {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "${OPENAI_API_KEY}",
                "fallbacks": [
                    "fast",
                    {"provider": "ollama", "model": "mistral:7b", "base_url": "http://x"},
                ],
                "fallback_on": ["timeout", "empty_response"],
            },
            "fast": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "${OPENAI_API_KEY}",
            },
        },
    }
    path = tmp_path / "fb.yaml"
    path.write_text(yaml.dump(stack_yaml), encoding="utf-8")

    monkey_env = {"OPENAI_API_KEY": "sk-test"}
    mgr = StackManager(tmp_path)
    # load via file content through StackManager.from_file style
    import os

    for key, value in monkey_env.items():
        os.environ[key] = value
    mgr.load("fb")
    entry = mgr.get_llm_config("default")
    assert len(entry.fallbacks) == 2
    resolved = mgr.resolve_fallbacks(entry)
    assert resolved[0]["model"] == "gpt-4o-mini"
    assert resolved[0]["api_key"] == "sk-test"
    assert resolved[1]["provider"] == "ollama"
    assert resolved[1]["model"] == "mistral:7b"


def test_apply_stack_defaults_wires_fallbacks(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack_yaml = {
        "name": "with_fb",
        "llm": {
            "default": {
                "provider": "dummy",
                "model": "primary",
                "fallbacks": [{"provider": "dummy", "model": "backup"}],
                "fallback_on": ["any_error"],
            }
        },
    }
    (tmp_path / "with_fb.yaml").write_text(yaml.dump(stack_yaml), encoding="utf-8")
    mgr = StackManager(tmp_path)
    mgr.load("with_fb")

    built: list[str] = []

    def fake_build(provider: str, **kwargs):  # noqa: ANN003
        built.append(kwargs.get("model", provider))
        return _FailThenSucceed(failures=0, content="ok")

    monkeypatch.setattr("agentomatic.providers.llm._build_llm", fake_build)
    instance = apply_stack_defaults(mgr)
    assert isinstance(instance, FallbackLLM)
    assert built == ["primary", "backup"]


def test_single_model_stack_unchanged(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack_yaml = {
        "name": "plain",
        "llm": {"default": {"provider": "dummy", "model": "only"}},
    }
    (tmp_path / "plain.yaml").write_text(yaml.dump(stack_yaml), encoding="utf-8")
    mgr = StackManager(tmp_path)
    mgr.load("plain")

    def fake_build(provider: str, **kwargs):  # noqa: ANN003
        return _FailThenSucceed(failures=0, content="ok")

    monkeypatch.setattr("agentomatic.providers.llm._build_llm", fake_build)
    instance = apply_stack_defaults(mgr)
    assert not isinstance(instance, FallbackLLM)
