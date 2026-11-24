# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Tests for LLM ``_build_llm`` base_url / api_base / openai_compatible support."""

from __future__ import annotations

import pytest

from agentomatic.providers import llm as llm_module


class _FakeChatOpenAI:
    """Stand-in for ``langchain_openai.ChatOpenAI`` capturing constructor kwargs."""

    instances: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeChatOpenAI.instances.append(kwargs)


class _FakeAzureChatOpenAI:
    """Stand-in for ``langchain_openai.AzureChatOpenAI``."""

    instances: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeAzureChatOpenAI.instances.append(kwargs)


@pytest.fixture(autouse=True)
def _reset_state():
    _FakeChatOpenAI.instances.clear()
    _FakeAzureChatOpenAI.instances.clear()
    llm_module.reset_llm()
    yield
    llm_module.reset_llm()


@pytest.fixture
def _patch_langchain_openai(monkeypatch):
    """Inject fake ChatOpenAI + AzureChatOpenAI into sys.modules."""
    import sys
    import types

    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = _FakeChatOpenAI
    fake_module.AzureChatOpenAI = _FakeAzureChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    return fake_module


def test_openai_accepts_base_url(_patch_langchain_openai):
    llm = llm_module._build_llm(
        "openai", api_key="k", model="gpt-4o", base_url="https://gw.example.com/v1"
    )
    assert isinstance(llm, _FakeChatOpenAI)
    assert llm.kwargs["base_url"] == "https://gw.example.com/v1"
    assert llm.kwargs["model"] == "gpt-4o"


def test_openai_accepts_api_base_alias(_patch_langchain_openai):
    """``api_base`` should be treated as an alias for ``base_url``."""
    llm = llm_module._build_llm(
        "openai", api_key="k", model="gpt-4o", api_base="https://gw.example.com/v1"
    )
    assert isinstance(llm, _FakeChatOpenAI)
    assert llm.kwargs["base_url"] == "https://gw.example.com/v1"


def test_openai_without_base_url_omits_kwarg(_patch_langchain_openai):
    llm = llm_module._build_llm("openai", api_key="k", model="gpt-4o")
    assert "base_url" not in llm.kwargs


def test_openai_compatible_requires_base_url(_patch_langchain_openai):
    with pytest.raises(ValueError, match="base_url"):
        llm_module._build_llm("openai_compatible", api_key="k", model="mistral")


def test_openai_compatible_builds_with_base_url(_patch_langchain_openai):
    llm = llm_module._build_llm(
        "openai_compatible",
        api_key="k",
        model="mistral",
        base_url="https://api.groq.com/openai/v1",
    )
    assert isinstance(llm, _FakeChatOpenAI)
    assert llm.kwargs["base_url"] == "https://api.groq.com/openai/v1"


def test_azure_maps_api_base_and_version_and_deployment(_patch_langchain_openai):
    llm = llm_module._build_llm(
        "azure",
        api_key="k",
        api_base="https://azuretenant.openai.azure.com/",
        api_version="2024-05-01-preview",
        deployment_name="gpt-4o-deploy",
    )
    assert isinstance(llm, _FakeAzureChatOpenAI)
    assert llm.kwargs["azure_endpoint"] == "https://azuretenant.openai.azure.com/"
    assert llm.kwargs["api_version"] == "2024-05-01-preview"
    assert llm.kwargs["azure_deployment"] == "gpt-4o-deploy"


def test_azure_accepts_base_url_alias(_patch_langchain_openai):
    """``base_url`` should also be accepted for Azure endpoints."""
    llm = llm_module._build_llm(
        "azure",
        api_key="k",
        base_url="https://tenant.openai.azure.com/",
        deployment_name="gpt-4o",
    )
    assert isinstance(llm, _FakeAzureChatOpenAI)
    assert llm.kwargs["azure_endpoint"] == "https://tenant.openai.azure.com/"


def test_apply_stack_defaults_no_stack_no_op():
    """apply_stack_defaults must be a safe no-op when no stack is loaded."""
    assert llm_module.apply_stack_defaults(None) is None


def test_search_space_to_dict_round_trip_includes_routing():
    """``PromptSearchSpace.to_dict()`` must round-trip routing/model_choice."""
    from agentomatic.optimize.search_space import PromptSearchSpace

    space = PromptSearchSpace(
        optimize_model_choice=True,
        model_choices=["ollama/mistral", "openai/gpt-4o"],
        fallback_models=["ollama/qwen2.5:7b"],
        routing_weight_space={"weight": [0.1, 0.5]},
    )
    data = space.to_dict()
    assert data["optimize_model_choice"] is True
    assert data["model_choices"] == ["ollama/mistral", "openai/gpt-4o"]
    assert data["fallback_models"] == ["ollama/qwen2.5:7b"]
    assert data["routing_weight_space"] == {"weight": [0.1, 0.5]}

    restored = PromptSearchSpace.from_dict(data)
    assert restored.optimize_model_choice is True
    assert restored.model_choices == space.model_choices
    assert restored.fallback_models == space.fallback_models
    assert restored.routing_weight_space == space.routing_weight_space


def test_apply_stack_defaults_promotes_to_global(_patch_langchain_openai):
    """apply_stack_defaults should build the default profile and set it as global."""
    from agentomatic.stacks.manager import LLMStackEntry, StackConfig, StackManager

    stack = StackConfig(
        name="test",
        llm={
            "default": LLMStackEntry(
                provider="openai",
                model="gpt-4o-mini",
                api_key="sk-test",
                base_url="https://gw.example.com/v1",
            ),
        },
    )
    mgr = StackManager()
    mgr._active_stack = stack

    llm_module.apply_stack_defaults(mgr)
    llm = llm_module.get_llm()
    assert isinstance(llm, _FakeChatOpenAI)
    assert llm.kwargs["base_url"] == "https://gw.example.com/v1"
    assert llm.kwargs["model"] == "gpt-4o-mini"


@pytest.mark.parametrize(
    "model",
    ["o1", "o1-mini", "o1-preview", "gpt-5", "gpt-5-mini", "o3-mini", "o4-mini"],
)
def test_build_llm_reasoning_models_omit_temperature_in_payload_and_http_body(model: str) -> None:
    """Real ChatOpenAI & AzureChatOpenAI must omit temperature for reasoning models."""
    import json

    import httpx

    sent_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    transport = httpx.MockTransport(handler)
    openai_llm = llm_module._build_llm(
        "openai",
        model=model,
        api_key="k",
        temperature=0.7,
        extra={"extra_body": {"temperature": 0.3}},
        model_kwargs={"temperature": 0.4, "extra_body": {"temperature": 0.2}},
    )
    openai_llm.root_client._client._transport = transport
    openai_llm.invoke("bonjour")
    assert len(sent_bodies) == 1
    assert "temperature" not in sent_bodies[0]
    assert "temperature" not in (sent_bodies[0].get("extra_body") or {})

    azure_llm = llm_module._build_llm(
        "azure",
        model=model,
        api_key="k",
        azure_endpoint="https://az.example.com",
        azure_deployment=model,
        temperature=0.7,
        extra={"extra_body": {"temperature": 0.3}},
        model_kwargs={"temperature": 0.4, "extra_body": {"temperature": 0.2}},
    )
    azure_llm.root_client._client._transport = transport
    azure_llm.invoke("bonjour azure")
    assert len(sent_bodies) == 2
    assert "temperature" not in sent_bodies[1]
    assert "temperature" not in (sent_bodies[1].get("extra_body") or {})


def test_build_llm_preserves_temperature_for_gpt4o_and_strips_extra_body_for_gpt5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-reasoning models keep temperature; reasoning models strip nested extras and vertex."""
    import json
    import sys
    import types

    import httpx

    sent_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-4o",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    llm_4o = llm_module._build_llm("openai", model="gpt-4o", api_key="k", temperature=0.35)
    llm_4o.root_client._client._transport = httpx.MockTransport(handler)
    llm_4o.invoke("hi")
    assert sent_bodies[0]["temperature"] == 0.35

    cleaned = llm_module._strip_disallowed_temperature(
        {
            "temperature": 0.7,
            "extra_body": {"temperature": 0.2},
            "model_kwargs": {"extra_body": {"temperature": 0.1}},
            "top_p": 0.9,
        }
    )
    assert cleaned == {"top_p": 0.9}

    vertex_calls: list[dict] = []

    class _FakeVertex:
        def __init__(self, **kwargs):
            vertex_calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_google_vertexai",
        types.SimpleNamespace(ChatVertexAI=_FakeVertex),
    )
    llm_module._build_llm("vertex", model="gpt-5-mini", project="p", temperature=0.6)
    llm_module._build_llm("vertex", model="o1-mini", project="p", temperature=0.6)
    llm_module._build_llm("vertex", model="gemini-2.5-flash", project="p", temperature=0.4)
    assert "temperature" not in vertex_calls[0]
    assert "temperature" not in vertex_calls[1]
    assert vertex_calls[2]["temperature"] == 0.4


@pytest.mark.asyncio
async def test_invoke_with_retry_redacts_secrets_in_warning_log() -> None:
    """Retry warnings must redact credentials, query strings, and prompt payloads."""
    from loguru import logger

    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")

    class _FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: list, **_kwargs) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "Bearer tok_secret_abc https://u:p@host/v1?api_key=q1 secret=mysecret"
                )
            return "recovered"

    try:
        res = await llm_module.invoke_with_retry(_FlakyLLM(), [], max_retries=1, retry_delay=0.0)
        assert getattr(res, "content", res) == "recovered"
        joined = "\n".join(captured)
        assert "tok_secret_abc" not in joined
        assert "api_key=q1" not in joined
        assert "mysecret" not in joined
        assert "[REDACTED]" in joined
    finally:
        logger.remove(sink_id)
