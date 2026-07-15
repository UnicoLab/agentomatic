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
    fake_module.ChatOpenAI = _FakeChatOpenAI  # type: ignore[attr-defined]
    fake_module.AzureChatOpenAI = _FakeAzureChatOpenAI  # type: ignore[attr-defined]
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
