# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
"""Tests for the pluggable LLM provider registry (:func:`register_llm_provider`)."""

from __future__ import annotations

import pytest

from agentomatic.providers import llm as llm_module
from agentomatic.providers.llm import (
    get_llm,
    register_llm_provider,
    registered_llm_providers,
    unregister_llm_provider,
)


@pytest.fixture(autouse=True)
def _reset_state():
    llm_module.reset_llm()
    for provider in registered_llm_providers():
        unregister_llm_provider(provider)
    yield
    llm_module.reset_llm()
    for provider in registered_llm_providers():
        unregister_llm_provider(provider)


def test_register_and_list_custom_provider():
    register_llm_provider("acme", lambda **kw: "acme-llm")
    assert "acme" in registered_llm_providers()


def test_registered_provider_is_used_by_build_llm():
    register_llm_provider("acme", lambda **kw: {"built": True, **kw})
    result = llm_module._build_llm("acme", model="x", api_key="k")
    assert result == {"built": True, "model": "x", "api_key": "k"}


def test_provider_name_matching_is_case_insensitive():
    register_llm_provider("  AcmeGateway  ", lambda **kw: "ok")
    assert llm_module._build_llm("acmegateway") == "ok"
    assert llm_module._build_llm("ACMEGATEWAY") == "ok"


def test_registered_provider_can_override_builtin():
    register_llm_provider("dummy", lambda **kw: "overridden-dummy")
    assert llm_module._build_llm("dummy") == "overridden-dummy"


def test_unknown_provider_raises_helpful_error():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm_module._build_llm("totally-not-a-thing")


def test_get_llm_end_to_end_with_custom_provider():
    register_llm_provider("acme", lambda **kw: kw.get("model", "no-model"))
    instance = get_llm(provider="acme", model="acme-1")
    assert instance == "acme-1"


@pytest.mark.parametrize("name", ["", "   "])
def test_register_rejects_empty_provider_name(name):
    with pytest.raises(ValueError, match="non-empty"):
        register_llm_provider(name, lambda **kw: kw)


def test_register_rejects_non_callable_builder():
    with pytest.raises(TypeError, match="callable"):
        register_llm_provider("broken", object())


def test_register_can_detect_accidental_overwrite():
    register_llm_provider("acme", lambda: "first")
    with pytest.raises(ValueError, match="already registered"):
        register_llm_provider("ACME", lambda: "second", overwrite=False)
    assert llm_module._build_llm("acme") == "first"


def test_unregister_provider_is_case_insensitive_and_idempotent():
    register_llm_provider("Acme", lambda: "value")
    assert unregister_llm_provider(" acme ") is True
    assert unregister_llm_provider("ACME") is False
    assert registered_llm_providers() == []


def test_registry_helpers_are_available_from_top_level_package():
    import agentomatic

    assert agentomatic.register_llm_provider is register_llm_provider
    assert agentomatic.unregister_llm_provider is unregister_llm_provider


def test_concurrent_registrations_remain_consistent():
    import threading

    threads = [
        threading.Thread(
            target=register_llm_provider,
            args=(f"provider-{index}", lambda **kw: kw),
        )
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert registered_llm_providers() == sorted(f"provider-{index}" for index in range(20))
