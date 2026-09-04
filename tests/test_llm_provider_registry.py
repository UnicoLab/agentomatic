# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
"""Tests for the pluggable LLM provider registry (:func:`register_llm_provider`)."""

from __future__ import annotations

import pytest

from agentomatic.providers import llm as llm_module
from agentomatic.providers.llm import (
    _LLM_PROVIDERS,
    get_llm,
    register_llm_provider,
    registered_llm_providers,
)


@pytest.fixture(autouse=True)
def _reset_state():
    llm_module.reset_llm()
    _LLM_PROVIDERS.clear()
    yield
    llm_module.reset_llm()
    _LLM_PROVIDERS.clear()


def test_register_and_list_custom_provider():
    register_llm_provider("acme", lambda **kw: "acme-llm")
    assert "acme" in registered_llm_providers()


def test_registered_provider_is_used_by_build_llm():
    register_llm_provider("acme", lambda **kw: {"built": True, **kw})
    result = llm_module._build_llm("acme", model="x", api_key="k")
    assert result == {"built": True, "model": "x", "api_key": "k"}


def test_provider_name_matching_is_case_insensitive():
    register_llm_provider("AcmeGateway", lambda **kw: "ok")
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
