# pyright: reportMissingParameterType=none
"""A configured LLM whose driver is missing must fail, not answer with a fake.

Regression: every failure to build an LLM — including "No module named
'langchain_openai'" — was caught, logged at WARNING, and replaced with a
dummy model. A deployment that had configured a real provider therefore
booted, reported healthy, and answered every request with fabricated text.

A missing client library is a defect in the image that no retry fixes, so it
is now fatal and the message names the extra to install. A backend that is
merely unreachable can recover, so that still degrades — loudly.
"""

from __future__ import annotations

import pytest

from agentomatic.providers import llm as llm_mod
from agentomatic.providers.llm import LLMDriverMissingError


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    """Named instances are cached process-wide; isolate each test."""
    llm_mod._named_instances.clear()  # noqa: SLF001
    yield
    llm_mod._named_instances.clear()  # noqa: SLF001


class TestMissingDriverIsFatal:
    def test_a_missing_client_library_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(provider: str, **kwargs: object) -> object:
            raise ImportError("No module named 'langchain_openai'")

        monkeypatch.setattr(llm_mod, "_build_llm", _boom)

        with pytest.raises(LLMDriverMissingError):
            llm_mod.get_named_llm("agent:default", provider="openai_compatible")

    @pytest.mark.parametrize(
        "provider,extra",
        [
            ("openai", "openai"),
            ("openai_compatible", "openai"),
            ("azure", "azure"),
            ("vertex", "vertex"),
            ("ollama", "ollama"),
        ],
    )
    def test_the_error_names_the_extra_to_install(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, extra: str
    ) -> None:
        def _boom(provider: str, **kwargs: object) -> object:
            raise ImportError("No module named 'whatever'")

        monkeypatch.setattr(llm_mod, "_build_llm", _boom)

        with pytest.raises(LLMDriverMissingError, match=rf"agentomatic\[{extra}\]"):
            llm_mod.get_named_llm(f"agent:{provider}", provider=provider)

    def test_the_error_says_why_a_dummy_was_not_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point is that a silent fake is worse than a failed boot."""

        def _boom(provider: str, **kwargs: object) -> object:
            raise ImportError("No module named 'langchain_openai'")

        monkeypatch.setattr(llm_mod, "_build_llm", _boom)

        with pytest.raises(LLMDriverMissingError) as excinfo:
            llm_mod.get_named_llm("agent:default", provider="openai")

        assert "dummy" in str(excinfo.value).lower()


class TestUnreachableBackendStillDegrades:
    def test_a_connection_error_falls_back_to_a_dummy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreachable backend can recover, so local development still runs."""

        def _boom(provider: str, **kwargs: object) -> object:
            raise ConnectionError("Connection refused")

        monkeypatch.setattr(llm_mod, "_build_llm", _boom)

        built = llm_mod.get_named_llm("agent:default", provider="ollama")

        assert built is not None

    def test_the_degraded_warning_says_responses_are_fabricated(
        self, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        def _boom(provider: str, **kwargs: object) -> object:
            raise ConnectionError("Connection refused")

        monkeypatch.setattr(llm_mod, "_build_llm", _boom)
        messages: list[str] = []
        handler_id = llm_mod.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            llm_mod.get_named_llm("agent:warn", provider="ollama")
        finally:
            llm_mod.logger.remove(handler_id)

        assert any("fabricated" in m.lower() for m in messages), messages


class TestWorkingProvidersAreUnaffected:
    def test_a_buildable_provider_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        monkeypatch.setattr(llm_mod, "_build_llm", lambda provider, **kw: sentinel)
        monkeypatch.setattr(llm_mod, "_wrap_with_fallbacks", lambda built, **kw: built)

        assert llm_mod.get_named_llm("agent:ok", provider="ollama") is sentinel

    def test_the_dummy_provider_still_works(self) -> None:
        """Explicitly asking for a dummy is a legitimate configuration."""
        built = llm_mod.get_named_llm("agent:dummy", provider="dummy")

        assert built is not None
