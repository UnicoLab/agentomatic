"""Tests for Agentomatic retry, failover, and build warning log sanitization."""

from __future__ import annotations

import time
from typing import Any

import pytest
from loguru import logger

from agentomatic.providers.fallback import FallbackLLM
from agentomatic.providers.llm import (
    _sanitize_error_for_log,
    invoke_with_retry,
    record_failover,
)


def test_sanitize_error_for_log_redacts_all_eight_leak_vectors_and_preserves_code() -> None:
    vectors = [
        (
            '401 {"error": {"api_key": "supersecret123", "client_secret": "s3cr3tVal"}}',
            ["supersecret123", "s3cr3tVal"],
        ),
        (
            '400 {"messages": [{"role": "user", "content": "SECRET_USER_MSG"}], '
            '"error": {"message": "bad", "code": "unsupported_value"}}',
            ["SECRET_USER_MSG"],
        ),
        (
            'error prompt="line1\nTOP SECRET LINE 2" code=oauth_secret_code',
            ["line1", "TOP SECRET LINE 2", "oauth_secret_code"],
        ),
        (
            'error secret="my top secret value" other=1',
            ["my top secret value"],
        ),
        (
            'error private_key="-----BEGIN PRIVATE KEY-----\n'
            'MIIE_SECRET_KEY_BODY\n-----END PRIVATE KEY-----"',
            ["MIIE_SECRET_KEY_BODY", "-----END PRIVATE KEY-----"],
        ),
        (
            "Failed https://myuser:s3cr3tPass@proxy.corp:8443/v1/chat?sig=1",
            ["myuser", "s3cr3tPass", "sig=1"],
        ),
        (
            "error AIzaSyD1234567890abcdefghijk",
            ["AIzaSyD1234567890abcdefghijk"],
        ),
        (
            "Ocp-Apim-Subscription-Key: abcdef1234567890",
            ["abcdef1234567890"],
        ),
    ]
    for raw, forbidden_list in vectors:
        san = _sanitize_error_for_log(raw)
        for forbidden in forbidden_list:
            assert forbidden not in san, f"{forbidden!r} leaked in {san!r}"

    # JSON error code must be preserved while user messages are redacted
    preserved = _sanitize_error_for_log(
        '400 {"messages": [{"role": "user", "content": "SECRET_USER_MSG"}], '
        '"error": {"message": "bad", "code": "unsupported_value"}}'
    )
    assert "unsupported_value" in preserved
    assert '"message": "bad"' in preserved
    assert "https://[REDACTED]@proxy.corp:8443/v1/chat" in _sanitize_error_for_log(
        "Failed https://myuser:s3cr3tPass@proxy.corp:8443/v1/chat?sig=1"
    )


def test_sanitize_error_for_log_resists_redos_recursion_error_and_quadratic_dos() -> None:
    unclosed_quote = 'prompt="' + ("x" * 50000)
    t0 = time.perf_counter()
    san_quote = _sanitize_error_for_log(unclosed_quote)
    assert time.perf_counter() - t0 < 0.1
    assert "xxxx" not in san_quote

    unclosed_brackets = ("[" * 20000) + ' secret="TOP_SECRET"'
    t0 = time.perf_counter()
    san_brackets = _sanitize_error_for_log(unclosed_brackets)
    assert time.perf_counter() - t0 < 0.1
    assert "TOP_SECRET" not in san_brackets

    non_contig = ('{"a": ' * 12000) + ' secret="TRAILING_SECRET"'
    t0 = time.perf_counter()
    san_nc = _sanitize_error_for_log(non_contig)
    assert time.perf_counter() - t0 < 0.1
    assert "TRAILING_SECRET" not in san_nc

    deep_valid = ("[" * 1500) + '{"secret": "TOP_SECRET"}' + ("]" * 1500)
    san_deep = _sanitize_error_for_log(deep_valid)
    assert "TOP_SECRET" not in san_deep


@pytest.mark.asyncio
async def test_invoke_with_retry_record_failover_and_fallback_llm_sanitize_logs() -> None:
    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="INFO")
    secret_payload = (
        '401 {"error": {"api_key": "supersecret123", "code": "unsupported_value"}, '
        '"messages": [{"role": "user", "content": "PRIVATE_PROMPT_BODY"}]} '
        "https://u1:p1@gw.corp/chat?sig=9 AIzaSyD1234567890abcdefghijk "
        "Ocp-Apim-Subscription-Key: apim-secret-999"
    )
    try:
        # 1. invoke_with_retry
        class FailingThenOkLLM:
            def __init__(self) -> None:
                self.calls = 0

            async def ainvoke(self, _messages: list[Any], **_kw: Any) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError(secret_payload)
                return "ok"

        res = await invoke_with_retry(FailingThenOkLLM(), [], max_retries=1, retry_delay=0.0)
        assert getattr(res, "content", res) == "ok"

        # 2. record_failover directly
        record_failover("securegpt", "gemini", secret_payload)

        # 3. FallbackLLM sync & async failover
        class TimeoutPrimary:
            def invoke(self, *_a: Any, **_kw: Any) -> str:
                raise TimeoutError(secret_payload)

            async def ainvoke(self, *_a: Any, **_kw: Any) -> str:
                raise TimeoutError(secret_payload)

        class OkBackup:
            def invoke(self, *_a: Any, **_kw: Any) -> str:
                return "backup-sync"

            async def ainvoke(self, *_a: Any, **_kw: Any) -> str:
                return "backup-async"

        fb = FallbackLLM(TimeoutPrimary(), [OkBackup()], labels=["primary", "backup"])
        assert fb.invoke(["hi"]) == "backup-sync"
        assert await fb.ainvoke(["hi"]) == "backup-async"

        joined = "\n".join(captured)
        assert len(captured) >= 4
        for leaked in (
            "supersecret123",
            "PRIVATE_PROMPT_BODY",
            "u1",
            "p1",
            "sig=9",
            "AIzaSyD1234567890abcdefghijk",
            "apim-secret-999",
        ):
            assert leaked not in joined, f"{leaked!r} leaked in logs:\n{joined}"
        assert "unsupported_value" in joined
    finally:
        logger.remove(sink_id)


def test_sanitize_malformed_nested_brackets_and_structured_output_fallback() -> None:
    from types import SimpleNamespace

    from pydantic import BaseModel

    from agentomatic.providers.llm import StructuredOutputFallbackWrapper

    malformed_nested = '{"messages": [["MY_PRIVATE_DOCUMENT_TEXT", "more"]], invalid}'
    san = _sanitize_error_for_log(malformed_nested)
    assert "MY_PRIVATE_DOCUMENT_TEXT" not in san
    assert "more" not in san
    assert "messages=[REDACTED]" in san

    malformed_with_error = (
        '400 {"messages": [["MY_PRIVATE_DOCUMENT_TEXT", "more"]], '
        '"error": {"message": "bad", "code": "unsupported_value"}, invalid}'
    )
    san_err = _sanitize_error_for_log(malformed_with_error)
    assert "MY_PRIVATE_DOCUMENT_TEXT" not in san_err
    assert '"message": "bad"' in san_err
    assert "unsupported_value" in san_err

    truncated = '{"messages": [["MY_PRIVATE_DOCUMENT_TEXT", "more"'
    san_trunc = _sanitize_error_for_log(truncated)
    assert "MY_PRIVATE_DOCUMENT_TEXT" not in san_trunc
    assert "more" not in san_trunc

    nested_dict_malformed = '{"raw_request": {"nested": ["MY_PRIVATE_DOCUMENT_TEXT"]}, invalid}'
    san_dict = _sanitize_error_for_log(nested_dict_malformed)
    assert "MY_PRIVATE_DOCUMENT_TEXT" not in san_dict
    assert "raw_request=[REDACTED]" in san_dict

    # Pathological nested/messages brackets stay strictly O(N) < 0.1s
    t0 = time.perf_counter()
    san_msg_brackets = _sanitize_error_for_log('{"messages": ' + ("[" * 20000) + "SECRET_TAIL")
    assert time.perf_counter() - t0 < 0.1
    assert "SECRET_TAIL" not in san_msg_brackets

    # StructuredOutputFallbackWrapper._parse_to_model must redact nested malformed content
    class _Out(BaseModel):
        answer: str

    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        wrapper = StructuredOutputFallbackWrapper(SimpleNamespace(), _Out)
        parsed = wrapper._parse_to_model(SimpleNamespace(content=malformed_nested))
        assert isinstance(parsed, _Out)
        joined = "\n".join(captured)
        assert "Structured output parse failed" in joined
        assert "MY_PRIVATE_DOCUMENT_TEXT" not in joined
    finally:
        logger.remove(sink_id)
