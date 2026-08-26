"""Contracts for the runnable Docker/oMLX demonstration agents."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).parents[1]


def test_default_compose_demo_contains_a_real_omlx_agent() -> None:
    """The advertised local stack must verify a model call, not only an echo."""
    source = (REPO / "e2e_demo" / "agents" / "omlx_echo" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "AsyncOpenAI" in source
    assert "host.docker.internal:8000/v1" in source
    assert "AGENTOMATIC_TASK_MODEL" in source
    assert "Do not reveal chain-of-thought" in source
    assert "enable_thinking" in source


def test_e2e_demo_does_not_advertise_a_stale_framework_version() -> None:
    """Demo output must derive its version from the installed package."""
    source = (REPO / "e2e_demo" / "run_demo.py").read_text(encoding="utf-8")

    assert "Agentomatic v{__version__}" in source
    assert "v0.4.1" not in source


def test_e2e_fixture_includes_a_structured_agent_without_a_query_field() -> None:
    """Studio must exercise deployed schemas that are not chat-shaped."""
    source = (REPO / "e2e_demo" / "agents" / "classifier" / "schemas.py").read_text(
        encoding="utf-8"
    )

    assert "class CustomInvokeRequest" in source
    assert "label: str" in source
    assert "query:" not in source
