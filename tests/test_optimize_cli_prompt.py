# pyright: reportMissingParameterType=none
"""``--prompt`` must seed the baseline in every optimization mode.

Regression: the option is documented as "Initial prompt (overrides
prompts.json)" and reached the legacy ``prompt_only`` path only. In all six
fitter modes it was accepted and silently dropped, so the run optimized from
the agent's own prompt and reported a *baseline score for a prompt the caller
never asked for* — which is the number every conclusion about the run rests on.
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from agentomatic.cli.commands import cli

FITTER_MODES = ("rewrite", "param_search", "gepa_like", "mipro_like", "few_shot", "apo")


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Invoke the CLI without running a fit, capturing PromptFitter kwargs."""
    seen: dict[str, Any] = {}

    class _FakeFitter:
        def __init__(self, **kwargs: Any) -> None:
            seen.clear()
            seen.update(kwargs)

    class _FakeAlgorithm:
        def __init__(self, _fitter: Any) -> None:
            pass

        async def run(self, *args: Any, **kwargs: Any) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(
                best_score=0.0,
                baseline_score=0.0,
                best_config=SimpleNamespace(system_prompt="x"),
                improvement=0.0,
                experiment_id="test",
                duration_seconds=0.0,
                trials=0,
                summary=lambda: "",
            )

    # The CLI imports these from `agentomatic.optimize` inside the function,
    # so patch them where they are looked up.
    import agentomatic.optimize as optimize_pkg

    monkeypatch.setattr(optimize_pkg, "PromptFitter", _FakeFitter)
    monkeypatch.setattr(optimize_pkg, "as_algorithm", lambda f: _FakeAlgorithm(f))

    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"query": "q", "expected_answer": "a"}\n')

    def _invoke(*extra: str) -> dict[str, Any]:
        result = CliRunner().invoke(
            cli, ["optimize", "agent_x", "--dataset", str(dataset), "--no-report", *extra]
        )
        assert result.exit_code == 0, result.output
        return seen

    return _invoke


@pytest.mark.parametrize("mode", FITTER_MODES)
def test_prompt_seeds_the_baseline_in_every_fitter_mode(captured, mode: str) -> None:
    kwargs = captured("--mode", mode, "--prompt", "You are a vague assistant.")

    assert kwargs.get("baseline_system_prompt") == "You are a vague assistant."


def test_no_baseline_is_set_when_prompt_is_omitted(captured) -> None:
    """Without the flag the agent's own prompt stays the baseline."""
    kwargs = captured("--mode", "rewrite")

    assert "baseline_system_prompt" not in kwargs


def test_an_empty_prompt_does_not_seed_a_blank_baseline(captured) -> None:
    kwargs = captured("--mode", "rewrite", "--prompt", "")

    assert "baseline_system_prompt" not in kwargs
