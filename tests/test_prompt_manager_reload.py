"""PromptManager live-reload and BaseGraphAgent.resolve_system_prompt tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agentomatic.agents.base import BaseGraphAgent
from agentomatic.prompts.manager import PromptManager


def _write_prompts(path: Path, system: str) -> None:
    path.write_text(
        json.dumps({"v1": {"system": system, "user_template": "{query}"}}),
        encoding="utf-8",
    )


def test_prompt_manager_remembers_source_and_reloads(tmp_path: Path) -> None:
    """Disk edits are visible after ``reload_from_disk`` without a new manager."""
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, "ORIGINAL")

    pm = PromptManager("demo", prompts_file=prompts)
    assert pm.source_path == prompts
    assert pm.get_prompt("v1", "system") == "ORIGINAL"

    _write_prompts(prompts, "UPDATED_FROM_UI")
    assert pm.get_prompt("v1", "system") == "ORIGINAL"  # still cached

    assert pm.reload_from_disk() is True
    assert pm.get_prompt("v1", "system") == "UPDATED_FROM_UI"


def test_reload_without_path_uses_source(tmp_path: Path) -> None:
    """``reload()`` with no args reloads the remembered source path."""
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, "A")
    pm = PromptManager("demo", prompts_file=prompts)
    _write_prompts(prompts, "B")
    pm.reload()
    assert pm.get_prompt("v1", "system") == "B"


def test_reload_from_disk_false_without_source() -> None:
    """Managers created without a file have nothing to reload."""
    pm = PromptManager("orphan")
    assert pm.source_path is None
    assert pm.reload_from_disk() is False


class _TinyAgent(BaseGraphAgent[dict[str, Any]]):
    """Minimal concrete agent for resolve_system_prompt coverage."""

    def __init__(self, prompt_manager: PromptManager) -> None:
        super().__init__()
        self.prompt_manager = prompt_manager
        self.llm = MagicMock()

    def input_to_state(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return input_data

    def state_to_output(self, state: dict[str, Any]) -> dict[str, Any]:
        return state

    def build_graph(self) -> Any:
        return MagicMock()


def test_resolve_system_prompt_reloads_from_disk(tmp_path: Path) -> None:
    """Saved prompts.json edits apply on the next resolve without restart."""
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, "DISK_V1")
    pm = PromptManager("tiny", prompts_file=prompts)
    agent = _TinyAgent(prompt_manager=pm)

    assert agent.resolve_system_prompt(default="FALLBACK") == "DISK_V1"

    _write_prompts(prompts, "DISK_V2_AFTER_SAVE")
    assert agent.resolve_system_prompt(default="FALLBACK") == "DISK_V2_AFTER_SAVE"


def test_resolve_system_prompt_override_beats_disk(tmp_path: Path) -> None:
    """Per-request override still wins over a reloaded disk prompt."""
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, "DISK")
    pm = PromptManager("tiny", prompts_file=prompts)
    agent = _TinyAgent(prompt_manager=pm)

    text = agent.resolve_system_prompt(
        {"system_prompt_override": "OVERRIDE"},
        default="FALLBACK",
    )
    assert text == "OVERRIDE"
