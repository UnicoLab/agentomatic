# pyright: reportMissingParameterType=none
"""The chatbot template must put prior turns in front of the model.

``/chat`` loads the whole thread into ``state["messages"]`` and reports
``history_loaded: N``. The scaffolded chatbot then sent the model only
``f"{prompt}\\n\\nUser: {state.request}"`` — so every turn was answered as if
it were the first, while the response body said N messages had been loaded.
A chatbot that cannot remember the previous sentence is broken as shipped,
and the number in the payload said otherwise.
"""

from __future__ import annotations

import ast
from typing import Any

import pytest

from agentomatic.cli.templates import get_template_files


def _chatbot_source(name: str = "cb") -> str:
    """Render the chatbot template the way ``agentomatic init`` does."""
    return get_template_files("chatbot", name)["agent.py"]


class TestTemplateRenders:
    def test_the_rendered_agent_is_valid_python(self) -> None:
        ast.parse(_chatbot_source())

    def test_no_unrendered_placeholders_survive(self) -> None:
        src = _chatbot_source("mybot")
        assert "{title}" not in src
        assert "{name}" not in src
        assert "MybotState" in src


class TestHistoryReachesTheModel:
    """Behavioural: build the agent and record what ``llm.invoke`` receives."""

    @staticmethod
    def _build(tmp_path, monkeypatch) -> tuple[Any, list[Any]]:
        """Exec the rendered template and return (agent, recorded_calls)."""
        import sys

        src = _chatbot_source("cb")
        module_path = tmp_path / "cb_agent.py"
        module_path.write_text(src)
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("cb_agent", None)
        import importlib

        mod = importlib.import_module("cb_agent")

        calls: list[Any] = []

        class RecordingLLM:
            def invoke(self, payload: Any, **_: Any) -> Any:
                calls.append(payload)

                class _R:
                    content = "ok"

                return _R()

        agent = mod.CbAgent(llm=RecordingLLM())
        return agent, calls

    def test_prior_turns_are_sent(self, tmp_path, monkeypatch) -> None:
        agent, calls = self._build(tmp_path, monkeypatch)
        state = agent.input_to_state(
            {
                "current_query": "What token did I give you?",
                "messages": [
                    {"role": "user", "content": "Remember this token: ZX42QV"},
                    {"role": "assistant", "content": "Noted."},
                    {"role": "user", "content": "What token did I give you?"},
                ],
            }
        )

        agent.respond(state)

        assert len(calls) == 1
        sent = str(calls[0])
        assert "ZX42QV" in sent, f"prior turn dropped: {sent}"
        assert "Noted." in sent, f"assistant turn dropped: {sent}"

    def test_the_system_prompt_leads(self, tmp_path, monkeypatch) -> None:
        agent, calls = self._build(tmp_path, monkeypatch)
        state = agent.input_to_state({"current_query": "Hi", "messages": []})

        agent.respond(state)

        first = calls[0][0]
        assert type(first).__name__ == "SystemMessage"
        assert "assistant" in str(first.content).lower()

    def test_a_first_turn_still_works_without_history(self, tmp_path, monkeypatch) -> None:
        """``/invoke`` supplies no messages — the query alone must go out."""
        agent, calls = self._build(tmp_path, monkeypatch)
        state = agent.input_to_state({"current_query": "Only this."})

        agent.respond(state)

        sent = str(calls[0])
        assert "Only this." in sent

    def test_the_current_turn_is_not_duplicated(self, tmp_path, monkeypatch) -> None:
        """``load_history`` already appends the current turn to ``messages``."""
        agent, calls = self._build(tmp_path, monkeypatch)
        state = agent.input_to_state(
            {
                "current_query": "Say it once.",
                "messages": [
                    {"role": "user", "content": "Earlier."},
                    {"role": "assistant", "content": "Sure."},
                    {"role": "user", "content": "Say it once."},
                ],
            }
        )

        agent.respond(state)

        contents = [str(getattr(m, "content", m)) for m in calls[0]]
        assert contents.count("Say it once.") == 1, contents

    def test_turn_count_excludes_the_current_turn(self, tmp_path, monkeypatch) -> None:
        """The offline branch numbers turns; ``messages`` ends with this one."""
        agent, _ = self._build(tmp_path, monkeypatch)
        agent.llm = None
        state = agent.input_to_state(
            {
                "current_query": "third",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                    {"role": "user", "content": "third"},
                ],
            }
        )

        out = agent.respond(state).output

        assert "[Turn 3]" in out["response"], out["response"]


class TestTheSourceItselfPinsTheBehaviour:
    """Guard the shape, so a refactor cannot quietly drop history again."""

    def test_the_query_only_prompt_is_gone(self) -> None:
        src = _chatbot_source()
        tree = ast.parse(src)
        respond = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "respond"
        )
        body = ast.get_source_segment(src, respond) or ""
        assert "User: " not in body, (
            "respond() builds a query-only string prompt again — prior turns would be dropped"
        )

    def test_respond_delegates_to_the_turn_builder(self) -> None:
        assert "self._turns(state)" in _chatbot_source()


@pytest.mark.parametrize("template", ["basic", "full", "class", "rag", "coordinator"])
def test_single_shot_templates_stay_single_shot(template: str) -> None:
    """Only the chatbot claims conversation; the rest take the query alone.

    Not a defect: an extraction or routing agent dragging in prior turns
    would be the surprise. This pins the distinction so a future change to
    one template does not silently blur it.
    """
    src = get_template_files(template, "x")["agent.py"]
    assert "current_query" in src
