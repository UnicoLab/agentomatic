# pyright: reportMissingParameterType=none
# pyright: reportAttributeAccessIssue=none
"""Every chat-capable template must chat, thread and run as a task out of the box.

Regression for a freshly scaffolded project (``agentomatic new`` +
``agentomatic init NAME --template basic|full|...``) where talking to the agent
from Studio failed with ``400``:

* Studio's "New Chat" creates a thread first (``POST /api/v1/{agent}/threads``).
  No store is configured in a fresh project, so every thread route answered
  ``400 "Thread storage not configured"`` and no conversation could start.
* Studio runs never loaded the thread's history nor saved the turn, so a
  chatbot answered each message as if it were the first and a reopened thread
  was empty.
* The ``basic`` / ``full`` templates sent the model a single string and ignored
  ``state.messages`` entirely.

The project is scaffolded with the real CLI and served from its own
``main.py`` — the same app ``agentomatic run`` / ``uvicorn main:app`` serve.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from agentomatic.cli.commands import cli
from agentomatic.cli.project import scaffold_project

#: template -> agent name. Names are unique so their ``agents.<name>`` modules
#: cannot collide with other suites' scaffolds in the same interpreter.
TEMPLATES = {
    "basic": "tct_basic",
    "full": "tct_full",
    "chatbot": "tct_chat",
    "rag": "tct_rag",
    "coordinator": "tct_coord",
    "extraction": "tct_extract",
}

#: Templates whose single LLM call is a conversation (system + turns).
HISTORY_AWARE = ("basic", "full", "chatbot")


class RecordingLLM:
    """Stand-in chat model that records every payload it is sent."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def invoke(self, payload: Any, **_: Any) -> Any:
        self.calls.append(payload)

        class _Reply:
            content = f"reply #{len(self.calls)}"

        return _Reply()


def _texts(payload: Any) -> list[str]:
    """Flatten an LLM payload (str, dicts or LangChain messages) to texts."""
    if isinstance(payload, str):
        return [payload]
    out: list[str] = []
    for item in payload:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
        out.append(str(content))
    return out


def _purge_agent_modules() -> None:
    for mod in [m for m in sys.modules if m == "agents" or m.startswith("agents.")]:
        sys.modules.pop(mod, None)


@pytest.fixture(scope="module")
def project_app(tmp_path_factory):
    """Scaffold a project with one agent per template and build its ``main:app``."""
    root = tmp_path_factory.mktemp("tpl_chat")
    project = root / "proj"
    scaffold_project(project, "proj", force=True)
    runner = CliRunner()
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(project)
        mp.delenv("AGENTOMATIC_EPHEMERAL_THREADS", raising=False)
        mp.setenv("AGENTOMATIC_LOG_LEVEL", "WARNING")
        for template, name in TEMPLATES.items():
            result = runner.invoke(cli, ["init", name, "--template", template])
            assert result.exit_code == 0, result.output
        _purge_agent_modules()
        mp.syspath_prepend(str(project))
        namespace: dict[str, Any] = {}
        source = (project / "main.py").read_text()
        exec(compile(source, str(project / "main.py"), "exec"), namespace)  # noqa: S102
        platform = namespace["_platform"]
        llms: dict[str, RecordingLLM] = {}
        for name in TEMPLATES.values():
            agent = platform._registry.get(name)
            assert agent is not None, f"agent {name} was not discovered"
            llms[name] = RecordingLLM()
            agent.class_instance.llm = llms[name]
        with TestClient(namespace["app"]) as client:
            yield client, llms
    _purge_agent_modules()


def _stream_turn(client: TestClient, agent: str, thread_id: str, query: str) -> list[str]:
    """Send one Studio chat turn and return its SSE data frames."""
    with client.stream(
        "POST",
        f"/studio/agents/{agent}/runs/stream",
        json={"query": query, "thread_id": thread_id, "user_id": "studio-user"},
    ) as response:
        assert response.status_code == 200
        return [line for line in response.iter_lines() if line.startswith("data:")]


def _wait(client: TestClient, task_id: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/v1/tasks/{task_id}").json()
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish")


@pytest.mark.parametrize("template", list(TEMPLATES))
class TestStudioChat:
    def test_new_chat_thread_can_be_created(self, project_app, template: str) -> None:
        client, _ = project_app
        agent = TEMPLATES[template]
        response = client.post(
            f"/api/v1/{agent}/threads", json={"user_id": "studio-user", "title": "New Chat"}
        )
        assert response.status_code == 200, response.text
        thread_id = response.json()["id"]
        assert client.get(f"/api/v1/{agent}/threads/{thread_id}").status_code == 200
        listed = client.get(f"/api/v1/{agent}/threads").json()
        assert thread_id in {t["id"] for t in listed["threads"]}

    def test_studio_turns_are_persisted(self, project_app, template: str) -> None:
        client, _ = project_app
        agent = TEMPLATES[template]
        thread_id = client.post(f"/api/v1/{agent}/threads", json={"title": "t"}).json()["id"]
        for query in ("My name is Ada.", "What is my name?"):
            frames = _stream_turn(client, agent, thread_id, query)
            assert not any('"run_error"' in f for f in frames), frames
        messages = client.get(f"/api/v1/{agent}/threads/{thread_id}/messages").json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
        assert messages[0]["content"] == "My name is Ada."
        assert messages[1]["content"].strip()

    def test_thread_lifecycle_routes(self, project_app, template: str) -> None:
        client, _ = project_app
        agent = TEMPLATES[template]
        thread_id = client.post(f"/api/v1/{agent}/threads", json={"title": "t"}).json()["id"]
        _stream_turn(client, agent, thread_id, "hello")
        renamed = client.patch(f"/api/v1/{agent}/threads/{thread_id}", json={"title": "Renamed"})
        assert renamed.status_code == 200
        forked = client.post(
            f"/api/v1/{agent}/threads/{thread_id}/fork", json={"message_index": 0}
        )
        assert forked.status_code == 200, forked.text
        assert client.delete(f"/api/v1/{agent}/threads/{thread_id}").status_code == 200


@pytest.mark.parametrize("template", HISTORY_AWARE)
def test_prior_turns_reach_the_model_in_studio(project_app, template: str) -> None:
    client, llms = project_app
    agent = TEMPLATES[template]
    thread_id = client.post(f"/api/v1/{agent}/threads", json={"title": "t"}).json()["id"]
    _stream_turn(client, agent, thread_id, "Remember this token: ZX42QV")
    _stream_turn(client, agent, thread_id, "What token did I give you?")
    last = _texts(llms[agent].calls[-1])
    assert any("ZX42QV" in text for text in last), last
    assert last[-1] == "What token did I give you?"


@pytest.mark.parametrize("template", ("basic", "full"))
def test_basic_templates_send_a_real_system_message(project_app, template: str) -> None:
    client, llms = project_app
    agent = TEMPLATES[template]
    response = client.post(f"/api/v1/{agent}/invoke", json={"query": "ping"})
    assert response.status_code == 200, response.text
    payload = llms[agent].calls[-1]
    assert isinstance(payload, list)
    assert payload[0]["role"] == "system"
    assert payload[0]["content"].strip()
    assert _texts(payload)[-1] == "ping"


@pytest.mark.parametrize("template", list(TEMPLATES))
class TestTasks:
    def test_async_task_succeeds(self, project_app, template: str) -> None:
        client, _ = project_app
        agent = TEMPLATES[template]
        response = client.post(
            "/api/v1/tasks",
            json={"target_type": "agent", "target": agent, "input": {"query": "hello"}},
        )
        assert response.status_code == 202, response.text
        record = _wait(client, response.json()["id"])
        assert record["status"] == "succeeded", record
        assert record["result"]

    def test_sync_task_succeeds(self, project_app, template: str) -> None:
        client, _ = project_app
        agent = TEMPLATES[template]
        response = client.post(
            "/api/v1/tasks",
            json={
                "target_type": "agent",
                "target": agent,
                "input": {"query": "hello"},
                "mode": "sync",
                "wait": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "succeeded", response.json()

    def test_studio_task_board_defaults_are_accepted(self, project_app, template: str) -> None:
        """The Task Board submits the defaults of the published input schema."""
        client, _ = project_app
        agent = TEMPLATES[template]
        schema = client.get(f"/studio/agents/{agent}/schemas").json()["input_schema"]
        payload: dict[str, Any] = {}
        for key, prop in (schema.get("properties") or {}).items():
            if "default" in prop:
                payload[key] = prop["default"]
            elif prop.get("type") == "object":
                payload[key] = {}
        payload["query"] = "from the task board"
        response = client.post(
            "/api/v1/tasks",
            json={
                "target_type": "agent",
                "target": agent,
                "input": payload,
                "mode": "sync",
                "wait": True,
                "retry": {"max_attempts": 3, "backoff": "exponential", "base_delay": 1},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "succeeded", response.json()

    def test_task_input_is_not_mutated_by_the_run(self, project_app, template: str) -> None:
        client, _ = project_app
        agent = TEMPLATES[template]
        response = client.post(
            "/api/v1/tasks",
            json={
                "target_type": "agent",
                "target": agent,
                "input": {"query": "hi", "metadata": {}},
                "mode": "sync",
                "wait": True,
            },
        )
        stored = response.json()["input"]
        # A custom request schema (``full``) may not publish ``metadata``.
        assert stored.get("metadata", {}) == {}


def test_a2a_task_round_trip(project_app) -> None:
    client, _ = project_app
    agent = TEMPLATES["basic"]
    response = client.post(f"/api/v1/{agent}/a2a/tasks", json={"message": {"content": "hi"}})
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        view = client.get(f"/api/v1/{agent}/a2a/tasks/{task_id}").json()
        if view["status"] in {"completed", "failed", "canceled"}:
            break
        time.sleep(0.05)
    assert view["status"] == "completed", view


def test_ephemeral_store_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    """``AGENTOMATIC_EPHEMERAL_THREADS=0`` restores the explicit no-store posture."""
    from agentomatic import AgentManifest, AgentPlatform

    async def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": "ok"}

    monkeypatch.setenv("AGENTOMATIC_EPHEMERAL_THREADS", "0")
    platform = AgentPlatform(agents_dir=tmp_path / "agents")
    platform.register_agent(manifest=AgentManifest(name="e1", slug="e1"), node_fn=echo)
    with TestClient(platform.build()) as client:
        response = client.post("/api/v1/e1/threads", json={"title": "t"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Thread storage not configured"
