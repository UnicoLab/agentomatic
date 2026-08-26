# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Quality gate for the code `agentomatic init` scaffolds.

Generated projects are the first thing a user sees, so a template that emits
code which does not compile — or that trips the linter the project itself
recommends — is a release defect. These tests render every template and check
the output the same way a user's own CI would.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import install_plugin_package

from agentomatic.cli.templates import TEMPLATES, get_template_files

# Templates render Python plus supporting files; only .py files are compiled.
_PY_SUFFIX = ".py"

# No allowances: every template must render lint-clean as-is. The scripts that
# genuinely need a ``sys.path`` bootstrap before their project imports carry a
# file-level E402 suppression of their own, so they satisfy the linter without
# this gate having to look the other way.
_ALLOWED_RUFF_CODES: set[str] = set()


def _rendered(template: str) -> dict[str, str]:
    return get_template_files(template, "sample_agent")


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_every_template_emits_syntactically_valid_python(template: str) -> None:
    """Every generated .py file must parse."""
    files = _rendered(template)
    py_files = {p: c for p, c in files.items() if p.endswith(_PY_SUFFIX)}
    assert py_files, f"template {template!r} generated no Python files"

    for path, content in py_files.items():
        try:
            ast.parse(content, filename=path)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{template}/{path} is not valid Python: {exc}")


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_every_template_passes_ruff(template: str, tmp_path: Path) -> None:
    """Generated code must be clean under the linter (bar allowed codes).

    A scaffold that ships lint-dirty code makes a new project fail its own
    first CI run.
    """
    files = _rendered(template)
    for rel_path, content in files.items():
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            # Deliberately ruff's *default* 88-char line length, not the
            # project's 99: a scaffolded project starts with no ruff config,
            # so the defaults are what its first CI run will enforce.
            "--select",
            "E,F,W,I",
            "--output-format",
            "concise",
            ".",
        ],
        capture_output=True,
        text=True,
        check=False,
        # Run from inside the scaffolded dir, exactly as a user's CI would.
        # From the repo root, ruff's isort resolves ``src/agentomatic`` and
        # would misclassify ``agentomatic`` as a first-party import.
        cwd=tmp_path,
    )
    if result.returncode == 0:
        return

    offending = [
        line
        for line in result.stdout.splitlines()
        if line.strip()
        # Drop ruff's trailing summary lines ("Found N errors.", "[*] N fixable…")
        and ":" in line
        and not any(f" {code} " in line for code in _ALLOWED_RUFF_CODES)
    ]
    assert not offending, f"template {template!r} generates lint-dirty code:\n" + "\n".join(
        offending
    )


def test_langchain_template_is_reachable_from_the_cli() -> None:
    """Regression: ``langchain`` shipped in the registry but was missing from
    the CLI's ``--template`` choices, so it could not actually be scaffolded.
    """
    from click.types import Choice

    from agentomatic.cli.commands import init as init_cmd

    template_opt = next(p for p in init_cmd.params if p.name == "template")
    assert isinstance(template_opt.type, Choice)
    choices = set(template_opt.type.choices)

    assert "langchain" in choices
    # The choice list is derived from the registry, so it can never drift again.
    assert choices == set(TEMPLATES), (
        "CLI --template choices have drifted from the TEMPLATES registry: "
        f"missing={set(TEMPLATES) - choices} extra={choices - set(TEMPLATES)}"
    )


def test_langchain_template_demonstrates_the_advertised_abstractions() -> None:
    """The ``langchain`` template's description promises specific LangChain
    abstractions — the generated code must actually use them.
    """
    agent_py = _rendered("langchain")["agent.py"]
    for expected in (
        "ChatPromptTemplate",
        "MessagesPlaceholder",
        "make_config",  # builds the RunnableConfig
        "self.prompt_template | self.llm",  # a real LCEL chain
    ):
        assert expected in agent_py, f"langchain template does not use {expected!r}"


# =====================================================================
# Scaffolded ML plugin must actually be usable
# =====================================================================


def test_plugin_template_sets_its_own_name() -> None:
    """Without ``plugin_name`` the scaffold inherits BaseMLPlugin's
    ``default_plugin``, so it mounts at /api/v1/plugins/default_plugin/* and a
    second scaffolded plugin silently collides with the first.
    """
    plugin_py = get_template_files("plugin", "sentiment")["plugin.py"]
    assert 'plugin_name = "sentiment"' in plugin_py


def test_plugin_template_marks_itself_loaded() -> None:
    """Overriding ``load_model`` without calling super() leaves ``_is_loaded``
    False: /predict answers 503 and /health reports the platform "degraded",
    while startup logs claim the plugin loaded successfully.
    """
    plugin_py = get_template_files("plugin", "sentiment")["plugin.py"]
    assert "await super().load_model()" in plugin_py


def test_scaffolded_plugin_serves_predictions_and_reports_healthy(tmp_path) -> None:
    """End-to-end: render the plugin template, mount it, and call /predict."""
    from fastapi.testclient import TestClient

    from agentomatic import AgentPlatform

    plugins_dir = tmp_path / "plugins"
    files = get_template_files("plugin", "sentiment")
    importable = install_plugin_package(plugins_dir, "sentiment", files["plugin.py"])
    for rel, content in files.items():
        if rel == "plugin.py":
            continue
        path = plugins_dir / "sentiment" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    with importable:
        platform = AgentPlatform(
            agents_dir=tmp_path / "agents",
            plugins_dir=plugins_dir,
            endpoints_dir=tmp_path / "endpoints",
        )
        with TestClient(platform.build()) as client:
            listed = client.get("/api/v1/plugins").json()
            entries = listed if isinstance(listed, list) else listed.get("plugins", [])
            assert [e.get("name") for e in entries] == ["sentiment"]

            assert client.get("/health").json()["status"] == "healthy"

            response = client.post("/api/v1/plugins/sentiment/predict", json={"text": "hi"})
            assert response.status_code == 200, response.text


def test_full_template_response_schema_matches_its_agent_output() -> None:
    """`schemas.py` required ``answer`` while the agent returned ``response``,
    so every invoke logged an output-validation warning.
    """
    files = get_template_files("full", "sample_agent")
    schemas_py, agent_py = files["schemas.py"], files["agent.py"]

    assert "response: str" in schemas_py
    assert "answer: str" not in schemas_py
    # The agent really does emit "response".
    assert '"response": text' in agent_py


def test_scaffolded_main_explains_a_version_skew_instead_of_a_bare_typeerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An older installed agentomatic must fail with actionable guidance.

    ``main.py`` is generated against the version that scaffolded it. If the
    image pins an older release (a stale ``requirements.txt`` line or a
    Dockerfile pin that predates a new option), ``AgentPlatform`` raises a bare
    ``unexpected keyword argument`` at import time and the container dies with
    no hint about the cause.
    """
    import agentomatic
    from agentomatic.cli.project import _main_py

    class _OldPlatform:
        @staticmethod
        def from_folder(*args: object, **kwargs: object) -> object:
            raise TypeError(
                "AgentPlatform.__init__() got an unexpected keyword argument "
                "'rate_limit_trust_proxy_headers'"
            )

    monkeypatch.setattr(agentomatic, "AgentPlatform", _OldPlatform)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        exec(compile(_main_py("demo"), "main.py", "exec"), {"__name__": "main"})

    message = str(excinfo.value)
    assert "rate_limit_trust_proxy_headers" in message
    assert "older than the one this project was generated with" in message
    assert "requirements.txt" in message
    # The original TypeError stays chained so the traceback is not lost.
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_scaffolded_main_does_not_swallow_unrelated_type_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only unknown-keyword failures are reframed; real bugs propagate as-is."""
    import agentomatic
    from agentomatic.cli.project import _main_py

    class _BrokenPlatform:
        @staticmethod
        def from_folder(*args: object, **kwargs: object) -> object:
            raise TypeError("unhashable type: 'dict'")

    monkeypatch.setattr(agentomatic, "AgentPlatform", _BrokenPlatform)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TypeError, match="unhashable"):
        exec(compile(_main_py("demo"), "main.py", "exec"), {"__name__": "main"})


def test_deepagent_template_names_its_missing_dependency() -> None:
    """The deepagent template imports a package agentomatic does not install.

    Without it the agent scaffolds, registers, and reports healthy, but every
    invocation returns a sanitised 500 whose only clue is the exception type
    ``ModuleNotFoundError`` — the caller cannot tell what to install.
    """
    files = get_template_files("deepagent", "mydeep")
    agent_py = files["agent.py"]

    assert "pip install deepagents" in agent_py
    assert "except ImportError" in agent_py
    # The bare import must not remain outside the guard.
    guarded = agent_py.split("try:", 1)[1]
    assert "from deepagents import create_deep_agent" in guarded


def test_extraction_template_uses_the_pipeline_context_step_shape() -> None:
    """Scaffolded extraction pipelines must address step outputs directly."""
    pipeline = get_template_files("extraction", "extractor")["pipeline.yaml"]

    assert "$.steps.to_md.path" in pipeline
    assert "$.steps.to_md.output.path" not in pipeline


def test_deepagent_scaffold_tells_the_user_to_install_it(tmp_path: Path) -> None:
    """``agentomatic init --template deepagent`` must surface the dependency."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agentomatic.cli.commands import cli; cli()",
            "init",
            "mydeep",
            "--template",
            "deepagent",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "pip install deepagents" in result.stdout + result.stderr
