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

from agentomatic.cli.templates import TEMPLATES, get_template_files

# Templates render Python plus supporting files; only .py files are compiled.
_PY_SUFFIX = ".py"

# ``train.py``/``eval.py`` scripts intentionally call ``load_dotenv()`` before
# importing project modules, which ruff flags as E402 (import not at top of
# file). That ordering is required for the scripts to work, so it is allowed.
_ALLOWED_RUFF_CODES = {"E402"}


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
            # Match the project's own line-length convention (see CLAUDE.md);
            # --isolated otherwise falls back to ruff's 88-char default.
            "--line-length",
            "99",
            "--select",
            "E,F,W",
            "--output-format",
            "concise",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
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
