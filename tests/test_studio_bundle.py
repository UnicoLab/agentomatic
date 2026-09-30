"""The prebuilt Studio bundle must be loadable and free of known defects.

The bundle under ``src/agentomatic/studio/static`` is built from the
``agentomatic-studio`` repository. These checks run on the shipped files:

* every asset ``index.html`` and the chunks reference exists (assets are
  served ``immutable``, so a changed file is re-hashed and every reference
  must follow);
* the Chat tab sends the typed message as ``query`` — a string message was
  dropped from the request body, so every agent answered an empty question;
* message ids are normalised to strings — SQL / memory stores return integer
  ids and ``id.includes(...)`` crashed the chat view;
* operation contracts resolve templated routes, so the Task Board can load
  the input form of ``/api/v1/pipelines/{name}/run``.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "agentomatic" / "studio" / "static"
ASSETS = STATIC / "assets"


def _entry() -> str:
    match = re.search(
        r'src="/studio/ui/assets/(index-[\w-]+\.js)"', (STATIC / "index.html").read_text()
    )
    assert match, "index.html has no entry script"
    return (ASSETS / match.group(1)).read_text(encoding="utf-8")


def test_every_referenced_asset_exists() -> None:
    names = {p.name for p in ASSETS.iterdir()}
    sources = [(STATIC / "index.html").read_text()] + [
        p.read_text(encoding="utf-8") for p in ASSETS.glob("*.js")
    ]
    referenced = {
        ref for text in sources for ref in re.findall(r"(?:assets/|\./)([\w-]+\.(?:js|css))", text)
    }
    missing = sorted(ref for ref in referenced if ref not in names)
    assert not missing, missing


def test_chat_sends_a_typed_message_as_query() -> None:
    entry = _entry()
    # The defect: ``query`` was only sent when the input *object* had one.
    buggy = re.search(r"\.\.\.(\w)&&\((\w)\|\|(\w)\.query\)\?\{query:\2\|\|\3\.query\}", entry)
    assert buggy is None, "string chat input is dropped from the run request"


def test_message_ids_are_strings() -> None:
    entry = _entry()
    assert "normalizeMessage(e,t){return{id:e?.id||" not in entry
    chat = next(ASSETS.glob("ChatView-*.js")).read_text(encoding="utf-8")
    assert not re.search(r"\b\w\.id\.includes\(", chat)


def test_operation_contracts_match_templated_paths() -> None:
    entry = _entry()
    start = entry.index("async getOperationContract(")
    body = entry[start : start + 1500]
    assert "Object.keys(" in body and "RegExp(" in body, "no templated-path fallback"
