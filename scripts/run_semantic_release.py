#!/usr/bin/env python3
"""Run python-semantic-release with supported GitPython versions.

python-semantic-release 10.6.1 still reads ``Actor.name_email_regex``, an
internal GitPython attribute removed in 3.1.45. Restore the identical parser
only for the release process, keeping the production and CI dependency set on
security-patched GitPython versions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from git import Actor
from semantic_release.__main__ import main

if not hasattr(Actor, "name_email_regex"):
    Actor.name_email_regex = re.compile(r"(.*) <(.*?)>")  # type: ignore[attr-defined]


CHANGELOG_MARKER = "<!-- version list -->"


def _validate_changelog_contract(args: list[str]) -> None:
    """Fail before a release when semantic-release cannot update the changelog."""
    if "version" not in args or "--no-changelog" in args:
        return
    changelog = Path("CHANGELOG.md")
    if not changelog.is_file() or CHANGELOG_MARKER not in changelog.read_text(encoding="utf-8"):
        raise SystemExit(
            "Release aborted: CHANGELOG.md must contain "
            f"{CHANGELOG_MARKER!r} for python-semantic-release."
        )


if __name__ == "__main__":
    _validate_changelog_contract(sys.argv[1:])
    main()
