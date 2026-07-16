#!/usr/bin/env python3
"""Strip ANSI escapes / bat line-number prefixes from a commit message file.

Root cause: shells that alias ``cat`` to ``bat`` corrupt Conventional Commit
messages when agents use::

    git commit -m "$(cat <<'EOF'
    feat(scope): subject
    EOF
    )"

``bat`` renders the heredoc with color codes and line numbers, which then land
in the git object. Python Semantic Release cannot parse those subjects.

This commit-msg hook rewrites the message file in place before
``conventional-pre-commit`` runs. Prefer ``/bin/cat`` (or ``command cat``) in
commit HEREDOCs so the message never needs cleaning.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# CSI / OSC style ANSI sequences (colors, cursor, etc.).
_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")
# bat-style leading line numbers after color strip: "   1\ttext" or bare "   2".
_BAT_LINE_PREFIX_RE = re.compile(r"^[ \t]*\d+[ \t]+")
_BAT_LINE_ONLY_RE = re.compile(r"^[ \t]*\d+[ \t]*$")


def sanitize_commit_message(text: str) -> str:
    """Return *text* with ANSI codes and bat line-number prefixes removed."""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = _ANSI_RE.sub("", line)
        if _BAT_LINE_ONLY_RE.match(line):
            cleaned_lines.append("")
            continue
        line = _BAT_LINE_PREFIX_RE.sub("", line)
        cleaned_lines.append(line)
    # Preserve a trailing newline if the original had one (git expects it).
    body = "\n".join(cleaned_lines)
    if text.endswith("\n"):
        body += "\n"
    return body


def main(argv: list[str]) -> int:
    """Rewrite the commit-msg file at ``argv[1]`` after sanitizing."""
    if len(argv) < 2:
        print("usage: strip_commit_msg_ansi.py COMMIT_MSG_FILE", file=sys.stderr)
        return 2
    path = Path(argv[1])
    original = path.read_text(encoding="utf-8")
    cleaned = sanitize_commit_message(original)
    if cleaned != original:
        path.write_text(cleaned, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
