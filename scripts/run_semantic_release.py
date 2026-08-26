#!/usr/bin/env python3
"""Run python-semantic-release with supported GitPython versions.

python-semantic-release 10.6.1 still reads ``Actor.name_email_regex``, an
internal GitPython attribute removed in 3.1.45. Restore the identical parser
only for the release process, keeping the production and CI dependency set on
security-patched GitPython versions.
"""

from __future__ import annotations

import re

from git import Actor
from semantic_release.__main__ import main


if not hasattr(Actor, "name_email_regex"):
    Actor.name_email_regex = re.compile(r"(.*) <(.*?)>")  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
