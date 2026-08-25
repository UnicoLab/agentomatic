"""Redaction helpers for displaying stack configuration.

Stack YAML is *meant* to reference secrets indirectly (``api_key: ${OPENAI_API_KEY}``),
but nothing enforces that convention — a literal key or a database URL with
embedded credentials is perfectly valid YAML. Commands that print a stack to a
terminal therefore have to assume the file may contain real secrets, because
that output lands in scrollback, CI logs, and screen shares.

These helpers redact secret-*looking* values while leaving ``${ENV_VAR}``
references intact (those are safe, and hiding them would obscure the very thing
an operator is trying to verify).
"""

from __future__ import annotations

import re

REDACTED = "***REDACTED***"

#: Key names whose values are treated as secrets.
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|credential|private[_-]?key|access[_-]?key)",
    re.IGNORECASE,
)

#: ``key: value`` on a YAML line (captures indent, key, value, trailing comment).
_YAML_PAIR_RE = re.compile(r"^(?P<indent>\s*(?:-\s*)?)(?P<key>[\w.\-]+)\s*:\s*(?P<value>.*)$")

#: ``scheme://user:password@host`` — credentials embedded in a URL.
_URL_CREDENTIALS_RE = re.compile(
    r"(?P<scheme>[a-zA-Z][\w+.-]*://)(?P<user>[^:/@\s]+):(?P<pw>[^@/\s]+)@"
)

#: A pure ``${VAR}`` / ``$VAR`` indirection — not a secret itself.
_ENV_REF_RE = re.compile(r"^\$\{[^}]+\}$|^\$[A-Za-z_][A-Za-z0-9_]*$")


def _is_env_reference(value: str) -> bool:
    """Whether *value* is only an environment-variable reference."""
    stripped = value.strip().strip("\"'")
    return bool(_ENV_REF_RE.match(stripped))


def redact_url_credentials(text: str) -> str:
    """Mask the password in any ``scheme://user:password@host`` URL in *text*."""
    return _URL_CREDENTIALS_RE.sub(
        lambda m: f"{m.group('scheme')}{m.group('user')}:{REDACTED}@", text
    )


def redact_secret_value(key: str, value: str) -> str:
    """Return *value* redacted when *key* names a secret.

    ``${ENV_VAR}`` references, empty values, and YAML block/flow openers are
    left alone — they carry no secret and hiding them would only obscure the
    structure an operator is inspecting.
    """
    bare = value.strip()
    if not bare or bare in {"~", "null", "{}", "[]", "|", ">"}:
        return value
    if _is_env_reference(bare):
        return value
    if _SECRET_KEY_RE.search(key):
        # Preserve any trailing comment so the file still reads naturally.
        comment = ""
        if " #" in value:
            bare, comment = value.split(" #", 1)
            comment = f" #{comment}"
        return f"{REDACTED}{comment}"
    return value


#: Placeholder written into generated ``.env.example`` files in place of a
#: literal secret. Actionable rather than merely masked, because the operator
#: is expected to fill it in.
ENV_EXAMPLE_PLACEHOLDER = "CHANGEME"


def env_example_value(value: str) -> str:
    """Return a value safe to write into a committed ``.env.example``.

    ``${ENV_VAR}`` references pass through — showing them is the whole point
    of the file. A literal secret is replaced with a placeholder: unlike a
    stack file (which may be gitignored), ``.env.example`` is conventionally
    committed, so writing a real key there publishes it.
    """
    if not value or _is_env_reference(value):
        return value
    return ENV_EXAMPLE_PLACEHOLDER


def redact_yaml_text(text: str) -> tuple[str, int]:
    """Redact secret-looking values in YAML *text*.

    Operates line-by-line so comments, ordering, and formatting survive — the
    point is to show the operator their real file, minus the secrets.

    Args:
        text: Raw YAML source.

    Returns:
        ``(redacted_text, number_of_redactions)``.
    """
    out: list[str] = []
    redactions = 0

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or not stripped:
            out.append(line)
            continue

        match = _YAML_PAIR_RE.match(line)
        if match:
            key, value = match.group("key"), match.group("value")
            new_value = redact_secret_value(key, value)
            if new_value != value:
                redactions += 1
                out.append(f"{match.group('indent')}{key}: {new_value}")
                continue

        # Even on non-secret keys, a URL may carry inline credentials.
        masked = redact_url_credentials(line)
        if masked != line:
            redactions += 1
        out.append(masked)

    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + trailing_newline, redactions
