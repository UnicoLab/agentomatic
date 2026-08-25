"""Prove conversation state survives losing the container that served it.

A deployment can pass every functional check and still lose a client's
history on the next restart: the store silently fell back to a SQLite file
inside the container, or a MEMORY-purpose connection quietly outranked
``DATABASE_URL``. Nothing in a single-process test run can tell the
difference — both write, both read back, and only a restart separates them.

Two phases, so the script stays deployment-agnostic. Run ``write``, replace
the deployment however you normally would (``docker rm -f`` and re-run,
``kubectl rollout restart``, redeploy), then run ``verify``::

    python scripts/durability_verify.py write  --base-url … --api-key … --agent ag_chatbot
    docker rm -f ag && docker run -d --name ag …            # or your equivalent
    python scripts/durability_verify.py verify --base-url … --api-key … --agent ag_chatbot

The thread id defaults to a fixed value so both phases address the same
thread; ``--thread`` overrides it. Exit code is ``0`` only when every check
passes.

The replacement has to be a *replacement*, not a restart: a container that
keeps its writable layer proves nothing about where the data lives.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx

#: Written in the ``write`` phase, each looked for again in ``verify``.
MESSAGES = (
    "Durability probe one: remember the token DURABLE-A1.",
    "Durability probe two: and the token DURABLE-B2.",
    "Durability probe three: that is all.",
)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record and print one assertion."""
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(f"{label}: {detail}")


def _client(base_url: str, api_key: str, timeout: float) -> httpx.Client:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    return httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)


def _messages(client: httpx.Client, agent: str, thread: str) -> tuple[int, list[Any]]:
    """Return ``(status, messages)`` for a thread.

    ``GET /threads/{id}`` returns thread *metadata* — the turns live one
    level down, under ``/messages``.
    """
    resp = client.get(f"/api/v1/{agent}/threads/{thread}/messages")
    if resp.status_code != 200:
        return resp.status_code, []
    body = resp.json()
    msgs = body.get("messages", body) if isinstance(body, dict) else body
    return 200, msgs if isinstance(msgs, list) else []


def phase_write(client: httpx.Client, agent: str, thread: str) -> None:
    """Write a thread and confirm it reads back before the restart."""
    print(f"Writing {len(MESSAGES)} message(s) to {agent}/{thread}\n")
    for msg in MESSAGES:
        resp = client.post(f"/api/v1/{agent}/chat", json={"content": msg, "thread_id": thread})
        check(f"POST /chat {msg[:34]!r}", resp.status_code == 200, f"HTTP {resp.status_code}")

    code, msgs = _messages(client, agent, thread)
    check(
        "thread readable before the restart",
        code == 200 and len(msgs) >= len(MESSAGES),
        f"HTTP {code}, {len(msgs)} message(s)",
    )

    print(
        "\nNow replace the deployment — destroy the container and start a new one "
        "from the same image against the same database — then re-run with 'verify'."
    )


def phase_verify(client: httpx.Client, agent: str, thread: str) -> None:
    """Read the thread back and confirm the deployment can continue it."""
    print(f"Reading {agent}/{thread} back after the restart\n")
    code, msgs = _messages(client, agent, thread)
    check(
        "thread survived the restart",
        code == 200 and bool(msgs),
        f"HTTP {code}, {len(msgs)} message(s) recovered",
    )

    text = " ".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
    for msg in MESSAGES:
        check(f"message survived: {msg[:34]!r}", msg in text)

    before = len(msgs)
    resp = client.post(
        f"/api/v1/{agent}/chat",
        json={
            "content": "Durability probe four: appended after the restart.",
            "thread_id": thread,
        },
    )
    check(
        "the new deployment can continue the thread",
        resp.status_code == 200,
        f"HTTP {resp.status_code}",
    )

    code, after = _messages(client, agent, thread)
    check(
        "the recovered thread grew",
        code == 200 and len(after) > before,
        f"{before} -> {len(after)} message(s)",
    )


def main() -> int:
    """Parse arguments, run the requested phase, print the verdict."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("phase", choices=("write", "verify"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--agent", required=True, help="a conversational agent, e.g. ag_chatbot")
    ap.add_argument("--thread", default="durability-probe")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    client = _client(args.base_url, args.api_key, args.timeout)
    try:
        if args.phase == "write":
            phase_write(client, args.agent, args.thread)
        else:
            phase_verify(client, args.agent, args.thread)
    finally:
        client.close()

    print()
    if failures:
        print(f"DURABILITY FAILED — {len(failures)} check(s)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if args.phase == "write":
        print("Written. Replace the deployment, then run the 'verify' phase.")
    else:
        print("DURABILITY PROVEN — conversation state survived losing the container.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
