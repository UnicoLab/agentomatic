# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
# pyright: reportArgumentType=none
# pyright: reportAttributeAccessIssue=none
"""Security regressions: path confinement, error sanitisation, secret redaction.

Each test here corresponds to a vulnerability that was reproduced by actually
executing it against a running platform, not inferred from reading code.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform

# =====================================================================
# Ingestion path confinement
# =====================================================================


class _StubIngestionContext:
    """Minimal ingestion context (the real one is a Protocol)."""

    cancelled = False

    async def report(self, **kwargs: Any) -> None:
        return None


@pytest.fixture
def ingest_env(monkeypatch, tmp_path):
    """An ingestion root plus a secret file safely outside it."""
    from agentomatic.ingestion.paths import INGESTION_ROOT_ENV

    root = tmp_path / "project"
    root.mkdir()
    (root / "legit.txt").write_text("legitimate project content", encoding="utf-8")

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("TOPSECRET-ABC123", encoding="utf-8")

    monkeypatch.setenv(INGESTION_ROOT_ENV, str(root))
    return {"root": root, "secret": secret, "out": root / "out"}


async def _ingest(**kwargs: Any):
    from agentomatic.ingestion.builtin.markdown import MarkdownIngestor, MarkdownIngestRequest

    kwargs.setdefault("engine", "plain")
    return await MarkdownIngestor().ingest(
        MarkdownIngestRequest(**kwargs), _StubIngestionContext()
    )


@pytest.mark.asyncio
async def test_ingestion_rejects_absolute_path_outside_root(ingest_env) -> None:
    """`source=/etc/passwd` must not exfiltrate arbitrary files."""
    result = await _ingest(source="/etc/passwd", output_dir=str(ingest_env["out"]))
    assert result.status == "failed"
    assert "outside the ingestion root" in result.errors[0]


@pytest.mark.asyncio
async def test_ingestion_rejects_reading_secret_outside_root(ingest_env) -> None:
    result = await _ingest(source=str(ingest_env["secret"]), output_dir=str(ingest_env["out"]))
    assert result.status == "failed"
    # The secret's contents must not have been copied anywhere.
    assert not list(ingest_env["out"].glob("*.md")) if ingest_env["out"].exists() else True


@pytest.mark.asyncio
async def test_ingestion_rejects_write_outside_root(ingest_env, tmp_path) -> None:
    target = tmp_path / "pwned_dir"
    result = await _ingest(source="legit.txt", output_dir=str(target))
    assert result.status == "failed"
    assert not target.exists(), "attacker-chosen directory must not be created"


@pytest.mark.asyncio
async def test_ingestion_rejects_filename_traversal(ingest_env) -> None:
    """output_filename is a name, not a path — traversal escapes the out dir."""
    result = await _ingest(
        source="legit.txt",
        output_dir=str(ingest_env["out"]),
        output_filename="../../pwned.txt",
    )
    assert result.status == "failed"
    assert "bare filename" in result.errors[0]


@pytest.mark.asyncio
async def test_ingestion_still_works_inside_the_root(ingest_env) -> None:
    """Confinement must not break legitimate in-project ingestion."""
    result = await _ingest(source="legit.txt", output_dir=str(ingest_env["out"]))
    assert result.status == "succeeded", result.errors
    assert (ingest_env["out"] / "legit.md").exists()


def test_ingestion_root_defaults_to_cwd(monkeypatch, tmp_path) -> None:
    from agentomatic.ingestion.paths import INGESTION_ROOT_ENV, ingestion_root

    monkeypatch.delenv(INGESTION_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert ingestion_root() == tmp_path.resolve()


def test_safe_output_filename_rejects_separators() -> None:
    from agentomatic.ingestion.paths import IngestionPathError, safe_output_filename

    assert safe_output_filename(None, default="x.md") == "x.md"
    assert safe_output_filename("report.md", default="x.md") == "report.md"
    for bad in ("../evil", "a/b.md", "..", "."):
        with pytest.raises(IngestionPathError):
            safe_output_filename(bad, default="x.md")


# =====================================================================
# Exception message sanitisation
# =====================================================================

_SECRET_DSN = "postgres://user:HUNTER2@db:5432"


@pytest.fixture
def leaky_client(tmp_path):
    async def boom(state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"connection failed: {_SECRET_DSN}")

    platform = AgentPlatform(
        agents_dir=tmp_path / "agents",
        plugins_dir=tmp_path / "plugins",
        endpoints_dir=tmp_path / "endpoints",
    )
    platform.register_agent(
        manifest=AgentManifest(name="boom", slug="boom", description="raises"),
        node_fn=boom,
    )
    with TestClient(platform.build(), raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/v1/boom/invoke", {"query": "x"}),
        ("/api/v1/boom/chat", {"content": "x"}),
        ("/api/v1/boom/invoke/stream", {"query": "x"}),
    ],
)
def test_exception_text_does_not_leak_to_clients(leaky_client, monkeypatch, path, payload) -> None:
    """A driver/exception message routinely carries credentials — it must not
    be interpolated into an HTTP response.
    """
    from agentomatic.core.errors import DEBUG_ERRORS_ENV

    monkeypatch.delenv(DEBUG_ERRORS_ENV, raising=False)
    response = leaky_client.post(path, json=payload)

    assert "HUNTER2" not in response.text
    assert _SECRET_DSN not in response.text
    # A correlation id is returned so operators can find the full server log.
    assert "error_id" in response.text


def test_debug_errors_opt_in_restores_raw_detail(leaky_client, monkeypatch) -> None:
    """Local development can opt back into raw exception text."""
    from agentomatic.core.errors import DEBUG_ERRORS_ENV

    monkeypatch.setenv(DEBUG_ERRORS_ENV, "1")
    response = leaky_client.post("/api/v1/boom/invoke", json={"query": "x"})
    assert "HUNTER2" in response.text


def test_client_safe_detail_shape() -> None:
    from agentomatic.core.errors import client_safe_detail

    payload = client_safe_detail(ValueError("secret-value"), context="Thing failed")
    assert payload["error"] == "Thing failed"
    assert payload["error_type"] == "ValueError"
    assert len(payload["error_id"]) == 12
    assert "secret-value" not in str(payload)


# =====================================================================
# Stack secret redaction
# =====================================================================

_STACK_YAML = """# Prod stack
name: prod
llm:
  default:
    provider: openai
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o
  legacy:
    api_key: sk-proj-REALSECRET123
database:
  url: postgresql+asyncpg://admin:SuperSecret99@db.internal:5432/app
  pool_size: 10
"""


def test_redaction_masks_literal_secrets_but_keeps_env_refs() -> None:
    from agentomatic.stacks.redaction import redact_yaml_text

    redacted, count = redact_yaml_text(_STACK_YAML)

    assert "sk-proj-REALSECRET123" not in redacted
    assert "SuperSecret99" not in redacted
    # ${ENV_VAR} indirections are not secrets and must stay visible.
    assert "${OPENAI_API_KEY}" in redacted
    # Non-secret structure survives so the file is still readable.
    assert "provider: openai" in redacted
    assert "pool_size: 10" in redacted
    assert "postgresql+asyncpg://admin:" in redacted  # host/user kept for debugging
    assert count == 2


def test_env_example_value_never_emits_a_literal_secret() -> None:
    """.env.example is conventionally committed — a literal key would publish it."""
    from agentomatic.stacks.redaction import ENV_EXAMPLE_PLACEHOLDER, env_example_value

    assert env_example_value("${OPENAI_API_KEY}") == "${OPENAI_API_KEY}"
    assert env_example_value("sk-proj-REALSECRET") == ENV_EXAMPLE_PLACEHOLDER
    assert env_example_value("") == ""


def test_generated_env_example_contains_no_literal_secret(tmp_path) -> None:
    """End-to-end: a stack with literal secrets must not leak into deploy output."""
    from agentomatic.cli import deploy as deploy_mod

    stacks_dir = tmp_path / "stacks"
    stacks_dir.mkdir()
    (stacks_dir / "prod.yaml").write_text(_STACK_YAML, encoding="utf-8")

    plan = deploy_mod.generate_deploy(
        out_dir=tmp_path / "out",
        stack_name="prod",
        stacks_dir=stacks_dir,
    )
    env_example = plan.files[".env.example"].read_text(encoding="utf-8")

    assert "sk-proj-REALSECRET123" not in env_example
    assert "SuperSecret99" not in env_example
