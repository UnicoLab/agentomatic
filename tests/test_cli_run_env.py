# pyright: reportMissingParameterType=none
# pyright: reportAttributeAccessIssue=none
"""``agentomatic run`` must honour the documented ``AGENTOMATIC_*`` switches.

The image this repo ships runs ``agentomatic run``, not ``uvicorn main:app``,
so any switch the CLI drops is silently inert in a container. That was a
security hazard for the auth switches in particular: a deployment started with
``AGENTOMATIC_ENABLE_AUTH=1`` and an API key served a fully unauthenticated
API while looking correctly configured.
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

from agentomatic.cli.commands import cli


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Run the CLI without serving, capturing the platform kwargs."""
    seen: dict[str, Any] = {}

    class _FakePlatform:
        def run(self, **_kwargs: Any) -> None:
            """Stand in for the blocking uvicorn call."""

    def _from_folder(agents_dir: str, **kwargs: Any) -> _FakePlatform:
        seen.clear()
        seen.update(kwargs)
        seen["_agents_dir"] = agents_dir
        return _FakePlatform()

    import agentomatic

    monkeypatch.setattr(agentomatic.AgentPlatform, "from_folder", staticmethod(_from_folder))
    # Clear every switch so each test starts from a known baseline.
    for var in (
        "AGENTOMATIC_ENABLE_AUTH",
        "AGENTOMATIC_API_KEY",
        "AGENTOMATIC_ENABLE_JWT",
        "AGENTOMATIC_ENABLE_ZERO_TRUST",
        "AGENTOMATIC_REQUIRE_AUTH",
        "AGENTOMATIC_ENABLE_CONTROL_PLANE",
        "AGENTOMATIC_CONTROL_TOKEN",
        "AGENTOMATIC_ENABLE_RATE_LIMIT",
        "AGENTOMATIC_RATE_LIMIT_TRUST_PROXY_HEADERS",
        "AGENTOMATIC_ENABLE_METRICS",
    ):
        monkeypatch.delenv(var, raising=False)

    agents = tmp_path / "agents"
    agents.mkdir()

    def _invoke(env: dict[str, str], *extra: str) -> dict[str, Any]:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        result = CliRunner().invoke(
            cli,
            ["run", "--agents-dir", str(agents), "--no-studio", *extra],
        )
        assert result.exit_code == 0, result.output
        return seen

    return _invoke


def test_api_key_auth_is_enabled_from_env(captured) -> None:
    """Regression: these two were dropped, leaving the API wide open."""
    kwargs = captured({"AGENTOMATIC_ENABLE_AUTH": "1", "AGENTOMATIC_API_KEY": "secret-key"})

    assert kwargs["enable_auth"] is True
    assert kwargs["auth_api_key"] == "secret-key"


def test_auth_stays_off_when_unset(captured) -> None:
    """The default must remain open — enabling auth is opt-in."""
    kwargs = captured({})

    assert kwargs["enable_auth"] is False
    assert kwargs["auth_api_key"] == ""


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", ""])
def test_falsey_values_do_not_enable_auth(captured, raw: str) -> None:
    """A disabled flag must not be read as enabled."""
    kwargs = captured({"AGENTOMATIC_ENABLE_AUTH": raw})

    assert kwargs["enable_auth"] is False


def test_jwt_auth_is_enabled_from_env(captured) -> None:
    kwargs = captured({"AGENTOMATIC_ENABLE_JWT": "1"})

    assert kwargs["enable_jwt_auth"] is True


def test_require_auth_env_implies_zero_trust_and_jwt(captured) -> None:
    """``AGENTOMATIC_REQUIRE_AUTH`` mirrors ``--require-auth-globally``."""
    kwargs = captured({"AGENTOMATIC_REQUIRE_AUTH": "1"})

    assert kwargs["require_auth_globally"] is True
    assert kwargs["enable_zero_trust"] is True
    assert kwargs["enable_jwt_auth"] is True


def test_require_auth_with_an_api_key_does_not_force_jwt(captured) -> None:
    """An API key is real verification, so JWT need not be forced on."""
    kwargs = captured({"AGENTOMATIC_REQUIRE_AUTH": "1", "AGENTOMATIC_API_KEY": "secret-key"})

    assert kwargs["require_auth_globally"] is True
    assert kwargs["auth_api_key"] == "secret-key"


def test_require_auth_flag_still_works(captured) -> None:
    """The CLI flag keeps its behaviour alongside the env var."""
    kwargs = captured({}, "--require-auth-globally")

    assert kwargs["require_auth_globally"] is True
    assert kwargs["enable_zero_trust"] is True


def test_control_plane_and_token_are_read(captured) -> None:
    kwargs = captured(
        {
            "AGENTOMATIC_ENABLE_CONTROL_PLANE": "1",
            "AGENTOMATIC_CONTROL_TOKEN": "control-token",
        }
    )

    assert kwargs["enable_control_plane"] is True
    assert kwargs["control_token"] == "control-token"


def test_rate_limit_is_read(captured) -> None:
    kwargs = captured({"AGENTOMATIC_ENABLE_RATE_LIMIT": "1"})

    assert kwargs["enable_rate_limit"] is True


def test_proxy_header_trust_defaults_off(captured) -> None:
    """Trusting X-Forwarded-For by default would let callers bypass limits."""
    kwargs = captured({"AGENTOMATIC_ENABLE_RATE_LIMIT": "1"})

    assert kwargs["rate_limit_trust_proxy_headers"] is False


def test_proxy_header_trust_is_opt_in(captured) -> None:
    kwargs = captured({"AGENTOMATIC_RATE_LIMIT_TRUST_PROXY_HEADERS": "1"})

    assert kwargs["rate_limit_trust_proxy_headers"] is True
