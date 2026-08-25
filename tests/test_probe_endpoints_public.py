# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
"""Probe endpoints must answer without credentials under every auth posture.

An orchestrator has no API key and no bearer token. A readiness probe that
answers 401 keeps every pod out of service, so the Deployment never rolls
out — and the platform looks healthy in logs the whole time.

Regression: the skip lists in both auth middlewares named ``/healthz``, which
the platform does not mount, but omitted ``/ready``, which it does. A
``readinessProbe: /ready`` — the conventional spelling — got 401.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform

#: Probe routes the platform actually mounts.
MOUNTED_PROBES = ("/health", "/ready", "/readiness")


def _platform(tmp_path, **kwargs):
    """Build a one-agent platform with the requested auth posture."""
    agents = tmp_path / "agents"
    agents.mkdir(exist_ok=True)
    platform = AgentPlatform(agents_dir=str(agents), **kwargs)
    platform.register_agent(
        AgentManifest(name="probe", slug="probe", description="probe agent"),
        node_fn=lambda state: {"response": "ok"},
    )
    return platform


@pytest.mark.parametrize("path", MOUNTED_PROBES)
def test_probes_are_public_without_auth(tmp_path, path: str) -> None:
    """Baseline: no auth configured at all."""
    with TestClient(_platform(tmp_path).build()) as client:
        assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", MOUNTED_PROBES)
def test_probes_are_public_under_api_key_auth(tmp_path, path: str) -> None:
    """An orchestrator does not carry the platform's API key."""
    platform = _platform(tmp_path, enable_auth=True, auth_api_key="secret-key")
    with TestClient(platform.build()) as client:
        assert client.get(path).status_code == 200
        # …while ordinary routes stay protected.
        assert client.get("/api/v1/agents").status_code == 401


@pytest.mark.parametrize("path", MOUNTED_PROBES)
def test_probes_are_public_under_jwt_auth(tmp_path, path: str) -> None:
    """An orchestrator does not carry a bearer token either."""
    from agentomatic.security.jwt_auth import JWTConfig

    platform = _platform(
        tmp_path,
        enable_jwt_auth=True,
        jwt_config=JWTConfig(
            enabled=True,
            jwks_url="https://issuer.example.com/.well-known/jwks.json",
            issuer="https://issuer.example.com",
            audience="agentomatic",
        ),
    )
    with TestClient(platform.build()) as client:
        assert client.get(path).status_code == 200
        assert client.get("/api/v1/agents").status_code == 401


@pytest.mark.parametrize("path", MOUNTED_PROBES)
def test_probes_are_public_under_global_auth_lock(tmp_path, path: str) -> None:
    """The strictest posture must still let an orchestrator probe."""
    platform = _platform(
        tmp_path,
        enable_auth=True,
        auth_api_key="secret-key",
        enable_zero_trust=True,
        require_auth_globally=True,
    )
    with TestClient(platform.build()) as client:
        assert client.get(path).status_code == 200


def test_every_mounted_probe_route_is_in_the_shared_set() -> None:
    """The skip set must not drift from the routes the platform mounts.

    Regression: the lists named ``/healthz`` (never mounted) while omitting
    ``/ready`` (mounted), so the omission was invisible by inspection.
    """
    from agentomatic.middleware.pathutils import PROBE_PATHS

    for path in MOUNTED_PROBES:
        assert path in PROBE_PATHS


def test_auth_middlewares_share_the_probe_set() -> None:
    """Both middlewares must exempt the same probes, or they drift again."""
    from agentomatic.middleware.auth import _SKIP_PATHS as api_key_skips
    from agentomatic.middleware.pathutils import PROBE_PATHS
    from agentomatic.security.jwt_auth import _DEFAULT_SKIP_PATHS as jwt_skips

    assert PROBE_PATHS <= api_key_skips
    assert PROBE_PATHS <= jwt_skips
