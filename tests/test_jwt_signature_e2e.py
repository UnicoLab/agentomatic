# pyright: reportMissingParameterType=none
# pyright: reportCallIssue=none
"""End-to-end JWT verification against a real JWKS and real RS256 signatures.

Every other JWT test in this suite exercises configuration or decoding in
isolation. This one mints genuine RS256 tokens, publishes a JWKS the
middleware fetches, and drives the whole HTTP path — the only way to catch a
middleware that accepts a token it should reject.

The forgeries below are the ones that matter in production: a token signed by
a key the issuer never published, that same key reusing a published ``kid``,
``alg=none``, and tokens whose ``exp`` / ``aud`` / ``iss`` do not hold.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentomatic import AgentManifest, AgentPlatform

jwt = pytest.importorskip("jwt", reason="pyjwt is required for JWT verification tests")
rsa = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.rsa",
    reason="cryptography is required for JWT verification tests",
)

ISSUER = "https://issuer.example.com"
AUDIENCE = "agentomatic-tests"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"


def _b64u(value: int) -> str:
    """Base64url-encode an unsigned integer as JWKS expects."""
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


@pytest.fixture(scope="module")
def keys() -> dict[str, Any]:
    """Return a published signing key and an unpublished rogue key."""
    return {
        "published": rsa.generate_private_key(public_exponent=65537, key_size=2048),
        "rogue": rsa.generate_private_key(public_exponent=65537, key_size=2048),
    }


@pytest.fixture(scope="module")
def jwks(keys) -> dict[str, Any]:
    """Return a JWKS document containing only the published key."""
    numbers = keys["published"].public_key().public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": "published-key",
                "n": _b64u(numbers.n),
                "e": _b64u(numbers.e),
            }
        ]
    }


@pytest.fixture
def client(tmp_path, jwks, monkeypatch):
    """Serve a platform whose JWKS fetches resolve to the fixture document."""
    import urllib.request

    from agentomatic.security.jwt_auth import JWTConfig

    body = json.dumps(jwks).encode()

    class _Response:
        def read(self) -> bytes:
            return body

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _urlopen(url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        target = getattr(url, "full_url", url)
        assert target == JWKS_URL, f"unexpected JWKS fetch: {target}"
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    class _HttpxResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return jwks

        def raise_for_status(self) -> None:
            return None

    try:
        import httpx

        monkeypatch.setattr(httpx, "get", lambda *a, **k: _HttpxResponse())
    except ImportError:  # pragma: no cover
        pass

    agents = tmp_path / "agents"
    agents.mkdir()
    platform = AgentPlatform(
        agents_dir=str(agents),
        enable_jwt_auth=True,
        jwt_config=JWTConfig(
            enabled=True,
            jwks_url=JWKS_URL,
            issuer=ISSUER,
            audience=AUDIENCE,
            require_signature=True,
        ),
    )
    platform.register_agent(
        AgentManifest(name="guarded", slug="guarded", description="guarded agent"),
        node_fn=lambda state: {"response": "ok"},
    )
    with TestClient(platform.build()) as test_client:
        yield test_client


def _mint(key: Any, kid: str, **overrides: Any) -> str:
    """Mint an RS256 token, letting the caller override any claim."""
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": "test-user",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 600,
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})


def _call(client: TestClient, token: str | None) -> int:
    """Call a protected route with an optional bearer token."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.get("/api/v1/agents", headers=headers).status_code


class TestValidTokensAreAccepted:
    def test_a_properly_signed_token_is_accepted(self, client, keys) -> None:
        assert _call(client, _mint(keys["published"], "published-key")) == 200


class TestForgeriesAreRejected:
    def test_a_token_signed_by_an_unpublished_key_is_rejected(self, client, keys) -> None:
        """The core property: only keys in the JWKS may sign."""
        assert _call(client, _mint(keys["rogue"], "rogue-key")) == 401

    def test_a_rogue_key_reusing_a_published_kid_is_rejected(self, client, keys) -> None:
        """``kid`` selects a key; it must not be trusted as proof of one."""
        assert _call(client, _mint(keys["rogue"], "published-key")) == 401

    def test_an_unsigned_alg_none_token_is_rejected(self, client) -> None:
        """The classic forgery: drop the signature and declare ``alg=none``."""
        now = int(time.time())
        forged = jwt.encode(
            {"sub": "admin", "iss": ISSUER, "aud": AUDIENCE, "exp": now + 600},
            None,
            algorithm="none",
        )

        assert _call(client, forged) == 401

    def test_a_tampered_payload_is_rejected(self, client, keys) -> None:
        """Editing claims after signing must invalidate the token."""
        token = _mint(keys["published"], "published-key")
        header, payload, signature = token.split(".")
        decoded = json.loads(base64.urlsafe_b64decode(payload + "=="))
        decoded["sub"] = "admin"
        tampered_payload = (
            base64.urlsafe_b64encode(json.dumps(decoded).encode()).rstrip(b"=").decode()
        )

        assert _call(client, f"{header}.{tampered_payload}.{signature}") == 401


class TestClaimsAreEnforced:
    def test_an_expired_token_is_rejected(self, client, keys) -> None:
        now = int(time.time())
        expired = _mint(keys["published"], "published-key", iat=now - 600, exp=now - 60)

        assert _call(client, expired) == 401

    def test_a_token_for_another_audience_is_rejected(self, client, keys) -> None:
        assert _call(client, _mint(keys["published"], "published-key", aud="other-api")) == 401

    def test_a_token_from_another_issuer_is_rejected(self, client, keys) -> None:
        forged = _mint(keys["published"], "published-key", iss="https://evil.example")

        assert _call(client, forged) == 401


class TestMissingCredentials:
    def test_no_token_is_rejected(self, client) -> None:
        assert _call(client, None) == 401

    def test_a_garbage_token_is_rejected(self, client) -> None:
        assert _call(client, "not-a-jwt") == 401

    def test_probes_still_answer_without_a_token(self, client) -> None:
        """An orchestrator has no token — see test_probe_endpoints_public."""
        for path in ("/health", "/ready", "/readiness"):
            assert client.get(path).status_code == 200
