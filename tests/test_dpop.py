"""Tests for DPoP (RFC 9449) proof validation."""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from agentomatic.security.dpop import (
    DPoPConfig,
    DPoPError,
    access_token_hash,
    jwk_thumbprint,
    reset_dpop_replay_cache,
    validate_dpop,
)


def _b64url_uint(val: int, length: int) -> str:
    raw = val.to_bytes(length, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _ec_keypair() -> tuple[ec.EllipticCurvePrivateKey, dict[str, Any]]:
    private_key = ec.generate_private_key(ec.SECP256R1())
    public = private_key.public_key().public_numbers()
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64url_uint(public.x, 32),
        "y": _b64url_uint(public.y, 32),
    }
    return private_key, jwk


def _make_dpop(
    private_key: ec.EllipticCurvePrivateKey,
    jwk: dict[str, Any],
    *,
    method: str = "GET",
    uri: str = "https://api.example.com/v1/agents",
    access_token: str | None = None,
    jti: str | None = None,
    iat: int | None = None,
) -> str:
    headers = {"typ": "dpop+jwt", "alg": "ES256", "jwk": jwk}
    claims: dict[str, Any] = {
        "htm": method,
        "htu": uri,
        "iat": iat if iat is not None else int(time.time()),
        "jti": jti or str(uuid.uuid4()),
    }
    if access_token is not None:
        claims["ath"] = access_token_hash(access_token)
    return jwt.encode(claims, private_key, algorithm="ES256", headers=headers)


@pytest.fixture(autouse=True)
def _clear_replay_cache() -> None:
    reset_dpop_replay_cache()


class TestJwkThumbprint:
    def test_stable_for_ec(self) -> None:
        _, jwk = _ec_keypair()
        assert jwk_thumbprint(jwk) == jwk_thumbprint(jwk)


class TestValidateDpop:
    def test_optional_no_header_and_unbound_token(self) -> None:
        validate_dpop(
            dpop_header=None,
            access_token="tok",
            access_claims={"sub": "u1"},
            http_method="GET",
            http_uri="https://api.example.com/v1/agents",
            config=DPoPConfig(require_dpop=False),
        )

    def test_require_dpop_missing_header(self) -> None:
        with pytest.raises(DPoPError, match="Missing DPoP"):
            validate_dpop(
                dpop_header=None,
                access_token="tok",
                access_claims={"sub": "u1"},
                http_method="GET",
                http_uri="https://api.example.com/v1/agents",
                config=DPoPConfig(require_dpop=True),
            )

    def test_bound_token_requires_proof(self) -> None:
        with pytest.raises(DPoPError, match="DPoP-bound"):
            validate_dpop(
                dpop_header=None,
                access_token="tok",
                access_claims={"sub": "u1", "cnf": {"jkt": "abc"}},
                http_method="GET",
                http_uri="https://api.example.com/v1/agents",
            )

    def test_valid_proof_with_jkt_and_ath(self) -> None:
        private_key, jwk = _ec_keypair()
        token = "access-token-value"
        proof = _make_dpop(private_key, jwk, access_token=token)
        validate_dpop(
            dpop_header=proof,
            access_token=token,
            access_claims={"sub": "u1", "cnf": {"jkt": jwk_thumbprint(jwk)}},
            http_method="GET",
            http_uri="https://api.example.com/v1/agents",
        )

    def test_htm_mismatch(self) -> None:
        private_key, jwk = _ec_keypair()
        proof = _make_dpop(private_key, jwk, method="POST")
        with pytest.raises(DPoPError, match="htm mismatch"):
            validate_dpop(
                dpop_header=proof,
                access_token="tok",
                access_claims={"sub": "u1"},
                http_method="GET",
                http_uri="https://api.example.com/v1/agents",
            )

    def test_htu_mismatch(self) -> None:
        private_key, jwk = _ec_keypair()
        proof = _make_dpop(private_key, jwk, uri="https://api.example.com/other")
        with pytest.raises(DPoPError, match="htu mismatch"):
            validate_dpop(
                dpop_header=proof,
                access_token="tok",
                access_claims={"sub": "u1"},
                http_method="GET",
                http_uri="https://api.example.com/v1/agents",
            )

    def test_jkt_mismatch(self) -> None:
        private_key, jwk = _ec_keypair()
        proof = _make_dpop(private_key, jwk)
        with pytest.raises(DPoPError, match="jkt"):
            validate_dpop(
                dpop_header=proof,
                access_token="tok",
                access_claims={"sub": "u1", "cnf": {"jkt": "wrong"}},
                http_method="GET",
                http_uri="https://api.example.com/v1/agents",
            )

    def test_replay_detected(self) -> None:
        private_key, jwk = _ec_keypair()
        jti = "fixed-jti-1"
        proof = _make_dpop(private_key, jwk, jti=jti)
        kwargs = {
            "dpop_header": proof,
            "access_token": "tok",
            "access_claims": {"sub": "u1"},
            "http_method": "GET",
            "http_uri": "https://api.example.com/v1/agents",
        }
        validate_dpop(**kwargs)  # type: ignore[arg-type]
        with pytest.raises(DPoPError, match="replay"):
            validate_dpop(**kwargs)  # type: ignore[arg-type]

    def test_expired_iat(self) -> None:
        private_key, jwk = _ec_keypair()
        proof = _make_dpop(private_key, jwk, iat=int(time.time()) - 120)
        with pytest.raises(DPoPError, match="expired"):
            validate_dpop(
                dpop_header=proof,
                access_token="tok",
                access_claims={"sub": "u1"},
                http_method="GET",
                http_uri="https://api.example.com/v1/agents",
                config=DPoPConfig(max_age_seconds=60),
            )

    def test_rejects_private_key_material_in_jwk(self) -> None:
        private_key, jwk = _ec_keypair()
        bad_jwk = {**jwk, "d": "private"}
        headers = {"typ": "dpop+jwt", "alg": "ES256", "jwk": bad_jwk}
        claims = {
            "htm": "GET",
            "htu": "https://api.example.com/v1/agents",
            "iat": int(time.time()),
            "jti": str(uuid.uuid4()),
        }
        proof = jwt.encode(claims, private_key, algorithm="ES256", headers=headers)
        with pytest.raises(DPoPError, match="public key"):
            validate_dpop(
                dpop_header=proof,
                access_token="tok",
                access_claims={"sub": "u1"},
                http_method="GET",
                http_uri="https://api.example.com/v1/agents",
            )
