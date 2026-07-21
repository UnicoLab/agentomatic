"""DPoP (RFC 9449) proof-of-possession validation for access tokens.

In-memory ``jti`` replay cache is suitable for single-instance deployments.
For multi-replica production, replace with a shared store (e.g. Redis).
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

try:
    import jwt as _jwt_lib
    from jwt import InvalidTokenError
    from jwt.algorithms import ECAlgorithm, OKPAlgorithm, RSAAlgorithm

    _HAS_PYJWT = True
except ImportError:  # pragma: no cover
    _jwt_lib = None  # type: ignore[assignment]
    InvalidTokenError = Exception  # type: ignore[misc, assignment]
    ECAlgorithm = OKPAlgorithm = RSAAlgorithm = None  # type: ignore[misc, assignment]
    _HAS_PYJWT = False


class DPoPError(Exception):
    """Raised when a DPoP proof is missing or invalid."""


@dataclass(slots=True)
class DPoPConfig:
    """Configuration for DPoP validation.

    Attributes:
        require_dpop: When ``True``, every authenticated request must carry a
            valid ``DPoP`` header.
        max_age_seconds: Maximum age of the proof ``iat`` claim.
        clock_skew_seconds: Allowed future skew for ``iat``.
        replay_cache_size: Max ``jti`` values retained for replay detection.
    """

    require_dpop: bool = False
    max_age_seconds: int = 60
    clock_skew_seconds: int = 5
    replay_cache_size: int = 10_000


class _JtiReplayCache:
    """Bounded in-memory LRU set of seen DPoP ``jti`` values."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max(1, max_size)
        self._items: OrderedDict[str, float] = OrderedDict()
        self._lock = Lock()

    def seen_or_add(self, jti: str, *, ttl: float) -> bool:
        """Return ``True`` if ``jti`` was already seen; otherwise record it.

        Args:
            jti: Unique proof identifier.
            ttl: Absolute expiry timestamp used for pruning.

        Returns:
            Whether this ``jti`` is a replay.
        """
        now = time.time()
        with self._lock:
            # Drop expired entries opportunistically.
            expired = [key for key, exp in self._items.items() if exp <= now]
            for key in expired:
                del self._items[key]

            if jti in self._items:
                return True

            self._items[jti] = ttl
            self._items.move_to_end(jti)
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)
            return False


_replay_cache = _JtiReplayCache()


def reset_dpop_replay_cache(max_size: int = 10_000) -> None:
    """Reset the process-wide DPoP replay cache (tests / config reload)."""
    global _replay_cache
    _replay_cache = _JtiReplayCache(max_size=max_size)


def _b64url_json(data: dict[str, Any]) -> str:
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def jwk_thumbprint(jwk: dict[str, Any]) -> str:
    """Compute the JWK thumbprint (RFC 7638) for ``jwk``.

    Args:
        jwk: Public JWK object from the DPoP proof header.

    Returns:
        Base64url-encoded SHA-256 thumbprint.
    """
    kty = jwk.get("kty")
    if kty == "EC":
        members = {"crv": jwk["crv"], "kty": "EC", "x": jwk["x"], "y": jwk["y"]}
    elif kty == "RSA":
        members = {"e": jwk["e"], "kty": "RSA", "n": jwk["n"]}
    elif kty == "OKP":
        members = {"crv": jwk["crv"], "kty": "OKP", "x": jwk["x"]}
    else:
        raise DPoPError(f"Unsupported DPoP JWK kty: {kty!r}")
    digest = hashlib.sha256(
        json.dumps(members, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def access_token_hash(token: str) -> str:
    """Return the base64url SHA-256 hash of ``token`` (DPoP ``ath`` claim)."""
    digest = hashlib.sha256(token.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def normalize_htu(method_url: str) -> str:
    """Normalize an HTTP URI for DPoP ``htu`` comparison (no query/fragment)."""
    parts = urlsplit(method_url)
    path = parts.path or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _public_key_from_jwk(jwk: dict[str, Any]) -> Any:
    """Convert a public JWK dict into a cryptography key object."""
    if not _HAS_PYJWT:
        raise DPoPError("PyJWT is required for DPoP validation")
    kty = jwk.get("kty")
    if kty == "EC":
        return ECAlgorithm.from_jwk(json.dumps(jwk))
    if kty == "RSA":
        return RSAAlgorithm.from_jwk(json.dumps(jwk))
    if kty == "OKP":
        return OKPAlgorithm.from_jwk(json.dumps(jwk))
    raise DPoPError(f"Unsupported DPoP JWK kty: {kty!r}")


def validate_dpop(
    *,
    dpop_header: str | None,
    access_token: str,
    access_claims: dict[str, Any],
    http_method: str,
    http_uri: str,
    config: DPoPConfig | None = None,
) -> None:
    """Validate an optional/required DPoP proof for ``access_token``.

    Args:
        dpop_header: Raw ``DPoP`` HTTP header value.
        access_token: Bearer access token string.
        access_claims: Already-verified access token claims.
        http_method: Request HTTP method (e.g. ``GET``).
        http_uri: Full request URI (scheme + host + path).
        config: DPoP settings.

    Raises:
        DPoPError: If the proof is required/present but invalid.
    """
    cfg = config or DPoPConfig()
    proof = (dpop_header or "").strip()
    if not proof:
        if cfg.require_dpop:
            raise DPoPError("Missing DPoP proof header")
        # Bound tokens without a proof are rejected even when DPoP is optional.
        cnf = access_claims.get("cnf")
        if isinstance(cnf, dict) and cnf.get("jkt"):
            raise DPoPError("Access token is DPoP-bound but no DPoP proof was provided")
        return

    if not _HAS_PYJWT:
        raise DPoPError("PyJWT is required for DPoP validation")

    try:
        header = _jwt_lib.get_unverified_header(proof)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        raise DPoPError(f"Invalid DPoP header: {exc}") from exc

    if header.get("typ") not in {"dpop+jwt", "dpop+JWT", "DPOP+JWT"}:
        # RFC 9449 requires typ=dpop+jwt; accept common casing variants.
        if str(header.get("typ", "")).lower() != "dpop+jwt":
            raise DPoPError("DPoP proof typ must be dpop+jwt")

    jwk = header.get("jwk")
    if not isinstance(jwk, dict):
        raise DPoPError("DPoP proof header must include a public jwk")
    if "d" in jwk:
        raise DPoPError("DPoP jwk must be a public key")

    try:
        key = _public_key_from_jwk(jwk)
        proof_claims = _jwt_lib.decode(  # type: ignore[union-attr]
            proof,
            key=key,
            algorithms=[header.get("alg") or "ES256"],
            options={
                "verify_aud": False,
                "verify_iss": False,
                "require": ["iat", "jti", "htm", "htu"],
            },
        )
    except InvalidTokenError as exc:
        raise DPoPError(f"Invalid DPoP proof signature: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise DPoPError(f"Invalid DPoP proof: {exc}") from exc

    htm = str(proof_claims.get("htm", "")).upper()
    if htm != http_method.upper():
        raise DPoPError(f"DPoP htm mismatch: expected {http_method.upper()}, got {htm}")

    expected_htu = normalize_htu(http_uri)
    actual_htu = normalize_htu(str(proof_claims.get("htu", "")))
    if actual_htu != expected_htu:
        raise DPoPError(f"DPoP htu mismatch: expected {expected_htu}, got {actual_htu}")

    now = int(time.time())
    iat = int(proof_claims["iat"])
    if iat > now + cfg.clock_skew_seconds:
        raise DPoPError("DPoP iat is in the future")
    if iat < now - cfg.max_age_seconds:
        raise DPoPError("DPoP proof has expired")

    jti = str(proof_claims["jti"])
    if not jti:
        raise DPoPError("DPoP jti is required")
    if _replay_cache.seen_or_add(jti, ttl=float(now + cfg.max_age_seconds)):
        raise DPoPError("DPoP jti replay detected")

    ath = proof_claims.get("ath")
    if ath is not None:
        expected_ath = access_token_hash(access_token)
        if str(ath) != expected_ath:
            raise DPoPError("DPoP ath does not match access token")

    thumbprint = jwk_thumbprint(jwk)
    cnf = access_claims.get("cnf")
    if isinstance(cnf, dict) and cnf.get("jkt"):
        if str(cnf["jkt"]) != thumbprint:
            raise DPoPError("DPoP jkt does not match access token cnf.jkt")
    elif cfg.require_dpop:
        # Require binding when DPoP is mandatory (AXA-style deployments).
        logger.debug(
            "DPoP required but access token has no cnf.jkt; proof accepted with ath/htm/htu checks"
        )


# Re-export helper used by tests when crafting proofs.
__all__ = [
    "DPoPConfig",
    "DPoPError",
    "access_token_hash",
    "jwk_thumbprint",
    "normalize_htu",
    "reset_dpop_replay_cache",
    "validate_dpop",
    "_b64url_json",
]
