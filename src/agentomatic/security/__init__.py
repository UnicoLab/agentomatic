"""Security, authentication, and zero-trust enforcement."""

from __future__ import annotations

from agentomatic.security.claims import extract_roles, extract_scopes
from agentomatic.security.dpop import (
    DPoPConfig,
    DPoPError,
    access_token_hash,
    jwk_thumbprint,
    normalize_htu,
    reset_dpop_replay_cache,
    validate_dpop,
)
from agentomatic.security.jwt_auth import JWTAuthMiddleware, JWTConfig
from agentomatic.security.policy import AgentSecurityPolicy
from agentomatic.security.zero_trust import ZeroTrustEnforcer

__all__ = [
    "AgentSecurityPolicy",
    "DPoPConfig",
    "DPoPError",
    "JWTAuthMiddleware",
    "JWTConfig",
    "ZeroTrustEnforcer",
    "access_token_hash",
    "extract_roles",
    "extract_scopes",
    "jwk_thumbprint",
    "normalize_htu",
    "reset_dpop_replay_cache",
    "validate_dpop",
]
