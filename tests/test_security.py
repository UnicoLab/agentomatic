"""Tests for the agentomatic.security module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentomatic.security.jwt_auth import JWTConfig
from agentomatic.security.policy import AgentSecurityPolicy
from agentomatic.security.zero_trust import ZeroTrustEnforcer

# =========================================================================
# 1. AgentSecurityPolicy — defaults & custom values
# =========================================================================


class TestAgentSecurityPolicyDefaults:
    """Verify default values are sensible and consistent."""

    def test_default_values(self) -> None:
        policy = AgentSecurityPolicy()

        assert policy.require_auth is False
        assert policy.allowed_roles == []
        assert policy.allowed_scopes == []
        assert policy.max_execution_time == 30.0
        assert policy.max_tokens_per_request == 8192
        assert policy.allowed_tools is None
        assert policy.blocked_tools == []
        assert policy.env_vars_whitelist is None
        assert policy.allowed_delegation_targets == []
        assert policy.rate_limit_requests is None
        assert policy.rate_limit_window is None

    def test_custom_values(self) -> None:
        policy = AgentSecurityPolicy(
            require_auth=True,
            allowed_roles=["admin", "editor"],
            allowed_scopes=["read", "write"],
            max_execution_time=60.0,
            max_tokens_per_request=4096,
            allowed_tools=["search", "calculator"],
            blocked_tools=["shell"],
            env_vars_whitelist=["API_KEY"],
            allowed_delegation_targets=["helper_agent"],
            rate_limit_requests=100,
            rate_limit_window=60,
        )

        assert policy.require_auth is True
        assert policy.allowed_roles == ["admin", "editor"]
        assert policy.allowed_scopes == ["read", "write"]
        assert policy.max_execution_time == 60.0
        assert policy.max_tokens_per_request == 4096
        assert policy.allowed_tools == ["search", "calculator"]
        assert policy.blocked_tools == ["shell"]
        assert policy.env_vars_whitelist == ["API_KEY"]
        assert policy.allowed_delegation_targets == ["helper_agent"]
        assert policy.rate_limit_requests == 100
        assert policy.rate_limit_window == 60


# =========================================================================
# 2. is_role_allowed
# =========================================================================


class TestIsRoleAllowed:
    def test_empty_allowed_roles_allows_everything(self) -> None:
        policy = AgentSecurityPolicy()
        assert policy.is_role_allowed("admin") is True
        assert policy.is_role_allowed("viewer") is True

    def test_specific_roles_filter(self) -> None:
        policy = AgentSecurityPolicy(allowed_roles=["admin", "editor"])
        assert policy.is_role_allowed("admin") is True
        assert policy.is_role_allowed("editor") is True
        assert policy.is_role_allowed("viewer") is False

    def test_empty_string_role(self) -> None:
        policy = AgentSecurityPolicy(allowed_roles=["admin"])
        assert policy.is_role_allowed("") is False


# =========================================================================
# 3. is_scope_allowed
# =========================================================================


class TestIsScopeAllowed:
    def test_empty_allowed_scopes_allows_everything(self) -> None:
        policy = AgentSecurityPolicy()
        assert policy.is_scope_allowed("read") is True
        assert policy.is_scope_allowed("write") is True

    def test_specific_scopes_filter(self) -> None:
        policy = AgentSecurityPolicy(allowed_scopes=["read", "write"])
        assert policy.is_scope_allowed("read") is True
        assert policy.is_scope_allowed("write") is True
        assert policy.is_scope_allowed("delete") is False


# =========================================================================
# 4. is_tool_allowed
# =========================================================================


class TestIsToolAllowed:
    def test_no_whitelist_no_blacklist_allows_all(self) -> None:
        """allowed_tools=None and blocked_tools=[] means everything passes."""
        policy = AgentSecurityPolicy()
        assert policy.is_tool_allowed("search") is True
        assert policy.is_tool_allowed("shell") is True

    def test_blacklist_blocks(self) -> None:
        policy = AgentSecurityPolicy(blocked_tools=["shell", "exec"])
        assert policy.is_tool_allowed("search") is True
        assert policy.is_tool_allowed("shell") is False
        assert policy.is_tool_allowed("exec") is False

    def test_whitelist_restricts(self) -> None:
        policy = AgentSecurityPolicy(allowed_tools=["search", "calculator"])
        assert policy.is_tool_allowed("search") is True
        assert policy.is_tool_allowed("calculator") is True
        assert policy.is_tool_allowed("shell") is False

    def test_blacklist_overrides_whitelist(self) -> None:
        """A tool on both lists should be blocked (blacklist takes priority)."""
        policy = AgentSecurityPolicy(
            allowed_tools=["search", "shell"],
            blocked_tools=["shell"],
        )
        assert policy.is_tool_allowed("search") is True
        assert policy.is_tool_allowed("shell") is False

    def test_empty_whitelist_blocks_all(self) -> None:
        policy = AgentSecurityPolicy(allowed_tools=[])
        assert policy.is_tool_allowed("search") is False
        assert policy.is_tool_allowed("anything") is False


# =========================================================================
# 5. is_delegation_allowed
# =========================================================================


class TestIsDelegationAllowed:
    def test_empty_targets_deny_all(self) -> None:
        policy = AgentSecurityPolicy()
        assert policy.is_delegation_allowed("helper") is False

    def test_explicit_targets(self) -> None:
        policy = AgentSecurityPolicy(
            allowed_delegation_targets=["helper", "summarizer"],
        )
        assert policy.is_delegation_allowed("helper") is True
        assert policy.is_delegation_allowed("summarizer") is True
        assert policy.is_delegation_allowed("attacker") is False


# =========================================================================
# 6. ZeroTrustEnforcer — register & get policy
# =========================================================================


class TestZeroTrustEnforcerPolicies:
    def test_register_and_get(self) -> None:
        enforcer = ZeroTrustEnforcer()
        policy = AgentSecurityPolicy(require_auth=True)
        enforcer.register_policy("agent_a", policy)

        retrieved = enforcer.get_policy("agent_a")
        assert retrieved is policy
        assert retrieved.require_auth is True

    def test_get_default_when_missing(self) -> None:
        enforcer = ZeroTrustEnforcer()
        default = enforcer.get_policy("nonexistent")

        assert isinstance(default, AgentSecurityPolicy)
        assert default.require_auth is False

    def test_init_with_policies(self) -> None:
        policies = {
            "a": AgentSecurityPolicy(max_execution_time=10.0),
            "b": AgentSecurityPolicy(max_execution_time=20.0),
        }
        enforcer = ZeroTrustEnforcer(policies=policies)

        assert enforcer.get_policy("a").max_execution_time == 10.0
        assert enforcer.get_policy("b").max_execution_time == 20.0

    def test_register_overwrites(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy("x", AgentSecurityPolicy(max_execution_time=5.0))
        enforcer.register_policy("x", AgentSecurityPolicy(max_execution_time=99.0))

        assert enforcer.get_policy("x").max_execution_time == 99.0


# =========================================================================
# 7. ZeroTrustEnforcer — verify_delegation
# =========================================================================


class TestVerifyDelegation:
    def test_allowed(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "source",
            AgentSecurityPolicy(allowed_delegation_targets=["target"]),
        )

        ok, reason = enforcer.verify_delegation("source", "target")
        assert ok is True
        assert reason == ""

    def test_denied(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "source",
            AgentSecurityPolicy(allowed_delegation_targets=["other"]),
        )

        ok, reason = enforcer.verify_delegation("source", "target")
        assert ok is False
        assert "not allowed" in reason

    def test_denied_empty_targets(self) -> None:
        enforcer = ZeroTrustEnforcer()

        ok, reason = enforcer.verify_delegation("any_agent", "target")
        assert ok is False
        assert "not allowed" in reason


# =========================================================================
# 8. ZeroTrustEnforcer — enforce_tool_access
# =========================================================================


class TestEnforceToolAccess:
    def test_default_allows_all(self) -> None:
        enforcer = ZeroTrustEnforcer()
        ok, reason = enforcer.enforce_tool_access("agent", "any_tool")
        assert ok is True
        assert reason == ""

    def test_whitelist_allows(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(allowed_tools=["search"]),
        )

        ok, _ = enforcer.enforce_tool_access("agent", "search")
        assert ok is True

    def test_whitelist_denies(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(allowed_tools=["search"]),
        )

        ok, reason = enforcer.enforce_tool_access("agent", "shell")
        assert ok is False
        assert "not allowed" in reason

    def test_blacklist_denies(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(blocked_tools=["shell"]),
        )

        ok, reason = enforcer.enforce_tool_access("agent", "shell")
        assert ok is False
        assert "not allowed" in reason


# =========================================================================
# 9. ZeroTrustEnforcer — verify_request (with mocked Request)
# =========================================================================


def _mock_request(
    claims: dict | None = None,
    user_id: str | None = None,
) -> MagicMock:
    """Build a mock Starlette Request with ``state`` attributes."""
    request = MagicMock()
    state = MagicMock()

    # MagicMock auto-creates attributes, so we need to explicitly control
    # which attributes are present on state.
    if claims is not None:
        state.jwt_claims = claims
    else:
        # Simulate attribute not being set.
        del state.jwt_claims

    if user_id is not None:
        state.user_id = user_id
    else:
        del state.user_id

    request.state = state
    return request


class TestVerifyRequest:
    def test_no_auth_required_passes(self) -> None:
        """Default policy + no claims → still allowed."""
        enforcer = ZeroTrustEnforcer()
        request = _mock_request()

        ok, reason = enforcer.verify_request(request, "agent")
        assert ok is True
        assert reason == ""

    def test_auth_required_globally_no_claims_denied(self) -> None:
        enforcer = ZeroTrustEnforcer(require_auth_globally=True)
        request = _mock_request()

        ok, reason = enforcer.verify_request(request, "agent")
        assert ok is False
        assert "Authentication is required" in reason

    def test_auth_required_per_policy_no_claims_denied(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy("agent", AgentSecurityPolicy(require_auth=True))
        request = _mock_request()

        ok, reason = enforcer.verify_request(request, "agent")
        assert ok is False
        assert "Authentication is required" in reason

    def test_auth_required_with_claims_passes(self) -> None:
        enforcer = ZeroTrustEnforcer(require_auth_globally=True)
        request = _mock_request(claims={"sub": "user1", "roles": [], "scopes": []})

        ok, reason = enforcer.verify_request(request, "agent")
        assert ok is True

    def test_role_check_passes(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(
                require_auth=True,
                allowed_roles=["admin"],
            ),
        )
        request = _mock_request(claims={"sub": "u1", "roles": ["admin"]})

        ok, _ = enforcer.verify_request(request, "agent")
        assert ok is True

    def test_role_check_keycloak_realm_access(self) -> None:
        """Roles nested under realm_access must satisfy allowed_roles."""
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(
                require_auth=True,
                allowed_roles=["metier"],
            ),
        )
        request = _mock_request(claims={"sub": "u1", "realm_access": {"roles": ["metier"]}})

        ok, _ = enforcer.verify_request(request, "agent")
        assert ok is True

    def test_scope_check_from_space_delimited_scope(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(
                require_auth=True,
                allowed_scopes=["openid"],
            ),
        )
        request = _mock_request(claims={"sub": "u1", "scope": "openid profile"})

        ok, _ = enforcer.verify_request(request, "agent")
        assert ok is True

    def test_role_check_denied(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(
                require_auth=True,
                allowed_roles=["admin"],
            ),
        )
        request = _mock_request(claims={"sub": "u1", "roles": ["viewer"]})

        ok, reason = enforcer.verify_request(request, "agent")
        assert ok is False
        assert "role" in reason.lower()

    def test_scope_check_passes(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(
                require_auth=True,
                allowed_scopes=["read"],
            ),
        )
        request = _mock_request(claims={"sub": "u1", "scopes": ["read", "write"]})

        ok, _ = enforcer.verify_request(request, "agent")
        assert ok is True

    def test_scope_check_denied(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.register_policy(
            "agent",
            AgentSecurityPolicy(
                require_auth=True,
                allowed_scopes=["admin"],
            ),
        )
        request = _mock_request(claims={"sub": "u1", "scopes": ["read"]})

        ok, reason = enforcer.verify_request(request, "agent")
        assert ok is False
        assert "scope" in reason.lower()


# =========================================================================
# 10. JWTConfig — defaults
# =========================================================================


class TestJWTConfigDefaults:
    def test_defaults(self) -> None:
        config = JWTConfig()

        assert config.enabled is False
        assert config.jwks_url == ""
        assert config.issuer == ""
        assert config.audience == ""
        assert config.algorithms == ["RS256"]
        assert config.header_name == "Authorization"
        assert config.header_prefix == "Bearer"
        assert "/health" in config.skip_paths
        assert "/docs" in config.skip_paths
        assert "/" in config.skip_paths

    def test_custom_config(self) -> None:
        config = JWTConfig(
            enabled=True,
            jwks_url="https://example.com/.well-known/jwks.json",
            issuer="https://example.com",
            audience="my-app",
            algorithms=["RS256", "ES256"],
            header_name="X-Auth",
            header_prefix="Token",
            skip_paths={"/ping"},
        )

        assert config.enabled is True
        assert config.jwks_url == "https://example.com/.well-known/jwks.json"
        assert config.issuer == "https://example.com"
        assert config.audience == "my-app"
        assert config.algorithms == ["RS256", "ES256"]
        assert config.header_name == "X-Auth"
        assert config.header_prefix == "Token"
        assert config.skip_paths == {"/ping"}

    def test_require_signature_defaults_false(self) -> None:
        assert JWTConfig().require_signature is False


# =========================================================================
# 10b. JWTAuthMiddleware — signature-required refusal + dev-mode exp (P1-2)
# =========================================================================


class TestJWTMiddlewareSafety:
    """The auth-required flag must not accept forged/unsigned/expired tokens."""

    def test_require_signature_without_jwks_raises(self) -> None:
        from agentomatic.security.jwt_auth import JWTAuthMiddleware

        cfg = JWTConfig(enabled=True, require_signature=True)  # no jwks_url
        with pytest.raises(ValueError, match="jwks_url"):
            JWTAuthMiddleware(MagicMock(), config=cfg)

    def test_dev_mode_still_rejects_expired_token(self) -> None:
        import datetime

        pytest.importorskip("jwt")
        import jwt
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agentomatic.security.jwt_auth import JWTAuthMiddleware

        app = FastAPI()

        @app.get("/api/v1/echo/invoke")
        async def _protected() -> dict[str, str]:
            return {"ok": "yes"}

        # Dev mode: no jwks_url (signatures not verified) but exp MUST be.
        app.add_middleware(JWTAuthMiddleware, config=JWTConfig(enabled=True))
        client = TestClient(app)

        expired = jwt.encode(
            {
                "sub": "user-1",
                "exp": datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=1),
            },
            "irrelevant-secret",
            algorithm="HS256",
        )
        resp = client.get("/api/v1/echo/invoke", headers={"Authorization": f"Bearer {expired}"})
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_options_preflight_bypasses_jwt(self) -> None:
        """CORS preflight must not require Authorization."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agentomatic.security.jwt_auth import JWTAuthMiddleware

        app = FastAPI()

        @app.get("/api/v1/agents")
        async def _list() -> dict[str, list]:
            return {"agents": []}

        app.add_middleware(
            JWTAuthMiddleware,
            config=JWTConfig(
                enabled=True, jwks_url="https://example/jwks", require_signature=True
            ),
        )
        # Avoid JWKS fetch on OPTIONS — middleware must short-circuit first.
        client = TestClient(app)
        resp = client.options(
            "/api/v1/agents",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code != 401


# =========================================================================
# 11. ZeroTrustEnforcer — audit_log (smoke test)
# =========================================================================


class TestAuditLog:
    def test_audit_log_does_not_raise(self) -> None:
        """Audit logging should never raise — it's a fire-and-forget operation."""
        enforcer = ZeroTrustEnforcer()
        enforcer.audit_log("test_event", "agent_x")
        enforcer.audit_log("test_event", "agent_x", {"key": "value"})

    def test_audit_log_with_none_context(self) -> None:
        enforcer = ZeroTrustEnforcer()
        enforcer.audit_log("test_event", "agent_x", None)
