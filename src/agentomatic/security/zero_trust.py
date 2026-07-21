"""Zero-trust enforcement engine.

The :class:`ZeroTrustEnforcer` sits between the platform routing layer and
individual agent invocations.  It verifies every request against the
registered :class:`~agentomatic.security.policy.AgentSecurityPolicy` and
emits structured audit logs via *loguru*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from agentomatic.security.claims import extract_roles, extract_scopes
from agentomatic.security.policy import AgentSecurityPolicy

if TYPE_CHECKING:
    from starlette.requests import Request


class ZeroTrustEnforcer:
    """Centralised policy enforcement for agent requests.

    Args:
        policies: Mapping of agent names to their security policies.
        require_auth_globally: When ``True``, authentication is required for
            *every* agent, regardless of individual policy settings.
    """

    def __init__(
        self,
        *,
        policies: dict[str, AgentSecurityPolicy] | None = None,
        require_auth_globally: bool = False,
    ) -> None:
        self._policies: dict[str, AgentSecurityPolicy] = dict(policies or {})
        self._require_auth_globally = require_auth_globally

    # -- policy management ---------------------------------------------------

    def register_policy(self, agent_name: str, policy: AgentSecurityPolicy) -> None:
        """Register or replace the security policy for *agent_name*.

        Args:
            agent_name: Unique agent identifier.
            policy: The :class:`AgentSecurityPolicy` to attach.
        """
        self._policies[agent_name] = policy
        self.audit_log("policy_registered", agent_name)

    def get_policy(self, agent_name: str) -> AgentSecurityPolicy:
        """Return the policy for *agent_name* or a permissive default.

        Args:
            agent_name: Unique agent identifier.

        Returns:
            The registered policy, or a default :class:`AgentSecurityPolicy`.
        """
        return self._policies.get(agent_name, AgentSecurityPolicy())

    # -- verification --------------------------------------------------------

    def verify_request(
        self,
        request: Request,
        agent_name: str,
    ) -> tuple[bool, str]:
        """Verify that *request* is authorised for *agent_name*.

        Checks performed (in order):
        1. Authentication present when required.
        2. Role claim matches ``allowed_roles``.
        3. Scope claim matches ``allowed_scopes``.

        Args:
            request: The incoming Starlette :class:`Request`.
            agent_name: Target agent name.

        Returns:
            A ``(ok, reason)`` tuple.  *reason* is empty on success.
        """
        policy = self.get_policy(agent_name)

        # --- 1. Authentication required? ---
        auth_required = self._require_auth_globally or policy.require_auth
        raw_claims = getattr(request.state, "jwt_claims", None)
        claims: dict[str, Any] = raw_claims if isinstance(raw_claims, dict) else {}

        if auth_required and not claims:
            self.audit_log(
                "request_denied",
                agent_name,
                {"reason": "authentication_required"},
            )
            return False, "Authentication is required but no valid JWT claims found"

        # Prefer middleware-normalized lists; fall back to claim extraction so
        # Keycloak-style realm_access / scope strings are honoured.
        state_roles = getattr(request.state, "roles", None)
        token_roles: list[str] = (
            list(state_roles) if isinstance(state_roles, list) else extract_roles(claims)
        )
        state_scopes = getattr(request.state, "scopes", None)
        token_scopes: list[str] = (
            list(state_scopes) if isinstance(state_scopes, list) else extract_scopes(claims)
        )

        # --- 2. Role check ---
        if policy.allowed_roles and claims:
            if not any(policy.is_role_allowed(r) for r in token_roles):
                self.audit_log(
                    "request_denied",
                    agent_name,
                    {"reason": "role_not_allowed", "token_roles": token_roles},
                )
                return False, (
                    f"None of the token roles {token_roles} are in "
                    f"allowed_roles {policy.allowed_roles}"
                )

        # --- 3. Scope check ---
        if policy.allowed_scopes and claims:
            if not any(policy.is_scope_allowed(s) for s in token_scopes):
                self.audit_log(
                    "request_denied",
                    agent_name,
                    {"reason": "scope_not_allowed", "token_scopes": token_scopes},
                )
                return False, (
                    f"None of the token scopes {token_scopes} are in "
                    f"allowed_scopes {policy.allowed_scopes}"
                )

        self.audit_log("request_allowed", agent_name)
        return True, ""

    def verify_delegation(
        self,
        source_agent: str,
        target_agent: str,
    ) -> tuple[bool, str]:
        """Check whether *source_agent* may delegate to *target_agent*.

        Args:
            source_agent: Agent initiating the delegation.
            target_agent: Agent being delegated to.

        Returns:
            A ``(ok, reason)`` tuple.
        """
        policy = self.get_policy(source_agent)

        if not policy.is_delegation_allowed(target_agent):
            self.audit_log(
                "delegation_denied",
                source_agent,
                {"target": target_agent},
            )
            return False, (
                f"Agent '{source_agent}' is not allowed to delegate to "
                f"'{target_agent}'. Allowed targets: {policy.allowed_delegation_targets}"
            )

        self.audit_log(
            "delegation_allowed",
            source_agent,
            {"target": target_agent},
        )
        return True, ""

    def enforce_tool_access(
        self,
        agent_name: str,
        tool_name: str,
    ) -> tuple[bool, str]:
        """Check whether *agent_name* may invoke *tool_name*.

        Args:
            agent_name: Agent requesting tool access.
            tool_name: Name of the tool.

        Returns:
            A ``(ok, reason)`` tuple.
        """
        policy = self.get_policy(agent_name)

        if not policy.is_tool_allowed(tool_name):
            self.audit_log(
                "tool_access_denied",
                agent_name,
                {"tool": tool_name},
            )
            return False, (f"Agent '{agent_name}' is not allowed to use tool '{tool_name}'")

        self.audit_log(
            "tool_access_allowed",
            agent_name,
            {"tool": tool_name},
        )
        return True, ""

    # -- audit ---------------------------------------------------------------

    def audit_log(
        self,
        event: str,
        agent_name: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Emit a structured security audit log entry.

        Args:
            event: Short event identifier (e.g. ``request_denied``).
            agent_name: Agent involved in the event.
            context: Optional additional key-value context.
        """
        logger.info(
            "security.audit | event={event} agent={agent} context={ctx}",
            event=event,
            agent=agent_name,
            ctx=context or {},
        )
