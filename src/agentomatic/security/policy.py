"""Per-agent security policy definitions.

An :class:`AgentSecurityPolicy` captures fine-grained access-control rules
that the :class:`~agentomatic.security.zero_trust.ZeroTrustEnforcer` evaluates
at runtime for every agent invocation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentSecurityPolicy(BaseModel):
    """Declarative security policy attached to a single agent.

    Attributes:
        require_auth: If ``True``, override the platform-level auth flag and
            force authentication for this agent.
        allowed_roles: JWT roles that may access this agent.  An empty list
            means *all roles are allowed*.
        allowed_scopes: OAuth2 scopes required to access this agent.  An empty
            list means *all scopes are accepted*.
        max_execution_time: Hard timeout for a single invocation (seconds).
        max_tokens_per_request: Token budget forwarded to the LLM provider.
        allowed_tools: Explicit tool whitelist.  ``None`` means every tool is
            allowed; an empty list means *no* tools are allowed.
        blocked_tools: Tool blacklist — always applied on top of the whitelist.
        env_vars_whitelist: If set, only these environment variables are visible
            to the agent's tool sandbox.
        allowed_delegation_targets: Agent names this agent may delegate to.
            An empty list means *no delegation allowed*.
        rate_limit_requests: Per-agent request cap (overrides platform default).
        rate_limit_window: Time window in seconds for the rate limit.
    """

    require_auth: bool = False
    allowed_roles: list[str] = Field(default_factory=list)
    allowed_scopes: list[str] = Field(default_factory=list)
    max_execution_time: float = Field(30.0, gt=0)
    max_tokens_per_request: int = Field(8192, ge=1)
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] = Field(default_factory=list)
    env_vars_whitelist: list[str] | None = None
    allowed_delegation_targets: list[str] = Field(default_factory=list)
    rate_limit_requests: int | None = None
    rate_limit_window: int | None = None

    # -- access-check helpers ------------------------------------------------

    def is_role_allowed(self, role: str) -> bool:
        """Return ``True`` if *role* passes the role filter.

        When ``allowed_roles`` is empty every role is accepted.

        Args:
            role: The role claim to check.

        Returns:
            Whether the role is permitted.
        """
        if not self.allowed_roles:
            return True
        return role in self.allowed_roles

    def is_scope_allowed(self, scope: str) -> bool:
        """Return ``True`` if *scope* passes the scope filter.

        When ``allowed_scopes`` is empty every scope is accepted.

        Args:
            scope: The OAuth2 scope to check.

        Returns:
            Whether the scope is permitted.
        """
        if not self.allowed_scopes:
            return True
        return scope in self.allowed_scopes

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* passes whitelist/blacklist checks.

        Evaluation order:
        1. If the tool appears in ``blocked_tools`` → **deny**.
        2. If ``allowed_tools`` is ``None`` → **allow** (no whitelist).
        3. If the tool appears in ``allowed_tools`` → **allow**.
        4. Otherwise → **deny**.

        Args:
            tool_name: Name of the tool being invoked.

        Returns:
            Whether the tool invocation is permitted.
        """
        if tool_name in self.blocked_tools:
            return False
        if self.allowed_tools is None:
            return True
        return tool_name in self.allowed_tools

    def is_delegation_allowed(self, target_agent: str) -> bool:
        """Return ``True`` if delegation to *target_agent* is permitted.

        Delegation is only allowed when the target appears in
        ``allowed_delegation_targets``.  An empty list means no delegation is
        allowed at all.

        Args:
            target_agent: The name of the agent to delegate to.

        Returns:
            Whether delegation is permitted.
        """
        return target_agent in self.allowed_delegation_targets
