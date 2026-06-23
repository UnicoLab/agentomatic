# Security & Zero Trust

Agentomatic takes security seriously, operating under a **Zero-Trust** model for agent communications and deployments.

## JWT Authentication

When exposing agents via REST APIs, you can protect the endpoints using the built-in `JWTAuthMiddleware`. This middleware validates incoming JSON Web Tokens (JWT) against an identity provider (e.g., Auth0, Firebase, Keycloak).

```python
from agentomatic.security import JWTAuthMiddleware, JWTConfig
from agentomatic import AgentPlatform

config = JWTConfig(
    jwks_url="https://auth.example.com/.well-known/jwks.json",
    audience="agentomatic-api",
    issuer="https://auth.example.com/"
)

platform = AgentPlatform()
platform._app.add_middleware(JWTAuthMiddleware, config=config)
```

## Agent Security Policies (RBAC)

Agentomatic allows you to enforce Role-Based Access Control (RBAC) on a per-agent basis using `AgentSecurityPolicy`.

If an agent requires specific permissions, you can declare them:

```python
from agentomatic.security import AgentSecurityPolicy

policy = AgentSecurityPolicy(
    required_roles=["admin", "ai_operator"],
    allow_anonymous=False
)

# Apply this policy to an agent's run endpoint to restrict execution.
```

## Zero-Trust Enforcer

In advanced multi-agent (A2A) setups, you may want to prevent certain agents from invoking others. The `ZeroTrustEnforcer` intercepts internal communication and checks if the calling agent is authorized to invoke the target agent.

```python
from agentomatic.security import ZeroTrustEnforcer

enforcer = ZeroTrustEnforcer()

# Only allow 'orchestrator' to call 'db_agent'
enforcer.allow("orchestrator", "db_agent")

enforcer.check_permission(source="orchestrator", target="db_agent") # OK
enforcer.check_permission(source="unauthorized", target="db_agent") # Raises UnauthorizedError
```

By combining JWT validation for external users and Zero-Trust enforcement for internal multi-agent communication, Agentomatic provides an enterprise-grade security layer natively.
