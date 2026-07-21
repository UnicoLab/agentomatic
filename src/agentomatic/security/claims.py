"""Normalize OIDC/JWT role and scope claims across IdP shapes.

Supports Keycloak-style ``realm_access.roles``, space-delimited ``scope``,
flat ``roles`` / ``groups`` / ``scopes`` arrays, and resource-access roles.
"""

from __future__ import annotations

from typing import Any


def extract_roles(claims: dict[str, Any]) -> list[str]:
    """Extract a de-duplicated, sorted role list from JWT claims.

    Args:
        claims: Verified JWT payload.

    Returns:
        Sorted unique role strings.
    """
    roles: list[str] = []
    for key in ("roles", "groups"):
        value = claims.get(key)
        if isinstance(value, list):
            roles.extend(str(item) for item in value)
        elif isinstance(value, str) and value.strip():
            roles.extend(value.replace(",", " ").split())

    realm = claims.get("realm_access")
    if isinstance(realm, dict):
        realm_roles = realm.get("roles")
        if isinstance(realm_roles, list):
            roles.extend(str(item) for item in realm_roles)

    resource_access = claims.get("resource_access")
    if isinstance(resource_access, dict):
        for entry in resource_access.values():
            if isinstance(entry, dict):
                entry_roles = entry.get("roles")
                if isinstance(entry_roles, list):
                    roles.extend(str(item) for item in entry_roles)

    return sorted(set(roles))


def extract_scopes(claims: dict[str, Any]) -> list[str]:
    """Extract OAuth2 scopes from JWT claims.

    Args:
        claims: Verified JWT payload.

    Returns:
        Sorted unique scope strings.
    """
    scopes: list[str] = []
    scope = claims.get("scope")
    if isinstance(scope, str) and scope.strip():
        scopes.extend(scope.split())
    elif isinstance(scope, list):
        scopes.extend(str(item) for item in scope)

    raw_scopes = claims.get("scopes")
    if isinstance(raw_scopes, list):
        scopes.extend(str(item) for item in raw_scopes)
    elif isinstance(raw_scopes, str) and raw_scopes.strip():
        scopes.extend(raw_scopes.replace(",", " ").split())

    scp = claims.get("scp")
    if isinstance(scp, list):
        scopes.extend(str(item) for item in scp)
    elif isinstance(scp, str) and scp.strip():
        scopes.extend(scp.split())

    return sorted(set(scopes))
