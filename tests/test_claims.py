"""Tests for OIDC claim normalization helpers."""

from __future__ import annotations

from agentomatic.security.claims import extract_roles, extract_scopes


class TestExtractRoles:
    def test_empty_claims(self) -> None:
        assert extract_roles({}) == []

    def test_flat_roles_and_groups(self) -> None:
        assert extract_roles({"roles": ["admin"], "groups": ["po"]}) == ["admin", "po"]

    def test_realm_access(self) -> None:
        claims = {"realm_access": {"roles": ["metier", "offline_access"]}}
        assert extract_roles(claims) == ["metier", "offline_access"]

    def test_resource_access(self) -> None:
        claims = {
            "resource_access": {
                "scooper-backend": {"roles": ["reader"]},
                "account": {"roles": ["manage-account"]},
            }
        }
        assert extract_roles(claims) == ["manage-account", "reader"]

    def test_dedupes_and_sorts(self) -> None:
        assert extract_roles({"roles": ["b", "a", "b"]}) == ["a", "b"]


class TestExtractScopes:
    def test_space_delimited_scope(self) -> None:
        assert extract_scopes({"scope": "openid profile email"}) == [
            "email",
            "openid",
            "profile",
        ]

    def test_scopes_array(self) -> None:
        assert extract_scopes({"scopes": ["read", "write"]}) == ["read", "write"]

    def test_scp_claim(self) -> None:
        assert extract_scopes({"scp": ["api"]}) == ["api"]
