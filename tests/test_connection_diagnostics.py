# pyright: reportMissingParameterType=none
"""Unconfigured connections must name the variable an operator has to set.

The scaffolded ``connections.py`` ships several example connections whose
targets are ``${ENV}`` placeholders. Unset, those resolve to an empty string
and every boot logged an error from whatever the driver made of it — four
opaque ERROR lines on a fresh scaffold, none of which said which variable was
missing.
"""

from __future__ import annotations

import pytest

from agentomatic.connections.manager import (
    ConnectionManager,
    _unconfigured_reason,
    unresolved_env_vars,
)
from agentomatic.connections.models import (
    ConnectionPurpose,
    DatabaseConnectionConfig,
    HttpConnectionConfig,
    VectorConnectionConfig,
)


class TestUnresolvedEnvVars:
    def test_reports_unset_placeholders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AG_TEST_URL", raising=False)

        assert unresolved_env_vars("${AG_TEST_URL}") == ["AG_TEST_URL"]

    def test_ignores_set_placeholders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AG_TEST_URL", "postgresql+asyncpg://host/db")

        assert unresolved_env_vars("${AG_TEST_URL}") == []

    def test_reports_every_missing_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AG_A", raising=False)
        monkeypatch.setenv("AG_B", "set")
        monkeypatch.delenv("AG_C", raising=False)

        assert unresolved_env_vars("${AG_A}/${AG_B}/${AG_C}") == ["AG_A", "AG_C"]

    def test_plain_strings_have_none(self) -> None:
        assert unresolved_env_vars("sqlite+aiosqlite:///data.db") == []

    def test_non_strings_are_ignored(self) -> None:
        assert unresolved_env_vars(None) == []
        assert unresolved_env_vars(5) == []


class TestUnconfiguredReason:
    def test_database_target_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AG_RAG_DB_URL", raising=False)
        config = DatabaseConnectionConfig(name="main", url="${AG_RAG_DB_URL}")

        reason = _unconfigured_reason(config)

        assert reason is not None
        assert "AG_RAG_DB_URL" in reason

    def test_http_target_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AG_SCORING_URL", raising=False)
        config = HttpConnectionConfig(name="api", base_url="${AG_SCORING_URL}")

        reason = _unconfigured_reason(config)

        assert reason is not None
        assert "AG_SCORING_URL" in reason

    def test_vector_target_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AG_QDRANT_URL", raising=False)
        config = VectorConnectionConfig(name="kb", provider="qdrant", url="${AG_QDRANT_URL}")

        reason = _unconfigured_reason(config)

        assert reason is not None
        assert "AG_QDRANT_URL" in reason

    def test_a_configured_connection_is_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AG_RAG_DB_URL", "sqlite+aiosqlite:///data.db")
        config = DatabaseConnectionConfig(name="main", url="${AG_RAG_DB_URL}")

        assert _unconfigured_reason(config) is None

    def test_a_literal_url_is_not_flagged(self) -> None:
        config = DatabaseConnectionConfig(name="main", url="sqlite+aiosqlite:///data.db")

        assert _unconfigured_reason(config) is None

    def test_a_partially_resolved_target_is_left_to_the_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Half-configured is a real misconfiguration — do not mask it."""
        monkeypatch.delenv("AG_DB_HOST", raising=False)
        config = DatabaseConnectionConfig(name="main", url="postgresql+asyncpg://${AG_DB_HOST}/db")

        assert _unconfigured_reason(config) is None


class TestManagerSkipsUnconfigured:
    @pytest.mark.asyncio
    async def test_unconfigured_connection_is_warned_not_errored(
        self, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        monkeypatch.delenv("AG_RAG_DB_URL", raising=False)
        manager = ConnectionManager("ag_rag")
        manager.add(DatabaseConnectionConfig(name="main", url="${AG_RAG_DB_URL}"))

        # Must not raise, and must not attempt a doomed driver connection.
        await manager.initialize()

        assert "main" in manager.list_names()

    @pytest.mark.asyncio
    async def test_configured_connections_still_initialize(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        db = tmp_path / "conn.db"
        monkeypatch.setenv("AG_RAG_DB_URL", f"sqlite+aiosqlite:///{db}")
        manager = ConnectionManager("ag_rag")
        manager.add(DatabaseConnectionConfig(name="main", url="${AG_RAG_DB_URL}"))

        await manager.initialize()

        assert manager.database("main") is not None
        await manager.close()


class TestHealthDistinguishesAbsentFromBroken:
    """An operator acts differently on "down" than on "never switched on".

    Regression: an unconfigured connection reported ``unhealthy`` with
    whatever the driver made of an empty URL — in the control plane that reads
    as a backend outage, when nothing is wrong and no variable was ever set.
    """

    @pytest.mark.asyncio
    async def test_unconfigured_reports_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AG_ABSENT_URL", raising=False)
        manager = ConnectionManager("scope")
        manager.add(DatabaseConnectionConfig(name="absent", url="${AG_ABSENT_URL}"))

        health = await manager.health_check()

        assert health["absent"]["status"] == "not_configured"
        assert "AG_ABSENT_URL" in health["absent"]["detail"]

    @pytest.mark.asyncio
    async def test_unconfigured_still_reports_its_kind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AG_ABSENT_URL", raising=False)
        manager = ConnectionManager("scope")
        manager.add(DatabaseConnectionConfig(name="absent", url="${AG_ABSENT_URL}"))

        health = await manager.health_check()

        assert "database" in health["absent"]["kind"]

    @pytest.mark.asyncio
    async def test_a_configured_connection_reports_healthy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("AG_REAL_URL", f"sqlite+aiosqlite:///{tmp_path / 'h.db'}")
        manager = ConnectionManager("scope")
        manager.add(DatabaseConnectionConfig(name="real", url="${AG_REAL_URL}"))
        await manager.initialize()

        health = await manager.health_check()

        assert health["real"]["status"] == "healthy"
        await manager.close()


class TestStorePrecedenceIsAnnounced:
    """A MEMORY connection silently outranking DATABASE_URL loses data.

    Both can be configured at once, and the connection wins. Left unsaid, an
    operator who pointed DATABASE_URL at a managed Postgres reads "store
    configured" and believes their threads live there — while they are going
    wherever the connection points, which for the scaffolded MEMORY example is
    a file inside the container that dies with it.
    """

    @pytest.mark.asyncio
    async def test_the_override_is_logged_as_a_warning(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentomatic import AgentPlatform
        from agentomatic.connections.manager import register_connections
        from agentomatic.core import platform as platform_mod

        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:secret@db:5432/prod")
        monkeypatch.setenv("AG_MEM_URL", f"sqlite+aiosqlite:///{tmp_path / 'mem.db'}")
        agents = tmp_path / "agents"
        agents.mkdir()
        register_connections(
            "scoped",
            [
                DatabaseConnectionConfig(
                    name="memory",
                    url="${AG_MEM_URL}",
                    purpose=ConnectionPurpose.MEMORY,
                )
            ],
        )
        messages: list[str] = []
        # Capture the call rather than adding a loguru sink: the platform
        # reconfigures logging, which tears sinks down mid-test.
        monkeypatch.setattr(
            platform_mod.logger,
            "warning",
            lambda msg, *a, **k: messages.append(str(msg)),
        )
        platform = AgentPlatform(agents_dir=str(agents))
        await platform._auto_derive_store_from_connections()  # noqa: SLF001

        assert any("OVERRIDES" in m for m in messages), messages

    @pytest.mark.asyncio
    async def test_the_warning_never_prints_the_password(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentomatic import AgentPlatform
        from agentomatic.connections.manager import register_connections
        from agentomatic.core import platform as platform_mod

        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:hunter2@db:5432/prod")
        monkeypatch.setenv("AG_MEM_URL2", f"sqlite+aiosqlite:///{tmp_path / 'mem2.db'}")
        agents = tmp_path / "agents"
        agents.mkdir()
        register_connections(
            "scoped2",
            [
                DatabaseConnectionConfig(
                    name="memory",
                    url="${AG_MEM_URL2}",
                    purpose=ConnectionPurpose.MEMORY,
                )
            ],
        )
        messages: list[str] = []
        # Capture the call rather than adding a loguru sink: the platform
        # reconfigures logging, which tears sinks down mid-test.
        monkeypatch.setattr(
            platform_mod.logger,
            "warning",
            lambda msg, *a, **k: messages.append(str(msg)),
        )
        platform = AgentPlatform(agents_dir=str(agents))
        await platform._auto_derive_store_from_connections()  # noqa: SLF001

        assert not any("hunter2" in m for m in messages), "credential leaked into the log"

    def test_the_url_helper_strips_credentials(self) -> None:
        from agentomatic.core.platform import _safe_db_url

        assert _safe_db_url("postgresql+asyncpg://u:pw@host:5432/db") == "host:5432/db"
        assert _safe_db_url("sqlite+aiosqlite:///data.db") == "sqlite+aiosqlite:///data.db"
