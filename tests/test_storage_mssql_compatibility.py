"""SQL Server compatibility contracts for persistent storage."""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import mssql, mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable

from agentomatic.storage.models import Base, ThreadModel
from agentomatic.storage.sqlalchemy import SQLAlchemyStore


def test_thread_self_reference_has_no_database_delete_action() -> None:
    """The thread table must not create a database-managed self cascade."""
    ddl = str(CreateTable(ThreadModel.__table__).compile(dialect=mssql.dialect())).upper()
    assert "FOREIGN KEY(PARENT_THREAD_ID) REFERENCES THREADS (ID)" in ddl
    assert "ON DELETE SET NULL" not in ddl


@pytest.mark.parametrize(
    "dialect",
    [
        postgresql.dialect(),
        mysql.dialect(),
        mssql.dialect(),
        sqlite.dialect(),
    ],
    ids=["postgresql", "mysql-mariadb", "mssql-azure-sql", "sqlite"],
)
def test_complete_storage_schema_compiles_for_supported_databases(dialect: object) -> None:
    """Every storage table must produce valid vendor-specific DDL."""
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert f"CREATE TABLE {table.name}" in ddl


async def test_deleting_parent_detaches_child_thread() -> None:
    """Application cleanup preserves a fork after its parent is deleted."""
    store = SQLAlchemyStore("sqlite+aiosqlite:///:memory:")
    await store.initialize()
    try:
        await store.create_thread("parent", "user", "assistant")
        await store.fork_thread("parent", 0, "child")

        assert await store.delete_thread("parent") is True
        child = await store.get_thread("child")

        assert child is not None
        assert child["parent_thread_id"] is None
    finally:
        await store.close()
