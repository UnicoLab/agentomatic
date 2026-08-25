# pyright: reportMissingParameterType=none
"""Committed rows must come back complete without a second database read.

``SQLAlchemyStore`` used to follow every ``commit()`` with ``refresh()``,
which issues a SELECT to re-read the row just written. Against a networked
Postgres that roughly doubled the latency of every write — measured at
+38ms p50 on ``POST /invoke`` with invocation logging on, a 6x increase over
the same call with logging off.

The refresh was never needed: the session factory sets
``expire_on_commit=False`` and every column default in ``storage.models`` is
Python-side, so a committed object is already fully populated. These tests
pin both halves of that reasoning, so a change to either is caught here
rather than by someone profiling production.
"""

from __future__ import annotations

import inspect

import pytest

from agentomatic.storage import models as storage_models
from agentomatic.storage.sqlalchemy import SQLAlchemyStore


@pytest.fixture
async def store(tmp_path):
    """A real SQLite-backed store."""
    s = SQLAlchemyStore(f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    await s.initialize()
    yield s
    await s.close()


class TestTheInvariantsThatMakeRefreshUnnecessary:
    def test_sessions_do_not_expire_on_commit(self) -> None:
        """If this flips to True, attributes reload lazily after commit."""
        source = inspect.getsource(SQLAlchemyStore.__init__)

        assert "expire_on_commit=False" in source

    def test_no_column_relies_on_a_server_side_default(self) -> None:
        """A server default would only be known after reading the row back."""
        source = inspect.getsource(storage_models)

        assert "server_default" not in source

    def test_write_paths_do_not_refresh_after_commit(self) -> None:
        """The redundant round trip must not creep back in."""
        import agentomatic.storage.sqlalchemy as module

        # Comments explain the absence, so match code lines only.
        source = "\n".join(
            line
            for line in inspect.getsource(module).splitlines()
            if not line.lstrip().startswith("#")
        )

        assert "session.refresh(" not in source, (
            "a post-commit refresh re-reads a row that is already fully "
            "loaded — it costs one extra database round trip per write"
        )


class TestWritesStillReturnCompleteRows:
    @pytest.mark.asyncio
    async def test_thread_comes_back_populated(self, store) -> None:
        thread = await store.create_thread("th_1", "u1", "a1", title="t")

        assert thread["id"] == "th_1"
        assert thread["created_at"]
        assert thread["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_message_comes_back_populated(self, store) -> None:
        thread = await store.create_thread("th_2", "u1", "a1")

        message = await store.add_message(thread["id"], "user", "hello")

        assert message["id"]
        assert message["timestamp"]
        assert message["content"] == "hello"

    @pytest.mark.asyncio
    async def test_invocation_log_comes_back_populated(self, store) -> None:
        entry = await store.create_invocation_log(
            agent_name="a1", endpoint="invoke", input_data={"query": "x"}
        )

        assert entry["id"].startswith("invlog_")
        assert entry["timestamp"]
        assert entry["endpoint"] == "invoke"

    @pytest.mark.asyncio
    async def test_feedback_comes_back_populated(self, store) -> None:
        thread = await store.create_thread("th_3", "u1", "a1")

        feedback = await store.add_feedback(thread["id"], "u1", "a1", rating=5, comment="good")

        assert feedback["id"]
        assert feedback["rating"] == 5

    @pytest.mark.asyncio
    async def test_written_rows_are_readable_afterwards(self, store) -> None:
        """The values returned must be what actually landed in the database."""
        thread = await store.create_thread("th_4", "u1", "a1", title="t")
        await store.add_message(thread["id"], "user", "hello")

        fetched = await store.get_thread(thread["id"])
        messages = await store.get_messages(thread["id"])

        assert fetched["id"] == thread["id"]
        assert fetched["created_at"] == thread["created_at"]
        assert [m["content"] for m in messages] == ["hello"]


class TestTimestampsAreUnambiguous:
    """Stored timestamps must carry an explicit UTC offset on every backend.

    ``DateTime(timezone=True)`` is a no-op on SQLite, so a value read back
    from there is naive while the same value from Postgres carries ``+00:00``.
    Emitting the raw ``isoformat()`` therefore made the API's timestamp format
    depend on which database was configured — and left a client no way to know
    a naive string was UTC.
    """

    def test_naive_values_are_stamped_utc(self) -> None:
        from datetime import datetime

        from agentomatic.storage.models import iso_utc

        assert iso_utc(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05+00:00"

    def test_aware_values_are_converted_not_relabelled(self) -> None:
        from datetime import datetime, timedelta, timezone

        from agentomatic.storage.models import iso_utc

        berlin = timezone(timedelta(hours=2))
        stamped = datetime(2026, 1, 2, 5, 4, 5, tzinfo=berlin)

        assert iso_utc(stamped) == "2026-01-02T03:04:05+00:00"

    def test_none_stays_none(self) -> None:
        from agentomatic.storage.models import iso_utc

        assert iso_utc(None) is None

    @pytest.mark.asyncio
    async def test_write_and_read_agree_on_format(self, store) -> None:
        """The value a write returns must match what a later read returns."""
        written = await store.create_thread("th_fmt", "u1", "a1", title="t")

        read = await store.get_thread("th_fmt")

        assert written["created_at"] == read["created_at"]
        assert written["created_at"].endswith("+00:00")

    @pytest.mark.asyncio
    async def test_message_timestamps_are_offset_aware(self, store) -> None:
        await store.create_thread("th_msg", "u1", "a1")
        written = await store.add_message("th_msg", "user", "hello")

        read = (await store.get_messages("th_msg"))[0]

        assert written["timestamp"] == read["timestamp"]
        assert read["timestamp"].endswith("+00:00")

    @pytest.mark.asyncio
    async def test_invocation_log_timestamps_are_offset_aware(self, store) -> None:
        written = await store.create_invocation_log(agent_name="a1", endpoint="invoke")

        read = await store.get_invocation_log(written["id"])

        assert written["timestamp"] == read["timestamp"]
        assert read["timestamp"].endswith("+00:00")
