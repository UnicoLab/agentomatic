"""Sequenced SSE frames and a bounded replay buffer for raw streams.

The task subsystem has its own event log, because a task's events outlive the
connection watching them. Raw response streams — ``/invoke/stream``, Studio's
run stream — are different: the work is bound to the request, so a client that
disconnects loses the *run*, not just the view of it.

That still leaves something worth keeping. The frames already produced are the
partial answer, and a client that reconnects can be handed them instead of
being told the whole exchange is gone. This module provides the numbering and
the bounded buffer that makes that possible, plus the one SSE encoder the
streaming endpoints share.

For a stream that must survive a dropped connection *and* keep working, run
the agent as a task (``/invoke/async``) and follow
``/api/v1/tasks/{id}/events``, which is resumable end to end.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from collections import OrderedDict, deque
from typing import Any

#: Frames retained per stream before the oldest are discarded.
DEFAULT_MAX_FRAMES_PER_STREAM = 512

#: Streams retained before the least recently used are discarded.
DEFAULT_MAX_STREAMS = 256

#: Bytes retained per stream. A frame carrying a long generated answer is
#: kilobytes, so a count-only bound is not a memory bound: 512 frames across
#: 256 streams is hundreds of megabytes of perfectly ordinary traffic.
DEFAULT_MAX_BYTES_PER_STREAM = 4 * 1024 * 1024

#: Bytes retained across all streams — the ceiling that actually protects a
#: container's memory limit.
DEFAULT_MAX_BYTES_TOTAL = 64 * 1024 * 1024


def new_stream_id() -> str:
    """Return an identifier for one response stream.

    Returns:
        A short, unique stream id.
    """
    return f"stream_{uuid.uuid4().hex[:16]}"


def sse_frame(payload: Any, *, event_id: int | None = None) -> str:
    """Encode one Server-Sent Events frame.

    Args:
        payload: A JSON-serialisable body, or a pre-rendered ``str``.
        event_id: Sequence to publish as the SSE ``id:`` field. Browsers echo
            the last id they saw back in ``Last-Event-ID`` when reconnecting.

    Returns:
        The encoded frame, terminated by a blank line.
    """
    from agentomatic.langchain_adapter import json_default

    body = payload if isinstance(payload, str) else json.dumps(payload, default=json_default)
    frame = f"data: {body}\n\n"
    if event_id:
        return f"id: {event_id}\n{frame}"
    return frame


async def numbered_stream(
    source: Any,
    *,
    stream_id: str,
    owner: str = "",
    buffer: StreamReplayBuffer | None = None,
) -> Any:
    """Record and number frames from a generator that already emits SSE text.

    Endpoints that build their own ``data: ...`` frames can be made replayable
    without rewriting them: this wraps the generator, stores each frame's
    payload against ``stream_id``, and re-emits it carrying the sequence as
    the SSE ``id:``. Non-data chunks (comments, keep-alives) pass through
    untouched and are not numbered.

    Args:
        source: Async iterator of SSE-encoded strings.
        stream_id: Identity the frames are retained under.
        owner: Ownership tag required to read the frames back.
        buffer: Replay buffer to use (default: the process-wide one).

    Yields:
        The same frames, numbered.
    """
    store = buffer if buffer is not None else get_replay_buffer()
    async for chunk in source:
        payload = _sse_payload(chunk)
        if payload is None:
            yield chunk
            continue
        sequence = await store.record(stream_id, payload, owner=owner)
        yield sse_frame(payload, event_id=sequence)


def _sse_payload(chunk: str) -> str | None:
    """Extract the body of a single-frame ``data:`` chunk.

    Args:
        chunk: A chunk yielded by an SSE generator.

    Returns:
        The payload, or None when the chunk is not a lone data frame (a
        comment, a keep-alive, or a multi-line frame this must not mangle).
    """
    if not isinstance(chunk, str) or not chunk.startswith("data: "):
        return None
    body = chunk.removeprefix("data: ").rstrip("\n")
    return None if "\n" in body else body


class StreamFrame:
    """One recorded frame of a response stream.

    Attributes:
        sequence: Position in the stream, from 1, with no gaps.
        data: The frame body exactly as it was sent.
        timestamp: When the frame was recorded.
        size: Measured in-memory footprint of ``data``, used for budgeting.
    """

    __slots__ = ("data", "sequence", "size", "timestamp")

    def __init__(self, sequence: int, data: str) -> None:
        self.sequence = sequence
        self.data = data
        self.timestamp = time.time()
        # Real in-memory footprint of the payload, measured once. Character
        # count would under-report multi-byte text, which is exactly the
        # traffic most likely to blow a budget.
        self.size = sys.getsizeof(data)

    def as_dict(self) -> dict[str, Any]:
        """Return the frame as a JSON-serialisable mapping.

        Returns:
            The frame's sequence, body and timestamp.
        """
        return {"sequence": self.sequence, "data": self.data, "timestamp": self.timestamp}


class StreamReplayBuffer:
    """Bounded, in-process history of the frames each stream produced.

    Retention is capped per stream and across streams, so a long-running
    process cannot accumulate transcripts without limit.

    Args:
        max_frames_per_stream: Frames retained per stream.
        max_streams: Number of streams retained.
    """

    def __init__(
        self,
        *,
        max_frames_per_stream: int = DEFAULT_MAX_FRAMES_PER_STREAM,
        max_streams: int = DEFAULT_MAX_STREAMS,
        max_bytes_per_stream: int = DEFAULT_MAX_BYTES_PER_STREAM,
        max_bytes_total: int = DEFAULT_MAX_BYTES_TOTAL,
    ) -> None:
        self._max_frames = max(1, max_frames_per_stream)
        self._max_streams = max(1, max_streams)
        self._max_bytes_per_stream = max(1, max_bytes_per_stream)
        self._max_bytes_total = max(1, max_bytes_total)
        # No ``maxlen`` on the deques: an implicit drop would evict a frame
        # without its size leaving the running total, and the byte accounting
        # would drift upwards until nothing could be retained at all.
        self._streams: OrderedDict[str, deque[StreamFrame]] = OrderedDict()
        self._owners: dict[str, str] = {}
        self._bytes: dict[str, int] = {}
        self._total_bytes = 0
        self._sequences: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def record(self, stream_id: str, data: str, *, owner: str = "") -> int:
        """Append one frame and return the sequence assigned to it.

        Args:
            stream_id: Stream the frame belongs to.
            data: The frame body as sent to the client.
            owner: Opaque tag identifying who may read this stream back —
                the agent that produced it, and the principal it was produced
                for. Reads with a different tag are refused. The buffer is
                process-wide, so without this a lookup by id alone would cross
                every authorization boundary the platform enforces per agent.

        Returns:
            The frame's sequence number.
        """
        frame = StreamFrame(sequence=0, data=data)
        async with self._lock:
            sequence = self._sequences.get(stream_id, 0) + 1
            self._sequences[stream_id] = sequence
            self._owners.setdefault(stream_id, owner)
            frame.sequence = sequence

            bucket = self._streams.setdefault(stream_id, deque())
            bucket.append(frame)
            self._bytes[stream_id] = self._bytes.get(stream_id, 0) + frame.size
            self._total_bytes += frame.size
            self._streams.move_to_end(stream_id)

            self._trim_stream(stream_id)
            self._trim_total(keep=stream_id)
            return sequence

    def _trim_stream(self, stream_id: str) -> None:
        """Drop this stream's oldest frames until it fits its own limits."""
        bucket = self._streams.get(stream_id)
        if bucket is None:
            return
        while len(bucket) > 1 and len(bucket) > self._max_frames:
            self._drop_oldest(stream_id)
        self._trim_bytes(stream_id, self._max_bytes_per_stream)

    def _trim_bytes(self, stream_id: str, limit: int) -> None:
        """Drop a stream's oldest frames until it fits ``limit`` bytes.

        The newest frame is always kept: a single frame larger than the budget
        should still be delivered and replayable, not silently dropped.

        Args:
            stream_id: Stream to trim.
            limit: Byte ceiling for this stream.
        """
        bucket = self._streams.get(stream_id)
        if bucket is None:
            return
        while len(bucket) > 1 and self._bytes.get(stream_id, 0) > limit:
            self._drop_oldest(stream_id)

    def _drop_oldest(self, stream_id: str) -> None:
        """Remove a stream's oldest frame, keeping the byte totals honest."""
        bucket = self._streams[stream_id]
        dropped = bucket.popleft()
        self._bytes[stream_id] = self._bytes.get(stream_id, 0) - dropped.size
        self._total_bytes -= dropped.size

    def _trim_total(self, *, keep: str) -> None:
        """Evict least-recently-used streams until the whole buffer fits.

        Args:
            keep: Stream that must survive — it is the one just written to.
        """
        while len(self._streams) > 1 and (
            len(self._streams) > self._max_streams or self._total_bytes > self._max_bytes_total
        ):
            oldest = next(iter(self._streams))
            if oldest == keep:
                # ``keep`` was just moved to the end, so it is the newest;
                # reaching it means every other stream is already gone.
                break
            self._forget(oldest)

        # Nothing left to evict but still over budget: the surviving stream is
        # the one holding the bytes, so trim its history rather than let the
        # total ceiling be quietly ignored.
        if self._total_bytes > self._max_bytes_total:
            self._trim_bytes(keep, self._max_bytes_total)

    def _forget(self, stream_id: str) -> None:
        """Remove one stream and its byte accounting."""
        self._streams.pop(stream_id, None)
        self._total_bytes -= self._bytes.pop(stream_id, 0)
        self._sequences.pop(stream_id, None)
        self._owners.pop(stream_id, None)

    async def replay(
        self, stream_id: str, *, after: int = 0, owner: str | None = None
    ) -> list[StreamFrame]:
        """Return retained frames with sequence greater than ``after``.

        Args:
            stream_id: Stream to read.
            after: Exclusive lower bound on sequence.
            owner: When given, return nothing unless the stream was recorded
                under this exact tag.

        Returns:
            The matching frames, oldest first.
        """
        async with self._lock:
            if owner is not None and self._owners.get(stream_id) != owner:
                return []
            bucket = self._streams.get(stream_id)
            if not bucket:
                return []
            return [frame for frame in bucket if frame.sequence > after]

    async def earliest_sequence(self, stream_id: str) -> int:
        """Return the lowest sequence still retained, or 0 when empty.

        Args:
            stream_id: Stream to inspect.

        Returns:
            The retention floor.
        """
        async with self._lock:
            bucket = self._streams.get(stream_id)
            return bucket[0].sequence if bucket else 0

    async def knows(self, stream_id: str, *, owner: str | None = None) -> bool:
        """Report whether a stream is retained and readable by ``owner``.

        A stream owned by someone else reports False rather than raising, so
        callers answer "not found" and do not confirm that the id exists.

        Args:
            stream_id: Stream to check.
            owner: When given, require this exact ownership tag.

        Returns:
            True when the stream has retained frames the caller may read.
        """
        async with self._lock:
            if owner is not None and self._owners.get(stream_id) != owner:
                return False
            return bool(self._streams.get(stream_id))

    async def drop(self, stream_id: str) -> None:
        """Discard a stream's retained frames.

        Args:
            stream_id: Stream to forget.
        """
        async with self._lock:
            self._forget(stream_id)

    async def stats(self) -> dict[str, int]:
        """Return current occupancy, for tests and operational visibility.

        Returns:
            Stream count, retained frame count and total retained bytes.
        """
        async with self._lock:
            return {
                "streams": len(self._streams),
                "frames": sum(len(bucket) for bucket in self._streams.values()),
                "bytes": self._total_bytes,
            }


_DEFAULT_BUFFER: StreamReplayBuffer | None = None


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment.

    Args:
        name: Environment variable to read.
        default: Value used when unset, unparseable, or not positive.

    Returns:
        The configured value, or ``default``.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


def buffer_from_env() -> StreamReplayBuffer:
    """Build the replay buffer a deployment's environment asks for.

    Reads ``AGENTOMATIC_STREAM_FRAMES``, ``AGENTOMATIC_STREAM_COUNT`` and
    ``AGENTOMATIC_STREAM_MAX_MB``. Set ``AGENTOMATIC_STREAM_REPLAY=0`` to keep
    nothing at all: streams still work, they just cannot be replayed.

    Returns:
        The configured :class:`StreamReplayBuffer`.
    """
    enabled = (os.getenv("AGENTOMATIC_STREAM_REPLAY") or "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        # A zero-capacity buffer keeps one frame per stream at most, which is
        # the cheapest way to disable replay without special-casing callers.
        return StreamReplayBuffer(
            max_frames_per_stream=1, max_streams=1, max_bytes_per_stream=1, max_bytes_total=1
        )
    return StreamReplayBuffer(
        max_frames_per_stream=_env_int("AGENTOMATIC_STREAM_FRAMES", DEFAULT_MAX_FRAMES_PER_STREAM),
        max_streams=_env_int("AGENTOMATIC_STREAM_COUNT", DEFAULT_MAX_STREAMS),
        max_bytes_total=_env_int(
            "AGENTOMATIC_STREAM_MAX_MB", DEFAULT_MAX_BYTES_TOTAL // (1024 * 1024)
        )
        * 1024
        * 1024,
    )


def get_replay_buffer() -> StreamReplayBuffer:
    """Return the process-wide replay buffer, creating it on first use.

    Returns:
        The shared :class:`StreamReplayBuffer`.
    """
    global _DEFAULT_BUFFER
    if _DEFAULT_BUFFER is None:
        _DEFAULT_BUFFER = buffer_from_env()
    return _DEFAULT_BUFFER


def set_replay_buffer(buffer: StreamReplayBuffer | None) -> None:
    """Replace the process-wide replay buffer.

    Args:
        buffer: The buffer to install, or None to reset to a fresh default.
    """
    global _DEFAULT_BUFFER
    _DEFAULT_BUFFER = buffer
