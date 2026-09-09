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
import time
import uuid
from collections import OrderedDict, deque
from typing import Any

#: Frames retained per stream before the oldest are discarded.
DEFAULT_MAX_FRAMES_PER_STREAM = 512

#: Streams retained before the least recently used are discarded.
DEFAULT_MAX_STREAMS = 256


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
        sequence = await store.record(stream_id, payload)
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
    """

    __slots__ = ("data", "sequence", "timestamp")

    def __init__(self, sequence: int, data: str) -> None:
        self.sequence = sequence
        self.data = data
        self.timestamp = time.time()

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
    ) -> None:
        self._max_frames = max(1, max_frames_per_stream)
        self._max_streams = max(1, max_streams)
        self._streams: OrderedDict[str, deque[StreamFrame]] = OrderedDict()
        self._sequences: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def record(self, stream_id: str, data: str) -> int:
        """Append one frame and return the sequence assigned to it.

        Args:
            stream_id: Stream the frame belongs to.
            data: The frame body as sent to the client.

        Returns:
            The frame's sequence number.
        """
        async with self._lock:
            sequence = self._sequences.get(stream_id, 0) + 1
            self._sequences[stream_id] = sequence
            bucket = self._streams.get(stream_id)
            if bucket is None:
                bucket = deque(maxlen=self._max_frames)
                self._streams[stream_id] = bucket
            bucket.append(StreamFrame(sequence, data))
            self._streams.move_to_end(stream_id)
            while len(self._streams) > self._max_streams:
                evicted, _ = self._streams.popitem(last=False)
                self._sequences.pop(evicted, None)
            return sequence

    async def replay(self, stream_id: str, *, after: int = 0) -> list[StreamFrame]:
        """Return retained frames with sequence greater than ``after``.

        Args:
            stream_id: Stream to read.
            after: Exclusive lower bound on sequence.

        Returns:
            The matching frames, oldest first.
        """
        async with self._lock:
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

    async def knows(self, stream_id: str) -> bool:
        """Report whether any frames are retained for a stream.

        Args:
            stream_id: Stream to check.

        Returns:
            True when the stream has retained frames.
        """
        async with self._lock:
            return bool(self._streams.get(stream_id))

    async def drop(self, stream_id: str) -> None:
        """Discard a stream's retained frames.

        Args:
            stream_id: Stream to forget.
        """
        async with self._lock:
            self._streams.pop(stream_id, None)
            self._sequences.pop(stream_id, None)


_DEFAULT_BUFFER: StreamReplayBuffer | None = None


def get_replay_buffer() -> StreamReplayBuffer:
    """Return the process-wide replay buffer, creating it on first use.

    Returns:
        The shared :class:`StreamReplayBuffer`.
    """
    global _DEFAULT_BUFFER
    if _DEFAULT_BUFFER is None:
        _DEFAULT_BUFFER = StreamReplayBuffer()
    return _DEFAULT_BUFFER


def set_replay_buffer(buffer: StreamReplayBuffer | None) -> None:
    """Replace the process-wide replay buffer.

    Args:
        buffer: The buffer to install, or None to reset to a fresh default.
    """
    global _DEFAULT_BUFFER
    _DEFAULT_BUFFER = buffer
