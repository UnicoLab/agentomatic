"""Shared plumbing for Server-Sent Events streams."""

from __future__ import annotations

from .replay import (
    DEFAULT_MAX_FRAMES_PER_STREAM,
    DEFAULT_MAX_STREAMS,
    StreamFrame,
    StreamReplayBuffer,
    get_replay_buffer,
    new_stream_id,
    numbered_stream,
    set_replay_buffer,
    sse_frame,
)

__all__ = [
    "DEFAULT_MAX_FRAMES_PER_STREAM",
    "DEFAULT_MAX_STREAMS",
    "StreamFrame",
    "StreamReplayBuffer",
    "get_replay_buffer",
    "new_stream_id",
    "numbered_stream",
    "set_replay_buffer",
    "sse_frame",
]
