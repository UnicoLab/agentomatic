"""LangGraph BaseCheckpointSaver implementation delegating to Agentomatic storage backends."""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from agentomatic.storage.base import BaseStore


class AgentomaticCheckpointer(BaseCheckpointSaver):
    """Custom LangGraph checkpointer mapping checkpoints to Agentomatic BaseStore interfaces."""

    def __init__(self, store: BaseStore) -> None:
        super().__init__()
        self.store = store

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(self.aget_tuple(config))

        if loop.is_running():
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(self.aget_tuple(config)))
                return future.result()  # type: ignore[no-any-return]
        return asyncio.run(self.aget_tuple(config))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"].get("thread_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id", "")

        if not thread_id:
            return None

        cp_data = await self.store.get_checkpoint(thread_id, checkpoint_ns, checkpoint_id)
        if not cp_data:
            return None

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cp_data["checkpoint_id"],
                }
            },
            checkpoint=cp_data["checkpoint"],
            metadata=cp_data["metadata"],
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cp_data["parent_checkpoint_id"],
                    }
                }
                if cp_data.get("parent_checkpoint_id")
                else None
            ),
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(self.aput(config, checkpoint, metadata, new_versions))

        if loop.is_running():
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.aput(config, checkpoint, metadata, new_versions))
                )
                return future.result()
        return asyncio.run(self.aput(config, checkpoint, metadata, new_versions))

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"].get("thread_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not thread_id:
            raise ValueError("thread_id is required in config['configurable']")

        checkpoint_id_str = str(checkpoint_id) if checkpoint_id is not None else ""
        parent_id = metadata.get("parent_checkpoint_id") or config["configurable"].get(
            "checkpoint_id"
        )
        parent_id_str = str(parent_id) if parent_id is not None else None

        await self.store.save_checkpoint(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id_str,
            parent_checkpoint_id=parent_id_str,
            checkpoint=dict(checkpoint),
            metadata=dict(metadata),
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            return iter(
                loop.run_until_complete(
                    self._alist_list(config, filter=filter, before=before, limit=limit)
                )
            )

        if loop.is_running():
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        self._alist_list(config, filter=filter, before=before, limit=limit)
                    )
                )
                return iter(future.result())
        return iter(
            asyncio.run(self._alist_list(config, filter=filter, before=before, limit=limit))
        )

    async def _alist_list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> builtins.list[CheckpointTuple]:
        if not config:
            return []

        thread_id = config["configurable"].get("thread_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if not thread_id:
            return []

        before_id = before["configurable"].get("checkpoint_id") if before else None
        cps_data = await self.store.list_checkpoints(
            thread_id, checkpoint_ns, before=before_id, limit=limit
        )

        tuples = []
        for cp in cps_data:
            tuples.append(
                CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": cp["checkpoint_id"],
                        }
                    },
                    checkpoint=cp["checkpoint"],
                    metadata=cp["metadata"],
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": cp["parent_checkpoint_id"],
                            }
                        }
                        if cp.get("parent_checkpoint_id")
                        else None
                    ),
                )
            )
        return tuples

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        cps = await self._alist_list(config, filter=filter, before=before, limit=limit)
        for cp in cps:  # type: ignore[attr-defined]
            yield cp
