"""Dataset mixin — load and save ``AgentDataset`` instances.

Supports JSONL (default) and JSON list formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import AgentDataset


class DatasetMixin:
    """Mixin for loading and saving agent datasets.

    Example::

        agent = MyAgent()
        dataset = agent.load_dataset("data.jsonl")
        agent.save_dataset(dataset, "output.jsonl")
    """

    def load_dataset(
        self,
        path: str | Path,
        format: str = "jsonl",  # noqa: A002
    ) -> AgentDataset:
        """Load a dataset from disk.

        Args:
            path: Path to the dataset file.
            format: File format — ``"jsonl"`` (default) or
                ``"json"`` (list of objects).

        Returns:
            Populated ``AgentDataset``.

        Raises:
            ValueError: If the format is unsupported.
            FileNotFoundError: If the path does not exist.
        """
        from ..types import AgentDataset as _AgentDataset

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if format == "jsonl":
            return _AgentDataset.from_jsonl(path)

        if format == "json":
            with open(path) as f:
                items: list[dict[str, Any]] = json.load(f)
            return _AgentDataset.from_list(items, name=path.stem)

        raise ValueError(f"Unsupported dataset format: '{format}'. Use 'jsonl' or 'json'.")

    def save_dataset(
        self,
        dataset: AgentDataset,
        path: str | Path,
    ) -> None:
        """Save a dataset to disk as JSONL.

        Args:
            dataset: The ``AgentDataset`` to save.
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_jsonl(path)
