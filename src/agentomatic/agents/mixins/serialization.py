"""Serialization mixin — save and load agent state as JSON files.

Persists:
- ``config.json`` — compiled configuration
- ``metadata.json`` — compilation metadata and agent info
- ``evaluation_history.json`` — past evaluation reports
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger


class SerializationMixin:
    """Mixin for persisting agent configuration and history.

    Saves and loads agent state from a directory containing
    JSON files.

    Example::

        agent.save("./checkpoints/my_agent")
        loaded = MyAgent.load("./checkpoints/my_agent")
    """

    def save(self, path: str | Path) -> None:
        """Save agent state to a directory.

        Creates the directory if it does not exist. Writes:
        - ``config.json`` — compiled configuration
        - ``metadata.json`` — compilation metadata + agent info
        - ``evaluation_history.json`` — evaluation history

        Args:
            path: Directory path to save into.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        # Config
        config: dict[str, Any] = getattr(self, "compiled_config", {})
        (directory / "config.json").write_text(json.dumps(config, indent=2, default=str))

        # Metadata
        metadata: dict[str, Any] = getattr(self, "compiled_metadata", {})
        agent_meta = {
            "agent_class": type(self).__qualname__,
            "agent_module": type(self).__module__,
            **metadata,
        }
        (directory / "metadata.json").write_text(json.dumps(agent_meta, indent=2, default=str))

        # Evaluation history
        history: list[Any] = getattr(self, "evaluation_history", [])
        serialized_history = [asdict(report) for report in history]
        (directory / "evaluation_history.json").write_text(
            json.dumps(serialized_history, indent=2, default=str)
        )

        logger.info("Agent saved to {}", directory)

    @classmethod
    def load(cls, path: str | Path) -> SerializationMixin:
        """Load agent state from a directory.

        Instantiates the class and restores ``compiled_config``,
        ``compiled_metadata``, and ``evaluation_history`` from
        JSON files.

        Args:
            path: Directory path to load from.

        Returns:
            A new instance of the agent class with restored state.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """

        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"Agent directory not found: {directory}")

        instance = cls()

        # Config
        config_path = directory / "config.json"
        if config_path.exists():
            instance.compiled_config = json.loads(  # type: ignore[attr-defined]
                config_path.read_text()
            )

        # Metadata
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            instance.compiled_metadata = json.loads(  # type: ignore[attr-defined]
                meta_path.read_text()
            )

        # Evaluation history
        history_path = directory / "evaluation_history.json"
        if history_path.exists():
            raw_history: list[dict[str, Any]] = json.loads(history_path.read_text())
            instance.evaluation_history = [  # type: ignore[attr-defined]
                _report_from_dict(entry) for entry in raw_history
            ]

        logger.info("Agent loaded from {}", directory)
        return instance


def _report_from_dict(data: dict[str, Any]) -> Any:
    """Reconstruct an ``EvaluationReport`` from a dictionary.

    Args:
        data: Serialized report dictionary.

    Returns:
        An ``EvaluationReport`` instance.
    """
    from ..types import EvaluationReport, ExampleResult

    example_results = [ExampleResult(**er) for er in data.get("example_results", [])]
    return EvaluationReport(
        agent_name=data.get("agent_name", ""),
        dataset_name=data.get("dataset_name", ""),
        scores=data.get("scores", {}),
        example_results=example_results,
        metadata=data.get("metadata", {}),
    )
