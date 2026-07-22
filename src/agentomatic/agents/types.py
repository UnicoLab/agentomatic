"""Type definitions, protocols, and data structures for class-owned agents.

Contains:
- ``StateT`` — generic type variable for agent state
- ``Metric`` / ``Optimizer`` — protocols for evaluation and optimization
- ``AgentExample`` / ``AgentDataset`` — rich dataset containers
- ``EvaluationReport`` / ``ExampleResult`` — evaluation output
- ``TraceEvent`` — observability trace event
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

# ---------------------------------------------------------------------------
# Generic state type
# ---------------------------------------------------------------------------

StateT = TypeVar("StateT")
"""Generic type variable for agent state (dataclass, TypedDict, etc.)."""


# ---------------------------------------------------------------------------
# Metric protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Metric(Protocol):
    """Protocol for evaluation metrics.

    Any object with ``name`` and ``score()`` can be used as a metric.

    Example::

        class MyMetric:
            name = "my_metric"

            def score(
                self,
                example: AgentExample,
                prediction: dict[str, Any],
            ) -> float:
                return 1.0 if "key" in prediction else 0.0
    """

    name: str

    def score(
        self,
        example: AgentExample,
        prediction: dict[str, Any],
    ) -> float:
        """Score a single prediction against an example.

        Args:
            example: The input example with expected output.
            prediction: The agent's actual output.

        Returns:
            A float score between 0.0 and 1.0.
        """
        ...


# ---------------------------------------------------------------------------
# Optimizer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for agent optimizers.

    An optimizer takes an agent, dataset, and metrics, and returns
    an optimized configuration dict.

    Example::

        class MyOptimizer:
            def optimize(
                self, agent, dataset, metrics,
            ) -> dict[str, Any]:
                return {"temperature": 0.2}
    """

    def optimize(
        self,
        agent: Any,
        dataset: AgentDataset,
        metrics: Sequence[Metric],
    ) -> dict[str, Any]:
        """Run optimization and return config changes.

        Args:
            agent: The agent to optimize.
            dataset: Training dataset.
            metrics: Metrics to evaluate.

        Returns:
            Dict of optimized configuration values.
        """
        ...


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class AgentExample:
    """Single evaluation example for an agent.

    Richer than ``optimize.DataPoint`` — supports structured I/O,
    rubrics, tags, and metadata.

    Attributes:
        id: Unique identifier for this example.
        input: Input data dictionary.
        expected_output: Expected output (ground truth).
        metadata: Arbitrary metadata (domain, difficulty, source, etc.).
        rubric: Evaluation rubric (per-dimension criteria).
        tags: Tags for filtering/grouping.
        split: Train/validation/test split assignment.
    """

    id: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    expected_output: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d: dict[str, Any] = {"id": self.id, "input": self.input}
        if self.expected_output is not None:
            d["expected_output"] = self.expected_output
        if self.metadata:
            d["metadata"] = self.metadata
        if self.rubric:
            d["rubric"] = self.rubric
        if self.tags:
            d["tags"] = self.tags
        if self.split:
            d["split"] = self.split
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentExample:
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", ""),
            input=data.get("input", {}),
            expected_output=data.get("expected_output"),
            metadata=data.get("metadata", {}),
            rubric=data.get("rubric", {}),
            tags=data.get("tags", []),
            split=data.get("split", "train"),
        )

    def to_datapoint(self) -> Any:
        """Convert to ``optimize.DataPoint`` for the existing pipeline.

        Returns:
            A ``DataPoint`` compatible with ``agentomatic.optimize``.
        """
        from agentomatic.optimize.dataset import DataPoint

        # Extract query from input — try common keys (including current_query
        # used by agents that follow the agentomatic convention).
        query = (
            self.input.get("query")
            or self.input.get("current_query")
            or self.input.get("request")
            or self.input.get("question")
            or json.dumps(self.input)
        )
        expected = None
        if self.expected_output:
            expected = (
                self.expected_output.get("response")
                or self.expected_output.get("answer")
                or json.dumps(self.expected_output)
            )
        meta = dict(self.metadata or {})
        if self.split and "split" not in meta:
            meta["split"] = self.split
        # Ensure invoke payload carries agent inputs when not already set.
        if "invoke" not in meta and self.input:
            invoke = {
                k: v for k, v in self.input.items() if k not in {"query", "request", "question"}
            }
            if invoke:
                meta["invoke"] = invoke
        return DataPoint(
            query=str(query),
            expected_answer=expected,
            metadata=meta,
        )


@dataclass
class AgentDataset:
    """Collection of agent examples with train/val/test splits.

    Example::

        dataset = AgentDataset.from_jsonl("data.jsonl")
        report = agent.evaluate(dataset.test, metrics=[...])
    """

    name: str = "dataset"
    examples: list[AgentExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Properties for split access ---

    @property
    def train(self) -> list[AgentExample]:
        """Return training examples."""
        return [e for e in self.examples if e.split == "train"]

    @property
    def validation(self) -> list[AgentExample]:
        """Return validation examples."""
        return [e for e in self.examples if e.split in ("validation", "val")]

    @property
    def test(self) -> list[AgentExample]:
        """Return test examples."""
        return [e for e in self.examples if e.split == "test"]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[AgentExample]:
        return iter(self.examples)

    def __getitem__(self, idx: int) -> AgentExample:
        return self.examples[idx]

    # --- I/O ---

    def to_jsonl(self, path: str | Path) -> None:
        """Save to JSONL file."""
        with open(path, "w") as f:
            for example in self.examples:
                f.write(json.dumps(example.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str = "") -> AgentDataset:
        """Load from JSONL file.

        Args:
            path: Path to the JSONL file.
            name: Dataset name (defaults to filename).

        Returns:
            Populated ``AgentDataset``.
        """
        path = Path(path)
        examples: list[AgentExample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if not data.get("id"):
                    data["id"] = f"example_{i:04d}"
                examples.append(AgentExample.from_dict(data))
        return cls(
            name=name or path.stem,
            examples=examples,
        )

    @classmethod
    def from_list(
        cls,
        items: list[dict[str, Any]],
        name: str = "dataset",
    ) -> AgentDataset:
        """Create from a list of dictionaries."""
        examples = []
        for i, item in enumerate(items):
            if not item.get("id"):
                item["id"] = f"example_{i:04d}"
            examples.append(AgentExample.from_dict(item))
        return cls(name=name, examples=examples)

    def to_optimize_dataset(self) -> Any:
        """Convert to ``optimize.Dataset`` for the existing pipeline.

        Returns:
            A ``Dataset`` compatible with ``agentomatic.optimize``.
        """
        from agentomatic.optimize.dataset import Dataset

        return Dataset(
            points=[e.to_datapoint() for e in self.examples],
        )

    def add(self, example: AgentExample) -> None:
        """Add an example."""
        self.examples.append(example)

    def filter_by_tags(self, *tags: str) -> list[AgentExample]:
        """Filter examples by tags (any match)."""
        tag_set = set(tags)
        return [e for e in self.examples if tag_set.intersection(e.tags)]


# ---------------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------------


@dataclass
class ExampleResult:
    """Result of evaluating a single example."""

    example_id: str
    prediction: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether all scores are above 0.5."""
        return all(s >= 0.5 for s in self.scores.values())


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""

    agent_name: str = ""
    dataset_name: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    example_results: list[ExampleResult] = field(
        default_factory=list,
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Fraction of examples that passed."""
        if not self.example_results:
            return 0.0
        passed = sum(1 for r in self.example_results if r.passed)
        return passed / len(self.example_results)

    @property
    def num_examples(self) -> int:
        """Number of evaluated examples."""
        return len(self.example_results)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Evaluation Report: {self.agent_name}",
            f"  Dataset: {self.dataset_name}",
            f"  Examples: {self.num_examples}",
            f"  Pass Rate: {self.pass_rate:.1%}",
            "  Scores:",
        ]
        for name, score in sorted(self.scores.items()):
            lines.append(f"    {name}: {score:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@dataclass
class TraceEvent:
    """Trace event for a single graph node execution."""

    node_name: str
    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    finished_at: datetime | None = None
    duration_ms: float = 0.0
    status: Literal["success", "error", "skipped"] = "success"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(
        self,
        *,
        status: Literal["success", "error"] = "success",
        error: str | None = None,
    ) -> None:
        """Mark this event as finished."""
        self.finished_at = datetime.now(UTC)
        self.duration_ms = (self.finished_at - self.started_at).total_seconds() * 1000
        self.status = status
        self.error = error
