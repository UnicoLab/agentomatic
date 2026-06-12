"""Dataset container for prompt optimization."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class DataPoint:
    """Single evaluation data point.

    Attributes:
        query: The input question/prompt.
        expected_answer: The expected/ideal response (ground truth).
        context: Optional context documents (for RAG evaluation).
        metadata: Arbitrary metadata for filtering/grouping.
    """

    query: str
    expected_answer: str | None = None
    context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d: dict[str, Any] = {"query": self.query}
        if self.expected_answer is not None:
            d["expected_answer"] = self.expected_answer
        if self.context:
            d["context"] = self.context
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Dataset:
    """Collection of data points for optimization.

    Supports loading from JSONL, CSV, or Python lists.

    Example::

        dataset = Dataset.from_jsonl("qa_pairs.jsonl")
        dataset = Dataset.from_list([
            {"query": "What is X?", "expected_answer": "X is ..."},
        ])
    """

    points: list[DataPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[DataPoint]:
        return iter(self.points)

    def __getitem__(self, idx: int) -> DataPoint:
        return self.points[idx]

    def add(self, point: DataPoint) -> None:
        """Add a data point."""
        self.points.append(point)

    def split(self, ratio: float = 0.8) -> tuple[Dataset, Dataset]:
        """Split into train/test sets."""
        split_idx = int(len(self.points) * ratio)
        return (
            Dataset(points=self.points[:split_idx]),
            Dataset(points=self.points[split_idx:]),
        )

    def to_jsonl(self, path: str) -> None:
        """Save to JSONL file."""
        with open(path, "w") as f:
            for point in self.points:
                f.write(json.dumps(point.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> Dataset:
        """Load from JSONL file.

        Each line must be a JSON object with at least a ``query`` field.
        Optional fields: ``expected_answer``, ``context``, ``metadata``.
        """
        points: list[DataPoint] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                points.append(DataPoint(
                    query=data["query"],
                    expected_answer=data.get("expected_answer"),
                    context=data.get("context", []),
                    metadata=data.get("metadata", {}),
                ))
        return cls(points=points)

    @classmethod
    def from_csv(cls, path: str, query_col: str = "query",
                 answer_col: str = "expected_answer") -> Dataset:
        """Load from CSV file."""
        points: list[DataPoint] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append(DataPoint(
                    query=row[query_col],
                    expected_answer=row.get(answer_col),
                    context=[v for k, v in row.items()
                             if k.startswith("context") and v],
                    metadata={k: v for k, v in row.items()
                              if k not in (query_col, answer_col)
                              and not k.startswith("context")},
                ))
        return cls(points=points)

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> Dataset:
        """Create from a list of dictionaries."""
        points = [
            DataPoint(
                query=item["query"],
                expected_answer=item.get("expected_answer"),
                context=item.get("context", []),
                metadata=item.get("metadata", {}),
            )
            for item in items
        ]
        return cls(points=points)
