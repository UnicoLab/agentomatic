"""Dataset container for prompt optimization."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

#: ``metadata`` keys the optimization stack adds to a point itself (agent
#: inputs, split label, per-run knobs). They are plumbing, not example context.
INTERNAL_METADATA_KEYS = frozenset(
    {
        "invoke",
        "split",
        "model_params",
        "optimize_nodes",
        "node_match",
        "resource_id",
        "parent_id",
        "id",
        # Folded into the expected reference by ``AgentExample.to_datapoint``.
        "judge_expected",
        "expected_reference",
        "quality_reference",
    }
)

#: ``metadata.invoke`` keys that are run knobs or the question itself.
INTERNAL_INVOKE_KEYS = frozenset(
    {
        "query",
        "current_query",
        "question",
        "request",
        "model_params",
        "temperature",
        "optimize_nodes",
        "node_match",
        "system_prompt_override",
        "user_id",
    }
)


#: Keys of a document dict that hold its text, in order of preference.
DOCUMENT_TEXT_KEYS = ("content", "text", "page_content", "snippet", "document")


def document_text(item: Any) -> str:
    """One document as text: ``"<text> (source: <name>)"`` for a document dict.

    Args:
        item: A string, or a dict with ``content`` / ``text`` /
            ``page_content`` … and an optional ``source`` / ``title`` / ``id``.

    Returns:
        The text; a dict without a text field is serialised as JSON.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        text = next((str(item[k]) for k in DOCUMENT_TEXT_KEYS if item.get(k)), "")
        source = item.get("source") or item.get("title") or item.get("id")
        if text:
            return f"{text} (source: {source})" if source else text
        return json.dumps(item, ensure_ascii=False, default=str) if item else ""
    return str(item) if item else ""


def normalize_context(raw: Any) -> list[str]:
    """Return context documents as a list of strings.

    Args:
        raw: A list of documents, one document string, or a context dict.
            A dict's ``documents`` become the documents; any other keys are
            kept as one extra JSON document. Document dicts are rendered by
            :func:`document_text`.

    Returns:
        The documents; empty when there are none.
    """
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, dict):
        if not raw:
            return []
        docs = raw.get("documents")
        if not isinstance(docs, (str, list, tuple)):
            return [json.dumps(raw, ensure_ascii=False, default=str)]
        rest = {k: v for k, v in raw.items() if k != "documents" and v not in (None, "", [], {})}
        extra = [json.dumps(rest, ensure_ascii=False, default=str)] if rest else []
        return normalize_context(docs) + extra
    if isinstance(raw, (list, tuple)):
        return [text for text in (document_text(item) for item in raw if item) if text]
    return []


def example_context(
    metadata: dict[str, Any] | None,
    context: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Everything an example carries beyond its question and expected answer.

    This is what the optimizer's rewrite model, the augmenter and the reports
    see next to each question, so a prompt is improved against the
    documents, extra inputs, labels and tags the agent actually ran with —
    not against bare question/answer pairs.

    Args:
        metadata: The point's metadata (``metadata.invoke`` holds the agent
            inputs other than the question).
        context: Context documents of the point (``DataPoint.context``).
        tags: The example's tags.

    Returns:
        ``{"inputs": ..., "context": ..., "metadata": ..., "tags": ...}`` with
        only the non-empty keys. ``context`` is omitted when the inputs
        already carry it (the agent received it there).
    """
    meta = dict(metadata or {})
    invoke = meta.get("invoke")
    view: dict[str, Any] = {}
    if isinstance(invoke, dict):
        inputs = {
            k: v
            for k, v in invoke.items()
            if k not in INTERNAL_INVOKE_KEYS and v not in (None, "", [], {})
        }
        if inputs:
            view["inputs"] = inputs
    docs = normalize_context(context)
    if docs and "context" not in view.get("inputs", {}):
        view["context"] = docs
    labels = {
        k: v
        for k, v in meta.items()
        if k not in INTERNAL_METADATA_KEYS and v not in (None, "", [], {})
    }
    if labels:
        view["metadata"] = labels
    if tags:
        view["tags"] = [str(t) for t in tags]
    return view


@dataclass
class DataPoint:
    """Single evaluation data point.

    Attributes:
        query: The input question/prompt.
        expected_answer: The expected/ideal response (ground truth).
        context: Optional context documents (for RAG evaluation). A single
            string or a structured dict is normalised to a list.
        metadata: Arbitrary metadata for filtering/grouping. ``metadata.invoke``
            holds extra agent inputs sent with the question.
        tags: Labels for filtering/grouping, shown to the optimizer.
    """

    query: str
    expected_answer: str | None = None
    context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.context, list) or any(not isinstance(c, str) for c in self.context):
            self.context = normalize_context(self.context)
        if isinstance(self.tags, str):
            self.tags = [self.tags]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d: dict[str, Any] = {"query": self.query}
        if self.expected_answer is not None:
            d["expected_answer"] = self.expected_answer
        if self.context:
            d["context"] = self.context
        if self.metadata:
            d["metadata"] = self.metadata
        if self.tags:
            d["tags"] = list(self.tags)
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
        Optional fields: ``expected_answer``, ``context``, ``metadata``, ``tags``.
        """
        points: list[DataPoint] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                points.append(
                    DataPoint(
                        query=data["query"],
                        expected_answer=data.get("expected_answer"),
                        context=data.get("context", []),
                        metadata=data.get("metadata", {}),
                        tags=list(data.get("tags") or []),
                    )
                )
        return cls(points=points)

    @classmethod
    def from_csv(
        cls, path: str, query_col: str = "query", answer_col: str = "expected_answer"
    ) -> Dataset:
        """Load from CSV file."""
        points: list[DataPoint] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append(
                    DataPoint(
                        query=row[query_col],
                        expected_answer=row.get(answer_col),
                        context=[v for k, v in row.items() if k.startswith("context") and v],
                        metadata={
                            k: v
                            for k, v in row.items()
                            if k not in (query_col, answer_col) and not k.startswith("context")
                        },
                    )
                )
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
                tags=list(item.get("tags") or []),
            )
            for item in items
        ]
        return cls(points=points)
