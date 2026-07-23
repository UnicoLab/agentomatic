"""Tests for AgentExample and AgentDataset data structures."""

from __future__ import annotations

import json

from agentomatic.agents.types import AgentDataset, AgentExample

# ===========================================================================
# AgentExample tests
# ===========================================================================


class TestAgentExampleSerialization:
    """Test AgentExample.to_dict() and from_dict()."""

    def test_to_dict_minimal(self):
        """to_dict() with minimal fields should include id and input."""
        ex = AgentExample(id="ex1", input={"query": "hello"})
        d = ex.to_dict()
        assert d["id"] == "ex1"
        assert d["input"] == {"query": "hello"}

    def test_to_dict_full(self):
        """to_dict() should include all populated fields."""
        ex = AgentExample(
            id="ex1",
            input={"query": "hello"},
            expected_output={"response": "world"},
            metadata={"domain": "test"},
            rubric={"quality": "high"},
            tags=["smoke", "unit"],
            split="test",
        )
        d = ex.to_dict()
        assert d["expected_output"] == {"response": "world"}
        assert d["metadata"] == {"domain": "test"}
        assert d["rubric"] == {"quality": "high"}
        assert d["tags"] == ["smoke", "unit"]
        assert d["split"] == "test"

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(x)) should produce an equivalent example."""
        original = AgentExample(
            id="ex1",
            input={"query": "test"},
            expected_output={"answer": "42"},
            metadata={"source": "manual"},
            tags=["regression"],
            split="validation",
        )
        restored = AgentExample.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.input == original.input
        assert restored.expected_output == original.expected_output
        assert restored.metadata == original.metadata
        assert restored.tags == original.tags
        assert restored.split == original.split

    def test_from_dict_defaults(self):
        """from_dict() should use sensible defaults for missing fields."""
        ex = AgentExample.from_dict({"id": "x", "input": {"q": "a"}})
        assert ex.expected_output is None
        assert ex.metadata == {}
        assert ex.tags == []
        assert ex.split == "train"

    def test_to_dict_omits_none_expected_output(self):
        """to_dict() should not include expected_output when None."""
        ex = AgentExample(id="ex1", input={"q": "a"})
        d = ex.to_dict()
        assert "expected_output" not in d

    def test_to_dict_omits_empty_metadata(self):
        """to_dict() should not include empty metadata."""
        ex = AgentExample(id="ex1", input={"q": "a"})
        d = ex.to_dict()
        assert "metadata" not in d

    def test_to_dict_omits_empty_tags(self):
        """to_dict() should not include empty tags list."""
        ex = AgentExample(id="ex1", input={"q": "a"})
        d = ex.to_dict()
        assert "tags" not in d


class TestAgentExampleDatapointBridge:
    """Test AgentExample.to_datapoint() bridge."""

    def test_to_datapoint_query_key(self):
        """to_datapoint() should extract 'query' from input."""
        ex = AgentExample(
            id="ex1",
            input={"query": "What is AI?"},
            expected_output={"response": "Artificial Intelligence"},
        )
        dp = ex.to_datapoint()
        assert dp.query == "What is AI?"
        # Rich judge reference includes answer text plus structured JSON block.
        assert dp.expected_answer is not None
        assert "Artificial Intelligence" in dp.expected_answer
        assert "## Expected answer" in dp.expected_answer

    def test_to_datapoint_request_fallback(self):
        """to_datapoint() should fallback to 'request' key."""
        ex = AgentExample(
            id="ex1",
            input={"request": "Explain ML"},
        )
        dp = ex.to_datapoint()
        assert dp.query == "Explain ML"


# ===========================================================================
# AgentDataset tests
# ===========================================================================


class TestAgentDatasetIO:
    """Test AgentDataset I/O methods."""

    def test_from_jsonl(self, tmp_path):
        """from_jsonl() should load examples from a JSONL file."""
        data = [
            {"id": "e1", "input": {"q": "a"}, "split": "train"},
            {"id": "e2", "input": {"q": "b"}, "split": "test"},
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        ds = AgentDataset.from_jsonl(path)
        assert len(ds) == 2
        assert ds[0].id == "e1"
        assert ds[1].id == "e2"

    def test_to_jsonl(self, tmp_path):
        """to_jsonl() should write examples as JSONL."""
        ds = AgentDataset(
            name="test",
            examples=[
                AgentExample(id="e1", input={"q": "a"}),
                AgentExample(id="e2", input={"q": "b"}),
            ],
        )
        path = tmp_path / "output.jsonl"
        ds.to_jsonl(path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        loaded = [json.loads(line) for line in lines]
        assert loaded[0]["id"] == "e1"
        assert loaded[1]["id"] == "e2"

    def test_jsonl_roundtrip(self, tmp_path):
        """from_jsonl(to_jsonl(ds)) should produce equivalent dataset."""
        original = AgentDataset(
            name="roundtrip",
            examples=[
                AgentExample(
                    id="e1",
                    input={"q": "hello"},
                    expected_output={"a": "world"},
                    tags=["unit"],
                    split="train",
                ),
            ],
        )
        path = tmp_path / "roundtrip.jsonl"
        original.to_jsonl(path)
        loaded = AgentDataset.from_jsonl(path, name="roundtrip")
        assert len(loaded) == 1
        assert loaded[0].id == "e1"
        assert loaded[0].input == {"q": "hello"}
        assert loaded[0].expected_output == {"a": "world"}

    def test_from_list(self):
        """from_list() should create dataset from plain dicts."""
        items = [
            {"input": {"q": "a"}},
            {"input": {"q": "b"}, "id": "custom_id"},
        ]
        ds = AgentDataset.from_list(items, name="list_ds")
        assert len(ds) == 2
        assert ds[0].id == "example_0000"  # auto-generated
        assert ds[1].id == "custom_id"
        assert ds.name == "list_ds"

    def test_from_jsonl_auto_ids(self, tmp_path):
        """from_jsonl() should auto-generate IDs when missing."""
        data = [{"input": {"q": "a"}}, {"input": {"q": "b"}}]
        path = tmp_path / "noid.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        ds = AgentDataset.from_jsonl(path)
        assert ds[0].id == "example_0000"
        assert ds[1].id == "example_0001"


class TestAgentDatasetSplits:
    """Test AgentDataset split properties."""

    def _make_dataset(self) -> AgentDataset:
        return AgentDataset(
            name="split_test",
            examples=[
                AgentExample(id="t1", input={}, split="train"),
                AgentExample(id="t2", input={}, split="train"),
                AgentExample(id="v1", input={}, split="validation"),
                AgentExample(id="v2", input={}, split="val"),
                AgentExample(id="e1", input={}, split="test"),
            ],
        )

    def test_train_split(self):
        """train property should return only training examples."""
        ds = self._make_dataset()
        assert len(ds.train) == 2
        assert all(e.split == "train" for e in ds.train)

    def test_validation_split(self):
        """validation property should return val/validation examples."""
        ds = self._make_dataset()
        assert len(ds.validation) == 2

    def test_test_split(self):
        """test property should return only test examples."""
        ds = self._make_dataset()
        assert len(ds.test) == 1
        assert ds.test[0].id == "e1"


class TestAgentDatasetOperations:
    """Test AgentDataset filtering, iteration, and indexing."""

    def test_filter_by_tags(self):
        """filter_by_tags() should return matching examples."""
        ds = AgentDataset(
            name="tags_test",
            examples=[
                AgentExample(id="e1", input={}, tags=["smoke", "regression"]),
                AgentExample(id="e2", input={}, tags=["integration"]),
                AgentExample(id="e3", input={}, tags=["smoke"]),
                AgentExample(id="e4", input={}, tags=[]),
            ],
        )
        filtered = ds.filter_by_tags("smoke")
        assert len(filtered) == 2
        assert {e.id for e in filtered} == {"e1", "e3"}

    def test_filter_by_tags_multiple(self):
        """filter_by_tags() should match any of the provided tags."""
        ds = AgentDataset(
            name="tags_test",
            examples=[
                AgentExample(id="e1", input={}, tags=["a"]),
                AgentExample(id="e2", input={}, tags=["b"]),
                AgentExample(id="e3", input={}, tags=["c"]),
            ],
        )
        filtered = ds.filter_by_tags("a", "c")
        assert len(filtered) == 2

    def test_iteration(self):
        """AgentDataset should be iterable."""
        ds = AgentDataset(
            name="iter",
            examples=[AgentExample(id=f"e{i}", input={}) for i in range(3)],
        )
        ids = [e.id for e in ds]
        assert ids == ["e0", "e1", "e2"]

    def test_indexing(self):
        """AgentDataset should support integer indexing."""
        ds = AgentDataset(
            name="idx",
            examples=[
                AgentExample(id="first", input={}),
                AgentExample(id="second", input={}),
            ],
        )
        assert ds[0].id == "first"
        assert ds[1].id == "second"

    def test_len(self):
        """len() should return the number of examples."""
        ds = AgentDataset(name="len", examples=[])
        assert len(ds) == 0
        ds.add(AgentExample(id="e1", input={}))
        assert len(ds) == 1

    def test_empty_dataset_handling(self):
        """Empty dataset should work correctly."""
        ds = AgentDataset(name="empty")
        assert len(ds) == 0
        assert ds.train == []
        assert ds.validation == []
        assert ds.test == []
        assert list(ds) == []
        assert ds.filter_by_tags("anything") == []
