# Testing Your Agents

<div align="center">
  <img src="../assets/logo.png" width="200" alt="agentomatic logo">
  <h3>Confidence through Automated Testing</h3>
</div>

---

## 📖 Overview

A robust test suite is essential for maintaining agent reliability as your
logic, prompts, and graph topology evolve. Agentomatic agents are **plain
Python classes**, so you can test them with standard Python tooling —
no special harness required.

| Layer | What You Test | Speed | Fidelity |
|-------|---------------|-------|----------|
| **Unit** | Individual node methods | ⚡ ms | Low |
| **Pipeline** | `transform()` end-to-end | ⚡ ms | Medium |
| **REST API** | HTTP endpoints via `TestClient` | 🔶 ~100 ms | High |
| **Evaluation** | Quality metrics over a dataset | 🐢 seconds | Highest |

!!! info "Test Framework"
    Agentomatic uses **pytest** with **pytest-asyncio** in `auto` mode.
    All `async def test_*` functions are automatically collected — no
    need for explicit `@pytest.mark.asyncio` in most cases (though we
    add it in this guide for clarity).

---

## 🧪 1. Unit Testing Node Methods

Node methods are plain Python methods that accept and return the agent's
state dataclass. Test them **directly** — no graph execution needed.

### Defining a Test Agent

First, let's define a minimal agent that we'll test throughout this guide:

```python
# agents/my_agent/agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentomatic.agents import BaseGraphAgent


@dataclass
class MyState:
    """Per-run transient state."""

    query: str = ""
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MyAgent(BaseGraphAgent[MyState]):
    """A simple Q&A agent."""

    agent_name = "my_agent"
    agent_description = "Answers questions"

    def __init__(self, *, llm=None) -> None:
        super().__init__()
        self.llm = llm

    def build_graph(self):
        g = self.new_graph()
        g.add_node("process", self.process)
        g.add_node("format_output", self.format_output)
        g.set_entry_point("process")
        g.add_edge("process", "format_output")
        g.set_finish_point("format_output")
        return g.compile()

    def process(self, state: MyState) -> MyState:
        """Core logic node."""
        if self.llm:
            # In production, call the LLM
            state.response = self.llm.invoke(state.query)
        else:
            state.response = f"Echo: {state.query}"
        return state

    def format_output(self, state: MyState) -> MyState:
        """Post-processing node."""
        state.metadata["processed"] = True
        return state

    def input_to_state(self, input_data: dict[str, Any]) -> MyState:
        return MyState(query=input_data.get("query", ""))

    def state_to_output(self, state: MyState) -> dict[str, Any]:
        return {"response": state.response, "metadata": state.metadata}
```

### Testing Individual Nodes

```python
# tests/test_my_agent.py
from __future__ import annotations

import pytest

from agents.my_agent.agent import MyAgent, MyState


class TestProcessNode:
    """Unit tests for the process node."""

    def test_echo_without_llm(self):
        """Without an LLM the node should echo the query."""
        agent = MyAgent()
        state = MyState(query="hello world")
        result = agent.process(state)
        assert result.response == "Echo: hello world"

    def test_empty_query(self):
        """An empty query should still produce a valid response."""
        agent = MyAgent()
        state = MyState(query="")
        result = agent.process(state)
        assert result.response == "Echo: "

    @pytest.mark.parametrize(
        "query",
        ["What is AI?", "Tell me a joke", "Summarise this document"],
    )
    def test_various_queries(self, query: str):
        """Node should handle diverse inputs."""
        agent = MyAgent()
        state = MyState(query=query)
        result = agent.process(state)
        assert query in result.response


class TestFormatOutputNode:
    """Unit tests for the format_output node."""

    def test_sets_processed_flag(self):
        """format_output should mark state as processed."""
        agent = MyAgent()
        state = MyState(query="test", response="done")
        result = agent.format_output(state)
        assert result.metadata["processed"] is True
```

!!! tip "Why test nodes in isolation?"
    Isolated node tests run in **microseconds**, give pinpoint failure
    messages, and let you develop nodes with a fast red-green-refactor
    loop before wiring them into the graph.

---

## 🔗 2. Testing the Full Pipeline

The `transform()` method runs the complete
`input_to_state → graph → state_to_output` pipeline. Use it to verify
that nodes compose correctly.

```python
class TestTransformPipeline:
    """End-to-end pipeline tests."""

    def test_basic_transform(self):
        """transform() should return a complete output dict."""
        agent = MyAgent()
        result = agent.transform({"query": "Hello"})
        assert "response" in result
        assert "metadata" in result
        assert result["metadata"]["processed"] is True

    def test_transform_preserves_query(self):
        """The original query should appear in the response."""
        agent = MyAgent()
        result = agent.transform({"query": "test input"})
        assert "test input" in result["response"]

    def test_invoke_is_alias(self):
        """invoke() is an alias for transform()."""
        agent = MyAgent()
        t_result = agent.transform({"query": "x"})
        i_result = agent.invoke({"query": "x"})
        assert t_result == i_result

    def test_graph_is_cached_across_transforms(self):
        """Multiple transform() calls should reuse the same graph."""
        agent = MyAgent()
        agent.transform({"query": "first"})
        graph_ref = agent._graph
        agent.transform({"query": "second"})
        assert agent._graph is graph_ref
```

### Testing Async Transform

```python
@pytest.mark.asyncio
async def test_async_transform():
    """atransform() should produce the same result as transform()."""
    agent = MyAgent()
    sync_result = agent.transform({"query": "async test"})
    async_result = await agent.atransform({"query": "async test"})
    assert sync_result == async_result
```

---

## 🌐 3. Integration Testing with the REST API

Agentomatic builds a **FastAPI** application via `AgentPlatform.build()`.
Use FastAPI's `TestClient` (synchronous) or `httpx.AsyncClient` (async)
to test the full HTTP layer — routing, serialization, error handling.

=== "Sync with TestClient"

    ```python
    from __future__ import annotations

    import pytest
    from fastapi.testclient import TestClient

    from agentomatic import AgentManifest, AgentPlatform


    @pytest.fixture
    def platform():
        """Create a test platform with an echo agent."""
        p = AgentPlatform(
            agents_dir="/tmp/agentomatic_test",
            title="Test Platform",
            version="0.0.1",
        )

        async def echo_fn(state):
            return {
                "response": f"Echo: {state.get('current_query', '')}",
                "agent_type": "echo",
            }

        p.register_agent(
            manifest=AgentManifest(
                name="echo",
                slug="test-echo",
                description="Echo agent for testing",
            ),
            node_fn=echo_fn,
        )
        return p


    @pytest.fixture
    def client(platform):
        """Provide a synchronous test client."""
        app = platform.build()
        with TestClient(app) as c:
            yield c


    class TestInvokeEndpoint:
        def test_invoke_returns_200(self, client):
            resp = client.post(
                "/api/v1/echo/invoke",
                json={"query": "test"},
            )
            assert resp.status_code == 200
            assert "Echo: test" in resp.json()["response"]

        def test_invoke_nonexistent_agent_404(self, client):
            resp = client.post(
                "/api/v1/nonexistent/invoke",
                json={"query": "x"},
            )
            assert resp.status_code == 404

        def test_health_endpoint(self, client):
            resp = client.get("/api/v1/echo/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"
    ```

=== "Async with httpx"

    ```python
    from __future__ import annotations

    import pytest
    from httpx import ASGITransport, AsyncClient

    from agentomatic import AgentManifest, AgentPlatform


    @pytest.fixture
    def app():
        """Build the ASGI app for async testing."""
        p = AgentPlatform(
            agents_dir="/tmp/agentomatic_test",
            title="Test Platform",
            version="0.0.1",
        )

        async def echo_fn(state):
            return {"response": f"Echo: {state.get('current_query', '')}"}

        p.register_agent(
            manifest=AgentManifest(
                name="echo",
                slug="test-echo",
                description="Echo agent",
            ),
            node_fn=echo_fn,
        )
        return p.build()


    @pytest.mark.asyncio
    async def test_invoke_endpoint_async(app):
        """Test the invoke endpoint using httpx AsyncClient."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/echo/invoke",
                json={"query": "async test"},
            )
        assert resp.status_code == 200
        assert "Echo: async test" in resp.json()["response"]
    ```

!!! note "No Real Server Needed"
    Both `TestClient` and `ASGITransport` run the ASGI app **in-process**.
    No ports are bound, no sockets are opened — tests stay fast and
    hermetic.

---

## 🤖 4. Mocking LLMs

Real LLM calls are slow, expensive, and non-deterministic. Mock them
with `unittest.mock` to keep tests **fast, free, and reproducible**.

### Basic LLM Mocking

```python
from __future__ import annotations

from unittest.mock import MagicMock

from agents.my_agent.agent import MyAgent


def test_with_mocked_llm():
    """Agent should use the LLM's response when one is provided."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "mocked response"

    agent = MyAgent(llm=mock_llm)
    result = agent.transform({"query": "test"})

    assert result["response"] == "mocked response"
    mock_llm.invoke.assert_called_once_with("test")
```

### Mocking Async LLM Calls

```python
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_with_async_mocked_llm():
    """Async LLM calls should be mockable with AsyncMock."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "async mocked response"

    agent = MyAgent(llm=mock_llm)
    # If your agent uses ainvoke in its async path:
    result = await agent.atransform({"query": "test"})
    assert "response" in result
```

### Patching External Services

```python
from unittest.mock import patch, MagicMock


def test_patch_vector_store():
    """Patch an external vector store used by the agent."""
    with patch(
        "agents.my_agent.agent.VectorStore",
    ) as MockStore:
        mock_instance = MockStore.return_value
        mock_instance.search.return_value = [
            {"text": "relevant doc", "score": 0.95},
        ]

        agent = MyAgent(llm=MagicMock())
        # ... test logic that triggers vector search
```

!!! warning "Mock Boundaries"
    Mock at the **boundary** of your code — the LLM client, the vector
    store, the HTTP client — not inside your node methods. This keeps
    tests coupled to the **contract**, not the implementation.

---

## 📊 5. Testing with AgentDataset

Use `AgentDataset` and built-in metrics to write **evaluation-as-tests**
that assert quality thresholds.

### Creating a Test Dataset

=== "Inline Dataset"

    ```python
    from __future__ import annotations

    from agentomatic.agents.types import AgentDataset, AgentExample


    def _make_test_dataset() -> AgentDataset:
        """Create a small inline test dataset."""
        return AgentDataset(
            name="qa_smoke_test",
            examples=[
                AgentExample(
                    id="q1",
                    input={"query": "What is Python?"},
                    expected_output={
                        "response": "Python is a programming language",
                    },
                    split="test",
                ),
                AgentExample(
                    id="q2",
                    input={"query": "What is 2+2?"},
                    expected_output={"response": "4"},
                    split="test",
                ),
            ],
        )
    ```

=== "From JSONL File"

    ```python
    # tests/fixtures/test_cases.jsonl
    # {"id": "q1", "input": {"query": "What is Python?"}, "expected_output": {"response": "..."}, "split": "test"}
    # {"id": "q2", "input": {"query": "What is 2+2?"}, "expected_output": {"response": "4"}, "split": "test"}

    dataset = AgentDataset.from_jsonl("tests/fixtures/test_cases.jsonl")
    ```

### Running Evaluations in Tests

```python
from __future__ import annotations

import pytest

from agentomatic.agents.metrics import (
    CallableMetric,
    ContainsTermsMetric,
    ExactKeyMatchMetric,
)
from agentomatic.agents.types import AgentDataset, AgentExample

from agents.my_agent.agent import MyAgent


class TestAgentQuality:
    """Quality-gate tests using evaluation metrics."""

    def test_output_keys_present(self):
        """All expected output keys should be present."""
        agent = MyAgent()
        dataset = _make_test_dataset()
        metric = ExactKeyMatchMetric(["response"])
        report = agent.evaluate(dataset.test, metrics=[metric])

        assert report.scores["exact_key_match"] >= 1.0

    def test_pass_rate_threshold(self):
        """At least 80% of examples should pass."""
        agent = MyAgent()
        dataset = _make_test_dataset()
        metric = ExactKeyMatchMetric(["response", "metadata"])
        report = agent.evaluate(dataset.test, metrics=[metric])

        assert report.pass_rate >= 0.8, (
            f"Pass rate {report.pass_rate:.1%} below threshold. "
            f"Report:\n{report.summary()}"
        )

    def test_custom_metric(self):
        """Use a custom callable metric for domain-specific checks."""
        def response_not_empty(example, prediction):
            return 1.0 if prediction.get("response", "") else 0.0

        agent = MyAgent()
        dataset = _make_test_dataset()
        metric = CallableMetric(
            "response_not_empty",
            response_not_empty,
        )
        report = agent.evaluate(dataset.test, metrics=[metric])

        assert report.scores["response_not_empty"] == 1.0
```

!!! tip "Available Built-in Metrics"
    | Metric | What It Checks |
    |--------|---------------|
    | `ExactKeyMatchMetric(keys)` | All specified keys exist in prediction |
    | `ContainsTermsMetric(terms)` | Fraction of terms found in output text |
    | `CallableMetric(name, fn)` | Any custom `(example, prediction) → float` |

---

## 📸 6. Snapshot / Golden File Testing

Compare agent output against **saved expected results** to detect
regressions. This is especially useful for prompt-sensitive agents where
even small changes can alter output.

### Writing Snapshot Tests

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.my_agent.agent import MyAgent

GOLDEN_DIR = Path(__file__).parent / "golden"


class TestGoldenFiles:
    """Snapshot tests against saved golden outputs."""

    def test_matches_golden_output(self):
        """Agent output should match the saved golden file."""
        agent = MyAgent()
        result = agent.transform({"query": "What is agentomatic?"})

        golden_path = GOLDEN_DIR / "what_is_agentomatic.json"

        if not golden_path.exists():
            # First run — save the golden file
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(
                json.dumps(result, indent=2, sort_keys=True)
            )
            pytest.skip("Golden file created — review and re-run")

        expected = json.loads(golden_path.read_text())
        assert result == expected, (
            f"Output drifted from golden file.\n"
            f"Expected: {expected}\n"
            f"Got:      {result}"
        )
```

### Updating Golden Files

```bash
# Delete all golden files and re-run to regenerate
rm -rf tests/golden/
uv run pytest tests/test_golden.py -v
# Review the generated files, then commit them
```

!!! note "When to Use Snapshots"
    Snapshot tests work best for **deterministic** agents (no LLM, or
    mocked LLM). For non-deterministic agents, prefer evaluation metrics
    with tolerance thresholds.

---

## 💡 7. Testing Tips & Best Practices

### Use `SimpleNamespace` for `.name` Attributes

!!! warning "MagicMock(name=...) Pitfall"
    `MagicMock(name="my_node")` sets the mock's **internal debug name**,
    not a `.name` attribute. When testing methods that read `.name`,
    use `SimpleNamespace` instead:

    ```python
    from types import SimpleNamespace

    # ❌ Wrong — .name returns the mock's internal repr
    node = MagicMock(name="extract")
    assert node.name == "extract"  # FAILS!

    # ✅ Correct — .name is a real attribute
    node = SimpleNamespace(name="extract")
    assert node.name == "extract"  # PASSES
    ```

### Use Fixtures for Agent Instantiation

```python
import pytest
from agents.my_agent.agent import MyAgent


@pytest.fixture
def agent():
    """Provide a fresh agent instance for each test."""
    return MyAgent()


@pytest.fixture
def agent_with_mock_llm():
    """Provide an agent with a mocked LLM."""
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "mocked"
    return MyAgent(llm=mock_llm)


def test_with_fixture(agent):
    result = agent.transform({"query": "fixture test"})
    assert "response" in result
```

### Parametrize for Multiple Inputs

```python
import pytest


@pytest.mark.parametrize(
    "query, expected_substring",
    [
        ("Hello", "Hello"),
        ("What is AI?", "AI"),
        ("", "Echo: "),
    ],
    ids=["greeting", "question", "empty"],
)
def test_diverse_inputs(query, expected_substring):
    """Agent should handle diverse inputs gracefully."""
    from agents.my_agent.agent import MyAgent

    agent = MyAgent()
    result = agent.transform({"query": query})
    assert expected_substring in result["response"]
```

### Reduce Log Noise in Tests

!!! tip "Silence Verbose Logs"
    Set the `AGENTOMATIC_LOG_LEVEL` environment variable to suppress
    debug output during test runs:

    ```bash
    AGENTOMATIC_LOG_LEVEL=WARNING uv run pytest tests/ -q
    ```

    Or configure it in `pyproject.toml`:

    ```toml
    [tool.pytest.ini_options]
    env = ["AGENTOMATIC_LOG_LEVEL=WARNING"]
    ```

### Test Graph Structure

```python
def test_graph_has_expected_nodes():
    """The compiled graph should contain all expected nodes."""
    from agents.my_agent.agent import MyAgent

    agent = MyAgent()
    graph = agent.graph
    assert "process" in graph.nodes
    assert "format_output" in graph.nodes


def test_graph_invalidation_after_compile():
    """compile() should invalidate the cached graph."""
    from agents.my_agent.agent import MyAgent
    from agentomatic.agents.types import AgentDataset

    agent = MyAgent()
    _ = agent.graph  # force build
    assert agent._graph is not None

    agent.compile(AgentDataset(name="empty"), metrics=[])
    assert agent._graph is None  # cleared
```

### Test Serialization Round-Trip

```python
import tempfile
from pathlib import Path


def test_save_and_load_round_trip():
    """Agent state should survive save/load cycle."""
    from agents.my_agent.agent import MyAgent
    from agentomatic.agents.metrics import ExactKeyMatchMetric
    from agentomatic.agents.types import AgentDataset

    agent = MyAgent()
    agent.compile(AgentDataset(name="test"), metrics=[])

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "agent_state"
        agent.save(save_path)

        new_agent = MyAgent()
        new_agent.load_compiled(save_path)

        assert new_agent.compiled_metadata == agent.compiled_metadata
```

---

## 🚀 8. Running Tests

### Basic Commands

```bash
# Run all tests
uv run pytest tests/ --override-ini='addopts='

# Run a specific test file
uv run pytest tests/test_my_agent.py -v

# Run a specific test class or method
uv run pytest tests/test_my_agent.py::TestProcessNode -v
uv run pytest tests/test_my_agent.py::TestProcessNode::test_echo_without_llm -v

# Run tests matching a keyword
uv run pytest tests/ -k "transform" -v
```

### Coverage

```bash
# Run with coverage report
uv run pytest tests/ --cov=agentomatic --cov-report=html

# Open the HTML report (macOS)
open htmlcov/index.html
```

### Useful Flags

| Flag | Purpose |
|------|---------|
| `-v` | Verbose — show each test name |
| `-q` | Quiet — minimal output |
| `-x` | Stop on first failure |
| `--tb=short` | Shorter tracebacks |
| `-k "pattern"` | Run only tests matching pattern |
| `--lf` | Re-run only last-failed tests |
| `--no-header` | Suppress the pytest header |

### CI Configuration

A minimal CI step for GitHub Actions:

```yaml
- name: Run tests
  run: |
    uv run pytest tests/ \
      --override-ini='addopts=' \
      --cov=agentomatic \
      --cov-report=xml \
      -q
```

---

## 🗂️ Recommended Test Structure

```
tests/
├── conftest.py                # Shared fixtures
├── fixtures/
│   └── test_cases.jsonl       # Evaluation datasets
├── golden/
│   └── what_is_agentomatic.json  # Snapshot golden files
├── test_my_agent.py           # Unit + pipeline tests
├── test_my_agent_api.py       # REST API integration tests
└── test_my_agent_eval.py      # Evaluation / quality-gate tests
```

!!! tip "Fixture Organization"
    Put shared fixtures (agent instances, datasets, mock LLMs) in
    `conftest.py` so they're automatically discovered by pytest across
    all test files.

---

## 📚 9. Related Documentation

| Guide | Description |
|-------|-------------|
| [Class-Based Agents](class-agents.md) | Full reference for `BaseGraphAgent` |
| [Agent Structure](agent-structure.md) | Folder layout and conventions |
| [Configuration](configuration.md) | `agent.yaml` and environment config |
| [Optimization](optimization.md) | `compile()` / `fit()` / `evaluate()` lifecycle |
| [Cookbook](cookbook.md) | Practical recipes and patterns |
| [Studio Debugging](debug-ui.md) | Visual debugging with Agentomatic Studio |

---

<div align="center">
  <em>Tests are the safety net that lets you refactor with confidence.
  Start with unit tests, add pipeline tests, then graduate to
  evaluation-based quality gates.</em>
</div>
