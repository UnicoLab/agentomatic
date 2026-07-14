"""BaseGraphAgent — the main public class for class-owned agents.

Combines graph execution, evaluation, optimization, serialization,
and observability into a single ML-like API.

Example::

    from dataclasses import dataclass, field
    from agentomatic.agents import BaseGraphAgent, GraphBuilder

    @dataclass
    class MyState:
        query: str = ""
        response: str = ""

    class MyAgent(BaseGraphAgent[MyState]):
        agent_name = "my_agent"
        agent_description = "A simple agent"

        def __init__(self, *, llm=None):
            super().__init__()
            self.llm = llm

        # --- Graph wiring (LangGraph-style) ---
        def build_graph(self):
            g = self.new_graph()
            g.add_node("process", self.process)
            g.add_node("format", self.format_output)
            g.set_entry_point("process")
            g.add_edge("process", "format")
            g.set_finish_point("format")
            return g.compile()

        # --- Node methods (plain methods, no decorators) ---
        def process(self, state: MyState) -> MyState:
            state.response = "processed"
            return state

        def format_output(self, state: MyState) -> MyState:
            return state

        def input_to_state(self, data):
            return MyState(query=data.get("query", ""))

        def state_to_output(self, state):
            return {"response": state.response}

    agent = MyAgent(llm="gpt-4")
    result = agent.transform({"query": "hello"})
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic

from loguru import logger

from .builder import GraphBuilder
from .decorators import get_node_meta
from .graph import AgentGraph
from .history import Callback, History, Loss, resolve_loss
from .types import (
    AgentDataset,
    AgentExample,
    EvaluationReport,
    ExampleResult,
    Metric,
    Optimizer,
    StateT,
    TraceEvent,
)


class BaseGraphAgent(ABC, Generic[StateT]):
    """Abstract base class for class-owned graph agents.

    Combines graph execution, evaluation, optimization, serialization,
    and observability into a single ML-like API.

    Users must implement:

    - ``build_graph()`` — wire nodes into a graph (LangGraph-style)
    - ``input_to_state()`` — convert raw input to state
    - ``state_to_output()`` — convert final state to output

    The framework provides:

    - ``new_graph()`` — get a ``GraphBuilder`` for graph construction
    - ``transform()`` — end-to-end inference
    - ``compile()`` / ``fit()`` — ML-like optimization
    - ``evaluate()`` — evaluate on dataset
    - ``save()`` / ``load()`` — serialize/deserialize
    - ``graph`` — lazy cached graph property
    """

    # --- Class-level metadata (override in subclass) ---
    agent_name: str = ""
    agent_description: str = ""
    agent_version: str = "1.0.0"
    agent_framework: str = "graph_agent"

    def __init__(self) -> None:
        """Initialize the agent with empty framework state.

        Subclasses should call ``super().__init__()`` and then
        set up their resources (llm, embedder, vector_store, etc.).
        """
        # Graph execution
        self._graph: AgentGraph[StateT] | None = None

        # Optimization
        self.compiled_config: dict[str, Any] = {}
        self.compiled_metadata: dict[str, Any] = {}
        self._optimizer: Optimizer | None = None
        self._compile_metrics: Sequence[Metric] = []
        self._compile_dataset: AgentDataset | None = None
        self._loss: Loss | None = None

        # Training lifecycle (Keras-style)
        self.history: History | None = None
        self.stop_training: bool = False

        # Evaluation
        self.evaluation_history: list[EvaluationReport] = []

        # Observability
        self._traces: list[list[TraceEvent]] = []

    # ==================================================================
    # Abstract methods (user implements)
    # ==================================================================

    @abstractmethod
    def input_to_state(
        self,
        input_data: dict[str, Any],
    ) -> StateT:
        """Convert raw input data to agent state.

        Args:
            input_data: Raw input dictionary.

        Returns:
            Initial state for graph execution.
        """
        ...

    @abstractmethod
    def state_to_output(
        self,
        state: StateT,
    ) -> dict[str, Any]:
        """Convert final state to output dictionary.

        Args:
            state: Final state after graph execution.

        Returns:
            Output dictionary.
        """
        ...

    # ==================================================================
    # Graph Execution
    # ==================================================================

    def build_graph(self) -> AgentGraph[StateT]:
        """Build the agent's execution graph.

        **This is the primary method to override.** Use ``new_graph()``
        to get a ``GraphBuilder`` with LangGraph-compatible API::

            def build_graph(self):
                g = self.new_graph()
                g.add_node("extract", self.extract)
                g.add_node("generate", self.generate)
                g.set_entry_point("extract")
                g.add_edge("extract", "generate")
                g.set_finish_point("generate")
                return g.compile()

        Returns:
            An ``AgentGraph`` instance.

        Raises:
            NotImplementedError: If not overridden and no
                ``@agent_node`` decorators found.
        """
        # Fallback: try auto-building from @agent_node decorators
        try:
            return self._build_graph_from_decorated_nodes()
        except ValueError:
            raise NotImplementedError(
                f"{type(self).__name__} must implement build_graph(). "
                f"Use self.new_graph() to create a GraphBuilder:\n"
                f"\n"
                f"    def build_graph(self):\n"
                f"        g = self.new_graph()\n"
                f"        g.add_node('step', self.step)\n"
                f"        g.set_entry_point('step')\n"
                f"        g.set_finish_point('step')\n"
                f"        return g.compile()\n"
            ) from None

    def new_graph(self) -> GraphBuilder[StateT]:
        """Create a new ``GraphBuilder`` for constructing the graph.

        Returns a builder with LangGraph-compatible API:
        ``add_node()``, ``add_edge()``, ``set_entry_point()``,
        ``set_finish_point()``, ``add_conditional_edge()``,
        ``compile()``.

        Returns:
            A fresh ``GraphBuilder`` instance.

        Example::

            def build_graph(self):
                g = self.new_graph()
                g.add_node("extract", self.extract)
                g.add_node("generate", self.generate)
                g.set_entry_point("extract")
                g.add_edge("extract", "generate")
                g.set_finish_point("generate")
                return g.compile()
        """
        return GraphBuilder()

    @property
    def graph(self) -> AgentGraph[StateT]:
        """Lazy-built, cached execution graph.

        The graph is built on first access and cached.
        Call ``invalidate_graph()`` to force rebuild.
        """
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    def invalidate_graph(self) -> None:
        """Clear the cached graph.

        The graph will be rebuilt on next access.
        Called automatically after ``compile()`` or ``fit()``.
        """
        self._graph = None

    def invoke_graph(self, state: StateT) -> StateT:
        """Execute the graph synchronously.

        Args:
            state: Initial state.

        Returns:
            Final state.
        """
        return self.graph.invoke(state)

    async def ainvoke_graph(self, state: StateT) -> StateT:
        """Execute the graph asynchronously.

        Args:
            state: Initial state.

        Returns:
            Final state.
        """
        return await self.graph.ainvoke(state)

    def _build_graph_from_decorated_nodes(
        self,
    ) -> AgentGraph[StateT]:
        """Auto-build graph from @agent_node decorated methods.

        This is a convenience fallback for simple linear graphs.
        For complex graphs, override ``build_graph()`` instead.

        Returns:
            A validated ``AgentGraph``.

        Raises:
            ValueError: If no decorated nodes found.
        """
        builder: GraphBuilder[StateT] = GraphBuilder()
        decorated: list[tuple[str, Any, Any]] = []

        # Introspect class hierarchy for decorated methods
        seen: set[str] = set()
        for cls in type(self).__mro__:
            for attr_name, attr_value in vars(cls).items():
                if attr_name.startswith("_"):
                    continue
                if attr_name in seen:
                    continue
                seen.add(attr_name)

                if isinstance(attr_value, (property, classmethod, staticmethod)):
                    continue
                if not callable(attr_value):
                    continue

                meta = get_node_meta(attr_value)
                if meta is None:
                    continue

                bound_method = getattr(self, attr_name)
                node_name = meta.name or attr_name
                decorated.append((node_name, bound_method, meta))

        if not decorated:
            raise ValueError(f"No @agent_node decorated methods found on {type(self).__name__}.")

        for node_name, method, meta in decorated:
            builder.node(
                node_name,
                method,
                description=meta.description,
                metadata=meta.metadata,
            )

        for node_name, _, meta in decorated:
            if meta.entrypoint:
                builder.entrypoint(node_name)
            if meta.finish:
                builder.finish(node_name)

        for node_name, _, meta in decorated:
            if meta.after:
                builder.edge(meta.after, node_name)

        return builder.build()

    # ==================================================================
    # Transform (inference)
    # ==================================================================

    def transform(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run end-to-end inference.

        Converts input → state → graph execution → output.

        Args:
            input_data: Raw input dictionary.

        Returns:
            Output dictionary.
        """
        t0 = time.perf_counter()

        state = self.input_to_state(input_data)
        final_state = self.invoke_graph(state)
        output = self.state_to_output(final_state)

        # Record trace
        if hasattr(self, "_graph") and self._graph is not None:
            trace = self._graph.last_trace
            self._traces.append(trace)

        duration = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Transform completed in {duration:.1f}ms "
            f"({len(self._graph.nodes) if self._graph else '?'} nodes)"
        )

        return output

    async def atransform(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run end-to-end inference asynchronously.

        Args:
            input_data: Raw input dictionary.

        Returns:
            Output dictionary.
        """
        t0 = time.perf_counter()

        state = self.input_to_state(input_data)
        final_state = await self.ainvoke_graph(state)
        output = self.state_to_output(final_state)

        if hasattr(self, "_graph") and self._graph is not None:
            trace = self._graph.last_trace
            self._traces.append(trace)

        duration = (time.perf_counter() - t0) * 1000
        logger.debug(f"Async transform completed in {duration:.1f}ms")

        return output

    def invoke(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Alias for ``transform()``.

        Args:
            input_data: Raw input dictionary.

        Returns:
            Output dictionary.
        """
        return self.transform(input_data)

    # ==================================================================
    # Evaluation
    # ==================================================================

    def evaluate(
        self,
        dataset: list[AgentExample] | AgentDataset,
        metrics: Sequence[Metric],
    ) -> EvaluationReport:
        """Evaluate the agent on a dataset.

        Runs ``transform()`` for each example, scores predictions
        with all metrics, and aggregates results.

        Args:
            dataset: Examples to evaluate on.
            metrics: Metrics to compute.

        Returns:
            An ``EvaluationReport`` with scores and per-example
            results.
        """
        examples = dataset.examples if isinstance(dataset, AgentDataset) else dataset
        dataset_name = dataset.name if isinstance(dataset, AgentDataset) else "inline"

        example_results: list[ExampleResult] = []
        metric_totals: dict[str, float] = {m.name: 0.0 for m in metrics}

        for example in examples:
            t0 = time.perf_counter()
            try:
                prediction = self.transform(example.input)
                duration = (time.perf_counter() - t0) * 1000

                scores: dict[str, float] = {}
                for metric in metrics:
                    try:
                        score = metric.score(example, prediction)
                        scores[metric.name] = score
                        metric_totals[metric.name] += score
                    except Exception as exc:
                        logger.warning(f"Metric '{metric.name}' error: {exc}")
                        scores[metric.name] = 0.0

                example_results.append(
                    ExampleResult(
                        example_id=example.id,
                        prediction=prediction,
                        scores=scores,
                        duration_ms=duration,
                    )
                )
            except Exception as exc:
                duration = (time.perf_counter() - t0) * 1000
                logger.warning(f"Transform error for '{example.id}': {exc}")
                example_results.append(
                    ExampleResult(
                        example_id=example.id,
                        error=str(exc),
                        duration_ms=duration,
                    )
                )

        # Aggregate scores
        n = len(examples) if examples else 1
        avg_scores = {k: v / n for k, v in metric_totals.items()}

        report = EvaluationReport(
            agent_name=self.agent_name or type(self).__name__,
            dataset_name=dataset_name,
            scores=avg_scores,
            example_results=example_results,
        )

        self.evaluation_history.append(report)
        return report

    # ==================================================================
    # Optimization
    # ==================================================================

    def compile(
        self,
        dataset: AgentDataset | None = None,
        metrics: Sequence[Metric] | None = None,
        optimizer: Optimizer | None = None,
        loss: Loss | Metric | Any | None = None,
    ) -> BaseGraphAgent[StateT]:
        """Prepare the agent for optimization (Keras-style ``compile``).

        Stores the optimization strategy (dataset, metrics, optimizer, and an
        optional ``loss`` objective) and invalidates the graph so it is rebuilt
        with any new config.

        Args:
            dataset: Training dataset (optional; can be supplied to ``fit``).
            metrics: Metrics tracked during ``fit``/``evaluate``.
            optimizer: Optional optimizer instance.
            loss: Optional objective to minimise. Accepts a :class:`Loss`, a
                metric-like object (turned into ``1 - score``), or a
                ``(example, prediction) -> float`` callable.

        Returns:
            Self for chaining.
        """
        metrics = list(metrics or [])
        self._optimizer = optimizer
        self._compile_metrics = metrics
        self._compile_dataset = dataset
        self._loss = resolve_loss(loss)
        if dataset is not None:
            self.compiled_metadata["dataset_name"] = dataset.name
            self.compiled_metadata["dataset_size"] = len(dataset)
        self.compiled_metadata["metrics"] = [m.name for m in metrics]
        self.compiled_metadata["optimizer"] = type(optimizer).__name__ if optimizer else "none"
        self.compiled_metadata["loss"] = self._loss.name if self._loss else "none"

        self.invalidate_graph()
        size = len(dataset) if dataset is not None else 0
        logger.info(f"Compiled {self.agent_name}: {size} examples, {len(metrics)} metrics")
        return self

    def fit(
        self,
        dataset: AgentDataset | list[AgentExample] | None = None,
        *,
        epochs: int = 1,
        verbose: int = 1,
        callbacks: Sequence[Callback] | None = None,
        validation_data: AgentDataset | list[AgentExample] | None = None,
    ) -> History:
        """Train the agent and return a Keras-style :class:`History`.

        Each epoch (optionally) runs the compiled optimizer, applies any config
        changes, then evaluates the agent on the training data (and
        ``validation_data`` if given), recording per-epoch metric and ``loss``
        values. Callbacks receive ``on_epoch_end(epoch, logs)`` and may set
        ``agent.stop_training`` to halt early (see :class:`EarlyStopping`).

        Args:
            dataset: Training data. Defaults to the dataset passed to
                ``compile``. An ``AgentDataset`` uses its ``train`` split when
                present, otherwise all examples.
            epochs: Number of optimization/evaluation rounds.
            verbose: ``0`` = silent, ``1`` = per-epoch log line.
            callbacks: Optional list of :class:`Callback` instances.
            validation_data: Optional held-out data scored as ``val_*`` keys.

        Returns:
            A :class:`History` (also stored on ``self.history``).
        """
        dataset = dataset if dataset is not None else self._compile_dataset
        train_examples = self._resolve_examples(dataset, split="train")
        val_examples = self._resolve_examples(validation_data, split="validation")
        if not val_examples and isinstance(dataset, AgentDataset):
            val_examples = dataset.validation

        metrics = list(self._compile_metrics)
        loss = self._loss
        history = History(
            params={
                "epochs": epochs,
                "optimizer": type(self._optimizer).__name__ if self._optimizer else "none",
                "metrics": [m.name for m in metrics],
                "loss": loss.name if loss else "none",
                "train_size": len(train_examples),
                "val_size": len(val_examples),
            }
        )
        self.history = history
        self.stop_training = False

        callbacks = list(callbacks or [])
        for cb in callbacks:
            cb.set_agent(self)
            cb.set_params(history.params)
            cb.on_train_begin()

        logger.info(
            f"Fitting {self.agent_name}: {epochs} epoch(s), {len(train_examples)} train example(s)"
        )

        for epoch in range(epochs):
            for cb in callbacks:
                cb.on_epoch_begin(epoch)

            # Optimization step (if an optimizer was compiled in).
            if self._optimizer is not None and dataset is not None:
                opt_dataset = dataset if isinstance(dataset, AgentDataset) else None
                if opt_dataset is None:
                    opt_dataset = AgentDataset(name="inline", examples=list(train_examples))
                config = self._optimizer.optimize(self, opt_dataset, metrics)
                if config:
                    self.compiled_config.update(config)
                    for key, value in config.items():
                        if hasattr(self, key) and not key.startswith("_"):
                            setattr(self, key, value)
                    self.invalidate_graph()

            logs = self._epoch_logs(train_examples, metrics, loss)
            if val_examples:
                val_logs = self._epoch_logs(val_examples, metrics, loss)
                logs.update({f"val_{k}": v for k, v in val_logs.items()})

            history.record(epoch, logs)

            if verbose:
                metric_str = " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
                logger.info(f"Epoch {epoch + 1}/{epochs} - {metric_str}")

            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)

            if self.stop_training:
                break

        for cb in callbacks:
            cb.on_train_end()

        return history

    def _epoch_logs(
        self,
        examples: Sequence[AgentExample],
        metrics: Sequence[Metric],
        loss: Loss | None,
    ) -> dict[str, float]:
        """Run one evaluation pass, returning averaged metric + loss logs.

        A single ``transform()`` per example feeds both metrics and the loss.
        Prediction failures score ``0.0`` for metrics and the maximum loss
        (``1.0``) so a broken agent surfaces as a poor epoch rather than an
        exception.
        """
        if not examples:
            return {}

        totals = {m.name: 0.0 for m in metrics}
        loss_total = 0.0
        n = len(examples)

        for example in examples:
            try:
                prediction = self.transform(example.input)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"fit: transform failed for '{example.id}': {exc}")
                if loss is not None:
                    loss_total += 1.0
                continue

            for metric in metrics:
                try:
                    totals[metric.name] += float(metric.score(example, prediction))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"fit: metric '{metric.name}' failed: {exc}")
            if loss is not None:
                try:
                    loss_total += loss.compute(example, prediction)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"fit: loss '{loss.name}' failed: {exc}")
                    loss_total += 1.0

        logs = {name: total / n for name, total in totals.items()}
        if loss is not None:
            logs["loss"] = loss_total / n
        return logs

    @staticmethod
    def _resolve_examples(
        data: AgentDataset | list[AgentExample] | None,
        split: str,
    ) -> list[AgentExample]:
        """Normalise a dataset / example list into a list of examples.

        For an ``AgentDataset`` the requested ``split`` is preferred, falling
        back to all examples when that split is empty.
        """
        if data is None:
            return []
        if isinstance(data, AgentDataset):
            if split == "train":
                return data.train or list(data.examples)
            if split == "validation":
                return data.validation
            return list(data.examples)
        return list(data)

    # ==================================================================
    # Observability
    # ==================================================================

    def get_last_trace(self) -> list[TraceEvent]:
        """Return the trace from the most recent execution."""
        if self._traces:
            return self._traces[-1]
        return []

    def get_trace_history(self) -> list[list[TraceEvent]]:
        """Return all recorded traces."""
        return list(self._traces)

    # ==================================================================
    # Serialization
    # ==================================================================

    def save(self, path: str | Path) -> None:
        """Save the agent's compiled state.

        Creates a directory with:
        - ``config.json`` — compiled configuration
        - ``metadata.json`` — compilation metadata
        - ``evaluation_history.json`` — past evaluation reports

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.compiled_config, f, indent=2)

        # Save metadata
        meta = {
            "agent_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            **self.compiled_metadata,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save evaluation history
        history = []
        for report in self.evaluation_history:
            history.append(
                {
                    "agent_name": report.agent_name,
                    "dataset_name": report.dataset_name,
                    "scores": report.scores,
                    "num_examples": report.num_examples,
                    "pass_rate": report.pass_rate,
                }
            )
        with open(path / "evaluation_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved agent state to {path}")

    def load_compiled(self, path: str | Path) -> None:
        """Load compiled config from a saved directory.

        Args:
            path: Directory containing saved state.
        """
        path = Path(path)

        config_file = path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                self.compiled_config = json.load(f)

        meta_file = path / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                self.compiled_metadata = json.load(f)

        # Apply loaded config
        for key, value in self.compiled_config.items():
            if hasattr(self, key) and not key.startswith("_"):
                setattr(self, key, value)

        self.invalidate_graph()
        logger.info(f"Loaded compiled config from {path}")

    # ==================================================================
    # Dataset
    # ==================================================================

    def load_dataset(
        self,
        path: str | Path,
        *,
        format: str = "jsonl",
    ) -> AgentDataset:
        """Load a dataset from file.

        Args:
            path: Path to dataset file.
            format: File format (currently only "jsonl").

        Returns:
            Loaded ``AgentDataset``.
        """
        if format == "jsonl":
            return AgentDataset.from_jsonl(path)
        raise ValueError(f"Unsupported format: {format}")

    # ==================================================================
    # Registry integration
    # ==================================================================

    def to_manifest(self) -> Any:
        """Generate an ``AgentManifest`` from class metadata.

        Returns:
            An ``AgentManifest`` instance.
        """
        from agentomatic.core.manifest import AgentManifest

        name = self.agent_name or type(self).__name__.lower()
        return AgentManifest(
            name=name,
            slug=f"class-agent-{name}",
            description=self.agent_description,
            version=self.agent_version,
            framework=self.agent_framework,
        )

    def as_registered_agent(self) -> Any:
        """Convert to a ``RegisteredAgent`` for the registry.

        Creates a ``RegisteredAgent`` with ``node_fn`` and
        ``graph_fn`` that delegates to this class instance.

        Returns:
            A ``RegisteredAgent`` instance.
        """
        from agentomatic.core.manifest import RegisteredAgent

        async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
            """Node function adapter for registry."""
            input_data = {
                "query": state.get("current_query", ""),
                **{k: v for k, v in state.items() if k not in ("messages", "thread_id")},
            }
            return await self.atransform(input_data)

        def graph_fn() -> AgentGraph[StateT]:
            """Graph function adapter for registry."""
            return self.graph

        return RegisteredAgent(
            manifest=self.to_manifest(),
            node_fn=node_fn,
            graph_fn=graph_fn,
        )

    # ==================================================================
    # Visualization
    # ==================================================================

    def visualize(self) -> str:
        """Generate a Mermaid diagram of the agent's graph.

        Returns:
            Mermaid-syntax string.
        """
        return self.graph.visualize()

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        name = self.agent_name or type(self).__name__
        # Don't trigger lazy graph build in repr
        node_count = len(self._graph.nodes) if self._graph else "?"
        return f"<{type(self).__name__}(name={name!r}, nodes={node_count})>"
