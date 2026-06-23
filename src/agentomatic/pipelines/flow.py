"""Decorator-based Flow for agentomatic agent pipelines.

Provides a reactive, DAG-driven execution model inspired by CrewAI's
Flow pattern, deeply integrated with agentomatic's ``AgentRegistry``.

Decorators:
    ``@start()``   — marks a method as a flow entry point.
    ``@listen(trigger)`` — triggers after *trigger* completes.
    ``@router(trigger)`` — triggers after *trigger*; return value selects
                           which ``@listen("<route_name>")`` to fire next.

Example::

    class ResearchFlow(Flow):
        @start()
        async def plan(self, input_data):
            return await self.agent("query_planner").run(input_data)

        @listen(plan)
        async def research(self, plan_output):
            return await self.parallel(
                [self.agent("web_researcher"), self.agent("knowledge_base")],
                input={"current_query": plan_output["response"]},
            )

        @router(research)
        def route(self, results):
            if len(results) > 2:
                return "deep_path"
            return "quick_path"

        @listen("deep_path")
        async def deep(self, results):
            return await self.agent("synthesizer").run(results)

        @listen("quick_path")
        async def quick(self, results):
            return await self.agent("summarizer").run(results)

    flow = ResearchFlow()
    result = await flow.run({"query": "..."})
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentomatic.core.registry import AgentRegistry


# ---------------------------------------------------------------------------
# FlowResult
# ---------------------------------------------------------------------------


class FlowResult(BaseModel):
    """Result of a complete flow execution.

    Attributes:
        output: Final output from the last executed step(s).
        steps: Results from every executed step, keyed by method name.
        duration_ms: Wall-clock execution time in milliseconds.
        status: Terminal status (``"success"`` or ``"failed"``).
    """

    output: dict[str, Any] = Field(default_factory=dict)
    steps: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    status: str = "success"


# ---------------------------------------------------------------------------
# AgentHandle
# ---------------------------------------------------------------------------


class AgentHandle:
    """Lightweight handle to a registered agent.

    Returned by ``Flow.agent(name)``; callers invoke the agent via
    :meth:`run`.  Resolution against the ``AgentRegistry`` is deferred
    until invocation time.

    Args:
        name: Agent name as registered in the ``AgentRegistry``.
        registry: Bound registry (may be ``None`` if not yet bound).
    """

    def __init__(self, name: str, registry: AgentRegistry | None) -> None:
        self.name = name
        self._registry = registry

    def __repr__(self) -> str:  # noqa: D105
        return f"AgentHandle({self.name!r})"

    async def run(self, input_data: dict[str, Any] | Any) -> dict[str, Any]:
        """Invoke the agent with given input.

        Resolves the agent from the registry, constructs a
        ``BaseAgentState``-compatible state dict, and invokes via
        ``graph_fn`` (compiled graph) or ``node_fn`` (simple function).

        Args:
            input_data: Input payload.  When a ``dict``, its keys are
                merged into the state; otherwise the value is placed
                under the ``current_query`` key.

        Returns:
            Agent output as a dictionary.

        Raises:
            ValueError: If the registry is not bound or the agent is
                not found.
            RuntimeError: If the agent has neither ``graph_fn`` nor
                ``node_fn``.
        """
        if self._registry is None:
            raise ValueError(
                f"AgentHandle({self.name!r}): no AgentRegistry bound. "
                "Call flow.bind_registry(registry) before running."
            )

        agent = self._registry.get(self.name)
        if agent is None:
            raise ValueError(
                f"Agent {self.name!r} not found in registry. "
                f"Available: {self._registry.list_names()}"
            )

        state = self._build_state(input_data)

        # Prefer compiled graph, fall back to node function
        if agent.graph_fn is not None:
            graph = agent.graph_fn()
            result = await graph.ainvoke(state)
            return self._normalise_output(result)

        if agent.node_fn is not None:
            result = await agent.node_fn(state)
            return self._normalise_output(result)

        raise RuntimeError(
            f"Agent {self.name!r} has neither graph_fn nor node_fn — cannot invoke."
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _build_state(input_data: Any) -> dict[str, Any]:
        """Convert heterogeneous input into a ``BaseAgentState`` dict."""
        if isinstance(input_data, dict):
            state: dict[str, Any] = dict(input_data)
            # Ensure current_query is populated
            if "current_query" not in state:
                # Heuristic: use the first string value found
                for v in state.values():
                    if isinstance(v, str):
                        state["current_query"] = v
                        break
            return state

        # Scalar / non-dict → wrap as current_query
        return {"current_query": str(input_data) if input_data is not None else ""}

    @staticmethod
    def _normalise_output(result: Any) -> dict[str, Any]:
        """Ensure the invocation result is a plain dict."""
        if isinstance(result, dict):
            return dict(result)
        return {"result": result}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

_ATTR_START = "_flow_start"
_ATTR_LISTEN = "_flow_listen"
_ATTR_ROUTER = "_flow_router"


def start() -> Callable[..., Any]:
    """Mark a method as a flow entry point.

    Entry-point methods receive the raw ``input_data`` passed to
    ``Flow.run()`` and are executed first.

    Returns:
        A decorator that tags the wrapped method.

    Example::

        @start()
        async def plan(self, input_data):
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, _ATTR_START, True)
        return fn

    return decorator


def listen(trigger: Callable[..., Any] | str) -> Callable[..., Any]:
    """Trigger a method after *trigger* completes.

    Args:
        trigger: A reference to the upstream method (function object)
            or a string route name emitted by a ``@router``.

    Returns:
        A decorator that tags the wrapped method.

    Example::

        @listen(plan)
        async def research(self, plan_output):
            ...

        @listen("deep_path")
        async def deep(self, results):
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, _ATTR_LISTEN, trigger)
        return fn

    return decorator


def router(trigger: Callable[..., Any] | str) -> Callable[..., Any]:
    """Trigger a routing method after *trigger* completes.

    The decorated method must return a string that matches a
    ``@listen("<route_name>")`` declaration in the same flow.

    Args:
        trigger: A reference to the upstream method or its string name.

    Returns:
        A decorator that tags the wrapped method.

    Example::

        @router(research)
        def route(self, results):
            return "deep_path" if len(results) > 2 else "quick_path"
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, _ATTR_ROUTER, trigger)
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Flow base class
# ---------------------------------------------------------------------------


class Flow:
    """Declarative, decorator-driven agent pipeline.

    Subclasses decorate methods with :func:`start`, :func:`listen`, and
    :func:`router` to define a reactive execution graph.  At runtime,
    ``await flow.run(input_data)`` introspects the class, builds a
    dependency graph, and walks it to completion.

    Attributes:
        _registry: Bound ``AgentRegistry`` (set via :meth:`bind_registry`).
        _results: Step outputs keyed by method name.
    """

    def __init__(self) -> None:
        self._registry: AgentRegistry | None = None
        self._results: dict[str, Any] = {}

    # -- public API -------------------------------------------------------

    def bind_registry(self, registry: AgentRegistry) -> None:
        """Bind an ``AgentRegistry`` for agent resolution.

        Args:
            registry: The registry instance to use.
        """
        self._registry = registry

    def agent(self, name: str) -> AgentHandle:
        """Get a handle to a registered agent.

        Args:
            name: Agent name as known to the registry.

        Returns:
            An :class:`AgentHandle` that can be ``await``-ed via ``.run()``.
        """
        return AgentHandle(name=name, registry=self._registry)

    async def parallel(
        self,
        agents: list[AgentHandle],
        *,
        input: dict[str, Any],  # noqa: A002
        strategy: str = "all",
    ) -> list[dict[str, Any]]:
        """Execute multiple agents concurrently.

        Args:
            agents: Agent handles to invoke.
            input: Input payload broadcast to every agent.
            strategy: Aggregation strategy — ``"all"`` (wait for every
                result), ``"first"`` (return the first to finish).

        Returns:
            List of output dicts (one per agent, in submission order for
            ``"all"``; single-element list for ``"first"``).
        """
        if strategy == "first":
            return await self._parallel_first(agents, input)
        return await self._parallel_all(agents, input)

    async def run(self, input_data: dict[str, Any]) -> FlowResult:
        """Execute the flow end-to-end.

        Algorithm:
            1. Introspect the class for ``@start``, ``@listen``, and
               ``@router`` methods.
            2. Build a dependency mapping (trigger → [methods]).
            3. Execute ``@start`` methods first.
            4. Propagate results through ``@listen`` / ``@router`` edges
               until the frontier is empty.
            5. Return a :class:`FlowResult`.

        Args:
            input_data: Initial input fed to ``@start`` methods.

        Returns:
            A :class:`FlowResult` with outputs, step results, timing,
            and status.
        """
        t0 = time.perf_counter()
        self._results = {}
        status = "success"

        try:
            starts, listeners, routers = self._introspect()
            await self._execute(starts, listeners, routers, input_data)
        except Exception as exc:
            logger.error(f"Flow execution failed: {exc}")
            status = "failed"
            self._results["__error__"] = str(exc)

        duration = (time.perf_counter() - t0) * 1000

        # Determine final output — last written result
        final_output = self._compute_final_output()

        return FlowResult(
            output=final_output,
            steps=dict(self._results),
            duration_ms=round(duration, 2),
            status=status,
        )

    # -- introspection ----------------------------------------------------

    def _introspect(
        self,
    ) -> tuple[
        list[Callable[..., Any]],
        dict[str, list[Callable[..., Any]]],
        dict[str, list[Callable[..., Any]]],
    ]:
        """Discover decorated methods on this instance.

        Returns:
            Tuple of ``(start_methods, listeners, routers)`` where
            *listeners* and *routers* are dicts mapping trigger names
            to method lists.
        """
        starts: list[Callable[..., Any]] = []
        # trigger_name → [bound methods]
        listeners: dict[str, list[Callable[..., Any]]] = defaultdict(list)
        routers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            method = getattr(self, attr_name, None)
            if method is None or not callable(method):
                continue

            # Retrieve the underlying function to read decorator attrs
            fn = _unwrap_method(method)

            if getattr(fn, _ATTR_START, False):
                starts.append(method)

            listen_trigger = getattr(fn, _ATTR_LISTEN, None)
            if listen_trigger is not None:
                trigger_name = _resolve_trigger_name(listen_trigger)
                listeners[trigger_name].append(method)

            router_trigger = getattr(fn, _ATTR_ROUTER, None)
            if router_trigger is not None:
                trigger_name = _resolve_trigger_name(router_trigger)
                routers[trigger_name].append(method)

        if not starts:
            raise RuntimeError("No @start() methods found. A Flow must have at least one.")

        logger.debug(
            f"Flow introspection: {len(starts)} start(s), "
            f"{sum(len(v) for v in listeners.values())} listener(s), "
            f"{sum(len(v) for v in routers.values())} router(s)"
        )
        return starts, dict(listeners), dict(routers)

    # -- execution engine -------------------------------------------------

    async def _execute(
        self,
        starts: list[Callable[..., Any]],
        listeners: dict[str, list[Callable[..., Any]]],
        routers: dict[str, list[Callable[..., Any]]],
        input_data: dict[str, Any],
    ) -> None:
        """Walk the dependency graph to completion.

        Args:
            starts: Entry-point methods.
            listeners: trigger_name → methods map.
            routers: trigger_name → router methods map.
            input_data: Initial input.
        """
        # Phase 1: execute all @start methods
        completed_names: list[str] = []
        for method in starts:
            name = method.__name__
            logger.info(f"▶ start: {name}")
            result = await self._invoke(method, input_data)
            self._results[name] = result
            completed_names.append(name)

        # Phase 2: propagate until nothing fires
        while completed_names:
            next_round: list[str] = []
            for trigger_name in completed_names:
                trigger_result = self._results.get(trigger_name)

                # Fire routers first — they produce route names
                for rmethod in routers.get(trigger_name, []):
                    rname = rmethod.__name__
                    logger.info(f"  ↳ router: {rname} (trigger={trigger_name})")
                    route_value = await self._invoke(rmethod, trigger_result)
                    self._results[rname] = route_value

                    # route_value should be a string naming the next path
                    if isinstance(route_value, str):
                        # Fire listeners matching that route name
                        for lmethod in listeners.get(route_value, []):
                            lname = lmethod.__name__
                            logger.info(f"    ↳ listen(route={route_value!r}): {lname}")
                            lr = await self._invoke(lmethod, trigger_result)
                            self._results[lname] = lr
                            next_round.append(lname)
                    else:
                        logger.warning(
                            f"Router {rname} returned non-string "
                            f"{type(route_value).__name__}; skipping."
                        )

                # Fire direct listeners
                for lmethod in listeners.get(trigger_name, []):
                    lname = lmethod.__name__
                    logger.info(f"  ↳ listen: {lname} (trigger={trigger_name})")
                    lr = await self._invoke(lmethod, trigger_result)
                    self._results[lname] = lr
                    next_round.append(lname)

            completed_names = next_round

    # -- method invocation ------------------------------------------------

    @staticmethod
    async def _invoke(method: Callable[..., Any], payload: Any) -> Any:
        """Invoke a method, handling both sync and async signatures.

        Args:
            method: Bound method to call.
            payload: Single positional argument passed to the method.

        Returns:
            The method's return value.
        """
        if asyncio.iscoroutinefunction(method):
            return await method(payload)
        # Sync method — call directly (already bound, no I/O expected)
        return method(payload)

    # -- parallel helpers -------------------------------------------------

    @staticmethod
    async def _parallel_all(
        agents: list[AgentHandle],
        input_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run all agents concurrently and collect every result."""
        coros = [a.run(input_data) for a in agents]
        results: list[dict[str, Any]] = await asyncio.gather(*coros)
        return results

    @staticmethod
    async def _parallel_first(
        agents: list[AgentHandle],
        input_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run agents concurrently, return the first to finish."""
        tasks = [asyncio.create_task(a.run(input_data)) for a in agents]
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Cancel stragglers
        for t in pending:
            t.cancel()
        first = next(iter(done)).result()
        return [first]

    # -- output helpers ---------------------------------------------------

    def _compute_final_output(self) -> dict[str, Any]:
        """Derive the final flow output from collected step results.

        Strategy: return the result from the last step that was stored,
        excluding internal keys.
        """
        last: Any = {}
        for key, val in self._results.items():
            if key.startswith("__"):
                continue
            last = val
        if isinstance(last, dict):
            return last
        return {"result": last}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _unwrap_method(method: Any) -> Any:
    """Return the underlying function of a bound method."""
    return getattr(method, "__func__", method)


def _resolve_trigger_name(trigger: Callable[..., Any] | str) -> str:
    """Convert a trigger (function ref or string) to a stable name.

    Args:
        trigger: Either a function/method reference or a plain string.

    Returns:
        The trigger's canonical name.
    """
    if isinstance(trigger, str):
        return trigger
    # Function object — use its __name__
    name = getattr(trigger, "__name__", None)
    if name is not None:
        return name
    return str(trigger)
