"""Execution-ordering for pipeline steps.

Pipelines normally execute as an ordered list.  When any step declares
``upstreams`` (explicit dependencies), the engine schedules execution as a
DAG: a step only runs after all of its upstreams have completed, and steps
with equal readiness run in list order (stable, deterministic).

This module is dependency-free (only imports the models) so both the
engine and the validators share the exact same ordering semantics.
"""

from __future__ import annotations

import heapq
from typing import Any


def upstreams_of(step: Any) -> list[str]:
    """Return the declared upstream step names for a step (or ``[]``)."""
    value = getattr(step, "upstreams", None)
    if not value:
        return []
    return [u for u in value if isinstance(u, str) and u]


def has_upstreams(steps: list[Any]) -> bool:
    """Whether any step in the pipeline declares upstream dependencies."""
    return any(upstreams_of(step) for step in steps)


def compute_execution_order(steps: list[Any]) -> list[int]:
    """Compute the execution order for a list of steps.

    Returns a permutation of ``range(len(steps))``:

    - No ``upstreams`` anywhere → plain list order (``[0, 1, 2, …]``),
      preserving the classic sequential behavior.
    - With ``upstreams`` → topological order (Kahn's algorithm) where a
      step appears after all of its declared upstreams.  Ready steps are
      picked in list-index order so the result is deterministic and
      degrades gracefully to list order for chains.

    Raises:
        ValueError: If an upstream references a missing step, a step
            references itself, or the dependencies contain a cycle.
    """
    names = [getattr(step, "name", f"#{i}") for i, step in enumerate(steps)]
    index_by_name: dict[str, int] = {}
    for i, name in enumerate(names):
        if name and name not in index_by_name:
            index_by_name[name] = i

    graph: list[list[int]] = [[] for _ in steps]
    indegree = [0] * len(steps)
    edge_count = 0

    for i, step in enumerate(steps):
        for upstream in upstreams_of(step):
            if upstream == names[i]:
                raise ValueError(f"Step '{names[i]}' cannot depend on itself")
            if upstream not in index_by_name:
                raise ValueError(f"Step '{names[i]}' references unknown upstream '{upstream}'")
            upstream_index = index_by_name[upstream]
            graph[upstream_index].append(i)
            indegree[i] += 1
            edge_count += 1

    if edge_count == 0:
        return list(range(len(steps)))

    # Kahn's algorithm — always run the lowest-index ready step next.
    ready: list[int] = [i for i in range(len(steps)) if indegree[i] == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for dependent in sorted(graph[i]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)

    if len(order) != len(steps):
        remaining = sorted(set(range(len(steps))) - set(order))
        cycle_names = [names[i] for i in remaining]
        raise ValueError(
            "Cycle detected in upstream dependencies involving steps: " + ", ".join(cycle_names)
        )
    return order
