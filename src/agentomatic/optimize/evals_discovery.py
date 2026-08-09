"""Per-agent evaluation auto-discovery.

Scans agent folders for ``evals.py`` files, dynamically imports them,
and extracts test cases, custom metrics, and validation thresholds.
Enables a pattern where each agent ships with its own evaluation
definitions — co-located with the agent code.

Example::

    # agents/my_agent/evals.py
    AGENT_NAME = "my_agent"
    THRESHOLDS = {"AnswerRelevancyMetric": 0.7}

    def get_test_cases():
        return [LLMTestCase(input="Hello", expected_output="A greeting")]

    def get_custom_metrics():
        return [GEval(name="Helpfulness", criteria="...")]

    # Then run:
    from agentomatic.optimize.evals_discovery import discover_agent_evals

    discovered = discover_agent_evals("agents/")
    for name, data in discovered.items():
        print(f"{name}: {len(data['test_cases'])} test cases")
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from loguru import logger

# =====================================================================
# Discovery
# =====================================================================


def discover_agent_evals(
    agents_dir: str | Path = "agents",
) -> dict[str, dict[str, Any]]:
    """Discover all agent evaluation definitions.

    Searches for ``evals.py`` files in each subdirectory of
    *agents_dir*.  Each discovered module is dynamically imported
    and its public symbols (``AGENT_NAME``, ``THRESHOLDS``,
    ``get_test_cases()``, ``get_custom_metrics()``) are extracted.

    Args:
        agents_dir: Path to the agents directory.

    Returns:
        Dict mapping agent folder names to eval metadata::

            {
                "default_agent": {
                    "module": <module>,
                    "agent_name": "default_agent",
                    "description": "General-purpose agent",
                    "thresholds": {"AnswerRelevancyMetric": 0.7},
                    "test_cases": [LLMTestCase(...), ...],
                    "custom_metrics": [GEval(...), ...],
                },
                ...
            }

    Example::

        discovered = discover_agent_evals()
        for agent_name, data in discovered.items():
            cases = get_agent_test_cases(agent_name)
            print(f"{agent_name}: {len(cases)} tests")
    """
    agents_path = Path(agents_dir)
    discovered: dict[str, dict[str, Any]] = {}

    if not agents_path.exists():
        logger.warning(f"Agents directory not found: {agents_path}")
        return discovered

    for agent_folder in sorted(agents_path.iterdir()):
        if not agent_folder.is_dir() or agent_folder.name.startswith("_"):
            continue

        evals_file = agent_folder / "evals.py"
        if not evals_file.exists():
            logger.debug(f"No evals.py in {agent_folder.name}")
            continue

        try:
            module = _import_module(f"agents.{agent_folder.name}.evals", evals_file)
            data: dict[str, Any] = {
                "module": module,
                "agent_name": getattr(module, "AGENT_NAME", agent_folder.name),
                "description": getattr(module, "AGENT_DESCRIPTION", ""),
                "thresholds": getattr(module, "THRESHOLDS", {}),
            }

            # Call test_cases getter if present
            if hasattr(module, "get_test_cases"):
                data["test_cases"] = module.get_test_cases()
            else:
                data["test_cases"] = []

            # Call custom metrics getter if present
            if hasattr(module, "get_custom_metrics"):
                data["custom_metrics"] = module.get_custom_metrics()
            else:
                data["custom_metrics"] = []

            discovered[agent_folder.name] = data
            logger.info(f"Discovered evals for agent: {agent_folder.name}")

        except Exception as exc:
            logger.error(f"Failed to load evals for {agent_folder.name}: {exc}")

    return discovered


# =====================================================================
# Accessors
# =====================================================================


def get_agent_test_cases(agent_name: str, agents_dir: str = "agents") -> list[Any]:
    """Return test cases for a specific agent.

    Args:
        agent_name: Name of the agent folder (e.g. ``"default_agent"``).
        agents_dir: Path to the agents directory.

    Returns:
        List of test case objects (typically DeepEval ``LLMTestCase``).
    """
    discovered = discover_agent_evals(agents_dir)
    data = discovered.get(agent_name, {})
    return data.get("test_cases", [])


def get_agent_metrics(agent_name: str, agents_dir: str = "agents") -> list[Any]:
    """Return custom + default metrics for an agent.

    Merges agent-specific custom metrics with sensible defaults
    derived from agent-type detection.

    Args:
        agent_name: Name of the agent folder.
        agents_dir: Path to the agents directory.

    Returns:
        List of metric instances.
    """
    from agentomatic.optimize.agent_detect import detect_agent_type, get_metrics_for_agent_type

    discovered = discover_agent_evals(agents_dir)
    data = discovered.get(agent_name, {})

    custom = data.get("custom_metrics", [])

    # Add default metrics based on agent type (if we can load the agent)
    try:
        module = data.get("module")
        if module and hasattr(module, "agent"):
            agent = module.agent
            agent_type = detect_agent_type(agent)
            default_names = get_metrics_for_agent_type(agent_type)
        else:
            default_names = ["answer_relevancy", "geval", "faithfulness"]
    except Exception:
        default_names = ["answer_relevancy", "geval", "faithfulness"]

    # Combine
    metrics = list(custom)
    # Add default DeepEval metrics for each name not already covered
    try:
        from deepeval.metrics import AnswerRelevancyMetric, GEval

        try:
            from deepeval.test_case import LLMTestCaseParams

            geval_params: Any = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ]
        except Exception:
            geval_params = ["input", "actual_output"]

        default_map: dict[str, Any] = {
            "answer_relevancy": lambda: AnswerRelevancyMetric(threshold=0.7),
            "geval": lambda: GEval(
                name="Helpfulness",
                criteria="Is the response helpful and accurate?",
                evaluation_params=geval_params,
                threshold=0.7,
            ),
        }
        for name in default_names:
            if name in default_map and not any(getattr(m, "name", "") == name for m in metrics):
                metrics.append(default_map[name]())
    except ImportError:
        pass

    return metrics


def get_agent_thresholds(agent_name: str, agents_dir: str = "agents") -> dict[str, float]:
    """Return validation thresholds for a specific agent.

    Args:
        agent_name: Name of the agent folder.
        agents_dir: Path to the agents directory.

    Returns:
        Dict mapping metric names to minimum pass thresholds.
    """
    discovered = discover_agent_evals(agents_dir)
    data = discovered.get(agent_name, {})
    return data.get("thresholds", {})


def list_agents_with_evals(agents_dir: str = "agents") -> list[str]:
    """Return the names of all agents that have ``evals.py`` files.

    Args:
        agents_dir: Path to the agents directory.

    Returns:
        Sorted list of agent folder names.
    """
    discovered = discover_agent_evals(agents_dir)
    return sorted(discovered.keys())


# =====================================================================
# Pytest integration
# =====================================================================


def generate_pytest_params(agents_dir: str = "agents") -> list[tuple[str, Any, str]]:
    """Generate parametrized test items for all discovered agent evals.

    Returns a list of ``(agent_name, test_case, test_id)`` tuples
    suitable for use with ``@pytest.mark.parametrize``.

    Example::

        import pytest
        from agentomatic.optimize.evals_discovery import generate_pytest_params

        _all_tests = generate_pytest_params()

        @pytest.mark.parametrize("agent_name,test_case,test_id", _all_tests)
        def test_agent_case(agent_name, test_case, test_id):
            metrics = get_agent_metrics(agent_name)
            assert_test(test_case, metrics)

    Args:
        agents_dir: Path to the agents directory.

    Returns:
        List of ``(agent_name, test_case, test_id)`` tuples.
    """
    discovered = discover_agent_evals(agents_dir)
    tests: list[tuple[str, Any, str]] = []

    for agent_name, data in discovered.items():
        for i, case in enumerate(data.get("test_cases", [])):
            test_id = f"{agent_name}_case_{i}"
            tests.append((agent_name, case, test_id))

    return tests


# =====================================================================
# Helpers
# =====================================================================


def _import_module(full_name: str, filepath: Path) -> Any:
    """Dynamically import a Python module from a file path.

    Args:
        full_name: Fully-qualified module name (e.g. ``"agents.foo.evals"``).
        filepath: Path to the ``.py`` file.

    Returns:
        The imported module object.
    """
    spec = importlib.util.spec_from_file_location(full_name, str(filepath))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
