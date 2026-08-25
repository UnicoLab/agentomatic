# pyright: reportMissingParameterType=none
"""A pipeline that fails to load must say why, and name itself.

A pipeline that does not load is simply absent at runtime: its routes 404 and
the workflow it drives silently stops existing. The only signal an operator
gets is the skip warning at boot, so that line has to be enough to act on.

Regression: a step missing its ``agent`` key surfaced the bare repr of a
``KeyError`` — ``failed to load: 'agent'`` — which named neither the step nor
what was missing.
"""

from __future__ import annotations

import pytest

from agentomatic.pipelines.loader import PipelineLoader

#: A parallel block whose sub-step is a plugin — only agent steps nest there.
PLUGIN_INSIDE_PARALLEL = """
name: bad_parallel
steps:
  - name: fanout
    parallel:
      steps:
        - name: scoring
          plugin: scorer
"""

#: A top-level step declaring nothing the loader recognises.
STEP_WITHOUT_A_TARGET = """
name: bad_step
steps:
  - name: mystery
    somethingelse: value
"""


def _write(tmp_path, text: str):
    """Write a pipeline YAML and return its path."""
    path = tmp_path / "pipeline.yaml"
    path.write_text(text)
    return path


class TestErrorMessagesAreActionable:
    def test_a_plugin_nested_under_parallel_names_the_step(self, tmp_path) -> None:
        path = _write(tmp_path, PLUGIN_INSIDE_PARALLEL)

        with pytest.raises(Exception) as excinfo:
            PipelineLoader.from_yaml(path)

        message = str(excinfo.value)
        assert "scoring" in message, f"the failing step is not named: {message}"
        assert "agent" in message
        # And it must explain the actual constraint, not just the missing key.
        assert "parallel" in message.lower()

    def test_the_message_lists_what_the_step_did_declare(self, tmp_path) -> None:
        """Seeing the keys present is what makes a typo obvious."""
        path = _write(tmp_path, PLUGIN_INSIDE_PARALLEL)

        with pytest.raises(Exception) as excinfo:
            PipelineLoader.from_yaml(path)

        assert "plugin" in str(excinfo.value)

    def test_a_parallel_block_without_a_name_says_so(self, tmp_path) -> None:
        """The outer block is checked first, and already reports clearly."""
        path = _write(
            tmp_path,
            "name: bad\nsteps:\n  - parallel:\n      steps:\n        - agent: alpha\n",
        )

        with pytest.raises(ValueError, match="require an explicit 'name'"):
            PipelineLoader.from_yaml(path)

    def test_a_top_level_step_with_no_recognised_target_is_rejected(self, tmp_path) -> None:
        """A step naming nothing the loader understands must not load."""
        path = _write(tmp_path, STEP_WITHOUT_A_TARGET)

        with pytest.raises(Exception):
            PipelineLoader.from_yaml(path)


class TestValidPipelinesStillLoad:
    def test_a_parallel_block_of_agent_steps_loads(self, tmp_path) -> None:
        path = _write(
            tmp_path,
            """
name: good_parallel
steps:
  - name: fanout
    parallel:
      strategy: all
      steps:
        - agent: alpha
        - agent: beta
""",
        )

        config = PipelineLoader.from_yaml(path)

        assert config.name == "good_parallel"
        assert len(config.steps) == 1
        assert len(config.steps[0].steps) == 2

    def test_a_plugin_step_at_the_top_level_loads(self, tmp_path) -> None:
        """The constraint is about nesting, not about plugin steps as such."""
        path = _write(
            tmp_path,
            "name: with_plugin\nsteps:\n  - agent: alpha\n  - name: score\n    plugin: scorer\n",
        )

        config = PipelineLoader.from_yaml(path)

        assert len(config.steps) == 2


class TestDiscoverySkipsLoudly:
    def test_a_broken_pipeline_is_skipped_with_a_named_warning(self, tmp_path, caplog) -> None:
        """Discovery must not drop a pipeline without saying so."""
        pipelines = tmp_path / "pipelines"
        pipelines.mkdir()
        (pipelines / "broken.yaml").write_text(PLUGIN_INSIDE_PARALLEL)
        (pipelines / "fine.yaml").write_text("name: fine\nsteps:\n  - agent: alpha\n")

        found = PipelineLoader.discover_pipelines(tmp_path)

        # The good one still loads; the broken one is absent, not fatal.
        assert "fine" in found
        assert "bad_parallel" not in found
