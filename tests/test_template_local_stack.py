"""Contract tests for the local-first scaffold.

A scaffolded project has to run and train against a local small-model server
with no cloud credentials. These tests pin the pieces that make that true:
the ``omlx`` provider, the generated ``local`` stack, and the dataset + fit
scripts every class-agent template now ships.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from agentomatic.cli.commands import init as init_cmd
from agentomatic.cli.templates import TEMPLATES, TRAINABLE_TEMPLATES, get_template_files
from agentomatic.optimize.train_api import _model_spec
from agentomatic.providers.llm import DEFAULT_OMLX_BASE_URL
from agentomatic.stacks.defaults import (
    DEFAULT_LOCAL_MODEL,
    default_local_model,
    get_default_local_stack,
    get_default_stack_yaml,
)


class TestOmlxProvider:
    """``omlx`` is a first-class provider, not just an optimizer prefix."""

    def test_builds_against_the_default_local_endpoint(self) -> None:
        from agentomatic.providers.llm import _build_llm

        llm = _build_llm("omlx", model="stub-slm")
        assert str(llm.openai_api_base).rstrip("/") == DEFAULT_OMLX_BASE_URL.rstrip("/")
        # ChatOpenAI refuses to build without a key; local servers ignore it.
        assert llm.openai_api_key.get_secret_value()

    def test_env_supplies_base_url_when_the_stack_leaves_it_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentomatic.providers.llm import _build_llm

        monkeypatch.setenv("OMLX_BASE_URL", "http://127.0.0.1:9999/v1")
        llm = _build_llm("omlx", model="stub-slm")
        assert str(llm.openai_api_base).rstrip("/") == "http://127.0.0.1:9999/v1"

    def test_explicit_base_url_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentomatic.providers.llm import _build_llm

        monkeypatch.setenv("OMLX_BASE_URL", "http://127.0.0.1:9999/v1")
        llm = _build_llm("omlx", model="stub-slm", base_url="http://10.0.0.4:8000/v1")
        assert str(llm.openai_api_base).rstrip("/") == "http://10.0.0.4:8000/v1"

    def test_unknown_provider_error_advertises_omlx(self) -> None:
        from agentomatic.providers.llm import _build_llm

        with pytest.raises(ValueError, match="omlx"):
            _build_llm("nope", model="x")


class TestModelSpecMapping:
    """Stack providers must reach the optimizer as a routable prefix."""

    def test_omlx_round_trips(self) -> None:
        entry = SimpleNamespace(provider="omlx", model="Qwen3.5-9B-MLX-4bit", base_url="")
        assert _model_spec(entry) == "omlx/Qwen3.5-9B-MLX-4bit"

    def test_openai_compatible_maps_to_openai(self) -> None:
        # The base_url reaches LLMCaller separately; only the prefix must be
        # one the caller understands, or the call silently hits Ollama.
        entry = SimpleNamespace(provider="openai_compatible", model="m", base_url="http://x/v1")
        assert _model_spec(entry) == "openai/m"

    def test_custom_provider_with_endpoint_is_treated_as_openai(self) -> None:
        entry = SimpleNamespace(provider="securegpt", model="m", base_url="http://x/v1")
        assert _model_spec(entry) == "openai/m"


class TestDefaultLocalStack:
    def test_targets_a_local_slm_server(self) -> None:
        stack = get_default_local_stack()
        default = stack.llm["default"]
        assert default.provider == "omlx"
        assert default.base_url == DEFAULT_OMLX_BASE_URL
        assert not default.api_key, "a local stack must need no credentials"

    def test_carries_the_profiles_train_py_resolves(self) -> None:
        assert {"default", "fast", "judge", "rewrite"} <= set(get_default_local_stack().llm)

    def test_embeddings_need_no_second_server(self) -> None:
        assert get_default_local_stack().embedding.provider == "hash"

    def test_model_is_overridable_at_scaffold_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTOMATIC_LOCAL_MODEL", "omlx/my-loaded-model")
        assert default_local_model() == "my-loaded-model"
        monkeypatch.delenv("AGENTOMATIC_LOCAL_MODEL")
        monkeypatch.setenv("OMLX_MODEL", "other-model")
        assert default_local_model() == "other-model"

    def test_generated_yaml_explains_itself(self) -> None:
        text = get_default_stack_yaml("local")
        assert text.startswith("#"), "the file a user edits first needs a header"
        assert "--port 8001" in text, "the 8000 collision must be called out"
        data = yaml.safe_load(text)
        assert data["llm"]["default"]["model"] == DEFAULT_LOCAL_MODEL


class TestTrainingBundle:
    """Every class-agent template ships a runnable train/eval loop."""

    @pytest.mark.parametrize("template", sorted(TRAINABLE_TEMPLATES))
    def test_dataset_lands_where_train_py_looks_for_it(self, template: str) -> None:
        files = get_template_files(template, "bot")
        # run_train defaults to <agent_dir>/datasets/all.jsonl — a dataset
        # written anywhere else makes `python train.py` fail on line one.
        assert "datasets/all.jsonl" in files
        assert "train.py" in files
        assert "eval.py" in files
        assert "Makefile" in files

    @pytest.mark.parametrize("template", sorted(TRAINABLE_TEMPLATES))
    def test_seed_dataset_is_loadable_and_split(self, template: str) -> None:
        from agentomatic.agents.types import AgentExample

        rows = [
            json.loads(line)
            for line in get_template_files(template, "bot")["datasets/all.jsonl"].splitlines()
            if line.strip()
        ]
        assert len(rows) >= 6
        splits = {row["split"] for row in rows}
        assert {"train", "validation", "test"} <= splits, (
            "EarlyStopping monitors val_loss, so a validation split must exist"
        )
        for row in rows:
            example = AgentExample.from_dict(row)
            assert example.input.get("current_query")
            assert (example.expected_output or {}).get("response")

    @pytest.mark.parametrize("template", sorted(TEMPLATES))
    def test_non_class_templates_ship_no_orphan_train_script(self, template: str) -> None:
        if template in TRAINABLE_TEMPLATES or template in ("plugin", "pipeline"):
            return
        files = get_template_files(template, "bot")
        assert "datasets/all.jsonl" not in files, (
            f"{template} has no BaseGraphAgent for train.py to fit"
        )

    def test_judge_rubric_matches_the_template(self) -> None:
        assert "grounded" in get_template_files("rag", "bot")["train.py"]
        assert "routed" in get_template_files("coordinator", "bot")["train.py"]

    def test_train_script_targets_the_shipped_dataset_path(self) -> None:
        train = get_template_files("basic", "bot")["train.py"]
        assert 'HERE / "datasets" / "all.jsonl"' in train


class TestInitBootstrapsAWorkspace:
    """`agentomatic init` in a bare directory must leave something runnable."""

    def test_creates_stacks_and_active_marker(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
            result = runner.invoke(init_cmd, ["helper", "--template", "chatbot"])
            assert result.exit_code == 0, result.output
            root = Path(cwd)
            assert (root / "stacks" / "local.yaml").exists()
            assert (root / "stacks" / "remote.yaml").exists()
            assert (root / ".agentomatic-stack").read_text().strip() == "local"
            assert (root / "agents" / "__init__.py").exists()
            assert (root / "agents" / "helper" / "datasets" / "all.jsonl").exists()

    def test_does_not_clobber_an_existing_stack(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
            root = Path(cwd)
            (root / "stacks").mkdir()
            (root / "stacks" / "local.yaml").write_text("name: local\nllm: {}\n")
            (root / ".agentomatic-stack").write_text("custom\n")
            result = runner.invoke(init_cmd, ["helper", "--template", "basic"])
            assert result.exit_code == 0, result.output
            assert (root / "stacks" / "local.yaml").read_text() == "name: local\nllm: {}\n"
            assert (root / ".agentomatic-stack").read_text().strip() == "custom"


class TestPlainTextFitMetric:
    """The fit metric must rank prose, not zero it out."""

    def test_plain_text_answers_are_scored_not_zeroed(self) -> None:
        from agentomatic.optimize.structured_metrics import structured_composite_score

        expected = json.dumps({"response": "Paris is the capital of France."})
        good = structured_composite_score(
            "What is the capital of France?",
            "Paris is the capital of France.",
            expected,
            required_keys=["response"],
        )
        poor = structured_composite_score(
            "What is the capital of France?",
            "I have no idea about that.",
            expected,
            required_keys=["response"],
        )
        assert good > poor > 0.0, "a flat 0.0 leaves the prompt fitter nothing to rank"

    def test_json_still_outranks_equivalent_plain_text(self) -> None:
        from agentomatic.optimize.structured_metrics import structured_composite_score

        answer = "Paris is the capital of France."
        expected = json.dumps({"response": answer})
        as_json = structured_composite_score(
            "q", json.dumps({"response": answer}), expected, required_keys=["response"]
        )
        as_text = structured_composite_score("q", answer, expected, required_keys=["response"])
        assert as_json > as_text

    def test_empty_response_still_scores_zero(self) -> None:
        from agentomatic.optimize.structured_metrics import structured_composite_score

        assert structured_composite_score("q", "   ", "{}", required_keys=["response"]) == 0.0


class TestDoctorLlmEndpointProbe:
    """`agentomatic doctor` must name the two ways a local model fails."""

    def _stacks(self, tmp_path: Path, model: str) -> Path:
        stacks = tmp_path / "stacks"
        stacks.mkdir()
        (stacks / "local.yaml").write_text(
            "name: local\n"
            "llm:\n"
            "  default:\n"
            "    provider: omlx\n"
            f"    model: {model}\n"
            "    base_url: http://127.0.0.1:8/v1\n"
        )
        return stacks

    def test_unreachable_server_is_reported(self, tmp_path: Path) -> None:
        from agentomatic.cli.commands import _check_stack_llm_endpoint

        label, ok, detail = _check_stack_llm_endpoint(self._stacks(tmp_path, "m"), "local")
        assert label == "LLM endpoint"
        assert ok is False
        assert "unreachable" in detail

    def test_model_mismatch_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from agentomatic.cli.commands import _check_stack_llm_endpoint

        served = {"data": [{"id": "stub-slm-1b-instruct"}]}
        monkeypatch.setattr(
            httpx,
            "get",
            lambda *a, **k: httpx.Response(200, json=served),
        )
        _, ok, detail = _check_stack_llm_endpoint(self._stacks(tmp_path, "not-loaded"), "local")
        # Reachable is not enough: an unknown model 404s on the first invoke.
        assert ok is False
        assert "stub-slm-1b-instruct" in detail
        assert "not-loaded" in detail

    def test_matching_model_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from agentomatic.cli.commands import _check_stack_llm_endpoint

        monkeypatch.setattr(
            httpx,
            "get",
            lambda *a, **k: httpx.Response(200, json={"data": [{"id": "loaded"}]}),
        )
        _, ok, detail = _check_stack_llm_endpoint(self._stacks(tmp_path, "loaded"), "local")
        assert ok is True
        assert "reachable" in detail

    def test_non_openai_shaped_body_is_not_treated_as_a_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from agentomatic.cli.commands import _check_stack_llm_endpoint

        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, text="pong"))
        _, ok, _ = _check_stack_llm_endpoint(self._stacks(tmp_path, "anything"), "local")
        assert ok is True, "an unknown response shape says nothing about the model"
