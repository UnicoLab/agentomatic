"""Tests for agentomatic CLI and template engine."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agentomatic.cli.templates import TEMPLATES, get_template_files

# =========================================================================
# Template Engine Tests
# =========================================================================


class TestTemplateRegistry:
    def test_all_templates_exist(self):
        assert "basic" in TEMPLATES
        assert "full" in TEMPLATES
        assert "rag" in TEMPLATES
        assert "chatbot" in TEMPLATES
        assert "custom" in TEMPLATES

    def test_template_descriptions(self):
        for name, desc in TEMPLATES.items():
            assert isinstance(desc, str)
            assert len(desc) > 5


class TestBasicTemplate:
    def test_basic_files(self):
        files = get_template_files("basic", "test_agent")
        assert "__init__.py" in files
        assert "agent.py" in files
        assert "graph.py" not in files
        assert "nodes.py" not in files
        assert "prompts.json" in files
        assert "langgraph.json" in files
        assert ".env.example" in files
        assert "README.md" in files

    def test_basic_init_content(self):
        files = get_template_files("basic", "my_agent")
        init = files["__init__.py"]
        assert "AgentManifest" in init
        assert 'name="my_agent"' in init
        assert "llm.py" in files
        assert "AgentLLMConfig" in files["llm.py"]

    def test_basic_agent_content(self):
        files = get_template_files("basic", "my_agent")
        agent = files["agent.py"]
        assert "BaseGraphAgent" in agent
        assert "MyAgentState" in agent
        assert "class MyAgentAgent" in agent
        assert "build_graph" in agent
        assert "def process" in agent


class TestFullTemplate:
    def test_full_has_all_files(self):
        files = get_template_files("full", "weather")
        assert "config.py" in files
        assert "schemas.py" in files
        assert "tools.py" in files
        assert "api.py" in files

    def test_full_config_content(self):
        files = get_template_files("full", "weather")
        config = files["config.py"]
        assert "WeatherConfig" in config
        assert "BaseModel" in config
        assert "prompt_version" in config

    def test_full_schemas_content(self):
        files = get_template_files("full", "weather")
        schemas = files["schemas.py"]
        assert "WeatherRequest" in schemas
        assert "WeatherResponse" in schemas

    def test_full_api_content(self):
        files = get_template_files("full", "weather")
        api = files["api.py"]
        assert "APIRouter" in api
        assert "router" in api


class TestRagTemplate:
    def test_rag_files(self):
        files = get_template_files("rag", "kb_search")
        assert "config.py" in files
        assert "tools.py" in files
        assert "agent.py" in files

    def test_rag_agent(self):
        files = get_template_files("rag", "kb_search")
        agent = files["agent.py"]
        assert "def retrieve" in agent
        assert "def generate" in agent
        assert "citations" in agent


class TestChatbotTemplate:
    def test_chatbot_files(self):
        files = get_template_files("chatbot", "assistant")
        assert "config.py" in files
        assert "agent.py" in files

    def test_chatbot_agent(self):
        files = get_template_files("chatbot", "assistant")
        agent = files["agent.py"]
        assert "def respond" in agent
        assert "messages" in agent


class TestCustomTemplate:
    def test_custom_minimal(self):
        files = get_template_files("custom", "simple")
        assert "__init__.py" in files
        assert "prompts.json" in files
        # No graph.py for custom
        assert "graph.py" not in files

    def test_custom_framework(self):
        files = get_template_files("custom", "simple")
        init = files["__init__.py"]
        assert 'framework="custom"' in init
        assert "node_fn" in init


class TestUnknownTemplate:
    def test_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template_files("nonexistent", "test")


# =========================================================================
# File Scaffolding Tests
# =========================================================================


class TestScaffolding:
    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp(prefix="agentomatic_test_")
        yield Path(d)
        shutil.rmtree(d)

    def test_write_basic_template(self, tmp_dir):
        files = get_template_files("basic", "hello")
        target = tmp_dir / "hello"
        target.mkdir()
        for rel_path, content in files.items():
            (target / rel_path).write_text(content)

        assert (target / "__init__.py").exists()
        assert (target / "agent.py").exists()
        assert not (target / "graph.py").exists()
        assert not (target / "nodes.py").exists()
        assert (target / "prompts.json").exists()

    def test_write_full_template(self, tmp_dir):
        files = get_template_files("full", "complete_agent")
        target = tmp_dir / "complete_agent"
        target.mkdir()
        for rel_path, content in files.items():
            file_path = target / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        assert len(list(target.iterdir())) >= 9

    def test_all_templates_produce_files(self, tmp_dir):
        for template in TEMPLATES:
            files = get_template_files(template, f"agent_{template}")
            if template == "pipeline":
                # Pipeline template produces only pipeline.yaml + README
                assert "pipeline.yaml" in files
                continue
            if template == "class":
                # Class template produces agent.py + dataset + train
                assert "agent.py" in files
                continue
            assert len(files) >= 3, f"Template {template} produced too few files"
            assert "__init__.py" in files, f"Template {template} missing __init__.py"

    def test_prompts_json_valid(self):
        import json

        for template in TEMPLATES:
            files = get_template_files(template, "test")
            if "prompts.json" in files:
                data = json.loads(files["prompts.json"])
                assert "v1" in data
                assert "system" in data["v1"]

    def test_langgraph_json_valid(self):
        import json

        for template in ["basic", "full", "rag", "chatbot"]:
            files = get_template_files(template, "test")
            if "langgraph.json" in files:
                data = json.loads(files["langgraph.json"])
                assert "graphs" in data


# =========================================================================
# UI Module Tests
# =========================================================================


class TestUIModule:
    def test_is_available_returns_bool(self):
        from agentomatic.ui import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_mount_without_chainlit(self):
        """Mount should gracefully handle missing Chainlit."""
        from agentomatic.ui import is_available

        if not is_available():
            # This should not raise
            from fastapi import FastAPI

            from agentomatic.ui import mount

            app = FastAPI()
            mount(app)  # Should log warning but not crash
