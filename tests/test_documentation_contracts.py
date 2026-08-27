"""High-value documentation contracts that are easy to accidentally stale."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import shlex
from pathlib import Path

from click.testing import CliRunner

from agentomatic._version import __version__
from agentomatic.cli.commands import cli
from agentomatic.cli.templates import TEMPLATES
from agentomatic.core.platform import AgentPlatform
from agentomatic.core.router_factory import (
    A2ATaskRequest,
    AgentChatRequest,
    AgentInvokeResponse,
    ApproveSuspendedRequest,
    FeedbackRequest,
    ForkThreadRequest,
    OptimizeInvokeRequest,
    RejectSuspendedRequest,
)
from agentomatic.tasks.models import TargetType, TaskProgress, TaskRecord

REPO = Path(__file__).parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MARKDOWN_IMAGE = re.compile(r"!\[[^]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)")
HTML_IMAGE = re.compile(r'<img\b[^>]*\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)
SHELL_FENCE = re.compile(r"```(?:bash|sh|shell|console)\n(.*?)```", re.DOTALL)
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)")


def _doc(path: str) -> str:
    """Read a documentation source relative to the repository root."""
    return (REPO / path).read_text(encoding="utf-8")


def test_cli_reference_lists_the_template_registry() -> None:
    """Every selectable scaffold template must be documented as selectable."""
    commands = _doc("docs/cli/commands.md")
    for template in TEMPLATES:
        assert f"`{template}`" in commands or f"{template}|" in commands


def test_cli_reference_lists_every_public_top_level_command() -> None:
    """Keep discoverable aliases such as ``new`` out of documentation limbo."""
    commands = _doc("docs/cli/commands.md")

    for command in cli.commands:
        assert f"agentomatic {command}" in commands


def test_cli_reference_covers_public_run_flags() -> None:
    """The long-lived CLI page must track the flags Click actually exposes."""
    commands = _doc("docs/cli/commands.md")
    help_text = CliRunner().invoke(cli, ["run", "--help"]).output

    for flag in (
        "--agents-dir",
        "--plugins-dir",
        "--endpoints-dir",
        "--ingestion-dir",
        "--stacks-dir",
        "--stack",
        "--ssl-certfile",
        "--ssl-keyfile",
        "--require-auth-globally",
    ):
        assert flag in help_text
        assert flag in commands


def test_cli_reference_covers_component_and_pipeline_commands() -> None:
    """Keep recently added public command families visible in the reference."""
    commands = _doc("docs/cli/commands.md")

    assert "  add" in commands
    assert "### `agentomatic add`" in commands
    assert "### `agentomatic pipeline`" in commands


def test_configuration_uses_current_auth_and_storage_variables() -> None:
    """Prevent stale variable names from breaking production authentication."""
    configuration = _doc("docs/guide/configuration.md")
    quickstart = _doc("docs/getting-started/quickstart.md")

    assert "AGENTOMATIC_API_KEY" in configuration
    assert "AGENTOMATIC_AUTH_API_KEY" not in configuration
    assert "DATABASE_URL` / `AGENTOMATIC_DB_URL" in quickstart
    assert "sqlite:///data/threads.db" not in quickstart


def test_configuration_reference_covers_every_platform_constructor_option() -> None:
    """A page labelled complete must track the actual platform public API."""
    configuration = _doc("docs/guide/configuration.md")

    for name in inspect.signature(AgentPlatform.__init__).parameters:
        if name != "self":
            assert f"`{name}`" in configuration


def test_install_and_deployment_guides_preserve_real_extra_and_release_contracts() -> None:
    """Keep install instructions usable in shells and tied to the package release."""
    installation = _doc("docs/getting-started/installation.md")
    quickstart = _doc("docs/getting-started/quickstart.md")
    deployment = _doc("docs/guide/deployment.md")

    assert 'pip install "agentomatic[all]"' in installation
    assert "Chainlit is intentionally separate" in installation
    assert 'pip install "agentomatic[all,ui]"' in quickstart
    assert f"agentomatic[all]=={__version__}" in deployment

    import tomllib

    with (REPO / "pyproject.toml").open("rb") as project_file:
        extras = tomllib.load(project_file)["project"]["optional-dependencies"]
    for extra in extras:
        assert f"`{extra}`" in installation

    for page in [REPO / "README.md", *(REPO / "docs").rglob("*.md")]:
        assert not re.search(r"pip install agentomatic\[[^\]]+\]", page.read_text()), page


def test_every_documentation_page_is_in_the_published_navigation() -> None:
    """Avoid shipping orphaned guides that users cannot discover."""
    nav_paths = set(
        re.findall(r"^\s*-\s+[^:]+:\s+([^\s]+\.md)\s*$", _doc("mkdocs.yml"), re.MULTILINE)
    )
    page_paths = {
        page.relative_to(REPO / "docs").as_posix() for page in (REPO / "docs").rglob("*.md")
    }

    assert nav_paths == page_paths


def test_local_markdown_links_target_existing_documentation_sources() -> None:
    """Catch stale relative Markdown links before a static deploy hides them."""
    for page in (REPO / "docs").rglob("*.md"):
        for target in MARKDOWN_LINK.findall(page.read_text(encoding="utf-8")):
            source = target.split("#", maxsplit=1)[0]
            if not source or source.startswith(("http://", "https://", "mailto:")):
                continue
            if source.endswith(".md"):
                assert (page.parent / source).resolve().is_file(), (page, target)


def test_frontend_guide_tracks_response_and_task_models() -> None:
    """Keep the hand-written TypeScript shapes faithful to server envelopes."""
    guide = _doc("docs/FRONTEND_API_GUIDE.md")

    for field in AgentInvokeResponse.model_fields:
        assert re.search(rf"\b{field}\s*[:?]", guide), field

    for field in (*TaskProgress.model_fields, *TaskRecord.model_fields, "duration_ms"):
        assert re.search(rf"\b{field}\s*[:?]", guide), field

    assert 'mode: "sync" | "async" | "batch";' in guide
    assert '"stream"' not in re.search(r"interface TaskRecord \{.*?\n\}", guide, re.DOTALL).group(
        0
    )
    assert "`/api/v1/ingestors`" in guide


def test_api_reference_tracks_the_standard_invoke_response_model() -> None:
    """The platform API page must not omit response fields added to the envelope."""
    api_reference = _doc("docs/architecture/api-reference.md")

    response_section = api_reference.split("**Response Body — `AgentInvokeResponse`:**", 1)[
        1
    ].split("**Response Example:**", 1)[0]
    for field in AgentInvokeResponse.model_fields:
        assert f"`{field}`" in response_section


def test_frontend_guide_tracks_the_feedback_request_model() -> None:
    """Feedback can be a rating, correction, comment, or thumbs signal."""
    frontend = _doc("docs/FRONTEND_API_GUIDE.md")
    feedback_section = frontend.split("interface FeedbackPayload {", 1)[1].split("}", 1)[0]

    for field in FeedbackRequest.model_fields:
        assert re.search(rf"\b{field}\s*\??:", feedback_section), field


def test_frontend_guide_tracks_a2a_optimization_and_hitl_request_models() -> None:
    """Every frontend-surface request body needs a usable typed shape."""
    frontend = _doc("docs/FRONTEND_API_GUIDE.md")

    for model in (
        A2ATaskRequest,
        OptimizeInvokeRequest,
        ApproveSuspendedRequest,
        RejectSuspendedRequest,
        ForkThreadRequest,
    ):
        section = frontend.split(f"interface {model.__name__} {{", 1)[1].split("}", 1)[0]
        for field in model.model_fields:
            assert re.search(rf"\b{field}\s*\??:", section), (model.__name__, field)


def test_provider_guide_tracks_supported_llm_and_embedding_factories() -> None:
    """Provider documentation must use the exact public factory vocabulary."""
    providers = _doc("docs/guide/llm-providers.md")

    for provider in ("ollama", "openai", "openai_compatible", "azure", "vertex", "dummy"):
        assert f"`{provider}`" in providers
    for provider in ("ollama", "openai", "azure_openai", "hash", "dummy"):
        assert f"`{provider}`" in providers

    assert 'provider="vertex",\n        model="gemini-1.5-pro"' in providers
    assert 'model_name="gemini-1.5-pro"' not in providers
    assert "POST /api/v1/{agent}/invoke/stream" in providers
    assert '"input": "Explain quantum computing"' not in providers


def test_task_guide_tracks_public_task_enums() -> None:
    """The TaskRecord reference must expose every routable kind and real mode."""
    task_record_section = _doc("docs/guide/tasks.md").split("### TaskRecord", maxsplit=1)[1]

    for target_type in TargetType:
        assert f"`{target_type.value}`" in task_record_section
    for mode in ("sync", "async", "batch"):
        assert f"`{mode}`" in task_record_section
    assert "`stream`" not in task_record_section


def test_getting_started_uses_the_actual_chat_and_sqlite_contracts() -> None:
    """Copy-paste onboarding examples must use the chat model and async SQL URL."""
    concepts = _doc("docs/getting-started/concepts.md")
    quickstart = _doc("docs/getting-started/quickstart.md")
    first_agent = _doc("docs/getting-started/first-agent.md")

    assert '"content": "Tell me more about that last point."' in concepts
    assert "/api/v1/search_bot/chat" in concepts
    assert '"query": "My name is Alice"' not in quickstart
    assert '"content": "My name is Alice"' in quickstart
    assert '"content": "What is my name?"' in quickstart
    assert "sqlite+aiosqlite:///./data/agents.db" in first_agent
    assert 'SQLAlchemyStore("sqlite:///./data/agents.db")' not in first_agent
    assert "content" in AgentChatRequest.model_fields


def test_storage_guide_does_not_describe_a_partial_schema_as_complete() -> None:
    """The five displayed tables intentionally cover state, not every SQL table."""
    storage = _doc("docs/guide/storage.md")

    assert "conversation and execution-state" in storage
    assert "invocation logs, analyses, and optimization" in storage
    assert "data model consists of **five**" not in storage


def test_architecture_and_status_docs_track_public_platform_contracts() -> None:
    """Avoid documenting removed constructor options or omitting status resources."""
    architecture = _doc("docs/architecture/overview.md")
    status = _doc("docs/guide/status.md")

    assert "enable_chainlit=True" not in architecture
    assert "app = platform.build()" in architecture
    assert '"connections": { "total": 2, "healthy": 2 }' in status
    assert "including a configured\nconnection" in status


def test_schema_and_security_examples_follow_state_and_edge_configuration() -> None:
    """Keep security commands on the deployment surface and custom schema fields usable."""
    schemas = _doc("docs/guide/schemas.md")
    middleware = _doc("docs/guide/middleware.md")
    security = _doc("docs/guide/security.md")

    assert 'location = state["location"]' in schemas
    assert 'state.get("metadata", {}).get("location")' not in schemas
    assert "AGENTOMATIC_ENABLE_AUTH=1" in security
    assert "FEATURES__ENABLE_AUTH=true" not in security
    assert "AGENTOMATIC_ENABLE_RATE_LIMIT=1" in security
    assert "trust_proxy_headers" in middleware
    assert "custom middleware is outermost" in middleware


def test_architecture_and_ingestion_guides_expose_real_routes() -> None:
    """Prevent removed app/chat aliases and task companions from drifting."""
    architecture = _doc("docs/architecture/overview.md")
    concepts = _doc("docs/getting-started/concepts.md")
    ingestion = _doc("docs/guide/ingestion.md")
    middleware = _doc("docs/guide/middleware.md")

    assert "platform.app" not in architecture
    assert "/chat/stream" not in concepts
    assert "POST /api/v1/ingestion/{name}/run/batch" in ingestion
    assert "`azure_openai`" in ingestion
    assert "(`/invoke/stream`)" in middleware


def test_langgraph_and_telemetry_guides_match_the_shipped_instrumentation() -> None:
    """The docs must not promise a removed ASGI attribute or automatic domain spans."""
    langgraph = _doc("docs/guide/langgraph.md")
    telemetry = _doc("docs/guide/telemetry.md")

    assert "platform.app" not in langgraph
    assert "uvicorn main:app --reload" in langgraph
    assert "Add `@traced`" in telemetry
    assert "automatically generates spans for every phase" not in telemetry
    assert "AGENTOMATIC_OTEL_CONSOLE=1" in telemetry


def test_chainlit_guide_only_promises_the_bundled_debug_ui_contract() -> None:
    """The shipped chat target is synchronous, stateless, and not an auth proxy."""
    debug_ui = _doc("docs/guide/debug-ui.md")

    assert "does not\n    forward an API key" in debug_ui
    assert "not stream `/invoke/stream`" in debug_ui
    assert "does not submit feedback" in debug_ui
    assert "Prompt Version Selector" not in debug_ui


def test_delegation_guide_does_not_claim_tool_creation_is_authorization() -> None:
    """A target list discovers handoffs; authorization is a separate policy check."""
    delegation = _doc("docs/guide/delegation.md")

    assert "ZeroTrustEnforcer blocks any delegation not in this list" not in delegation
    assert "Pair this with an AgentSecurityPolicy" in delegation


def test_deployment_guide_matches_root_compose_contract() -> None:
    """Keep the documented local oMLX instructions executable as written."""
    deployment = _doc("docs/guide/deployment.md")
    compose = _doc("docker-compose.yml")

    assert "AGENTOMATIC_AGENT_MOUNT=./agents docker compose up --build" in deployment
    assert "docker compose exec platform" in deployment
    assert "${AGENTOMATIC_PORT:-8010}:8000" in compose
    assert "${AGENTOMATIC_AGENT_MOUNT:-./e2e_demo/agents}" in compose
    assert "${AGENTOMATIC_REQUIRE_AUTH:-1}" in compose
    assert "${AGENTOMATIC_API_KEY:?Set AGENTOMATIC_API_KEY" in compose
    assert "AGENTOMATIC_CONTROL_TOKEN" in deployment
    assert "X-API-Key: $AGENTOMATIC_API_KEY" in deployment
    assert "per policy (not control token)" in deployment
    assert "--read-endpoint lookup" in _doc("docs/guide/verifying-a-deployment.md")


def test_deployment_verifier_docs_cover_discovered_schema_contracts() -> None:
    """The post-deploy guide must describe the dynamic live-schema gate."""
    verification = _doc("docs/guide/verifying-a-deployment.md")

    assert "`schema-contracts`" in verification
    assert "every deployed agent, plugin, endpoint, ingestor and pipeline" in verification
    assert "schema-contracts` group runs after the deliberate rate-limit" in verification


def test_task_guide_documents_pre_queue_resource_schema_validation() -> None:
    """The generic task route must not be documented as a validation bypass."""
    tasks = _doc("docs/guide/tasks.md")
    normalized = " ".join(tasks.split())

    assert "Resource input validation happens before queueing" in tasks
    assert "exact same published input contract" in normalized
    assert "plugin prediction schemas, custom-endpoint schemas, and ingestor schemas" in normalized
    assert "A malformed payload returns `422`, and an unknown resource returns `404`" in normalized
    assert "before** creating a durable task record" in normalized
    assert "strict_schema: true" in tasks


def test_documentation_png_assets_are_really_pngs() -> None:
    """Static hosts must not serve JPEG bytes with an ``image/png`` MIME type."""
    for asset in (
        "docs/assets/logo.png",
        "docs/assets/architecture_diagram.png",
        "docs/assets/optimization_flow.png",
    ):
        assert (REPO / asset).read_bytes().startswith(PNG_SIGNATURE), asset


def test_docs_shell_uses_the_portable_brand_mark_and_site_relative_banner() -> None:
    """The docs shell must work equally from the home page and nested pages."""
    config = _doc("mkdocs.yml")
    override = _doc("docs/overrides/main.html")

    assert "assets/agentomatic-mark.svg" in config
    assert "{{ 'changelog/' | url }}" in override


def test_raw_guide_images_use_paths_relative_to_built_nested_pages() -> None:
    """Raw HTML image URLs are not rewritten by MkDocs during the build."""
    for path in (REPO / "docs").rglob("*.md"):
        content = path.read_text(encoding="utf-8")
        assert '<img src="../assets/logo.png"' not in content, path


def test_local_documentation_images_resolve_from_their_published_pages() -> None:
    """Catch missing images and raw-HTML paths MkDocs cannot rewrite for us."""
    docs_root = REPO / "docs"
    site_root = REPO / "site"

    for page in docs_root.rglob("*.md"):
        content = page.read_text(encoding="utf-8")
        for source in MARKDOWN_IMAGE.findall(content):
            if source.startswith(("http://", "https://", "data:")):
                continue
            assert (page.parent / source).resolve().is_file(), (page, source)

        # Raw ``<img>`` paths are resolved by the browser after MkDocs has
        # converted ``guide/foo.md`` to ``site/guide/foo/index.html``.
        relative_page = page.relative_to(docs_root)
        published_page = (
            site_root / "index.html"
            if relative_page.name == "index.md"
            else site_root / relative_page.with_suffix("") / "index.html"
        )
        for source in HTML_IMAGE.findall(content):
            if source.startswith(("http://", "https://", "data:")):
                continue
            published_target = (published_page.parent / source).resolve()
            relative_target = published_target.relative_to(site_root.resolve())
            assert (docs_root / relative_target).is_file(), (page, source)


def test_optimization_guide_documents_only_supported_cli_paths() -> None:
    """Prevent removed optimization commands and modes returning to the docs."""
    optimization = _doc("docs/guide/optimization.md")

    assert "`PromptFitter`" in optimization
    assert "`PromptOptimizer`" in optimization
    for mode in ("rewrite", "param_search", "gepa_like", "mipro_like", "few_shot", "apo"):
        assert f"`{mode}`" in optimization
    for removed in ("route", "promote", "eval"):
        assert not re.search(rf"^agentomatic {removed}\b", optimization, re.MULTILINE)


def test_readme_does_not_advertise_removed_optimization_commands() -> None:
    """The README's workflow must use the public CLI, too."""
    readme = _doc("README.md")

    for removed in ("agentomatic dataset synth", "agentomatic eval", "agentomatic route"):
        assert removed not in readme
    assert "agentomatic optimize scope_agent" in readme


def test_documented_shell_commands_exist_in_the_current_cli() -> None:
    """Every fenced ``agentomatic`` command must start with a real command."""
    public_commands = set(cli.commands)
    pages = [REPO / "README.md", *(REPO / "docs").rglob("*.md")]

    for page in pages:
        for block in SHELL_FENCE.findall(page.read_text(encoding="utf-8")):
            for command in re.findall(r"(?:^|[ \t])agentomatic[ \t]+([a-z][a-z-]*)", block):
                assert command in public_commands, (page, command)


def test_documented_cli_options_exist_in_their_current_command_help() -> None:
    """Catch stale flags in multiline examples, including CLI subcommands."""
    help_by_command: dict[tuple[str, ...], str] = {}
    pages = [REPO / "README.md", *(REPO / "docs").rglob("*.md")]

    for page in pages:
        for block in SHELL_FENCE.findall(page.read_text(encoding="utf-8")):
            for line in block.replace("\\\n", " ").splitlines():
                match = re.search(r"(?:^|[ \t])agentomatic[ \t]+(.+)", line)
                if not match:
                    continue
                try:
                    tokens = shlex.split(match.group(1), comments=True)
                except ValueError:
                    continue
                if not tokens:
                    continue

                command = [tokens[0]]
                group = cli.commands.get(tokens[0])
                if group is None:
                    continue
                if (
                    hasattr(group, "commands")
                    and len(tokens) > 1
                    and not tokens[1].startswith("-")
                ):
                    command.append(tokens[1])

                key = tuple(command)
                help_text = help_by_command.setdefault(
                    key,
                    CliRunner().invoke(cli, [*command, "--help"]).output,
                )
                for option in re.findall(r"--[a-z][a-z0-9-]*", " ".join(tokens)):
                    assert option in help_text, (page, command, option)


def test_parsable_documented_agentomatic_imports_resolve() -> None:
    """Public imports in executable Python snippets must remain real exports."""
    for page in [REPO / "README.md", *(REPO / "docs").rglob("*.md")]:
        for snippet in PYTHON_FENCE.findall(page.read_text(encoding="utf-8")):
            try:
                tree = ast.parse(snippet)
            except SyntaxError:
                # Ellipsis-only fragments document a narrow API shape rather
                # than a standalone program; MkDocs still validates rendering.
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if not node.module.startswith("agentomatic"):
                    continue
                module = importlib.import_module(node.module)
                for imported in node.names:
                    if imported.name != "*":
                        assert hasattr(module, imported.name), (page, node.module, imported.name)


def test_optimization_guide_matches_current_cli_contract() -> None:
    """Keep mode names and important flags tied to Click's public interface."""
    optimization = _doc("docs/guide/optimization.md")
    help_text = CliRunner().invoke(cli, ["optimize", "--help"]).output

    for token in (
        "prompt_only",
        "rewrite",
        "param_search",
        "gepa_like",
        "mipro_like",
        "few_shot",
        "apo",
        "--search-space",
        "--search-method",
        "--node-match",
        "--n-runners",
    ):
        assert token in help_text
        assert token in optimization


def test_production_guides_do_not_advertise_removed_or_unavailable_features() -> None:
    """Keep versioned production guidance free of retired release-era claims."""
    plugins = _doc("docs/guide/ml-plugins.md")
    providers = _doc("docs/guide/llm-providers.md")
    stacks = _doc("docs/guide/stacks.md")
    platform_features = _doc("docs/guide/platform-features.md")

    assert "AGENTOMATIC_PLUGIN_AUTORELOAD=1" in plugins
    assert "prior\nin-memory state and readiness stay active" in plugins
    assert "install from git / editable checkout" not in providers
    assert re.search(
        r"\| `google_genai` \|\s+—\s+\|[^|]+\|\s+—\s+\|\s+Not available\s+\|",
        providers,
    )
    assert "requires **agentomatic >= 1.8.0**" not in stacks
    assert "requires **agentomatic >= 1.8.0**" not in platform_features


def test_api_and_pipeline_reference_cover_current_authoring_and_reload_routes() -> None:
    """The operator references must not hide current production route families."""
    api_reference = _doc("docs/architecture/api-reference.md")
    pipelines = _doc("docs/guide/pipelines.md")

    for path in (
        "/api/v1/plugins/reload",
        "/api/v1/ingestors",
        "/a2a/tasks/{task_id}/cancel",
    ):
        assert path in api_reference

    for path in (
        "/pipelines/validate-draft",
        "/pipelines/{name}",
        "/pipelines/{name}/run/async",
        "/pipelines/{name}/run/batch",
    ):
        assert path in pipelines


def test_docs_home_actions_link_to_built_routes_not_markdown_sources() -> None:
    """Raw HTML links bypass MkDocs' Markdown URL rewriting."""
    home = _doc("docs/index.md")

    assert 'href="getting-started/quickstart/"' in home
    assert 'href="guide/deployment/"' in home
    assert 'href="getting-started/quickstart.md"' not in home
    assert 'href="guide/deployment.md"' not in home


def test_studio_guide_describes_the_shipped_schema_form_contract() -> None:
    """Keep operator docs aligned with the dynamic forms in the packaged UI."""
    studio = _doc("docs/guide/studio.md")

    assert "`GET /openapi.json`" in studio
    assert "nested object" in studio
    assert 'it does not fabricate a `{ "query": ... }`' in studio
    assert "does not\nquietly substitute an empty payload" in studio
    assert "cache-busting hashed JavaScript and CSS chunks are immutable" in studio
    assert "**Reload Studio** action instead of leaving the page blank" in studio
    assert "schema-aware routing is available for outputs" in studio
    assert "Save & Run** dialog uses that draft pipeline input schema" in studio
    assert "Use current form as first item" in studio
    assert "documented path/query/header parameters" in studio
    assert "sends body fields as JSON, query" in studio
    assert "**Input shape** selector" in studio
    assert "Enum choices retain the JSON type" in studio
    assert "accepts any valid union branch" in studio
    assert "restores real conversations instead" in studio
    assert "browser-only thread identifiers" in studio
    assert "current browser tab session" in studio
    assert "closing the tab removes the secret" in studio
    assert "not production validation" in studio
    assert "deployment verifier" in studio
    assert "also provide an in-place **Refresh** action" in studio
    assert "without reconnecting Studio" in studio
    assert "Map** step" in studio
    assert "`items`, `by_key`, `count`, or `succeeded`" in studio
    assert "**Edit in Builder**" in studio
    assert "current\nserver configuration" in studio


def test_docker_omlx_example_uses_agentomatic_provider_model_syntax() -> None:
    """The oMLX server's raw id needs Agentomatic's ``omlx/`` provider prefix."""
    deployment = _doc("docs/guide/deployment.md")

    assert 'AGENTOMATIC_LIVE_MODEL="omlx/Qwen3.5-9B-MLX-4bit"' in deployment
    assert 'AGENTOMATIC_LIVE_MODEL="Qwen3.5-9B-MLX-4bit"' not in deployment


def test_security_audit_is_documented_and_required_by_ci() -> None:
    """The release gate must remain reproducible from the contributor docs."""
    assert "make audit" in _doc("README.md")
    assert "make audit" in _doc("CONTRIBUTING.md")
    assert "make audit" in _doc("docs/contributing.md")

    ci = _doc(".github/workflows/ci.yml")
    assert "security:" in ci
    assert "uv run pip-audit --progress-spinner off" in ci
    assert "needs: [lint, security, test, typecheck, docs, smoke]" in ci


def test_testing_guide_separates_bounded_suite_from_real_model_verification() -> None:
    """A reachable local oMLX server must not make the default test gate unbounded."""
    testing = _doc("docs/guide/testing.md")

    assert 'uv run pytest -m "not live"' in testing
    assert "uv run pytest -m live --override-ini='addopts='" in testing
    assert "deployment verifier" in testing


def test_endpoint_guide_describes_method_specific_input_transport() -> None:
    """Keep browser-safe GET endpoint behaviour discoverable to users."""
    endpoints = _doc("docs/guide/endpoints.md")

    assert 'methods = ["GET", "POST"]' in endpoints
    assert "query parameters" in endpoints
    assert "forbidden from sending a body" in endpoints
    assert "?tag=red&tag=blue" in endpoints
    assert '"region":"eu"' in endpoints
    assert "`GET`-only endpoint" in endpoints
    assert "Studio **Endpoints** page" in endpoints
