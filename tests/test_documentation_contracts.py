"""High-value documentation contracts that are easy to accidentally stale."""

from __future__ import annotations

import ast
import importlib
import re
import shlex
from pathlib import Path

from click.testing import CliRunner

from agentomatic.cli.commands import cli
from agentomatic.cli.templates import TEMPLATES

REPO = Path(__file__).parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MARKDOWN_IMAGE = re.compile(r"!\[[^]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)")
HTML_IMAGE = re.compile(r'<img\b[^>]*\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)
SHELL_FENCE = re.compile(r"```(?:bash|sh|shell|console)\n(.*?)```", re.DOTALL)
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _doc(path: str) -> str:
    """Read a documentation source relative to the repository root."""
    return (REPO / path).read_text(encoding="utf-8")


def test_cli_reference_lists_the_template_registry() -> None:
    """Every selectable scaffold template must be documented as selectable."""
    commands = _doc("docs/cli/commands.md")
    for template in TEMPLATES:
        assert f"`{template}`" in commands or f"{template}|" in commands


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

    assert "AGENTOMATIC_PLUGIN_AUTORELOAD" not in plugins
    assert "call the authenticated reload endpoint" in plugins
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
