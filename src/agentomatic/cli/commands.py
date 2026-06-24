"""Agentomatic CLI — beautiful terminal experience for agent lifecycle.

Commands:
    agentomatic init <name>     Interactive agent scaffolding
    agentomatic run             Start the platform
    agentomatic list            List discovered agents
    agentomatic test <name>     Test an agent interactively
    agentomatic inspect <name>  Show agent details
    agentomatic doctor          Check environment health
    agentomatic ui              Launch debug UI standalone
    agentomatic optimize <name> Run prompt optimization

Requires: pip install agentomatic[cli]
Fallback: works with basic output if Rich is not installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import click
from loguru import logger

# Graceful Rich fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console: Console = Console() if HAS_RICH else None  # type: ignore[assignment]


def _get_version() -> str:
    """Get the package version."""
    try:
        from importlib.metadata import version
        return version("agentomatic")
    except Exception:
        return "dev"


# =====================================================================
# Utilities
# =====================================================================


def _echo(msg: str) -> None:
    """Print with Rich if available, plain click.echo otherwise."""
    if console:
        console.print(msg)
    else:
        click.echo(msg)


def _print_banner() -> None:
    """Show the agentomatic banner."""
    ver = _get_version()
    if HAS_RICH:
        logo = (
            "[bold magenta]"
            "   _____                 __                        __  _     \n"
            "  /  _  \\ _____ ___ ___/  |_  ____   _____ _____ /  |_|__|____ \n"
            " /  /_\\  \\__  \\/ __ |   __\\/  _ \\ /     \\\\__  \\    _|  |  ___/\n"
            "/    |    \\/ __ |  __|  | (  <_> |  Y Y  \\/ __ \\|  | |  |  (__ \n"
            "\\____|__  (____/\\___|  |  \\____/|__|_|  (____/  |__| |__|\\___/\n"
            "        \\/                           \\/\n"
            "[/bold magenta]"
            f"[dim]  v{ver} — Drop agents, not code[/dim]"
        )
        console.print(Panel.fit(logo, border_style="magenta", padding=(0, 1)))
    else:
        click.echo(f"⚡ agentomatic v{ver} — Drop agents, not code")
        click.echo()


def _print_success(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold green]✅ {msg}[/bold green]")
    else:
        logger.success(msg)


def _print_error(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold red]❌ {msg}[/bold red]")
    else:
        logger.error(msg)


def _print_warning(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold yellow]⚠️  {msg}[/bold yellow]")
    else:
        logger.warning(msg)


# =====================================================================
# CLI Group
# =====================================================================


@click.group(invoke_without_command=True)
@click.option("--version", "-V", is_flag=True, help="Show version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """⚡ Agentomatic — Drop agents, not code."""
    if version:
        click.echo(f"agentomatic {_get_version()}")
        ctx.exit()
    elif ctx.invoked_subcommand is None:
        _print_banner()
        if HAS_RICH:
            # Categorized command table
            table = Table(
                show_header=True,
                header_style="bold magenta",
                show_lines=False,
                pad_edge=True,
                box=None,
            )
            table.add_column("Command", style="bold cyan", min_width=20)
            table.add_column("Description")

            table.add_row("[dim]── Agent Lifecycle ──[/dim]", "")
            table.add_row("  init <name>", "Create a new agent from a template")
            table.add_row("  list", "Discover and display all agents")
            table.add_row("  inspect <name>", "Deep-dive into agent structure & ML lifecycle")
            table.add_row("  test <name>", "Test an agent interactively")
            table.add_row("")
            table.add_row("[dim]── Platform ──[/dim]", "")
            table.add_row("  run", "Start the platform server")
            table.add_row("  demo", "Launch demo with built-in agent + Studio")
            table.add_row("  doctor", "Diagnose environment & dependencies")
            table.add_row("")
            table.add_row("[dim]── ML & Optimization ──[/dim]", "")
            table.add_row("  optimize <name>", "Run prompt optimization")
            table.add_row("")
            table.add_row("[dim]── Debug & UI ──[/dim]", "")
            table.add_row("  ui", "Launch Chainlit debug interface")
            table.add_row("")
            table.add_row("[dim]── Advanced ──[/dim]", "")
            table.add_row("  stack <cmd>", "Manage environment stacks")
            table.add_row("  pipeline <cmd>", "Manage and execute pipelines")

            console.print(table)
            console.print()
            console.print(
                "[dim]Run [bold]agentomatic <command> --help[/bold] "
                "for details on any command.[/dim]"
            )
        else:
            click.echo(ctx.get_help())


# =====================================================================
# INIT — Scaffold a new agent
# =====================================================================


@cli.command()
@click.argument("name")
@click.option(
    "--dir", "-d", "agents_dir", default="agents", help="Agents directory (default: agents)"
)
@click.option(
    "--template",
    "-t",
    type=click.Choice(
        [
            "basic",
            "full",
            "rag",
            "chatbot",
            "deepagent",
            "custom",
            "legacy_dict",
            "plugin",
        ]
    ),
    default=None,
    help="Template to use (default: interactive selection)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def init(name: str, agents_dir: str, template: str | None, force: bool) -> None:
    """Create a new agent from a template (basic, full, rag, chatbot, ...)."""
    from .templates import TEMPLATES, get_template_files

    target = Path(agents_dir) / name

    _print_banner()

    # Template selection
    if not template:
        # Interactive selection if questionary is available
        try:
            import questionary

            choices = [questionary.Choice(f"{k} — {v}", value=k) for k, v in TEMPLATES.items()]
            template = questionary.select(
                "Select a template:",
                choices=choices,
                default="basic",
            ).ask()
            if not template:
                _print_error("Cancelled")
                return
        except ImportError:
            template = "basic"
            _print_warning("Install questionary for interactive mode: pip install questionary")

    # Validate template
    if template not in TEMPLATES:
        _print_error(f"Unknown template: {template}. Choose from: {list(TEMPLATES.keys())}")
        sys.exit(1)

    # Check if target exists
    if target.exists() and any(target.iterdir()):
        _print_warning(f"Directory {target} already exists and is not empty")
        if not force:
            if not click.confirm("Overwrite?", default=False):
                logger.info("Cancelled")
                return

    # Generate files
    target.mkdir(parents=True, exist_ok=True)
    files = get_template_files(template, name)

    if HAS_RICH:
        tree = Tree(f"[bold cyan]📁 {target}[/bold cyan]")
        for rel_path in sorted(files.keys()):
            tree.add(f"[green]📄 {rel_path}[/green]")
        console.print(tree)
    else:
        click.echo(f"📁 {target}")
        for rel_path in sorted(files.keys()):
            click.echo(f"  📄 {rel_path}")

    for rel_path, content in files.items():
        file_path = target / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    click.echo()
    logger.success(f"Created agent '{name}' with template '{template}'")
    click.echo(f"   📍 Location: {target}")
    click.echo(f"   📦 Files: {len(files)}")
    click.echo()

    if template == "plugin":
        edit_file = "plugin.py"
    elif template in ["legacy_dict", "custom"]:
        edit_file = "nodes.py" if template == "legacy_dict" else "__init__.py"
    else:
        edit_file = "agent.py"

    # Build next-steps text
    steps = (
        f"[bold]Next steps:[/bold]\n\n"
        f"  1. [cyan]cd {agents_dir}[/cyan]\n"
        f"  2. Edit [yellow]{name}/{edit_file}[/yellow] with your logic\n"
        f"  3. [cyan]agentomatic run[/cyan] to start\n"
        f"  4. [cyan]agentomatic test {name}[/cyan] to test\n"
        f"  5. Open [blue]http://localhost:8000/docs[/blue] for API docs"
    )

    if template == "full":
        steps += (
            f"\n\n"
            f"[bold]ML Lifecycle:[/bold]\n\n"
            f"  [cyan]python -m agents.{name}.train[/cyan]     "
            f"[dim]— Compile, fit & save[/dim]\n"
            f"  [cyan]python -m agents.{name}.eval[/cyan]      "
            f"[dim]— Evaluate quality[/dim]\n"
            f"  [cyan]python -m agents.{name}.optimize[/cyan]  "
            f"[dim]— Prompt optimization[/dim]\n"
            f"  [cyan]python -m agents.{name}.predict[/cyan]   "
            f"[dim]— Batch / interactive inference[/dim]\n"
            f"  [cyan]make all[/cyan]                          "
            f"[dim]— Full pipeline (train → eval)[/dim]"
        )

    if HAS_RICH:
        console.print(Panel(steps, title="🚀 What's next?", border_style="green"))
    else:
        click.echo("Next steps:")
        click.echo(f"  1. Edit {name}/{edit_file} with your logic")
        click.echo("  2. agentomatic run")
        click.echo(f"  3. agentomatic test {name}")
        if template == "full":
            click.echo(f"  4. python -m agents.{name}.train")
            click.echo(f"  5. python -m agents.{name}.eval")
            click.echo(f"  6. python -m agents.{name}.optimize")


# =====================================================================
# RUN — Start the platform
# =====================================================================


@cli.command()
@click.option("--agents-dir", default="agents", help="Agents directory")
@click.option("--plugins-dir", default="plugins", help="Plugins directory")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", type=int, default=8000, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--title", default=None, help="Platform title")
@click.option("--log-level", default="INFO", help="Log level")
@click.option("--with-ui", "--ui", is_flag=True, help="Enable Chainlit debug UI at /chat")
@click.option("--studio", is_flag=True, help="Enable Agentomatic Studio debug UI at /studio/ui")
def run(
    agents_dir: str,
    plugins_dir: str,
    host: str,
    port: int,
    reload: bool,
    title: str | None,
    log_level: str,
    with_ui: bool,
    studio: bool,
) -> None:
    """Run the platform with Rich status output."""
    _print_banner()
    logger.info(f"Starting platform from {agents_dir} (plugins: {plugins_dir})...")

    from agentomatic import AgentPlatform

    kwargs: dict[str, Any] = {
        "plugins_dir": plugins_dir,
        "title": title or "Agentomatic Platform",
        "log_level": log_level,
        "enable_studio": studio,
    }

    # Auto-detect and enable UI
    if with_ui:
        from agentomatic.ui import is_available

        if is_available():
            logger.success("Debug UI will be available at /chat")
        else:
            logger.warning("Chainlit not installed. Install: pip install agentomatic[ui]")

    platform = AgentPlatform.from_folder(agents_dir, **kwargs)

    # Mount UI if requested
    if with_ui:

        @platform.on_startup
        async def _mount_ui():
            from agentomatic.ui import mount

            if platform._app:
                mount(platform._app)

    platform.run(host=host, port=port, reload=reload)


# =====================================================================
# DEMO — Launch demo platform with Studio
# =====================================================================


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", type=int, default=8000, help="Port to listen on")
def demo(host: str, port: int) -> None:
    """Launch a demo platform with a built-in agent and Studio for E2E testing."""
    _print_banner()

    if HAS_RICH:
        console.print(
            Panel(
                "[bold cyan]🎪 Demo Mode[/bold cyan]\n\n"
                "Starting a self-contained platform with:\n"
                "  • [green]demo_assistant[/green] — multi-step reasoning agent\n"
                "  • [green]Studio UI[/green] — graph visualisation & state inspection",
                title="Agentomatic Demo",
                border_style="cyan",
            )
        )
    else:
        click.echo("🎪 Demo Mode")
        click.echo("  • demo_assistant — multi-step reasoning agent")
        click.echo("  • Studio UI — graph visualisation & state inspection")

    _echo("")
    _echo(f"  📡 API:    http://{host}:{port}")
    _echo(f"  📖 Docs:   http://{host}:{port}/docs")
    _echo(f"  🎨 Studio: http://{host}:{port}/studio/ui/")
    _echo("")

    from agentomatic.demo.server import create_demo_platform

    platform = create_demo_platform(host=host, port=port, enable_studio=True)
    platform.run(host=host, port=port)


# =====================================================================
# LIST — Show discovered agents
# =====================================================================


@cli.command("list")
@click.option("--agents-dir", default="agents", help="Agents directory")
def list_agents(agents_dir: str) -> None:
    """Discover and display all agents with their patterns and capabilities."""
    agents_path = Path(agents_dir)
    _print_banner()

    if not agents_path.exists():
        _print_error(f"Directory not found: {agents_path}")
        sys.exit(1)

    agents = []
    for entry in sorted(agents_path.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue

        has_init = (entry / "__init__.py").exists()
        has_agent_py = (entry / "agent.py").exists()

        if not has_init and not has_agent_py:
            continue

        info: dict[str, Any] = {"name": entry.name, "path": str(entry)}
        info["files"] = len([f for f in entry.iterdir() if f.is_file()])

        # Detect pattern: class-based vs functional
        if has_agent_py:
            agent_source = (entry / "agent.py").read_text(errors="ignore")
            if "BaseGraphAgent" in agent_source:
                info["pattern"] = "class"
            else:
                info["pattern"] = "functional"
        else:
            info["pattern"] = "functional"

        # Read manifest / description from source
        try:
            if has_init:
                source = (entry / "__init__.py").read_text(errors="ignore")
                if "AgentManifest" in source:
                    info["has_manifest"] = True
                    # Try to extract description
                    import re
                    desc_match = re.search(
                        r'description\s*=\s*["\']([^"\']+)["\']', source
                    )
                    if desc_match:
                        info["description"] = desc_match.group(1)[:50]
                    # Extract framework
                    fw_match = re.search(
                        r'framework\s*=\s*["\']([^"\']+)["\']', source
                    )
                    if fw_match:
                        info["framework"] = fw_match.group(1)
            elif has_agent_py:
                info["has_manifest"] = True  # class agents self-describe
                agent_source = (entry / "agent.py").read_text(errors="ignore")
                import re
                desc_match = re.search(
                    r'agent_description\s*=\s*["\']([^"\']+)["\']', agent_source
                )
                if desc_match:
                    info["description"] = desc_match.group(1)[:50]
                info["framework"] = "graph_agent"
        except Exception:
            pass

        # Detect features
        info["has_graph"] = (entry / "graph.py").exists()
        info["has_schemas"] = (entry / "schemas.py").exists()
        info["has_config"] = (entry / "config.py").exists()
        info["has_tools"] = (entry / "tools.py").exists()

        # ML lifecycle files
        info["has_train"] = (entry / "train.py").exists()
        info["has_eval"] = (entry / "eval.py").exists()
        info["has_dataset"] = (entry / "dataset.jsonl").exists()
        info["has_optimize"] = (entry / "optimize.py").exists()

        agents.append(info)

    if not agents:
        _print_warning(f"No agents found in {agents_path}")
        click.echo("\n  Create one with: agentomatic init my_agent")
        return

    if HAS_RICH:
        table = Table(
            title=f"🤖 Agents in {agents_path} ({len(agents)} found)",
            show_lines=True,
        )
        table.add_column("Name", style="bold cyan", min_width=15)
        table.add_column("Pattern", justify="center")
        table.add_column("Framework", justify="center")
        table.add_column("Files", justify="center")
        table.add_column("ML Lifecycle", justify="center")
        table.add_column("Description")

        for a in agents:
            pattern = a.get("pattern", "?")
            if pattern == "class":
                pattern_str = "[bold green]⬡ class[/bold green]"
            else:
                pattern_str = "[yellow]ƒ functional[/yellow]"

            framework = a.get("framework", "—")

            # ML lifecycle badges
            ml_parts = []
            if a.get("has_train"):
                ml_parts.append("T")
            if a.get("has_eval"):
                ml_parts.append("E")
            if a.get("has_optimize"):
                ml_parts.append("O")
            if a.get("has_dataset"):
                ml_parts.append("D")
            if ml_parts:
                ml_str = "[green]" + "·".join(ml_parts) + "[/green]"
            else:
                ml_str = "[dim]—[/dim]"

            desc = a.get("description", "[dim]—[/dim]")

            table.add_row(
                a["name"],
                pattern_str,
                framework,
                str(a.get("files", "?")),
                ml_str,
                desc,
            )

        console.print(table)
        console.print(
            "[dim]  ML Lifecycle: T=train  E=eval  O=optimize  D=dataset[/dim]"
        )
    else:
        click.echo(f"📂 {agents_path} ({len(agents)} agents)")
        for a in agents:
            pattern = "⬡" if a.get("pattern") == "class" else "ƒ"
            desc = a.get("description", "")
            desc_str = f" — {desc}" if desc else ""
            click.echo(
                f"  {pattern} {a['name']} ({a.get('files', '?')} files){desc_str}"
            )

    _echo(f"\n   Total: {len(agents)} agent(s)")


# =====================================================================
# TEST — Interactive agent testing
# =====================================================================


@cli.command()
@click.argument("name")
@click.option("--host", default="localhost", help="Platform API host")
@click.option("--port", type=int, default=8000, help="Platform API port")
@click.option("--agents-dir", default="agents", help="Agents directory")
def test(name: str, host: str, port: int, agents_dir: str) -> None:
    """Test an agent interactively in the terminal."""
    import asyncio

    _print_banner()
    base_url = f"http://{host}:{port}"

    if HAS_RICH:
        _echo(f"🧪 Testing agent: [bold cyan]{name}[/bold cyan]")
    else:
        logger.info(f"Testing agent: {name}")

    click.echo(f"   API: {base_url}/api/v1/{name}/invoke")
    click.echo("   Type 'quit' or 'exit' to stop\n")

    async def _test_loop():
        import httpx

        async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
            # Health check
            try:
                resp = await client.get(f"/api/v1/{name}/health")
                if resp.status_code == 200:
                    logger.success(f"Agent '{name}' is healthy")
                else:
                    logger.error(f"Agent '{name}' health check failed: {resp.status_code}")
                    return
            except httpx.ConnectError:
                logger.error(f"Cannot connect to {base_url}. Is the platform running?")
                click.echo("   Start with: agentomatic run")
                return

            # Interactive loop
            thread_id = None
            while True:
                try:
                    query = input("\n🗣️  You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    break

                if query.lower() in ("quit", "exit", "q"):
                    break
                if not query:
                    continue

                try:
                    payload = {"query": query, "user_id": "cli-tester"}
                    if thread_id:
                        payload["thread_id"] = thread_id

                    resp = await client.post(
                        f"/api/v1/{name}/invoke",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    thread_id = data.get("thread_id", thread_id)

                    if HAS_RICH:
                        console.print(
                            f"\n🤖 [bold green]{name}[/bold green]: {data.get('response', '')}"
                        )
                        if data.get("steps_taken"):
                            console.print(
                                f"   [dim]Steps: {' → '.join(data['steps_taken'])}[/dim]"
                            )
                        if data.get("suggestions"):
                            console.print(
                                f"   [dim]Suggestions: {', '.join(data['suggestions'])}[/dim]"
                            )
                        console.print(f"   [dim]⏱ {data.get('duration_ms', 0):.0f}ms[/dim]")
                    else:
                        click.echo(f"\n🤖 {name}: {data.get('response', '')}")
                        if data.get("duration_ms"):
                            click.echo(f"   ⏱ {data['duration_ms']:.0f}ms")

                except httpx.HTTPStatusError as exc:
                    logger.error(f"API Error: {exc.response.status_code}")
                except Exception as exc:
                    logger.error(f"Error: {exc}")

            click.echo("\n👋 Test session ended")

    asyncio.run(_test_loop())


# =====================================================================
# INSPECT — Show agent details
# =====================================================================


@cli.command()
@click.argument("name")
@click.option("--agents-dir", default="agents", help="Agents directory")
def inspect(name: str, agents_dir: str) -> None:
    """Deep-dive into an agent's structure, files, and ML lifecycle status."""
    _print_banner()

    target = Path(agents_dir) / name
    if not target.exists():
        _print_error(f"Agent not found: {target}")
        sys.exit(1)

    files = sorted([f for f in target.rglob("*") if f.is_file() and not f.name.startswith(".")])

    if HAS_RICH:
        # Header
        console.print(
            Panel(
                f"[bold cyan]{name}[/bold cyan]\n[dim]{target}[/dim]",
                title="🔍 Agent Inspector",
                border_style="cyan",
            )
        )

        # File tree
        tree = Tree(f"[bold]📁 {name}/[/bold]")
        for f in files:
            rel = f.relative_to(target)
            size = f.stat().st_size
            tree.add(f"📄 {rel} [dim]({size:,} bytes)[/dim]")
        console.print(tree)

        # Read manifest info
        init_file = target / "__init__.py"
        if init_file.exists():
            source = init_file.read_text()
            console.print(Panel(source, title="__init__.py", border_style="green"))

        # Check for config
        config_file = target / "config.py"
        if config_file.exists():
            console.print(Panel(config_file.read_text(), title="config.py", border_style="yellow"))

        # Check for prompts
        prompts_file = target / "prompts.json"
        if prompts_file.exists():
            data = json.loads(prompts_file.read_text())
            console.print(
                Panel(
                    json.dumps(data, indent=2),
                    title=f"prompts.json ({len(data)} versions)",
                    border_style="blue",
                )
            )

        # v0.6 panels
        llm_file = target / "llm.py"
        if llm_file.exists():
            console.print(Panel(llm_file.read_text(), title="llm.py", border_style="magenta"))

        delegation_file = target / "delegation.py"
        if delegation_file.exists():
            console.print(
                Panel(
                    delegation_file.read_text(),
                    title="delegation.py",
                    border_style="magenta",
                )
            )

        security_file = target / "security.py"
        if security_file.exists():
            console.print(
                Panel(
                    security_file.read_text(),
                    title="security.py",
                    border_style="red",
                )
            )

        schemas_file = target / "schemas.py"
        if schemas_file.exists():
            console.print(
                Panel(
                    schemas_file.read_text(),
                    title="schemas.py",
                    border_style="yellow",
                )
            )

        evals_file = target / "evals.py"
        if evals_file.exists():
            console.print(
                Panel(
                    evals_file.read_text(),
                    title="evals.py",
                    border_style="blue",
                )
            )

        # Detect agent pattern
        agent_py = target / "agent.py"
        if agent_py.exists():
            agent_src = agent_py.read_text(errors="ignore")
            if "BaseGraphAgent" in agent_src:
                pattern_str = "[bold green]⬡ Class-Based[/bold green] (BaseGraphAgent)"
            else:
                pattern_str = "[yellow]ƒ Functional[/yellow]"
        else:
            pattern_str = "[yellow]ƒ Functional[/yellow] (legacy)"

        console.print(
            Panel(
                f"Pattern: {pattern_str}",
                title="🏗️ Agent Pattern",
                border_style="cyan",
            )
        )

        # Summary table — Core
        cap_table = Table(title="Agent Capabilities", show_lines=True)
        cap_table.add_column("Feature", style="bold", min_width=30)
        cap_table.add_column("Status", justify="center")

        core_caps = [
            ("🏗️  Agent Class (agent.py)", (target / "agent.py").exists()),
            ("📊 Graph (graph.py)", (target / "graph.py").exists()),
            ("⚙️  Config (config.py)", (target / "config.py").exists()),
            ("📝 Prompts (prompts.json)", (target / "prompts.json").exists()),
            ("📐 Schemas (schemas.py)", (target / "schemas.py").exists()),
            ("🔧 Tools (tools.py)", (target / "tools.py").exists()),
            ("🌐 Custom API (api.py)", (target / "api.py").exists()),
        ]
        for feat, exists in core_caps:
            cap_table.add_row(
                feat,
                "[green]✅[/green]" if exists else "[dim]—[/dim]",
            )

        # Separator
        cap_table.add_row("[dim]── ML Lifecycle ──[/dim]", "")
        ml_caps = [
            ("📦 Dataset (dataset.jsonl)", (target / "dataset.jsonl").exists()),
            ("🏋️  Train Script (train.py)", (target / "train.py").exists()),
            ("📊 Eval Script (eval.py)", (target / "eval.py").exists()),
            ("🔧 Optimize Script (optimize.py)", (target / "optimize.py").exists()),
            ("🔮 Predict Script (predict.py)", (target / "predict.py").exists()),
            ("📋 Makefile", (target / "Makefile").exists()),
        ]
        for feat, exists in ml_caps:
            cap_table.add_row(
                feat,
                "[green]✅[/green]" if exists else "[dim]—[/dim]",
            )

        # Separator
        cap_table.add_row("[dim]── Advanced ──[/dim]", "")
        adv_caps = [
            ("🤖 LLM Config (llm.py)", (target / "llm.py").exists()),
            ("🔗 Delegation (delegation.py)", (target / "delegation.py").exists()),
            ("🔒 Security (security.py)", (target / "security.py").exists()),
            ("📊 Evals (evals.py)", (target / "evals.py").exists()),
            ("🃏 Model Card (model_card.yaml)", (target / "model_card.yaml").exists()),
        ]
        for feat, exists in adv_caps:
            cap_table.add_row(
                feat,
                "[green]✅[/green]" if exists else "[dim]—[/dim]",
            )

        console.print(cap_table)
    else:
        click.echo(f"🔍 Agent: {name}")
        click.echo(f"   Path: {target}")
        click.echo(f"   Files: {len(files)}")
        for f in files:
            click.echo(f"   📄 {f.relative_to(target)}")
        click.echo("\n   Agent Capabilities:")
        capabilities = [
            ("Agent Class (agent.py)", (target / "agent.py").exists()),
            ("Graph (graph.py)", (target / "graph.py").exists()),
            ("Config (config.py)", (target / "config.py").exists()),
            ("Prompts (prompts.json)", (target / "prompts.json").exists()),
            ("Schemas (schemas.py)", (target / "schemas.py").exists()),
            ("Tools (tools.py)", (target / "tools.py").exists()),
            ("Custom API (api.py)", (target / "api.py").exists()),
            ("Dataset (dataset.jsonl)", (target / "dataset.jsonl").exists()),
            ("Train (train.py)", (target / "train.py").exists()),
            ("Eval (eval.py)", (target / "eval.py").exists()),
            ("Optimize (optimize.py)", (target / "optimize.py").exists()),
            ("Predict (predict.py)", (target / "predict.py").exists()),
            ("Makefile", (target / "Makefile").exists()),
        ]
        for feat, exists in capabilities:
            status = "✅" if exists else "—"
            click.echo(f"   {status} {feat}")


# =====================================================================
# DOCTOR — Environment health check
# =====================================================================


@cli.command()
@click.option("--agents-dir", default="agents", help="Agents directory")
def doctor(agents_dir: str) -> None:
    """Diagnose environment health, dependencies, and configuration."""
    _print_banner()

    checks: list[tuple[str, bool, str]] = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 11)
    checks.append(("Python", ok, f"{py_ver} {'✅' if ok else '(need ≥3.11)'}"))

    # Core deps
    for pkg in ["fastapi", "uvicorn", "pydantic", "loguru", "httpx"]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "installed")
            checks.append((pkg, True, ver))
        except ImportError:
            checks.append((pkg, False, "not installed"))

    # Optional deps
    for pkg, extra in [
        ("langgraph", "langgraph"),
        ("langchain_core", "langchain"),
        ("rich", "cli"),
        ("questionary", "cli"),
        ("chainlit", "ui"),
        ("sqlalchemy", "db"),
        ("prometheus_client", "metrics"),
        ("dotenv", "dotenv"),
        ("jwt", "security"),
        ("cryptography", "security"),
        ("langgraph_swarm", "swarm"),
    ]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "installed")
            checks.append((f"{pkg} [{extra}]", True, ver))
        except ImportError:
            checks.append((f"{pkg} [{extra}]", False, f"pip install agentomatic[{extra}]"))

    # Agents directory
    agents_path = Path(agents_dir)
    if agents_path.exists():
        count = len(
            [d for d in agents_path.iterdir()
             if d.is_dir() and (
                 (d / "__init__.py").exists() or (d / "agent.py").exists()
             )]
        )
        checks.append(("Agents directory", True, f"{count} agent(s) in {agents_path}"))
    else:
        checks.append(("Agents directory", False, f"Not found: {agents_path}"))

    # Stacks directory
    stacks_path = Path("stacks")
    if stacks_path.exists():
        yaml_count = len(list(stacks_path.glob("*.yaml"))) + len(list(stacks_path.glob("*.yml")))
        checks.append(("Stacks directory", True, f"{yaml_count} stack(s) in {stacks_path}"))
    else:
        checks.append(("Stacks directory", False, "Not found — run: agentomatic stack init"))

    # Active stack
    active_file = Path(".agentomatic-stack")
    if active_file.exists():
        active_name = active_file.read_text().strip()
        checks.append(("Active stack", True, active_name))
    else:
        checks.append(
            ("Active stack", False, "No active stack — run: agentomatic stack use <name>")
        )

    if HAS_RICH:
        table = Table(title="🩺 Environment Health Check", show_lines=True)
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for check_name, ok, detail in checks:
            status = "[green]✅[/green]" if ok else "[red]❌[/red]"
            style = "" if ok else "dim"
            table.add_row(check_name, status, f"[{style}]{detail}[/{style}]" if style else detail)

        console.print(table)

        all_ok = all(ok for _, ok, _ in checks[:6])  # Core deps only
        if all_ok:
            logger.success("All core dependencies satisfied!")
        else:
            logger.error("Some core dependencies are missing")
    else:
        click.echo("🩺 Environment Health Check")
        for check_name, ok, detail in checks:
            status = "✅" if ok else "❌"
            click.echo(f"  {status} {check_name}: {detail}")


# =====================================================================
# UI — Launch debug UI
# =====================================================================


@cli.command()
@click.option("--host", default="localhost", help="Platform API host")
@click.option("--port", type=int, default=8000, help="Platform API port")
@click.option("--ui-port", type=int, default=8001, help="UI port")
def ui(host: str, port: int, ui_port: int) -> None:
    """Launch the Chainlit debug UI."""
    _print_banner()

    from agentomatic.ui import is_available

    if not is_available():
        logger.error("Chainlit not installed")
        click.echo("   Install: pip install agentomatic[ui]")
        sys.exit(1)

    logger.success("Launching debug UI...")
    click.echo(f"   Platform API: http://{host}:{port}")

    import subprocess

    chat_path = Path(__file__).parent.parent / "ui" / "chat.py"
    subprocess.run(
        [sys.executable, "-m", "chainlit", "run", str(chat_path), "--port", str(ui_port)],
        env={
            **__import__("os").environ,
            "AGENTOMATIC_API_URL": f"http://{host}:{port}",
        },
    )


# =====================================================================
# OPTIMIZE — Prompt optimization
# =====================================================================


@cli.command()
@click.argument("agent")
@click.option("--dataset", "-d", required=True, help="Path to JSONL/CSV dataset")
@click.option("--metrics", "-m", default="exact_match", help="Comma-separated metrics")
@click.option(
    "--strategy",
    "-s",
    default="iterative_rewrite",
    type=click.Choice(["iterative_rewrite", "few_shot", "chain_of_thought"]),
    help="Optimization strategy",
)
@click.option("--max-iterations", type=int, default=10, help="Max iterations")
@click.option("--target-score", type=float, default=0.9, help="Target score")
@click.option("--rewrite-llm", default=None, help="LLM for prompt rewriting")
@click.option("--eval-llm", default=None, help="LLM for evaluation")
@click.option("--llm", default="ollama/mistral:7b", help="Default LLM")
@click.option("--patience", type=int, default=3, help="Early stopping patience")
@click.option("--prompt", default=None, help="Initial prompt (overrides prompts.json)")
@click.option("--no-report", is_flag=True, help="Skip HTML report generation")
@click.option("--apply", "auto_apply", is_flag=True, help="Auto-apply best prompt")
@click.option("--host", default="http://localhost:8000", help="Platform API base")
def optimize(
    agent: str,
    dataset: str,
    metrics: str,
    strategy: str,
    max_iterations: int,
    target_score: float,
    rewrite_llm: str | None,
    eval_llm: str | None,
    llm: str,
    patience: int,
    prompt: str | None,
    no_report: bool,
    auto_apply: bool,
    host: str,
) -> None:
    """Run prompt optimization for an agent."""
    import asyncio

    try:
        from agentomatic.optimize import Dataset, PromptOptimizer
    except ImportError:
        logger.error("Optimization module not available.")
        click.echo("Install: pip install agentomatic[optimize]")
        return

    # Load dataset
    if dataset.endswith(".csv"):
        ds = Dataset.from_csv(dataset)
    else:
        ds = Dataset.from_jsonl(dataset)

    logger.info(f"Dataset: {len(ds)} points from {dataset}")

    # Parse metrics
    metric_list = [m.strip() for m in metrics.split(",")]

    # Create optimizer
    optimizer = PromptOptimizer(
        agent=agent,
        metrics=metric_list,  # type: ignore[arg-type]
        llm=llm,
        rewrite_llm=rewrite_llm,
        eval_llm=eval_llm,
        strategy=strategy,
        api_base=host,
        auto_report=not no_report,
    )

    # Run optimization
    result = asyncio.run(
        optimizer.optimize(
            dataset=ds,
            initial_prompt=prompt,
            max_iterations=max_iterations,
            target_score=target_score,
            patience=patience,
        )
    )

    # Print report
    click.echo(result.report())

    # Auto-apply if requested
    if auto_apply:
        version = result.apply()
        logger.success(f"Applied as '{version}'")


# =====================================================================
# STACK — Multi-environment stack management
# =====================================================================


@cli.group()
def stack():
    """Manage environment stacks (local, remote, custom)."""
    pass


@stack.command("init")
@click.option(
    "--dir", "-d", "stacks_dir", default="stacks", help="Stacks directory (default: stacks)"
)
def stack_init(stacks_dir: str) -> None:
    """Create default stack configuration files."""
    from agentomatic.stacks.defaults import get_default_stack_yaml

    stacks_path = Path(stacks_dir)
    stacks_path.mkdir(parents=True, exist_ok=True)

    created = []
    for stack_name in ["local", "remote"]:
        file_path = stacks_path / f"{stack_name}.yaml"
        if file_path.exists():
            _print_warning(f"Stack '{stack_name}' already exists at {file_path}")
            continue
        content = get_default_stack_yaml(stack_name)
        file_path.write_text(content)
        created.append(stack_name)

    if created:
        _print_success(f"Created stack(s): {', '.join(created)} in {stacks_path}/")
    else:
        _echo("All default stacks already exist.")

    _echo("")
    _echo(
        "  Next: [cyan]agentomatic stack use local[/cyan]"
        if HAS_RICH
        else "  Next: agentomatic stack use local"
    )


@stack.command("list")
@click.option("--dir", "-d", "stacks_dir", default="stacks", help="Stacks directory")
def stack_list(stacks_dir: str) -> None:
    """List available stacks."""
    stacks_path = Path(stacks_dir)
    _print_banner()

    if not stacks_path.exists():
        _print_warning(f"No stacks directory found at {stacks_path}")
        _echo("  Run: agentomatic stack init")
        return

    yaml_files = sorted(stacks_path.glob("*.yaml")) + sorted(stacks_path.glob("*.yml"))
    if not yaml_files:
        _print_warning("No stack files found")
        return

    # Check active stack
    active = ""
    active_file = Path(".agentomatic-stack")
    if active_file.exists():
        active = active_file.read_text().strip()

    if HAS_RICH:
        table = Table(title="📦 Available Stacks", show_lines=True)
        table.add_column("Name", style="bold cyan")
        table.add_column("Active", justify="center")
        table.add_column("Path")

        for f in yaml_files:
            name = f.stem
            is_active = "✅" if name == active else "—"
            table.add_row(name, is_active, str(f))

        console.print(table)
    else:
        click.echo("📦 Available Stacks:")
        for f in yaml_files:
            name = f.stem
            marker = " (active)" if name == active else ""
            click.echo(f"  • {name}{marker} — {f}")


@stack.command("show")
@click.argument("name")
@click.option("--dir", "-d", "stacks_dir", default="stacks", help="Stacks directory")
def stack_show(name: str, stacks_dir: str) -> None:
    """Show the contents of a stack configuration."""
    stacks_path = Path(stacks_dir)
    stack_file = stacks_path / f"{name}.yaml"
    if not stack_file.exists():
        stack_file = stacks_path / f"{name}.yml"

    if not stack_file.exists():
        _print_error(f"Stack '{name}' not found in {stacks_path}")
        return

    content = stack_file.read_text()
    if HAS_RICH:
        from rich.syntax import Syntax

        console.print(
            Panel(
                Syntax(content, "yaml", theme="monokai"),
                title=f"📦 Stack: {name}",
                border_style="cyan",
            )
        )
    else:
        click.echo(f"📦 Stack: {name}")
        click.echo(content)


@stack.command("use")
@click.argument("name")
@click.option("--dir", "-d", "stacks_dir", default="stacks", help="Stacks directory")
def stack_use(name: str, stacks_dir: str) -> None:
    """Set the active stack for this project."""
    stacks_path = Path(stacks_dir)
    stack_file = stacks_path / f"{name}.yaml"
    if not stack_file.exists():
        stack_file = stacks_path / f"{name}.yml"

    if not stack_file.exists():
        _print_error(f"Stack '{name}' not found in {stacks_path}")
        return

    active_file = Path(".agentomatic-stack")
    active_file.write_text(name)
    _print_success(f"Active stack set to '{name}'")
    _echo(f"  Stack file: {stack_file}")


# =====================================================================
# Pipeline Commands
# =====================================================================


@cli.group()
def pipeline():
    """Manage and execute pipelines."""


@pipeline.command("list")
@click.option("--dir", "pipelines_dir", default=".", help="Project root directory")
def pipeline_list(pipelines_dir: str) -> None:
    """List discovered pipelines."""
    from pathlib import Path

    try:
        from agentomatic.pipelines.loader import PipelineLoader
    except ImportError:
        _print_error("Pipeline module not available.")
        return

    root = Path(pipelines_dir).resolve()
    pipelines: dict = {}
    for d in [root / "pipelines", root / "agents"]:
        if d.exists():
            pipelines.update(PipelineLoader.discover_pipelines(d))

    if not pipelines:
        _echo("No pipelines found.")
        _echo("  Create a pipeline.yaml in pipelines/ or agents/<name>/")
        return

    try:
        from rich.table import Table

        table = Table(title="📋 Discovered Pipelines")
        table.add_column("Name", style="cyan bold")
        table.add_column("Version")
        table.add_column("Steps", style="magenta")
        table.add_column("Agents", style="green")
        table.add_column("Description")

        for _name, config in sorted(pipelines.items()):
            agents = sorted(config.get_agent_names())
            table.add_row(
                config.name,
                config.version,
                str(len(config.steps)),
                ", ".join(agents) if agents else "—",
                config.description[:60] or "—",
            )
        console.print(table)
    except ImportError:
        for name, config in sorted(pipelines.items()):
            agents = sorted(config.get_agent_names())
            _echo(f"  {name} (v{config.version}) — {len(config.steps)} steps")
            if agents:
                _echo(f"    agents: {', '.join(agents)}")


@pipeline.command("validate")
@click.argument("name")
@click.option("--dir", "pipelines_dir", default=".", help="Project root directory")
@click.option("--agents-dir", default="agents", help="Agents directory")
def pipeline_validate(name: str, pipelines_dir: str, agents_dir: str) -> None:
    """Validate a pipeline configuration."""
    import sys
    from pathlib import Path

    try:
        from agentomatic.core.registry import AgentRegistry
        from agentomatic.pipelines.engine import PipelineEngine
        from agentomatic.pipelines.loader import PipelineLoader
    except ImportError:
        _print_error("Pipeline module not available.")
        return

    root = Path(pipelines_dir).resolve()
    pipelines: dict = {}
    for d in [root / "pipelines", root / agents_dir]:
        if d.exists():
            pipelines.update(PipelineLoader.discover_pipelines(d))

    if name not in pipelines:
        _print_error(f"Pipeline '{name}' not found. Available: {list(pipelines.keys())}")
        return

    config = pipelines[name]
    registry = AgentRegistry()
    ad = root / agents_dir
    if ad.exists():
        parent = str(ad.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        registry.discover(ad, ad.name)

    engine = PipelineEngine(config, registry)
    errors = engine.validate()

    if errors:
        _print_error(f"Pipeline '{name}' has {len(errors)} error(s):")
        for err in errors:
            _echo(f"  ❌ {err}")
    else:
        _print_success(f"Pipeline '{name}' is valid ✅")
        _echo(f"  Steps: {', '.join(config.step_names)}")
        _echo(f"  Agents: {', '.join(sorted(config.get_agent_names()))}")


@pipeline.command("visualize")
@click.argument("name")
@click.option("--dir", "pipelines_dir", default=".", help="Project root directory")
def pipeline_visualize(name: str, pipelines_dir: str) -> None:
    """Print a Mermaid diagram of the pipeline."""
    import sys
    from pathlib import Path

    try:
        from agentomatic.core.registry import AgentRegistry
        from agentomatic.pipelines.engine import PipelineEngine
        from agentomatic.pipelines.loader import PipelineLoader
    except ImportError:
        _print_error("Pipeline module not available.")
        return

    root = Path(pipelines_dir).resolve()
    pipelines: dict = {}
    for d in [root / "pipelines", root / "agents"]:
        if d.exists():
            pipelines.update(PipelineLoader.discover_pipelines(d))

    if name not in pipelines:
        _print_error(f"Pipeline '{name}' not found.")
        return

    config = pipelines[name]
    registry = AgentRegistry()
    ad = root / "agents"
    if ad.exists():
        parent = str(ad.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        registry.discover(ad, ad.name)

    engine = PipelineEngine(config, registry)
    mermaid = engine.visualize()

    try:
        from rich.panel import Panel
        from rich.syntax import Syntax

        console.print(
            Panel(
                Syntax(mermaid, "mermaid"),
                title=f"🔁 Pipeline: {name}",
                border_style="blue",
            )
        )
    except ImportError:
        _echo(f"--- Pipeline: {name} ---")
        _echo(mermaid)


@pipeline.command("run")
@click.argument("name")
@click.option("--input", "input_json", default="{}", help="JSON input")
@click.option("--dir", "pipelines_dir", default=".", help="Project root")
@click.option("--agents-dir", default="agents", help="Agents directory")
def pipeline_run(name: str, input_json: str, pipelines_dir: str, agents_dir: str) -> None:
    """Execute a pipeline from the CLI."""
    import asyncio
    import json
    import sys
    from pathlib import Path

    try:
        from agentomatic.core.registry import AgentRegistry
        from agentomatic.pipelines.engine import PipelineEngine
        from agentomatic.pipelines.loader import PipelineLoader
    except ImportError:
        _print_error("Pipeline module not available.")
        return

    root = Path(pipelines_dir).resolve()
    pipelines: dict = {}
    for d in [root / "pipelines", root / agents_dir]:
        if d.exists():
            pipelines.update(PipelineLoader.discover_pipelines(d))

    if name not in pipelines:
        _print_error(f"Pipeline '{name}' not found.")
        return

    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as exc:
        _print_error(f"Invalid JSON input: {exc}")
        return

    config = pipelines[name]
    registry = AgentRegistry()
    ad = root / agents_dir
    if ad.exists():
        parent = str(ad.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        registry.discover(ad, ad.name)

    engine = PipelineEngine(config, registry)
    errors = engine.validate()
    if errors:
        _print_error("Pipeline validation failed:")
        for err in errors:
            _echo(f"  ❌ {err}")
        return

    _echo(f"🚀 Running pipeline '{name}'...")
    result = asyncio.run(engine.run(input_data))

    try:
        from rich.panel import Panel
        from rich.syntax import Syntax

        output_json = json.dumps(result.model_dump(), indent=2, default=str)
        status = "✅" if result.succeeded else "❌"
        console.print(
            Panel(
                Syntax(output_json, "json"),
                title=(f"{status} Pipeline: {name} ({result.duration_ms:.0f}ms)"),
                border_style="green" if result.succeeded else "red",
            )
        )
    except ImportError:
        _echo(f"Status: {result.status.value}")
        _echo(f"Duration: {result.duration_ms:.0f}ms")
        _echo(f"Output: {result.output}")


# =====================================================================
# Backward-compatible main() entry point
# =====================================================================


def main() -> None:
    """CLI entry point (backward compatible)."""
    cli()


if __name__ == "__main__":
    cli()
