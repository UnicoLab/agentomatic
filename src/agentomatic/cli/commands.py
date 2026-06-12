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

console = Console() if HAS_RICH else None


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
    if HAS_RICH:
        console.print(
            Panel.fit(
                "[bold magenta]⚡ agentomatic[/bold magenta]\n[dim]Drop agents, not code[/dim]",
                border_style="magenta",
            )
        )
    else:
        click.echo("⚡ agentomatic — Drop agents, not code")
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


@click.group()
def cli():
    """⚡ Agentomatic — Drop agents, not code."""
    pass


# =====================================================================
# INIT — Scaffold a new agent
# =====================================================================


@cli.command()
@click.argument("name")
@click.option("--dir", "-d", "agents_dir", default="agents", help="Agents directory (default: agents)")
@click.option(
    "--template",
    "-t",
    type=click.Choice(["basic", "full", "rag", "chatbot", "custom"]),
    default=None,
    help="Template to use (default: interactive selection)",
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def init(name: str, agents_dir: str, template: str | None, force: bool) -> None:
    """Scaffold a new agent with template selection."""
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

    if HAS_RICH:
        console.print(
            Panel(
                f"[bold]Next steps:[/bold]\n\n"
                f"  1. [cyan]cd {agents_dir}[/cyan]\n"
                f"  2. Edit [yellow]{name}/nodes.py[/yellow] with your logic\n"
                f"  3. [cyan]agentomatic run[/cyan] to start\n"
                f"  4. [cyan]agentomatic test {name}[/cyan] to test\n"
                f"  5. Open [blue]http://localhost:8000/docs[/blue] for API docs",
                title="🚀 What's next?",
                border_style="green",
            )
        )
    else:
        click.echo("Next steps:")
        click.echo(f"  1. Edit {name}/nodes.py with your logic")
        click.echo("  2. agentomatic run")
        click.echo(f"  3. agentomatic test {name}")


# =====================================================================
# RUN — Start the platform
# =====================================================================


@cli.command()
@click.option("--agents-dir", default="agents", help="Agents directory")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", type=int, default=8000, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--title", default=None, help="Platform title")
@click.option("--log-level", default="INFO", help="Log level")
@click.option("--with-ui", "--ui", is_flag=True, help="Enable Chainlit debug UI at /chat")
def run(
    agents_dir: str,
    host: str,
    port: int,
    reload: bool,
    title: str | None,
    log_level: str,
    with_ui: bool,
) -> None:
    """Run the platform with Rich status output."""
    _print_banner()
    logger.info(f"Starting platform from {agents_dir}...")

    from agentomatic import AgentPlatform

    kwargs: dict[str, Any] = {
        "title": title or "Agentomatic Platform",
        "log_level": log_level,
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
# LIST — Show discovered agents
# =====================================================================


@cli.command("list")
@click.option("--agents-dir", default="agents", help="Agents directory")
def list_agents(agents_dir: str) -> None:
    """List agents in a directory with Rich table."""
    agents_path = Path(agents_dir)
    _print_banner()

    if not agents_path.exists():
        _print_error(f"Directory not found: {agents_path}")
        sys.exit(1)

    agents = []
    for entry in sorted(agents_path.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists() and not entry.name.startswith("_"):
            info: dict[str, Any] = {"name": entry.name, "path": str(entry)}

            # Try to read manifest
            try:
                spec = importlib.util.spec_from_file_location(
                    f"_agent_{entry.name}",
                    entry / "__init__.py",
                )
                if spec and spec.loader:
                    _ = importlib.util.module_from_spec(spec)
                    # Don't actually exec — just check for manifest in source
                    source = (entry / "__init__.py").read_text()
                    if "AgentManifest" in source:
                        info["has_manifest"] = True
                    if "graph.py" in [f.name for f in entry.iterdir()]:
                        info["has_graph"] = True
                    info["files"] = len([f for f in entry.iterdir() if f.is_file()])
            except Exception:
                pass

            agents.append(info)

    if not agents:
        _print_warning(f"No agents found in {agents_path}")
        return

    if HAS_RICH:
        table = Table(title=f"🤖 Agents in {agents_path}", show_lines=True)
        table.add_column("Name", style="bold cyan")
        table.add_column("Files", justify="center")
        table.add_column("Manifest", justify="center")
        table.add_column("Graph", justify="center")

        for a in agents:
            table.add_row(
                a["name"],
                str(a.get("files", "?")),
                "✅" if a.get("has_manifest") else "❌",
                "✅" if a.get("has_graph") else "—",
            )

        console.print(table)
    else:
        click.echo(f"📂 {agents_path}")
        for a in agents:
            click.echo(f"  🤖 {a['name']} ({a.get('files', '?')} files)")

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
    """Inspect an agent's structure and configuration."""
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
    else:
        click.echo(f"🔍 Agent: {name}")
        click.echo(f"   Path: {target}")
        click.echo(f"   Files: {len(files)}")
        for f in files:
            click.echo(f"   📄 {f.relative_to(target)}")


# =====================================================================
# DOCTOR — Environment health check
# =====================================================================


@cli.command()
@click.option("--agents-dir", default="agents", help="Agents directory")
def doctor(agents_dir: str) -> None:
    """Check environment health and dependencies."""
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
            [d for d in agents_path.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
        )
        checks.append(("Agents directory", True, f"{count} agent(s) in {agents_path}"))
    else:
        checks.append(("Agents directory", False, f"Not found: {agents_path}"))

    if HAS_RICH:
        table = Table(title="🩺 Environment Health Check", show_lines=True)
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for check_name, ok, detail in checks:
            status = "[green]✅[/green]" if ok else "[red]❌[/red]"
            style = "" if ok else "dim"
            table.add_row(
                check_name, status, f"[{style}]{detail}[/{style}]" if style else detail
            )

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
        metrics=metric_list,
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
# Backward-compatible main() entry point
# =====================================================================


def main() -> None:
    """CLI entry point (backward compatible)."""
    cli()


if __name__ == "__main__":
    cli()
