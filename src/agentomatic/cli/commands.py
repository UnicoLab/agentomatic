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

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

# Graceful Rich fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


# =====================================================================
# Utilities
# =====================================================================

def _print(msg: str) -> None:
    """Print with Rich if available, plain otherwise."""
    if console:
        console.print(msg)
    else:
        print(msg)


def _print_banner() -> None:
    """Show the agentomatic banner."""
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold magenta]⚡ agentomatic[/bold magenta]\n"
            "[dim]Drop agents, not code[/dim]",
            border_style="magenta",
        ))
    else:
        print("⚡ agentomatic — Drop agents, not code")
        print()


def _print_success(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold green]✅ {msg}[/bold green]")
    else:
        print(f"✅ {msg}")


def _print_error(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold red]❌ {msg}[/bold red]")
    else:
        print(f"❌ {msg}")


def _print_warning(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold yellow]⚠️  {msg}[/bold yellow]")
    else:
        print(f"⚠️  {msg}")


# =====================================================================
# INIT — Scaffold a new agent
# =====================================================================

def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a new agent with template selection."""
    from .templates import TEMPLATES, get_template_files

    name = args.name
    agents_dir = Path(args.dir)
    target = agents_dir / name

    _print_banner()

    # Template selection
    template = args.template

    if not template:
        # Interactive selection if questionary is available
        try:
            import questionary
            choices = [
                questionary.Choice(f"{k} — {v}", value=k)
                for k, v in TEMPLATES.items()
            ]
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
        if not args.force:
            response = input("Overwrite? [y/N]: ").strip().lower()
            if response != "y":
                _print("Cancelled")
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
        print(f"📁 {target}")
        for rel_path in sorted(files.keys()):
            print(f"  📄 {rel_path}")

    for rel_path, content in files.items():
        file_path = target / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    _print("")
    _print_success(f"Created agent '{name}' with template '{template}'")
    _print(f"   📍 Location: {target}")
    _print(f"   📦 Files: {len(files)}")
    _print("")

    if HAS_RICH:
        console.print(Panel(
            f"[bold]Next steps:[/bold]\n\n"
            f"  1. [cyan]cd {agents_dir}[/cyan]\n"
            f"  2. Edit [yellow]{name}/nodes.py[/yellow] with your logic\n"
            f"  3. [cyan]agentomatic run[/cyan] to start\n"
            f"  4. [cyan]agentomatic test {name}[/cyan] to test\n"
            f"  5. Open [blue]http://localhost:8000/docs[/blue] for API docs",
            title="🚀 What's next?",
            border_style="green",
        ))
    else:
        print("Next steps:")
        print(f"  1. Edit {name}/nodes.py with your logic")
        print(f"  2. agentomatic run")
        print(f"  3. agentomatic test {name}")


# =====================================================================
# RUN — Start the platform
# =====================================================================

def cmd_run(args: argparse.Namespace) -> None:
    """Run the platform with Rich status output."""
    _print_banner()
    _print(f"🚀 Starting platform from [bold]{args.agents_dir}[/bold]..." if HAS_RICH
           else f"🚀 Starting platform from {args.agents_dir}...")

    from agentomatic import AgentPlatform

    kwargs: dict[str, Any] = {
        "title": args.title or "Agentomatic Platform",
        "log_level": args.log_level,
    }

    # Auto-detect and enable UI
    if args.with_ui or args.ui:
        from agentomatic.ui import is_available
        if is_available():
            _print_success("Debug UI will be available at /chat")
        else:
            _print_warning("Chainlit not installed. Install: pip install agentomatic[ui]")

    platform = AgentPlatform.from_folder(args.agents_dir, **kwargs)

    # Mount UI if requested
    if args.with_ui or args.ui:
        @platform.on_startup
        async def _mount_ui():
            from agentomatic.ui import mount
            if platform._app:
                mount(platform._app)

    platform.run(host=args.host, port=args.port, reload=args.reload)


# =====================================================================
# LIST — Show discovered agents
# =====================================================================

def cmd_list(args: argparse.Namespace) -> None:
    """List agents in a directory with Rich table."""
    agents_dir = Path(args.agents_dir)
    _print_banner()

    if not agents_dir.exists():
        _print_error(f"Directory not found: {agents_dir}")
        sys.exit(1)

    agents = []
    for entry in sorted(agents_dir.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists() and not entry.name.startswith("_"):
            info: dict[str, Any] = {"name": entry.name, "path": str(entry)}

            # Try to read manifest
            try:
                spec = importlib.util.spec_from_file_location(
                    f"_agent_{entry.name}", entry / "__init__.py",
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
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
        _print_warning(f"No agents found in {agents_dir}")
        return

    if HAS_RICH:
        table = Table(title=f"🤖 Agents in {agents_dir}", show_lines=True)
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
        print(f"📂 {agents_dir}")
        for a in agents:
            print(f"  🤖 {a['name']} ({a.get('files', '?')} files)")

    _print(f"\n   Total: {len(agents)} agent(s)")


# =====================================================================
# TEST — Interactive agent testing
# =====================================================================

def cmd_test(args: argparse.Namespace) -> None:
    """Test an agent interactively in the terminal."""
    import asyncio

    _print_banner()
    agent_name = args.name
    base_url = f"http://{args.host}:{args.port}"

    _print(f"🧪 Testing agent: [bold cyan]{agent_name}[/bold cyan]" if HAS_RICH
           else f"🧪 Testing agent: {agent_name}")
    _print(f"   API: {base_url}/api/v1/{agent_name}/invoke")
    _print("   Type 'quit' or 'exit' to stop\n")

    async def _test_loop():
        import httpx
        async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
            # Health check
            try:
                resp = await client.get(f"/api/v1/{agent_name}/health")
                if resp.status_code == 200:
                    _print_success(f"Agent '{agent_name}' is healthy")
                else:
                    _print_error(f"Agent '{agent_name}' health check failed: {resp.status_code}")
                    return
            except httpx.ConnectError:
                _print_error(f"Cannot connect to {base_url}. Is the platform running?")
                _print("   Start with: agentomatic run")
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
                        f"/api/v1/{agent_name}/invoke",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    thread_id = data.get("thread_id", thread_id)

                    if HAS_RICH:
                        console.print(f"\n🤖 [bold green]{agent_name}[/bold green]: {data.get('response', '')}")
                        if data.get("steps_taken"):
                            console.print(f"   [dim]Steps: {' → '.join(data['steps_taken'])}[/dim]")
                        if data.get("suggestions"):
                            console.print(f"   [dim]Suggestions: {', '.join(data['suggestions'])}[/dim]")
                        console.print(f"   [dim]⏱ {data.get('duration_ms', 0):.0f}ms[/dim]")
                    else:
                        print(f"\n🤖 {agent_name}: {data.get('response', '')}")
                        if data.get("duration_ms"):
                            print(f"   ⏱ {data['duration_ms']:.0f}ms")

                except httpx.HTTPStatusError as exc:
                    _print_error(f"API Error: {exc.response.status_code}")
                except Exception as exc:
                    _print_error(f"Error: {exc}")

            _print("\n👋 Test session ended")

    asyncio.run(_test_loop())


# =====================================================================
# INSPECT — Show agent details
# =====================================================================

def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect an agent's structure and configuration."""
    _print_banner()

    target = Path(args.agents_dir) / args.name
    if not target.exists():
        _print_error(f"Agent not found: {target}")
        sys.exit(1)

    files = sorted([f for f in target.rglob("*") if f.is_file() and not f.name.startswith(".")])

    if HAS_RICH:
        # Header
        console.print(Panel(
            f"[bold cyan]{args.name}[/bold cyan]\n"
            f"[dim]{target}[/dim]",
            title="🔍 Agent Inspector",
            border_style="cyan",
        ))

        # File tree
        tree = Tree(f"[bold]📁 {args.name}/[/bold]")
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
            console.print(Panel(
                json.dumps(data, indent=2),
                title=f"prompts.json ({len(data)} versions)",
                border_style="blue",
            ))
    else:
        print(f"🔍 Agent: {args.name}")
        print(f"   Path: {target}")
        print(f"   Files: {len(files)}")
        for f in files:
            print(f"   📄 {f.relative_to(target)}")


# =====================================================================
# DOCTOR — Environment health check
# =====================================================================

def cmd_doctor(args: argparse.Namespace) -> None:
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
    agents_dir = Path(args.agents_dir)
    if agents_dir.exists():
        count = len([d for d in agents_dir.iterdir()
                     if d.is_dir() and (d / "__init__.py").exists()])
        checks.append(("Agents directory", True, f"{count} agent(s) in {agents_dir}"))
    else:
        checks.append(("Agents directory", False, f"Not found: {agents_dir}"))

    if HAS_RICH:
        table = Table(title="🩺 Environment Health Check", show_lines=True)
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for name, ok, detail in checks:
            status = "[green]✅[/green]" if ok else "[red]❌[/red]"
            style = "" if ok else "dim"
            table.add_row(name, status, f"[{style}]{detail}[/{style}]" if style else detail)

        console.print(table)

        all_ok = all(ok for _, ok, _ in checks[:6])  # Core deps only
        if all_ok:
            _print_success("All core dependencies satisfied!")
        else:
            _print_error("Some core dependencies are missing")
    else:
        print("🩺 Environment Health Check")
        for name, ok, detail in checks:
            status = "✅" if ok else "❌"
            print(f"  {status} {name}: {detail}")


# =====================================================================
# UI — Launch debug UI
# =====================================================================

def cmd_ui(args: argparse.Namespace) -> None:
    """Launch the Chainlit debug UI."""
    _print_banner()

    from agentomatic.ui import is_available

    if not is_available():
        _print_error("Chainlit not installed")
        _print("   Install: pip install agentomatic[ui]")
        sys.exit(1)

    _print_success("Launching debug UI...")
    _print(f"   Platform API: http://{args.host}:{args.port}")

    import subprocess
    chat_path = Path(__file__).parent.parent / "ui" / "chat.py"
    subprocess.run(
        [sys.executable, "-m", "chainlit", "run", str(chat_path), "--port", str(args.ui_port)],
        env={
            **__import__("os").environ,
            "AGENTOMATIC_API_URL": f"http://{args.host}:{args.port}",
        },
    )


# =====================================================================
# OPTIMIZE — Prompt optimization
# =====================================================================

def cmd_optimize(args: argparse.Namespace) -> None:
    """Handle optimize command."""
    import asyncio
    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        console = None

    try:
        from agentomatic.optimize import PromptOptimizer, Dataset
    except ImportError:
        print("ERROR: Optimization module not available.")
        print("Install: pip install agentomatic[optimize]")
        return

    # Load dataset
    dataset_path = args.dataset
    if dataset_path.endswith(".csv"):
        dataset = Dataset.from_csv(dataset_path)
    else:
        dataset = Dataset.from_jsonl(dataset_path)

    if console:
        console.print(f"[bold]Dataset:[/bold] {len(dataset)} points from {dataset_path}")

    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(",")]

    # Create optimizer
    optimizer = PromptOptimizer(
        agent=args.agent,
        metrics=metrics,
        llm=args.llm,
        rewrite_llm=args.rewrite_llm,
        eval_llm=args.eval_llm,
        strategy=args.strategy,
        api_base=args.host,
        auto_report=not args.no_report,
    )

    # Run optimization
    result = asyncio.run(optimizer.optimize(
        dataset=dataset,
        initial_prompt=args.prompt,
        max_iterations=args.max_iterations,
        target_score=args.target_score,
        patience=args.patience,
    ))

    # Print report
    print(result.report())

    # Auto-apply if requested
    if args.apply:
        version = result.apply()
        if console:
            console.print(f"\n[bold green]✅ Applied as '{version}'[/bold green]")


# =====================================================================
# Main CLI entry point
# =====================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="agentomatic",
        description="⚡ Agentomatic — Drop agents, not code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agentomatic init my_agent                    # Basic template
  agentomatic init my_agent --template full    # All files
  agentomatic init my_agent --template rag     # RAG pipeline
  agentomatic run                              # Start platform
  agentomatic run --with-ui                    # Start with debug UI
  agentomatic list                             # Show agents
  agentomatic test my_agent                    # Interactive test
  agentomatic inspect my_agent                 # Show details
  agentomatic doctor                           # Check environment
  agentomatic optimize my_agent -d qa.jsonl    # Optimize prompts
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- init ---
    p_init = sub.add_parser("init", help="Scaffold a new agent")
    p_init.add_argument("name", help="Agent name (snake_case)")
    p_init.add_argument("--dir", default="agents", help="Agents directory (default: agents)")
    p_init.add_argument(
        "--template", "-t",
        choices=["basic", "full", "rag", "chatbot", "custom"],
        default=None,
        help="Template to use (default: interactive selection)",
    )
    p_init.add_argument("--force", "-f", action="store_true", help="Overwrite existing files")

    # --- run ---
    p_run = sub.add_parser("run", help="Start the platform")
    p_run.add_argument("--agents-dir", default="agents", help="Agents directory")
    p_run.add_argument("--host", default="0.0.0.0")
    p_run.add_argument("--port", type=int, default=8000)
    p_run.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_run.add_argument("--title", default=None, help="Platform title")
    p_run.add_argument("--log-level", default="INFO")
    p_run.add_argument("--with-ui", "--ui", action="store_true", dest="with_ui",
                        help="Enable Chainlit debug UI at /chat")
    # compat alias
    p_run.set_defaults(ui=False)

    # --- list ---
    p_list = sub.add_parser("list", help="List discovered agents")
    p_list.add_argument("--agents-dir", default="agents", help="Agents directory")

    # --- test ---
    p_test = sub.add_parser("test", help="Test an agent interactively")
    p_test.add_argument("name", help="Agent name")
    p_test.add_argument("--host", default="localhost")
    p_test.add_argument("--port", type=int, default=8000)
    p_test.add_argument("--agents-dir", default="agents", help="Agents directory")

    # --- inspect ---
    p_inspect = sub.add_parser("inspect", help="Show agent details")
    p_inspect.add_argument("name", help="Agent name")
    p_inspect.add_argument("--agents-dir", default="agents", help="Agents directory")

    # --- doctor ---
    p_doctor = sub.add_parser("doctor", help="Check environment health")
    p_doctor.add_argument("--agents-dir", default="agents", help="Agents directory")

    # --- ui ---
    p_ui = sub.add_parser("ui", help="Launch debug UI standalone")
    p_ui.add_argument("--host", default="localhost", help="Platform API host")
    p_ui.add_argument("--port", type=int, default=8000, help="Platform API port")
    p_ui.add_argument("--ui-port", type=int, default=8001, help="UI port")

    # --- optimize ---
    p_optimize = sub.add_parser("optimize", help="Run prompt optimization")
    p_optimize.add_argument("agent", help="Agent name to optimize")
    p_optimize.add_argument("--dataset", "-d", required=True, help="Path to JSONL/CSV dataset")
    p_optimize.add_argument("--metrics", "-m", default="exact_match", help="Comma-separated metrics")
    p_optimize.add_argument("--strategy", "-s", default="iterative_rewrite",
                            choices=["iterative_rewrite", "few_shot", "chain_of_thought"],
                            help="Optimization strategy")
    p_optimize.add_argument("--max-iterations", type=int, default=10, help="Max iterations")
    p_optimize.add_argument("--target-score", type=float, default=0.9, help="Target score")
    p_optimize.add_argument("--rewrite-llm", default=None, help="LLM for prompt rewriting")
    p_optimize.add_argument("--eval-llm", default=None, help="LLM for evaluation")
    p_optimize.add_argument("--llm", default="ollama/mistral:7b", help="Default LLM")
    p_optimize.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    p_optimize.add_argument("--prompt", default=None, help="Initial prompt (overrides prompts.json)")
    p_optimize.add_argument("--no-report", action="store_true", help="Skip HTML report generation")
    p_optimize.add_argument("--apply", action="store_true", help="Auto-apply best prompt")
    p_optimize.add_argument("--host", default="http://localhost:8000", help="Platform API base")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "run": cmd_run,
        "list": cmd_list,
        "test": cmd_test,
        "inspect": cmd_inspect,
        "doctor": cmd_doctor,
        "ui": cmd_ui,
        "optimize": cmd_optimize,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        if HAS_RICH:
            _print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
