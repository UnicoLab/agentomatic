"""Agentomatic CLI commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a new agent."""
    name = args.name
    target = Path(args.dir) / name
    target.mkdir(parents=True, exist_ok=True)

    # __init__.py
    (target / "__init__.py").write_text(f'''"""Agent: {name}."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\nfrom agentomatic import AgentManifest\n\nmanifest = AgentManifest(\n    name="{name}",\n    slug="agent-{name}",\n    description="{name.title()} agent",\n    intent_keywords=["{name}"],\n)\n\n\nasync def node_fn(state: dict[str, Any]) -> dict[str, Any]:\n    from .graph import get_graph\n    return await get_graph().ainvoke(state)\n''')

    # graph.py
    (target / "graph.py").write_text(f'''"""LangGraph graph for {name}."""\nfrom __future__ import annotations\n\nfrom functools import lru_cache\n\nfrom langgraph.graph import END, StateGraph\n\nfrom agentomatic import BaseAgentState\n\nfrom . import nodes\n\n\ndef build_graph() -> StateGraph:\n    g = StateGraph(BaseAgentState)\n    g.add_node("process", nodes.process)\n    g.set_entry_point("process")\n    g.add_edge("process", END)\n    return g\n\n\n@lru_cache(maxsize=1)\ndef get_graph():\n    return build_graph().compile()\n''')

    # nodes.py
    (target / "nodes.py").write_text(f'''"""Node functions for {name}."""\nfrom __future__ import annotations\n\nfrom typing import Any\n\n\nasync def process(state: dict[str, Any]) -> dict[str, Any]:\n    query = state.get("current_query", "")\n    return {{\n        "response": f"Hello from {name}! You asked: {{query}}",\n        "agent_type": "agent-{name}",\n    }}\n''')

    print(f"✅ Created agent '{name}' at {target}")
    print(f"   Files: __init__.py, graph.py, nodes.py")
    print(f"   Next: restart your server to auto-discover it!")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the platform."""
    from agentomatic import AgentPlatform
    platform = AgentPlatform.from_folder(
        args.agents_dir,
        title=args.title or "Agentomatic Platform",
        log_level=args.log_level,
    )
    platform.run(host=args.host, port=args.port, reload=args.reload)


def cmd_list(args: argparse.Namespace) -> None:
    """List agents in a directory."""
    agents_dir = Path(args.agents_dir)
    if not agents_dir.exists():
        print(f"❌ Directory not found: {agents_dir}")
        sys.exit(1)

    print(f"📂 Scanning {agents_dir}...")
    for entry in sorted(agents_dir.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists() and not entry.name.startswith("_"):
            print(f"  🤖 {entry.name}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="agentomatic",
        description="Agentomatic — Drop agents, not code",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Scaffold a new agent")
    p_init.add_argument("name", help="Agent name")
    p_init.add_argument("--dir", default="agents", help="Agents directory")

    # run
    p_run = sub.add_parser("run", help="Run the platform")
    p_run.add_argument("--agents-dir", default="agents", help="Agents directory")
    p_run.add_argument("--host", default="0.0.0.0")
    p_run.add_argument("--port", type=int, default=8000)
    p_run.add_argument("--reload", action="store_true")
    p_run.add_argument("--title", default=None)
    p_run.add_argument("--log-level", default="INFO")

    # list
    p_list = sub.add_parser("list", help="List discovered agents")
    p_list.add_argument("--agents-dir", default="agents", help="Agents directory")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
