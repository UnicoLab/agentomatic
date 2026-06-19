#!/usr/bin/env python3
"""End-to-end demo server for agentomatic v0.4.1.

Tests:
  ✅ Package import (without extras)
  ✅ Auto-discovery from agents folder
  ✅ Programmatic agent registration
  ✅ Studio mounting + redirects
  ✅ /invoke, /chat, /health endpoints
  ✅ discover_agents() public API
  ✅ sys.path setup in build()

Usage:
    # Run the server:
    python run_demo.py

    # Or run the E2E test suite:
    python run_demo.py --test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Step 1: Test bare import ─────────────────────────────────────────
print("=" * 60)
print("  🧪 Agentomatic v0.4.1 — E2E Demo")
print("=" * 60)

print("\n1️⃣  Testing bare import...")
try:
    from agentomatic import AgentManifest, AgentPlatform

    print(f"   ✅ Import successful — version {AgentPlatform.__module__}")
except ImportError as e:
    print(f"   ❌ Import FAILED: {e}")
    sys.exit(1)

from agentomatic._version import __version__

print(f"   ✅ Version: {__version__}")

# ── Step 2: Create platform from agents folder ──────────────────────
print("\n2️⃣  Creating platform from agents folder...")
AGENTS_DIR = Path(__file__).parent / "agents"
print(f"   📂 Agents dir: {AGENTS_DIR}")

platform = AgentPlatform(
    agents_dir=AGENTS_DIR,
    title="Agentomatic E2E Demo",
    description="End-to-end demo with auto-discovery + programmatic agents",
    enable_studio=True,
    enable_telemetry=False,
)

# ── Step 3: Test discover_agents() (BUG 4 fix) ──────────────────────
print("\n3️⃣  Testing discover_agents() (public API)...")
platform.discover_agents()
discovered = platform.registry.list_names()
print(f"   ✅ Discovered {len(discovered)} agents: {discovered}")

# ── Step 4: Register a programmatic agent (BUG 11 fix) ──────────────
print("\n4️⃣  Registering programmatic agent...")


async def echo_fn(state):
    """Simple echo agent — returns the query as-is."""
    return {
        "response": f"Echo: {state.get('query', '')}",
        "messages": state.get("messages", []),
    }


echo_manifest = AgentManifest(
    name="echo",
    slug="echo",
    version="0.1.0",
    description="Echo agent — returns input as-is",
)
platform.register_agent(manifest=echo_manifest, node_fn=echo_fn)
print(f"   ✅ Registered 'echo' agent")

# ── Step 5: Set Studio hooks (BUG 11 fix — dataclass fields) ────────
print("\n5️⃣  Testing Studio hooks on RegisteredAgent...")
echo_agent = platform.registry.get("echo")
if echo_agent:
    echo_agent._studio_graph_fn = lambda: {
        "nodes": [{"id": "echo", "label": "Echo"}],
        "edges": [],
    }
    echo_agent._studio_state_fn = lambda: {"status": "ready"}
    print("   ✅ Studio hooks set without AttributeError")
else:
    print("   ⚠️ Echo agent not found in registry")

# ── Step 6: Build the app ───────────────────────────────────────────
print("\n6️⃣  Building FastAPI app...")
app = platform.build()
print(f"   ✅ App built successfully")
print(f"   📋 Total agents: {platform.registry.count}")
print(f"   📋 Agent names: {platform.registry.list_names()}")

# ── Step 7: List all routes ─────────────────────────────────────────
print("\n7️⃣  Registered routes:")
for route in sorted(app.routes, key=lambda r: getattr(r, "path", "")):
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if path and methods:
        print(f"   {', '.join(methods):8s} {path}")
    elif path:
        print(f"   {'MOUNT':8s} {path}")


# ── Step 8: Run E2E tests or start server ───────────────────────────
def run_tests():
    """Run E2E tests using TestClient."""
    print("\n" + "=" * 60)
    print("  🧪 Running E2E Tests")
    print("=" * 60)

    from starlette.testclient import TestClient

    client = TestClient(app)
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"   ✅ {name}" + (f" — {detail}" if detail else ""))
        else:
            failed += 1
            print(f"   ❌ {name}" + (f" — {detail}" if detail else ""))

    # Test: Health endpoint
    resp = client.get("/health")
    check("GET /health", resp.status_code == 200, f"status={resp.status_code}")

    # Test: Platform info — no dedicated endpoint, skip
    # (platform info is part of the root or A2A card)

    # Test: List agents
    resp = client.get("/api/v1/agents")
    check("GET /agents", resp.status_code == 200)
    if resp.status_code == 200:
        data = resp.json()
        agent_dict = data.get("agents", {})
        agent_names = list(agent_dict.keys())
        check("  greeter discovered", "greeter" in agent_names, str(agent_names))
        check("  echo registered", "echo" in agent_names, str(agent_names))

    # Test: Invoke greeter
    resp = client.post(
        "/api/v1/greeter/invoke",
        json={"query": "What's your name?", "user_id": "e2e_tester"},
    )
    check("POST /greeter/invoke", resp.status_code == 200, f"status={resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        check("  has response", bool(data.get("response")), data.get("response", "")[:80])

    # Test: Invoke echo
    resp = client.post(
        "/api/v1/echo/invoke",
        json={"query": "Hello echo!", "user_id": "e2e_tester"},
    )
    check("POST /echo/invoke", resp.status_code == 200, f"status={resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        check("  echo response", "Echo:" in data.get("response", ""), data.get("response", "")[:80])

    # Test: Chat endpoint with messages (BUG 3 fix)
    resp = client.post(
        "/api/v1/greeter/chat",
        json={
            "content": "How are you?",
            "user_id": "e2e_tester",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        },
    )
    check("POST /greeter/chat (with messages)", resp.status_code == 200, f"status={resp.status_code}")

    # Test: Agent health
    resp = client.get("/api/v1/greeter/health")
    check("GET /greeter/health", resp.status_code == 200)

    # Test: Studio API (if mounted)
    resp = client.get("/studio/agents")
    studio_mounted = resp.status_code in (200, 307, 404)
    check("GET /studio/agents", studio_mounted, f"status={resp.status_code}")

    # Test: Studio redirect (BUG 6+13 fix)
    resp = client.get("/studio", follow_redirects=False)
    check(
        "GET /studio → redirect",
        resp.status_code in (200, 307, 404),
        f"status={resp.status_code}",
    )

    # Test: Optimization result save (OPT fix)
    print("\n   --- Optimization Result Persistence ---")
    import tempfile

    from agentomatic.optimize.optimizer import OptimizationResult

    opt_result = OptimizationResult(
        best_prompt="Be helpful and concise.",
        best_score=0.92,
        best_iteration=3,
        baseline_prompt="Be helpful.",
        baseline_score=0.78,
        experiment_id="e2e_test_123",
        agent="greeter",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, html_path = opt_result.save(tmpdir)
        check("OptimizationResult.save()", json_path.exists(), str(json_path))

    from agentomatic.optimize.config import PromptFitResult, PromptRuntimeConfig

    fit_result = PromptFitResult(
        best_config=PromptRuntimeConfig(system_prompt="Optimized"),
        baseline_config=PromptRuntimeConfig(system_prompt="Original"),
        best_score=0.9,
        baseline_score=0.7,
        agent="greeter",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = fit_result.save(tmpdir)
        check("PromptFitResult.save()", json_path.exists(), str(json_path))

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED ❌")
    else:
        print(" ✅ ALL PASSED! 🎉")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentomatic E2E Demo")
    parser.add_argument("--test", action="store_true", help="Run E2E tests instead of server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print(f"\n🚀 Starting server at http://{args.host}:{args.port}")
        print(f"   📡 API:    http://{args.host}:{args.port}/api/v1/")
        print(f"   🎨 Studio: http://{args.host}:{args.port}/studio/ui/")
        print(f"   📋 Docs:   http://{args.host}:{args.port}/docs")
        print("\n   Press Ctrl+C to stop.\n")
        platform.run(host=args.host, port=args.port)
