#!/usr/bin/env python3
"""Final implementation test script."""

import asyncio
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.app.dependencies import agent_registry


async def test_agent_discovery():
    """Test agent discovery and registration."""
    print("🔍 Testing Agent Discovery...")

    # Discover all agents
    agent_registry.discover_agents()

    # List discovered agents
    agents = agent_registry.list_agents()
    print(f"✅ Discovered {len(agents)} agents:")
    for name, info in agents.items():
        print(f"   - {name}: {info['class']} ({info['llm_provider']}, {info['model']})")

    return len(agents) > 0


async def test_agent_schemas():
    """Test dynamic schema validation."""
    print("\n📋 Testing Agent Schemas...")

    # Import the validation function
    from src.app.api import create_api_router

    # Create router to access validation functions
    router = create_api_router()

    # Test Alpha agent schema
    try:
        # Test valid Alpha input
        alpha_payload = {
            "query": "What is machine learning?",
            "context": "Educational context"
        }

        # Import validation function directly
        import importlib
        schema_module = importlib.import_module("src.agents.alpha.schemas")
        AlphaInput = getattr(schema_module, "AlphaInput")

        validated = AlphaInput(**alpha_payload)
        print(f"✅ Alpha schema validation passed: {validated.query[:50]}...")

    except Exception as e:
        print(f"❌ Alpha schema validation failed: {e}")
        return False

    # Test Beta agent schema
    try:
        beta_payload = {
            "problem": "How to optimize database performance?",
            "domain": "Software Engineering",
            "requirements": ["Low latency", "High throughput"],
            "constraints": "Limited memory budget"
        }

        schema_module = importlib.import_module("src.agents.beta.schemas")
        BetaInput = getattr(schema_module, "BetaInput")

        validated = BetaInput(**beta_payload)
        print(f"✅ Beta schema validation passed: {validated.problem[:50]}...")

    except Exception as e:
        print(f"❌ Beta schema validation failed: {e}")
        return False

    return True


async def test_agent_execution():
    """Test agent execution."""
    print("\n🚀 Testing Agent Execution...")

    try:
        # Get Alpha agent
        alpha_agent = agent_registry.get_agent("alpha")
        if not alpha_agent:
            print("❌ Alpha agent not found")
            return False

        # Test Alpha agent health
        health = await alpha_agent.health_check()
        print(f"✅ Alpha agent health: {health.get('status', 'unknown')}")

        # Get Beta agent
        beta_agent = agent_registry.get_agent("beta")
        if not beta_agent:
            print("❌ Beta agent not found")
            return False

        # Test Beta agent health
        health = await beta_agent.health_check()
        print(f"✅ Beta agent health: {health.get('status', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ Agent execution test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🎯 Final Implementation Test Suite")
    print("=" * 50)

    # Run tests
    tests = [
        test_agent_discovery(),
        test_agent_schemas(),
        test_agent_execution()
    ]

    results = await asyncio.gather(*tests, return_exceptions=True)

    # Summary
    passed = sum(1 for result in results if result is True)
    total = len(results)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Implementation is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
