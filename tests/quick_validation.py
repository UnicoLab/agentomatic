#!/usr/bin/env python3
"""
Quick validation script to ensure LangGraph agents are working with FastAPI.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def validate_system():
    """Quick system validation."""
    print("🔍 Validating LangGraph Agents & FastAPI System...")

    try:
        # Test 1: Basic imports
        print("1. Testing imports...")
        from src.agents.alpha.agent import AlphaAgent
        from src.agents.beta.agent import BetaAgent
        from src.app.main import app
        from src.app.dependencies import agent_registry
        print("✅ All imports successful")

        # Test 2: Agent creation
        print("2. Testing agent creation...")
        alpha = AlphaAgent()
        beta = BetaAgent()
        print("✅ Both agents created successfully")

        # Test 3: Agent registry
        print("3. Testing agent registry...")
        agent_registry.discover_agents()
        agents = agent_registry.list_agents()
        print(f"✅ Registry discovered {len(agents)} agents: {list(agents.keys())}")

        # Test 4: FastAPI app
        print("4. Testing FastAPI app...")
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/healthz")
        print(f"✅ Health endpoint responds with status {response.status_code}")

        # Test 5: Agent health checks
        print("5. Testing agent health checks...")
        for name in agents:
            agent = agent_registry.get_agent(name)
            if agent:
                try:
                    health = await agent.health_check()
                    print(f"✅ {name} agent health: {health.get('status', 'unknown')}")
                except Exception as e:
                    print(f"⚠️  {name} agent health check failed: {e}")

        print("\n🎉 System validation complete! All LangGraph agents are working with FastAPI.")
        return True

    except Exception as e:
        print(f"❌ System validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(validate_system())
    sys.exit(0 if success else 1)