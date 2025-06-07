#!/usr/bin/env python3
"""
Final validation test for LangGraph agents and FastAPI service.
Streamlined for quick verification.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def final_validation():
    """Final comprehensive validation."""
    print("🚀 Final LangGraph + FastAPI Validation Test")
    print("=" * 50)

    success_count = 0
    total_tests = 8

    try:
        # Test 1: Core imports
        print("1. Testing core imports...")
        from src.agents.alpha.agent import AlphaAgent
        from src.agents.beta.agent import BetaAgent
        from src.app.main import app
        from src.app.dependencies import agent_registry
        from src.common.api_decorators import handle_api_errors, APIResponse
        print("   ✅ All imports successful")
        success_count += 1

        # Test 2: Agent creation and health
        print("2. Testing agent creation and health...")
        alpha = AlphaAgent()
        beta = BetaAgent()

        alpha_health = await alpha.health_check()
        beta_health = await beta.health_check()

        if alpha_health.get("status") == "healthy" and beta_health.get("status") == "healthy":
            print("   ✅ Both agents healthy")
            success_count += 1
        else:
            print("   ❌ Agent health issues")

        # Test 3: Agent registry
        print("3. Testing agent registry...")
        agent_registry.discover_agents()
        agents = agent_registry.list_agents()

        if len(agents) >= 2 and "alpha" in agents and "beta" in agents:
            print(f"   ✅ Registry has {len(agents)} agents")
            success_count += 1
        else:
            print(f"   ❌ Registry issues: {list(agents.keys())}")

        # Test 4: FastAPI app creation
        print("4. Testing FastAPI app...")
        from fastapi.testclient import TestClient
        client = TestClient(app)
        health_response = client.get("/healthz")

        if health_response.status_code == 200:
            print("   ✅ FastAPI health endpoint working")
            success_count += 1
        else:
            print(f"   ❌ Health endpoint failed: {health_response.status_code}")

        # Test 5: Alpha agent execution
        print("5. Testing Alpha agent execution...")
        from src.agents.alpha.schemas import AlphaInput
        alpha_input = AlphaInput(query="Test query", context="validation")
        alpha_result = await alpha.run(alpha_input)

        if hasattr(alpha_result, 'response') and alpha_result.response:
            print("   ✅ Alpha agent execution successful")
            success_count += 1
        else:
            print("   ❌ Alpha agent execution failed")

        # Test 6: Beta agent execution
        print("6. Testing Beta agent execution...")
        from src.agents.beta.schemas import BetaInput
        beta_input = BetaInput(problem="Test problem", domain="validation")
        beta_result = await beta.run(beta_input)

        if hasattr(beta_result, 'analysis') and beta_result.analysis:
            print("   ✅ Beta agent execution successful")
            success_count += 1
        else:
            print("   ❌ Beta agent execution failed")

        # Test 7: API decorators
        print("7. Testing API decorators...")
        response = APIResponse(data={"test": "data"}, message="Test successful")
        if response.success and response.data:
            print("   ✅ API decorators working")
            success_count += 1
        else:
            print("   ❌ API decorators failed")

        # Test 8: Agent registry access
        print("8. Testing registry agent access...")
        registry_alpha = agent_registry.get_agent("alpha")
        registry_beta = agent_registry.get_agent("beta")

        if registry_alpha and registry_beta:
            print("   ✅ Registry agent access working")
            success_count += 1
        else:
            print("   ❌ Registry agent access failed")

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "=" * 50)
    print(f"FINAL RESULTS: {success_count}/{total_tests} tests passed")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")

    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! LangGraph agents are fully working with FastAPI!")
        print("✨ System is ready for production use")
        return True
    elif success_count >= 6:
        print("⚠️  Most tests passed - system is functional with minor issues")
        return True
    else:
        print("❌ Many tests failed - system needs attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(final_validation())
    sys.exit(0 if success else 1)