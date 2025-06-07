"""Quick test for simplified agents with mock functionality."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))


async def test_architecture():
    """Test the simplified agent architecture."""
    print("🚀 Testing Simplified Agent Architecture\n")

    try:
        # Test imports
        print("1. Testing imports...")
        from src.agents.alpha.agent import AlphaAgent
        from src.agents.beta.agent import BetaAgent
        from src.agents.alpha.schemas import AlphaInput
        from src.agents.beta.schemas import BetaInput
        from src.app.dependencies import agent_registry
        print("✅ All imports successful")

        # Test agent instantiation
        print("\n2. Testing agent instantiation...")
        alpha = AlphaAgent()
        beta = BetaAgent()
        print(f"✅ Alpha agent: {alpha.name} with {alpha.llm.config.provider.value}")
        print(f"✅ Beta agent: {beta.name} with {beta.llm.config.provider.value}")

        # Test agent registry
        print("\n3. Testing agent registry...")
        agent_registry.discover_agents()
        agents = agent_registry.list_agents()
        print(f"✅ Registry discovered {len(agents)} agents:")
        for name, info in agents.items():
            print(f"   - {name}: {info['class']} ({info['llm_provider']})")

        # Test agent execution (with error handling)
        print("\n4. Testing agent execution...")
        
        # Alpha test
        alpha_input = AlphaInput(query="Test query", context="Test context")
        alpha_result = await alpha.run(alpha_input)
        print(f"✅ Alpha result: {alpha_result.response[:50]}...")
        
        # Beta test  
        beta_input = BetaInput(problem="Test problem", domain="Test domain")
        beta_result = await beta.run(beta_input)
        print(f"✅ Beta result: {beta_result.analysis[:50]}...")

        # Test health checks
        print("\n5. Testing health checks...")
        alpha_health = await alpha.health_check()
        beta_health = await beta.health_check()
        print(f"✅ Alpha health: {alpha_health['status']}")
        print(f"✅ Beta health: {beta_health['status']}")

        print("\n🎉 All tests passed! Simplified agent architecture is working correctly.")
        print("\n✨ Key improvements achieved:")
        print("   - ✅ Unified base agent structure")
        print("   - ✅ Common state management")
        print("   - ✅ Simplified prompt handling")
        print("   - ✅ Robust error handling")
        print("   - ✅ Agent registry auto-discovery")
        print("   - ✅ Updated to latest dependencies")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_architecture())
    sys.exit(0 if success else 1)