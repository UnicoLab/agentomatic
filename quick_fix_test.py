#!/usr/bin/env python3
"""Quick test to verify major fixes."""

import sys
from fastapi.testclient import TestClient


def test_pydantic_validation():
    """Test Pydantic V2 validation messages."""
    print("🔍 Testing Pydantic V2 validation...")

    try:
        from src.agents.alpha.schemas import AlphaInput
        from pydantic import ValidationError

        # Test empty query
        try:
            AlphaInput(query="")
        except ValidationError as e:
            error_msg = str(e)
            print(f"Error message: {error_msg}")
            if "Query cannot be empty" in error_msg:
                print("✅ Custom validation message works")
            else:
                print("❌ Custom validation message failed")

        # Test too long query
        try:
            AlphaInput(query="x" * 10001)
        except ValidationError as e:
            error_msg = str(e)
            if "String should have at most 10000 characters" in error_msg:
                print("✅ Pydantic V2 length validation works")
            else:
                print("❌ Pydantic V2 length validation failed")
                print(f"Got: {error_msg}")

    except Exception as e:
        print(f"❌ Pydantic validation test failed: {e}")
        import traceback
        traceback.print_exc()


def test_api_endpoints():
    """Test API endpoints."""
    print("\n🔍 Testing API endpoints...")

    try:
        from src.app.main import app
        client = TestClient(app)

        # Test /healthz
        response = client.get("/healthz")
        print(f"/healthz status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if "status" in data and data["status"] == "healthy":
                print("✅ /healthz endpoint works")
            else:
                print(f"❌ /healthz wrong format: {data}")
        else:
            print(f"❌ /healthz failed: {response.status_code}")

        # Test /metrics
        response = client.get("/metrics")
        print(f"/metrics status: {response.status_code}")
        if response.status_code == 200:
            print("✅ /metrics endpoint works")
        else:
            print(f"❌ /metrics failed: {response.status_code}")

        # Test /api/v1/agents
        response = client.get("/api/v1/agents")
        print(f"/api/v1/agents status: {response.status_code}")
        if response.status_code == 200:
            print("✅ /api/v1/agents endpoint works")
        else:
            print(f"❌ /api/v1/agents failed: {response.status_code}")

    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()


def test_agent_imports():
    """Test agent imports."""
    print("\n🔍 Testing agent imports...")

    try:
        from src.agents.alpha.agent import AlphaAgent
        from src.agents.beta.agent import BetaAgent
        print("✅ Agent imports work")

        # Try creating agents (this will test LLMFactory)
        try:
            alpha = AlphaAgent()
            print("✅ Alpha agent creation works")
        except Exception as e:
            print(f"❌ Alpha agent creation failed: {e}")

        try:
            beta = BetaAgent()
            print("✅ Beta agent creation works")
        except Exception as e:
            print(f"❌ Beta agent creation failed: {e}")

    except Exception as e:
        print(f"❌ Agent import test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Running quick fix verification tests...")

    test_pydantic_validation()
    test_api_endpoints()
    test_agent_imports()

    print("\n✨ Quick test completed!")
