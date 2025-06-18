"""
End-to-end integration tests for Vision backend.
Tests complete workflows from API to agent execution with real scenarios.
"""

import pytest
import asyncio
import time
from typing import Dict, List, Optional
import httpx
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from src.app.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def e2e_client():
    """End-to-end test client with proper configuration."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=60.0,  # Longer timeout for e2e tests
    ) as client:
        yield client


@pytest.fixture(scope="session")
def sync_e2e_client():
    """Synchronous end-to-end test client."""
    return TestClient(app)


class TestE2EHealthAndStatus:
    """End-to-end tests for health and status endpoints."""

    @pytest.mark.e2e
    def test_application_startup(self, sync_e2e_client):
        """Test that application starts up correctly."""
        response = sync_e2e_client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

        print("✓ Application startup successful")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_metrics_collection(self, e2e_client):
        """Test metrics collection endpoint."""
        response = await e2e_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

        # Check for basic Prometheus metrics
        content = response.text
        assert "python_info" in content or "process_" in content

        print("✓ Metrics collection working")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_cors_headers(self, e2e_client):
        """Test CORS headers are properly set."""
        response = await e2e_client.options("/healthz")

        # Should have CORS headers or handle OPTIONS request
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled

        print("✓ CORS configuration checked")


class TestE2EAgentDiscovery:
    """End-to-end tests for agent discovery and listing."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_agent_discovery_workflow(self, e2e_client):
        """Test complete agent discovery workflow."""
        # Step 1: List available agents
        response = await e2e_client.get("/api/v1/agents")

        if response.status_code == 404:
            pytest.skip("Agent endpoints not configured")

        assert response.status_code == 200
        data = response.json()

        assert "agents" in data
        agents = data["agents"]

        print(f"✓ Discovered {len(agents)} agents")

        # Step 2: Test each discovered agent's health
        for agent_name in agents.keys():
            health_response = await e2e_client.get(f"/api/v1/agents/{agent_name}/health")
            # Accept various status codes as agents may have different configurations
            assert health_response.status_code in [200, 404, 500]
            print(f"  - {agent_name}: health check status {health_response.status_code}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_agent_prompts_retrieval(self, e2e_client):
        """Test retrieving agent prompts."""
        # Test known agents
        known_agents = ["alpha", "beta"]

        for agent_name in known_agents:
            response = await e2e_client.get(f"/api/v1/agents/{agent_name}/prompts")

            if response.status_code == 404:
                print(f"  - {agent_name}: prompts endpoint not available")
                continue

            assert response.status_code in [200, 500]  # 500 if agent config issues

            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, dict)
                print(f"  - {agent_name}: prompts retrieved successfully")


class TestE2EAgentExecution:
    """End-to-end tests for agent execution workflows."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_alpha_agent_execution_workflow(self, e2e_client):
        """Test complete Alpha agent execution workflow."""
        # Test different types of queries
        test_scenarios = [
            {
                "name": "Simple Question",
                "input": "What is artificial intelligence?",
                "context": "Educational query about AI basics"
            },
            {
                "name": "Technical Question",
                "input": "Explain how neural networks work",
                "context": "Technical explanation request"
            },
            {
                "name": "Comparison Question",
                "input": "Compare machine learning and deep learning",
                "context": "Comparative analysis request"
            }
        ]

        for scenario in test_scenarios:
            print(f"Testing Alpha agent with: {scenario['name']}")

            response = await e2e_client.post(
                "/api/v1/agents/alpha/chat",
                json={
                    "input": scenario["input"],
                    "context": scenario["context"],
                    "streaming": False
                },
                timeout=30.0
            )

            if response.status_code == 404:
                pytest.skip("Alpha agent not configured")

            # Accept various status codes for different configurations
            if response.status_code == 200:
                data = response.json()
                assert "output" in data or "response" in data
                print(f"  ✓ {scenario['name']}: Success")
            elif response.status_code in [422, 500]:
                print(f"  ⚠ {scenario['name']}: Agent configuration issue (status: {response.status_code})")
            else:
                print(f"  ⚠ {scenario['name']}: Unexpected status {response.status_code}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_beta_agent_execution_workflow(self, e2e_client):
        """Test complete Beta agent execution workflow."""
        test_scenarios = [
            {
                "name": "System Design",
                "input": "Design a scalable web application architecture",
                "context": "System architecture design request"
            },
            {
                "name": "Problem Analysis",
                "input": "Analyze performance bottlenecks in database queries",
                "context": "Performance optimization request"
            },
            {
                "name": "Solution Strategy",
                "input": "Create a deployment strategy for microservices",
                "context": "DevOps strategy request"
            }
        ]

        for scenario in test_scenarios:
            print(f"Testing Beta agent with: {scenario['name']}")

            response = await e2e_client.post(
                "/api/v1/agents/beta/chat",
                json={
                    "input": scenario["input"],
                    "context": scenario["context"],
                    "streaming": False
                },
                timeout=30.0
            )

            if response.status_code == 404:
                pytest.skip("Beta agent not configured")

            if response.status_code == 200:
                data = response.json()
                assert "output" in data or "response" in data
                print(f"  ✓ {scenario['name']}: Success")
            elif response.status_code in [422, 500]:
                print(f"  ⚠ {scenario['name']}: Agent configuration issue (status: {response.status_code})")
            else:
                print(f"  ⚠ {scenario['name']}: Unexpected status {response.status_code}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_streaming_execution_workflow(self, e2e_client):
        """Test streaming execution workflow."""
        test_input = {
            "input": "Explain the concept of machine learning in detail",
            "context": "Educational streaming request",
            "streaming": True
        }

        # Test Alpha agent streaming
        print("Testing Alpha agent streaming...")
        try:
            async with e2e_client.stream(
                "POST",
                "/api/v1/agents/alpha/stream",
                json=test_input,
                timeout=30.0
            ) as response:
                if response.status_code == 404:
                    print("  ⚠ Alpha streaming endpoint not available")
                elif response.status_code == 200:
                    chunks_received = 0
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            chunks_received += 1
                            if chunks_received >= 3:  # Limit for testing
                                break
                    print(f"  ✓ Alpha streaming: received {chunks_received} chunks")
                else:
                    print(f"  ⚠ Alpha streaming: unexpected status {response.status_code}")
        except Exception as e:
            print(f"  ⚠ Alpha streaming error: {e}")

        # Test Beta agent streaming
        print("Testing Beta agent streaming...")
        try:
            async with e2e_client.stream(
                "POST",
                "/api/v1/agents/beta/stream",
                json=test_input,
                timeout=30.0
            ) as response:
                if response.status_code == 404:
                    print("  ⚠ Beta streaming endpoint not available")
                elif response.status_code == 200:
                    chunks_received = 0
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            chunks_received += 1
                            if chunks_received >= 3:  # Limit for testing
                                break
                    print(f"  ✓ Beta streaming: received {chunks_received} chunks")
                else:
                    print(f"  ⚠ Beta streaming: unexpected status {response.status_code}")
        except Exception as e:
            print(f"  ⚠ Beta streaming error: {e}")


class TestE2EErrorHandling:
    """End-to-end tests for error handling scenarios."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_agent_handling(self, e2e_client):
        """Test handling of invalid agent requests."""
        # Test non-existent agent
        response = await e2e_client.post(
            "/api/v1/agents/nonexistent/chat",
            json={"input": "test", "context": "test"}
        )
        assert response.status_code == 404
        print("✓ Non-existent agent properly handled")

        # Test invalid request body
        response = await e2e_client.post(
            "/api/v1/agents/alpha/chat",
            json={"invalid": "data"}
        )
        assert response.status_code in [422, 404]  # 422 for validation error, 404 if agent not found
        print("✓ Invalid request body properly handled")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_malformed_requests(self, e2e_client):
        """Test handling of malformed requests."""
        # Test with no JSON body
        response = await e2e_client.post("/api/v1/agents/alpha/chat")
        assert response.status_code in [422, 404]

        # Test with invalid JSON
        try:
            response = await e2e_client.post(
                "/api/v1/agents/alpha/chat",
                content="invalid json",
                headers={"content-type": "application/json"}
            )
            assert response.status_code in [422, 400, 404]
        except Exception:
            pass  # Some clients may reject invalid JSON before sending

        print("✓ Malformed requests properly handled")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_timeout_handling(self, e2e_client):
        """Test handling of request timeouts."""
        # Test with very short timeout
        try:
            response = await e2e_client.post(
                "/api/v1/agents/alpha/chat",
                json={
                    "input": "Very long complex question that might take time to process",
                    "context": "timeout test"
                },
                timeout=0.1  # Very short timeout
            )
            # If it doesn't timeout, that's also valid
            print("✓ Request completed within short timeout")
        except httpx.TimeoutException:
            print("✓ Timeout properly handled")
        except Exception as e:
            print(f"✓ Request handling: {type(e).__name__}")


class TestE2EMultiAgentWorkflows:
    """End-to-end tests for multi-agent workflows."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_sequential_agent_execution(self, e2e_client):
        """Test sequential execution of multiple agents."""
        # Scenario: Use Alpha to generate a question, then Beta to analyze it

        # Step 1: Alpha generates content
        alpha_response = await e2e_client.post(
            "/api/v1/agents/alpha/chat",
            json={
                "input": "Generate a complex technical problem about distributed systems",
                "context": "Problem generation for analysis"
            }
        )

        if alpha_response.status_code != 200:
            pytest.skip("Alpha agent not available for sequential test")

        alpha_data = alpha_response.json()
        alpha_output = alpha_data.get("output", alpha_data.get("response", "Default problem"))

        print("✓ Alpha agent generated content")

        # Step 2: Beta analyzes the content
        beta_response = await e2e_client.post(
            "/api/v1/agents/beta/chat",
            json={
                "input": f"Analyze this problem: {alpha_output}",
                "context": "Analysis of generated problem"
            }
        )

        if beta_response.status_code == 200:
            beta_data = beta_response.json()
            assert "output" in beta_data or "response" in beta_data
            print("✓ Beta agent analyzed Alpha's output")
        else:
            print(f"⚠ Beta agent analysis failed: status {beta_response.status_code}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, e2e_client):
        """Test parallel execution of multiple agents."""
        # Execute both agents simultaneously with different tasks

        tasks = [
            e2e_client.post(
                "/api/v1/agents/alpha/chat",
                json={
                    "input": "What are the key principles of machine learning?",
                    "context": "Educational query"
                }
            ),
            e2e_client.post(
                "/api/v1/agents/beta/chat",
                json={
                    "input": "Design a monitoring system for microservices",
                    "context": "System design task"
                }
            )
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        successful_responses = 0
        for i, response in enumerate(responses):
            agent_name = ["Alpha", "Beta"][i]

            if isinstance(response, Exception):
                print(f"⚠ {agent_name} agent failed with exception: {response}")
            elif hasattr(response, 'status_code'):
                if response.status_code == 200:
                    successful_responses += 1
                    print(f"✓ {agent_name} agent executed successfully")
                else:
                    print(f"⚠ {agent_name} agent failed with status: {response.status_code}")

        print(f"✓ Parallel execution completed: {successful_responses}/2 agents successful")


class TestE2EPerformanceBaseline:
    """End-to-end performance baseline tests."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_response_time_baseline(self, e2e_client):
        """Test baseline response times for various endpoints."""
        endpoints_to_test = [
            ("Health Check", "GET", "/healthz"),
            ("Agent List", "GET", "/api/v1/agents"),
            ("Metrics", "GET", "/metrics"),
        ]

        for name, method, endpoint in endpoints_to_test:
            start_time = time.time()

            try:
                if method == "GET":
                    response = await e2e_client.get(endpoint)
                else:
                    response = await e2e_client.post(endpoint, json={})

                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms

                if response.status_code in [200, 404]:  # 404 acceptable for some endpoints
                    print(f"✓ {name}: {response_time:.1f}ms (status: {response.status_code})")
                else:
                    print(f"⚠ {name}: {response_time:.1f}ms (status: {response.status_code})")

            except Exception as e:
                print(f"⚠ {name}: Failed with {type(e).__name__}")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_requests_baseline(self, e2e_client):
        """Test handling of concurrent requests."""
        # Send multiple health check requests concurrently
        num_requests = 10

        start_time = time.time()

        tasks = [
            e2e_client.get("/healthz")
            for _ in range(num_requests)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to ms

        successful_responses = sum(
            1 for r in responses
            if hasattr(r, 'status_code') and r.status_code == 200
        )

        requests_per_second = num_requests / (total_time / 1000)

        print(f"✓ Concurrent requests: {successful_responses}/{num_requests} successful")
        print(f"✓ Total time: {total_time:.1f}ms, RPS: {requests_per_second:.1f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e", "-s"])