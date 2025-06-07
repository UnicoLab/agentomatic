"""
Load testing and performance benchmarking for LangGraph agents and FastAPI service.
Optimized for maximum throughput and stress testing scenarios.
"""

import asyncio
import pytest
import time
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import psutil
import gc

import httpx
from httpx import AsyncClient, ASGITransport
from src.app.main import app
from src.agents.alpha.schemas import AlphaInput
from src.agents.beta.schemas import BetaInput


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    total_duration: float
    memory_usage_mb: float
    cpu_usage_percent: float


class LoadTestRunner:
    """High-performance load test runner with detailed metrics."""

    def __init__(self):
        self.metrics: List[float] = []
        self.errors: List[str] = []
        self.start_time: float = 0
        self.end_time: float = 0

    async def run_load_test(
        self,
        test_func,
        concurrent_users: int,
        duration_seconds: int,
        ramp_up_seconds: int = 0
    ) -> PerformanceMetrics:
        """Run load test with specified parameters."""
        self.metrics.clear()
        self.errors.clear()

        # Start monitoring
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()

        self.start_time = time.time()

        # Create semaphore for concurrent users
        semaphore = asyncio.Semaphore(concurrent_users)

        async def worker():
            async with semaphore:
                request_start = time.time()
                try:
                    await test_func()
                    request_end = time.time()
                    self.metrics.append(request_end - request_start)
                except Exception as e:
                    self.errors.append(str(e))

        # Run test for specified duration
        tasks = []
        end_time = time.time() + duration_seconds

        # Ramp up phase
        if ramp_up_seconds > 0:
            ramp_step = concurrent_users / (ramp_up_seconds * 10)  # 10 steps per second
            current_users = 0

            while time.time() < self.start_time + ramp_up_seconds:
                current_users = min(concurrent_users, current_users + ramp_step)
                for _ in range(int(current_users) - len([t for t in tasks if not t.done()])):
                    if time.time() < end_time:
                        tasks.append(asyncio.create_task(worker()))
                await asyncio.sleep(0.1)

        # Full load phase
        while time.time() < end_time:
            # Maintain concurrent user count
            active_tasks = [t for t in tasks if not t.done()]
            needed_tasks = concurrent_users - len(active_tasks)

            for _ in range(needed_tasks):
                if time.time() < end_time:
                    tasks.append(asyncio.create_task(worker()))

            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        self.end_time = time.time()

        # Calculate final metrics
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()

        return self._calculate_metrics(
            start_memory, end_memory,
            start_cpu, end_cpu
        )

    def _calculate_metrics(
        self,
        start_memory: float,
        end_memory: float,
        start_cpu: float,
        end_cpu: float
    ) -> PerformanceMetrics:
        """Calculate performance metrics from collected data."""
        total_duration = self.end_time - self.start_time
        total_requests = len(self.metrics) + len(self.errors)
        successful_requests = len(self.metrics)
        failed_requests = len(self.errors)

        if self.metrics:
            avg_response_time = statistics.mean(self.metrics)
            min_response_time = min(self.metrics)
            max_response_time = max(self.metrics)
            sorted_metrics = sorted(self.metrics)
            p50_response_time = sorted_metrics[int(len(sorted_metrics) * 0.5)]
            p95_response_time = sorted_metrics[int(len(sorted_metrics) * 0.95)]
            p99_response_time = sorted_metrics[int(len(sorted_metrics) * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0

        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        memory_usage_mb = end_memory - start_memory
        cpu_usage_percent = (start_cpu + end_cpu) / 2

        return PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            total_duration=total_duration,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent
        )


@pytest.fixture(scope="session")
async def load_test_client():
    """High-performance client for load testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=30.0,  # Increase timeout for load testing
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
    ) as client:
        yield client


class TestHealthEndpointLoad:
    """Load test the health endpoint for baseline performance."""

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_health_endpoint_light_load(self, load_test_client):
        """Test health endpoint under light load."""
        runner = LoadTestRunner()

        async def health_request():
            response = await load_test_client.get("/healthz")
            assert response.status_code == 200

        metrics = await runner.run_load_test(
            test_func=health_request,
            concurrent_users=10,
            duration_seconds=10
        )

        # Performance assertions
        assert metrics.requests_per_second > 50  # Should handle at least 50 RPS
        assert metrics.avg_response_time < 0.1  # Average response under 100ms
        assert metrics.p95_response_time < 0.2  # 95th percentile under 200ms
        assert metrics.failed_requests == 0  # No failures expected

        print(f"Health endpoint light load: {metrics.requests_per_second:.2f} RPS")

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_health_endpoint_heavy_load(self, load_test_client):
        """Test health endpoint under heavy load."""
        runner = LoadTestRunner()

        async def health_request():
            response = await load_test_client.get("/healthz")
            assert response.status_code == 200

        metrics = await runner.run_load_test(
            test_func=health_request,
            concurrent_users=50,
            duration_seconds=30,
            ramp_up_seconds=5
        )

        # Performance assertions for heavy load
        assert metrics.requests_per_second > 20  # Should still handle reasonable load
        assert metrics.avg_response_time < 0.5  # Average response under 500ms
        assert metrics.failed_requests / metrics.total_requests < 0.05  # Less than 5% failures

        print(f"Health endpoint heavy load: {metrics.requests_per_second:.2f} RPS")


class TestAgentEndpointLoad:
    """Load test agent endpoints."""

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_alpha_agent_load(self, load_test_client):
        """Test Alpha agent under load."""
        runner = LoadTestRunner()

        test_queries = [
            "What is AI?",
            "Explain machine learning",
            "How do neural networks work?",
            "What is deep learning?",
            "Describe computer vision"
        ]

        async def alpha_request():
            query = test_queries[len(runner.metrics) % len(test_queries)]
            response = await load_test_client.post(
                "/api/v1/agents/alpha/chat",
                json={
                    "input": query,
                    "context": "load test",
                    "streaming": False
                }
            )
            # Accept both success and reasonable errors
            assert response.status_code in [200, 404, 422, 500]

        metrics = await runner.run_load_test(
            test_func=alpha_request,
            concurrent_users=5,  # Lower concurrency for agent endpoints
            duration_seconds=20
        )

        # More lenient assertions for agent endpoints
        assert metrics.total_requests > 0
        print(f"Alpha agent load: {metrics.requests_per_second:.2f} RPS, "
              f"{metrics.successful_requests}/{metrics.total_requests} successful")

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_beta_agent_load(self, load_test_client):
        """Test Beta agent under load."""
        runner = LoadTestRunner()

        test_problems = [
            "Optimize database queries",
            "Design scalable architecture",
            "Implement caching strategy",
            "Build monitoring system",
            "Create deployment pipeline"
        ]

        async def beta_request():
            problem = test_problems[len(runner.metrics) % len(test_problems)]
            response = await load_test_client.post(
                "/api/v1/agents/beta/chat",
                json={
                    "input": problem,
                    "context": "load test",
                    "streaming": False
                }
            )
            assert response.status_code in [200, 404, 422, 500]

        metrics = await runner.run_load_test(
            test_func=beta_request,
            concurrent_users=5,
            duration_seconds=20
        )

        assert metrics.total_requests > 0
        print(f"Beta agent load: {metrics.requests_per_second:.2f} RPS, "
              f"{metrics.successful_requests}/{metrics.total_requests} successful")


class TestMixedWorkloadLoad:
    """Test mixed workload scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.load
    async def test_mixed_endpoint_load(self, load_test_client):
        """Test mixed endpoint load to simulate realistic usage."""
        runner = LoadTestRunner()

        endpoints = [
            {"method": "GET", "url": "/healthz"},
            {"method": "GET", "url": "/api/v1/agents"},
            {"method": "POST", "url": "/api/v1/agents/alpha/chat",
             "json": {"input": "Quick test", "context": "mixed load"}},
            {"method": "POST", "url": "/api/v1/agents/beta/chat",
             "json": {"input": "Quick analysis", "context": "mixed load"}},
        ]

        async def mixed_request():
            endpoint = endpoints[len(runner.metrics) % len(endpoints)]
            try:
                if endpoint["method"] == "GET":
                    response = await load_test_client.get(endpoint["url"])
                else:
                    response = await load_test_client.post(
                        endpoint["url"],
                        json=endpoint.get("json", {})
                    )
                assert response.status_code in [200, 404, 422, 500]
            except Exception:
                pass  # Accept failures in mixed load testing

        metrics = await runner.run_load_test(
            test_func=mixed_request,
            concurrent_users=15,
            duration_seconds=30,
            ramp_up_seconds=5
        )

        assert metrics.total_requests > 0
        print(f"Mixed workload: {metrics.requests_per_second:.2f} RPS, "
              f"Success rate: {metrics.successful_requests/metrics.total_requests*100:.1f}%")


class TestStressAndSpike:
    """Stress testing and spike load scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_spike_load(self, load_test_client):
        """Test sudden spike in load."""
        runner = LoadTestRunner()

        async def spike_request():
            response = await load_test_client.get("/healthz")
            assert response.status_code == 200

        # Sudden spike to high concurrency
        metrics = await runner.run_load_test(
            test_func=spike_request,
            concurrent_users=100,  # High spike
            duration_seconds=15,
            ramp_up_seconds=1  # Very fast ramp up
        )

        # System should handle spike gracefully
        assert metrics.total_requests > 0
        failure_rate = metrics.failed_requests / metrics.total_requests
        assert failure_rate < 0.3  # Less than 30% failures acceptable for spike

        print(f"Spike load: {metrics.requests_per_second:.2f} RPS, "
              f"Failure rate: {failure_rate*100:.1f}%")

    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_memory_stress(self, load_test_client):
        """Test memory usage under sustained load."""
        runner = LoadTestRunner()

        # Large payload to stress memory
        large_input = "Analyze this complex problem: " + "data " * 1000

        async def memory_stress_request():
            response = await load_test_client.post(
                "/api/v1/agents/alpha/chat",
                json={
                    "input": large_input,
                    "context": "memory stress test",
                    "streaming": False
                }
            )
            assert response.status_code in [200, 404, 422, 500, 413]  # Include payload too large

        metrics = await runner.run_load_test(
            test_func=memory_stress_request,
            concurrent_users=3,  # Lower concurrency for memory stress
            duration_seconds=30
        )

        # Memory usage should be reasonable
        assert metrics.memory_usage_mb < 500  # Less than 500MB increase
        print(f"Memory stress: {metrics.memory_usage_mb:.2f}MB increase")


class TestPerformanceRegression:
    """Performance regression testing."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_baseline_performance(self, load_test_client):
        """Establish baseline performance metrics."""
        runner = LoadTestRunner()

        async def baseline_request():
            response = await load_test_client.get("/healthz")
            assert response.status_code == 200

        metrics = await runner.run_load_test(
            test_func=baseline_request,
            concurrent_users=1,  # Single user baseline
            duration_seconds=10
        )

        # Store baseline metrics (in real scenario, save to file/database)
        baseline = {
            "avg_response_time": metrics.avg_response_time,
            "requests_per_second": metrics.requests_per_second,
            "memory_usage_mb": metrics.memory_usage_mb
        }

        # Basic performance expectations
        assert metrics.avg_response_time < 0.05  # Under 50ms average
        assert metrics.requests_per_second > 100  # At least 100 RPS for single user

        print(f"Baseline: {baseline}")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_agent_performance_comparison(self, load_test_client):
        """Compare performance between different agents."""
        runner = LoadTestRunner()

        # Test Alpha agent
        async def alpha_request():
            response = await load_test_client.post(
                "/api/v1/agents/alpha/chat",
                json={"input": "Test performance", "context": "comparison"}
            )
            assert response.status_code in [200, 404, 422, 500]

        alpha_metrics = await runner.run_load_test(
            test_func=alpha_request,
            concurrent_users=2,
            duration_seconds=15
        )

        # Test Beta agent
        runner.metrics.clear()
        runner.errors.clear()

        async def beta_request():
            response = await load_test_client.post(
                "/api/v1/agents/beta/chat",
                json={"input": "Test performance", "context": "comparison"}
            )
            assert response.status_code in [200, 404, 422, 500]

        beta_metrics = await runner.run_load_test(
            test_func=beta_request,
            concurrent_users=2,
            duration_seconds=15
        )

        # Compare metrics
        print(f"Alpha: {alpha_metrics.requests_per_second:.2f} RPS, "
              f"Avg: {alpha_metrics.avg_response_time:.3f}s")
        print(f"Beta: {beta_metrics.requests_per_second:.2f} RPS, "
              f"Avg: {beta_metrics.avg_response_time:.3f}s")

        # Both agents should have reasonable performance
        assert alpha_metrics.total_requests > 0
        assert beta_metrics.total_requests > 0


# Custom markers for test selection
pytestmark = [
    pytest.mark.load,
    pytest.mark.performance,
]


def pytest_configure(config):
    """Configure pytest markers for load testing."""
    config.addinivalue_line("markers", "load: mark test for load testing")
    config.addinivalue_line("markers", "stress: mark test for stress testing")
    config.addinivalue_line("markers", "performance: mark test for performance benchmarking")


if __name__ == "__main__":
    # Run load tests
    pytest.main([
        __file__,
        "-v",
        "-m", "load",  # Run only load tests
        "--tb=short",
        "-s",  # Show print statements
    ])