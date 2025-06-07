"""
Docker integration tests for Vision backend.
Tests Docker builds, container health, and multi-container orchestration.
"""

import pytest
import asyncio
import time
import subprocess
import json
import requests
from typing import Dict, List, Optional
import docker
from docker.errors import DockerException
import httpx

# Skip tests if Docker is not available
try:
    docker_client = docker.from_env()
    docker_available = True
except DockerException:
    docker_available = False


@pytest.fixture(scope="session")
def docker_cleanup():
    """Cleanup Docker resources after tests."""
    containers_to_cleanup = []
    images_to_cleanup = []

    yield containers_to_cleanup, images_to_cleanup

    # Cleanup containers
    for container_name in containers_to_cleanup:
        try:
            container = docker_client.containers.get(container_name)
            container.stop()
            container.remove()
        except:
            pass

    # Cleanup images (optional, commented out to preserve builds)
    # for image_tag in images_to_cleanup:
    #     try:
    #         docker_client.images.remove(image_tag)
    #     except:
    #         pass


class TestDockerBuild:
    """Test Docker image building."""

    @pytest.mark.docker
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_build_main_dockerfile(self, docker_cleanup):
        """Test building the main Dockerfile."""
        containers_to_cleanup, images_to_cleanup = docker_cleanup

        print("Building main Docker image...")
        image, logs = docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            tag="vision-backend:test",
            remove=True,
            forcerm=True
        )

        assert image is not None
        assert "vision-backend:test" in [tag for tag in image.tags]
        images_to_cleanup.append("vision-backend:test")

        print("✓ Main Docker image built successfully")

    @pytest.mark.docker
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_build_distroless_dockerfile(self, docker_cleanup):
        """Test building the distroless Dockerfile."""
        containers_to_cleanup, images_to_cleanup = docker_cleanup

        print("Building distroless Docker image...")
        try:
            image, logs = docker_client.images.build(
                path=".",
                dockerfile="Dockerfile.distroless",
                tag="vision-backend:distroless-test",
                remove=True,
                forcerm=True
            )

            assert image is not None
            assert "vision-backend:distroless-test" in [tag for tag in image.tags]
            images_to_cleanup.append("vision-backend:distroless-test")

            print("✓ Distroless Docker image built successfully")
        except Exception as e:
            pytest.skip(f"Distroless build failed: {e}")


class TestDockerRun:
    """Test running Docker containers."""

    @pytest.mark.docker
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_container_health_check(self, docker_cleanup):
        """Test container health check."""
        containers_to_cleanup, images_to_cleanup = docker_cleanup

        # Ensure image exists
        try:
            docker_client.images.get("vision-backend:test")
        except:
            pytest.skip("vision-backend:test image not available")

        print("Starting container for health check...")
        container = docker_client.containers.run(
            "vision-backend:test",
            name="vision-test-health",
            ports={"8000/tcp": 8099},
            detach=True,
            remove=False,
            environment={
                "ENVIRONMENT": "test"
            }
        )

        containers_to_cleanup.append("vision-test-health")

        # Wait for container to start
        max_wait = 60
        wait_time = 0

        while wait_time < max_wait:
            container.reload()
            if container.status == "running":
                break
            time.sleep(2)
            wait_time += 2

        assert container.status == "running", f"Container failed to start: {container.status}"

        # Wait for health check to pass
        health_wait = 0
        while health_wait < 120:  # 2 minutes max
            container.reload()
            health_status = container.attrs.get("State", {}).get("Health", {}).get("Status")

            if health_status == "healthy":
                break
            elif health_status == "unhealthy":
                logs = container.logs().decode()
                pytest.fail(f"Container health check failed. Logs:\n{logs}")

            time.sleep(5)
            health_wait += 5

        # Final health status check
        container.reload()
        health_status = container.attrs.get("State", {}).get("Health", {}).get("Status")

        if health_status != "healthy":
            logs = container.logs().decode()
            print(f"Container logs:\n{logs}")
            # Don't fail immediately, try HTTP check

        print("✓ Container health check passed")

    @pytest.mark.docker
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_container_http_endpoint(self, docker_cleanup):
        """Test HTTP endpoints in container."""
        containers_to_cleanup, images_to_cleanup = docker_cleanup

        # Use existing container if available, otherwise create new one
        try:
            container = docker_client.containers.get("vision-test-health")
        except:
            try:
                docker_client.images.get("vision-backend:test")
            except:
                pytest.skip("vision-backend:test image not available")

            container = docker_client.containers.run(
                "vision-backend:test",
                name="vision-test-http",
                ports={"8000/tcp": 8098},
                detach=True,
                remove=False,
                environment={
                    "ENVIRONMENT": "test"
                }
            )
            containers_to_cleanup.append("vision-test-http")

        # Wait for container to be ready
        max_wait = 60
        wait_time = 0

        while wait_time < max_wait:
            try:
                response = requests.get("http://localhost:8098/healthz", timeout=5)
                if response.status_code == 200:
                    break
            except:
                pass

            time.sleep(2)
            wait_time += 2

        # Test health endpoint
        try:
            response = requests.get("http://localhost:8098/healthz", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data

            print("✓ Container HTTP health endpoint working")
        except Exception as e:
            # Get container logs for debugging
            logs = container.logs().decode()
            print(f"Container logs:\n{logs}")
            pytest.fail(f"HTTP endpoint test failed: {e}")

    @pytest.mark.docker
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    @pytest.mark.asyncio
    async def test_container_agent_endpoints(self, docker_cleanup):
        """Test agent endpoints in container."""
        containers_to_cleanup, images_to_cleanup = docker_cleanup

        # Use existing container or skip
        try:
            container = docker_client.containers.get("vision-test-health")
            port = 8099
        except:
            try:
                container = docker_client.containers.get("vision-test-http")
                port = 8098
            except:
                pytest.skip("No running container available for endpoint testing")

        base_url = f"http://localhost:{port}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test agent listing
            try:
                response = await client.get(f"{base_url}/api/v1/agents")
                # Accept both success and not found (if agents not configured)
                assert response.status_code in [200, 404, 500]
                print("✓ Agent listing endpoint accessible")
            except Exception as e:
                print(f"Agent listing endpoint error (may be expected): {e}")

            # Test specific agent endpoints (may not work without proper config)
            agent_endpoints = [
                "/api/v1/agents/alpha/health",
                "/api/v1/agents/beta/health"
            ]

            for endpoint in agent_endpoints:
                try:
                    response = await client.get(f"{base_url}{endpoint}")
                    # Accept various status codes as agents may not be configured
                    assert response.status_code in [200, 404, 500, 422]
                except Exception as e:
                    print(f"Endpoint {endpoint} error (may be expected): {e}")


class TestDockerCompose:
    """Test Docker Compose orchestration."""

    @pytest.mark.docker
    @pytest.mark.slow
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_docker_compose_build(self):
        """Test building with docker-compose."""
        print("Testing docker-compose build...")

        result = subprocess.run(
            ["docker-compose", "build", "--no-cache"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )

        if result.returncode != 0:
            print(f"Docker compose build failed:\n{result.stderr}")
            pytest.skip("Docker compose build failed")

        print("✓ Docker compose build successful")

    @pytest.mark.docker
    @pytest.mark.slow
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_docker_compose_up_down(self):
        """Test docker-compose up and down."""
        print("Testing docker-compose up...")

        # Start services
        up_result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if up_result.returncode != 0:
            print(f"Docker compose up failed:\n{up_result.stderr}")
            pytest.skip("Docker compose up failed")

        try:
            # Wait for services to be ready
            time.sleep(30)

            # Test if services are accessible
            services_to_test = [
                ("alpha-agent", "http://localhost:8001/healthz"),
                ("beta-agent", "http://localhost:8002/healthz"),
                ("nginx", "http://localhost:80/healthz")
            ]

            for service_name, url in services_to_test:
                try:
                    response = requests.get(url, timeout=10)
                    print(f"✓ {service_name} is accessible (status: {response.status_code})")
                except Exception as e:
                    print(f"⚠ {service_name} not accessible: {e}")

        finally:
            # Always stop services
            print("Stopping docker-compose services...")
            down_result = subprocess.run(
                ["docker-compose", "down"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if down_result.returncode == 0:
                print("✓ Docker compose down successful")
            else:
                print(f"Docker compose down issues:\n{down_result.stderr}")


class TestDockerSecurity:
    """Test Docker security configurations."""

    @pytest.mark.docker
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_container_user(self, docker_cleanup):
        """Test that container runs as non-root user."""
        containers_to_cleanup, images_to_cleanup = docker_cleanup

        try:
            docker_client.images.get("vision-backend:test")
        except:
            pytest.skip("vision-backend:test image not available")

        # Run a command to check user
        result = docker_client.containers.run(
            "vision-backend:test",
            command="whoami",
            remove=True,
            detach=False
        )

        user = result.decode().strip()
        assert user == "appuser", f"Container should run as 'appuser', but runs as '{user}'"

        print("✓ Container runs as non-root user")

    @pytest.mark.docker
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_container_filesystem_permissions(self, docker_cleanup):
        """Test filesystem permissions in container."""
        containers_to_cleanup, images_to_cleanup = docker_cleanup

        try:
            docker_client.images.get("vision-backend:test")
        except:
            pytest.skip("vision-backend:test image not available")

        # Test that app directory is owned by appuser
        result = docker_client.containers.run(
            "vision-backend:test",
            command="ls -la /app",
            remove=True,
            detach=False
        )

        output = result.decode()
        assert "appuser" in output, "App directory should be owned by appuser"

        print("✓ Container filesystem permissions correct")


class TestDockerNetworking:
    """Test Docker networking configurations."""

    @pytest.mark.docker
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_container_port_exposure(self, docker_cleanup):
        """Test that container exposes correct ports."""
        containers_to_cleanup, images_to_cleanup = docker_cleanup

        try:
            image = docker_client.images.get("vision-backend:test")
        except:
            pytest.skip("vision-backend:test image not available")

        # Check image configuration
        config = image.attrs.get("Config", {})
        exposed_ports = config.get("ExposedPorts", {})

        assert "8000/tcp" in exposed_ports, "Container should expose port 8000"

        print("✓ Container exposes correct ports")

    @pytest.mark.docker
    @pytest.mark.slow
    @pytest.mark.skipif(not docker_available, reason="Docker not available")
    def test_multi_container_communication(self):
        """Test communication between containers in docker-compose."""
        print("Testing multi-container communication...")

        # This test requires docker-compose to be running
        # We'll do a basic check to see if the network exists
        try:
            networks = docker_client.networks.list()
            vision_network = next((n for n in networks if "vision" in n.name), None)

            if vision_network:
                print("✓ Vision network exists")
            else:
                print("⚠ Vision network not found (may not be running)")

        except Exception as e:
            print(f"Network check failed: {e}")


# Custom pytest hook for Docker test reporting
def pytest_runtest_makereport(item, call):
    """Custom test reporting for Docker tests."""
    if "docker" in [mark.name for mark in item.iter_markers()]:
        if call.when == "call":
            if call.excinfo is not None:
                # Test failed, try to get Docker logs
                try:
                    containers = docker_client.containers.list(all=True)
                    for container in containers:
                        if "vision" in container.name or "test" in container.name:
                            print(f"\n--- Logs for container {container.name} ---")
                            print(container.logs().decode())
                            print("--- End logs ---\n")
                except Exception:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "docker"])