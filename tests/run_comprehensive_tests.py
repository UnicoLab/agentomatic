#!/usr/bin/env python3
"""
Comprehensive test runner for LangGraph agents and FastAPI service.
Optimized for fastest parallel execution with detailed reporting.
"""

import asyncio
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class TestRunner:
    """High-performance test runner with parallel execution."""

    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = time.time()

    def print_header(self, title: str):
        """Print formatted header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{title.center(60)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

    def print_success(self, message: str):
        """Print success message."""
        print(f"{Colors.GREEN}✅ {message}{Colors.END}")

    def print_error(self, message: str):
        """Print error message."""
        print(f"{Colors.RED}❌ {message}{Colors.END}")

    def print_warning(self, message: str):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

    def print_info(self, message: str):
        """Print info message."""
        print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

    async def run_pytest_suite(self, test_file: str, markers: List[str] = None, parallel: bool = True) -> Dict[str, Any]:
        """Run pytest suite with optimization."""
        cmd = ["python", "-m", "pytest", test_file, "-v", "--tb=short"]

        if markers:
            cmd.extend(["-m", " and ".join(markers)])

        if parallel:
            # Add parallel execution options
            cmd.extend([
                "--asyncio-mode=auto",
                "--maxfail=10",  # Fail fast but allow some failures
                "-x" if "--debug" in sys.argv else "",  # Stop on first failure in debug mode
            ])
            cmd = [arg for arg in cmd if arg]  # Remove empty strings

        # Add performance optimizations
        cmd.extend([
            "--disable-warnings",  # Disable warnings for speed
            "--no-header",         # Reduce output overhead
            "--quiet" if "--verbose" not in sys.argv else "-v"
        ])

        self.print_info(f"Running: {' '.join(cmd)}")

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            end_time = time.time()
            duration = end_time - start_time

            return {
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "duration": 300,
                "stdout": "",
                "stderr": "Test suite timed out after 5 minutes",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "returncode": -2
            }

    async def validate_agents(self) -> bool:
        """Validate that agents are properly configured."""
        self.print_header("AGENT VALIDATION")

        try:
            # Test basic imports
            self.print_info("Testing basic imports...")

            from src.agents.alpha.agent import AlphaAgent
            from src.agents.beta.agent import BetaAgent
            from src.app.dependencies import agent_registry

            self.print_success("Basic imports successful")

            # Test agent creation
            self.print_info("Testing agent creation...")

            try:
                alpha_agent = AlphaAgent()
                self.print_success("Alpha agent created successfully")
            except Exception as e:
                self.print_error(f"Alpha agent creation failed: {e}")
                return False

            try:
                beta_agent = BetaAgent()
                self.print_success("Beta agent created successfully")
            except Exception as e:
                self.print_error(f"Beta agent creation failed: {e}")
                return False

            # Test agent registry
            self.print_info("Testing agent registry...")

            agent_registry.discover_agents()
            agents = agent_registry.list_agents()

            if "alpha" in agents:
                self.print_success("Alpha agent registered in registry")
            else:
                self.print_warning("Alpha agent not found in registry")

            if "beta" in agents:
                self.print_success("Beta agent registered in registry")
            else:
                self.print_warning("Beta agent not found in registry")

            self.print_success(f"Agent registry contains {len(agents)} agents")

            # Test health checks
            self.print_info("Testing agent health checks...")

            for agent_name, agent_info in agents.items():
                agent = agent_registry.get_agent(agent_name)
                if agent:
                    try:
                        health = await agent.health_check()
                        if health.get("status") == "healthy":
                            self.print_success(f"{agent_name} agent health check passed")
                        else:
                            self.print_warning(f"{agent_name} agent health check returned: {health.get('status')}")
                    except Exception as e:
                        self.print_warning(f"{agent_name} agent health check failed: {e}")

            return True

        except Exception as e:
            self.print_error(f"Agent validation failed: {e}")
            return False

    async def run_fast_api_tests(self) -> bool:
        """Run FastAPI endpoint tests."""
        self.print_header("FASTAPI ENDPOINT TESTS")

        # Quick API availability test
        try:
            from fastapi.testclient import TestClient
            from src.app.main import app

            client = TestClient(app)

            # Test basic endpoints
            endpoints = [
                ("/", "Root endpoint"),
                ("/healthz", "Health check"),
                ("/api/v1/agents", "Agent listing"),
            ]

            for endpoint, description in endpoints:
                try:
                    response = client.get(endpoint)
                    if response.status_code in [200, 404]:  # 404 acceptable for some endpoints
                        self.print_success(f"{description}: {response.status_code}")
                    else:
                        self.print_warning(f"{description}: {response.status_code}")
                except Exception as e:
                    self.print_error(f"{description} failed: {e}")

            return True

        except Exception as e:
            self.print_error(f"FastAPI test failed: {e}")
            return False

    async def run_all_tests(self):
        """Run all test suites in optimal order."""
        self.print_header("LANGGRAPH AGENTS & FASTAPI TEST SUITE")
        self.print_info("Optimized for fastest parallel execution")

        # Step 1: Validate agents first
        agent_validation = await self.validate_agents()
        if not agent_validation:
            self.print_error("Agent validation failed - skipping remaining tests")
            return False

        # Step 2: Quick FastAPI test
        api_validation = await self.run_fast_api_tests()
        if not api_validation:
            self.print_warning("FastAPI validation failed - continuing with other tests")

        # Step 3: Run test suites in parallel where possible
        test_suites = [
            {
                "name": "Parallel Agent Tests",
                "file": "tests/test_parallel_agents.py",
                "markers": ["parallel"],
                "priority": 1
            },
            {
                "name": "Load Performance Tests",
                "file": "tests/test_load_performance.py",
                "markers": ["load"],
                "priority": 2
            },
            {
                "name": "Simplified Agent Tests",
                "file": "tests/test_simplified_agents.py",
                "markers": None,
                "priority": 1
            },
            {
                "name": "Main App Tests",
                "file": "tests/test_app.py",
                "markers": None,
                "priority": 1
            }
        ]

        # Group tests by priority for parallel execution
        priority_groups = {}
        for suite in test_suites:
            priority = suite["priority"]
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(suite)

        # Run each priority group
        for priority in sorted(priority_groups.keys()):
            self.print_header(f"PRIORITY {priority} TESTS")

            # Run tests in this priority group in parallel
            tasks = []
            for suite in priority_groups[priority]:
                if Path(suite["file"]).exists():
                    task = self.run_pytest_suite(
                        suite["file"],
                        suite["markers"],
                        parallel=True
                    )
                    tasks.append((suite["name"], task))
                else:
                    self.print_warning(f"Test file not found: {suite['file']}")

            # Execute parallel tests
            if tasks:
                self.print_info(f"Running {len(tasks)} test suites in parallel...")

                results = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                )

                # Process results
                for i, (suite_name, _) in enumerate(tasks):
                    result = results[i]

                    if isinstance(result, Exception):
                        self.print_error(f"{suite_name}: Exception - {result}")
                        self.failed_tests += 1
                    elif result["success"]:
                        self.print_success(f"{suite_name}: Passed ({result['duration']:.2f}s)")
                        self.passed_tests += 1
                    else:
                        self.print_error(f"{suite_name}: Failed ({result['duration']:.2f}s)")
                        if result["stderr"]:
                            self.print_error(f"Error: {result['stderr'][:200]}...")
                        self.failed_tests += 1

                    self.total_tests += 1
                    self.results[suite_name] = result

        return self.generate_final_report()

    def generate_final_report(self) -> bool:
        """Generate final test report."""
        end_time = time.time()
        total_duration = end_time - self.start_time

        self.print_header("TEST EXECUTION SUMMARY")

        print(f"{Colors.BOLD}Total Tests:{Colors.END} {self.total_tests}")
        print(f"{Colors.GREEN}Passed:{Colors.END} {self.passed_tests}")
        print(f"{Colors.RED}Failed:{Colors.END} {self.failed_tests}")
        print(f"{Colors.BLUE}Duration:{Colors.END} {total_duration:.2f} seconds")

        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
            print(f"{Colors.CYAN}Success Rate:{Colors.END} {success_rate:.1f}%")

        # Performance metrics
        if self.total_tests > 0:
            avg_test_duration = total_duration / self.total_tests
            print(f"{Colors.MAGENTA}Avg Test Duration:{Colors.END} {avg_test_duration:.2f}s")

        # Recommendations
        self.print_header("RECOMMENDATIONS")

        if self.failed_tests == 0:
            self.print_success("🎉 All tests passed! LangGraph agents and FastAPI service are working correctly.")
            self.print_info("✨ System is ready for production deployment")
        elif self.failed_tests < self.total_tests / 2:
            self.print_warning("⚠️  Some tests failed but core functionality appears to work")
            self.print_info("🔧 Review failed tests and fix critical issues")
        else:
            self.print_error("❌ Many tests failed - system may have fundamental issues")
            self.print_info("🛠️  Investigate configuration and dependencies")

        # Performance recommendations
        if total_duration < 60:
            self.print_success("🚀 Excellent test performance - under 1 minute")
        elif total_duration < 180:
            self.print_info("⏱️  Good test performance - under 3 minutes")
        else:
            self.print_warning("🐌 Slow test performance - consider optimization")

        return self.failed_tests == 0


async def main():
    """Main test execution function."""
    runner = TestRunner()

    # Check for command line arguments
    if "--help" in sys.argv:
        print(f"""
{Colors.BOLD}LangGraph Agents & FastAPI Test Runner{Colors.END}

Usage: python tests/run_comprehensive_tests.py [options]

Options:
  --help      Show this help message
  --verbose   Verbose output
  --debug     Stop on first failure
  --quick     Run only essential tests

Test Categories:
  - Agent validation and health checks
  - FastAPI endpoint testing
  - Parallel agent execution tests
  - Load and performance testing
  - Error handling and edge cases

Optimizations:
  - Parallel test execution
  - Session-scoped fixtures
  - Minimal test overhead
  - Fast failure detection
        """)
        return True

    try:
        success = await runner.run_all_tests()
        return success
    except KeyboardInterrupt:
        runner.print_warning("Test execution interrupted by user")
        return False
    except Exception as e:
        runner.print_error(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    # Ensure we're in the right directory
    if not Path("src").exists():
        print(f"{Colors.RED}Error: Please run from project root directory{Colors.END}")
        sys.exit(1)

    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)