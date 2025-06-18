#!/usr/bin/env python3
"""Comprehensive test runner to check our fixes."""

import subprocess
import sys


def run_test(test_command, description):
    """Run a test and report results."""
    print(f"\n🔍 {description}")
    print("=" * 50)

    try:
        result = subprocess.run(
            test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✅ PASSED: {description}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return False


def main():
    """Run comprehensive test suite."""
    print("🚀 Running comprehensive test validation...")

    # Change to the project directory
    import os
    os.chdir("/Users/10176445/Desktop/PROJECTS/Vision/ideator")

    tests = [
        # Validation tests
        ("poetry run pytest tests/test_input_validation.py::TestAlphaInputValidation::test_empty_query_validation -v",
         "Alpha empty query validation"),
        ("poetry run pytest tests/test_input_validation.py::TestBetaInputValidation::test_empty_problem_validation -v",
         "Beta empty problem validation"),
        ("poetry run pytest tests/test_input_validation.py::TestAlphaInputValidation::test_query_too_long -v",
         "Alpha query length validation"),
        ("poetry run pytest tests/test_input_validation.py::TestBetaInputValidation::test_problem_too_long -v",
         "Beta problem length validation"),

        # API endpoint tests
        ("poetry run pytest tests/test_e2e.py::TestE2EHealthAndStatus::test_application_startup -v",
         "Health endpoint test"),

        # Async test
        ("poetry run pytest tests/test_simplified_agents.py::test_basic_imports -v",
         "Basic async imports test"),

        # App settings test
        ("poetry run pytest tests/test_app.py::TestSettings::test_settings_initialization -v",
         "App settings test"),
    ]

    passed = 0
    failed = 0

    for test_cmd, description in tests:
        if run_test(test_cmd, description):
            passed += 1
        else:
            failed += 1

    print(f"\n📊 SUMMARY")
    print("=" * 30)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed}/{passed+failed} ({100*passed/(passed+failed):.1f}%)")

    if failed == 0:
        print("\n🎉 All tests passed! Fixes are working correctly.")
    else:
        print(f"\n⚠️  {failed} tests still failing. Further investigation needed.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
