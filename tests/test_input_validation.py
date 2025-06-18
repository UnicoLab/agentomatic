#!/usr/bin/env python3
"""
Test input data validation for all agents.
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.alpha.schemas import AlphaInput, AlphaOutput
from src.agents.beta.schemas import BetaInput, BetaOutput
from src.app.main import app


class TestAlphaInputValidation:
    """Test Alpha agent input validation."""

    def test_valid_input(self):
        """Test valid Alpha input."""
        input_data = AlphaInput(
            query="What is machine learning?",
            context="Educational context"
        )
        assert input_data.query == "What is machine learning?"
        assert input_data.context == "Educational context"

    def test_empty_query_validation(self):
        """Test that empty query raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AlphaInput(query="")
        # Pydantic v2 built-in validation message for min_length=1
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_whitespace_only_query_validation(self):
        """Test that whitespace-only query raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AlphaInput(query="   \n\t   ")
        assert "Query cannot be empty" in str(exc_info.value)

    def test_query_too_long(self):
        """Test that excessively long query raises validation error."""
        long_query = "x" * 10001  # Over the 10000 limit
        with pytest.raises(ValidationError) as exc_info:
            AlphaInput(query=long_query)
        assert "String should have at most 10000 characters" in str(exc_info.value)

    def test_context_too_long(self):
        """Test that excessively long context raises validation error."""
        long_context = "x" * 5001  # Over the 5000 limit
        with pytest.raises(ValidationError) as exc_info:
            AlphaInput(query="Valid query", context=long_context)
        assert "String should have at most 5000 characters" in str(exc_info.value)

    def test_whitespace_trimming(self):
        """Test that whitespace is properly trimmed."""
        input_data = AlphaInput(
            query="  What is AI?  ",
            context="  Some context  "
        )
        assert input_data.query == "What is AI?"
        assert input_data.context == "Some context"

    def test_minimal_valid_input(self):
        """Test minimal valid input."""
        input_data = AlphaInput(query="Hi")
        assert input_data.query == "Hi"
        assert input_data.context == ""


class TestBetaInputValidation:
    """Test Beta agent input validation."""

    def test_valid_input(self):
        """Test valid Beta input."""
        input_data = BetaInput(
            problem="Analyze the impact of AI on employment",
            domain="Technology",
            requirements=["Consider economic factors", "Include timeline"],
            constraints="Focus on next 5 years"
        )
        assert input_data.problem == "Analyze the impact of AI on employment"
        assert input_data.domain == "Technology"
        assert len(input_data.requirements) == 2
        assert input_data.constraints == "Focus on next 5 years"

    def test_empty_problem_validation(self):
        """Test that empty problem raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BetaInput(problem="")
        # Pydantic v2 built-in validation message for min_length=1
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_whitespace_only_problem_validation(self):
        """Test that whitespace-only problem raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BetaInput(problem="   \n\t   ")
        assert "Problem description cannot be empty" in str(exc_info.value)

    def test_problem_too_long(self):
        """Test that excessively long problem raises validation error."""
        long_problem = "x" * 15001  # Over the 15000 limit
        with pytest.raises(ValidationError) as exc_info:
            BetaInput(problem=long_problem)
        assert "String should have at most 15000 characters" in str(exc_info.value)

    def test_domain_too_long(self):
        """Test that excessively long domain raises validation error."""
        long_domain = "x" * 501  # Over the 500 limit
        with pytest.raises(ValidationError) as exc_info:
            BetaInput(problem="Valid problem", domain=long_domain)
        assert "String should have at most 500 characters" in str(exc_info.value)

    def test_constraints_too_long(self):
        """Test that excessively long constraints raises validation error."""
        long_constraints = "x" * 2001  # Over the 2000 limit
        with pytest.raises(ValidationError) as exc_info:
            BetaInput(problem="Valid problem", constraints=long_constraints)
        assert "String should have at most 2000 characters" in str(exc_info.value)

    def test_requirements_cleaning(self):
        """Test that requirements list is properly cleaned."""
        input_data = BetaInput(
            problem="Test problem",
            requirements=["  req1  ", "", "  req2  ", "   ", "req3"]
        )
        # Should remove empty/whitespace entries and trim
        assert input_data.requirements == ["req1", "req2", "req3"]

    def test_whitespace_trimming(self):
        """Test that whitespace is properly trimmed."""
        input_data = BetaInput(
            problem="  Test problem  ",
            domain="  Tech  ",
            constraints="  Some constraints  "
        )
        assert input_data.problem == "Test problem"
        assert input_data.domain == "Tech"
        assert input_data.constraints == "Some constraints"

    def test_minimal_valid_input(self):
        """Test minimal valid input."""
        input_data = BetaInput(problem="Test")
        assert input_data.problem == "Test"
        assert input_data.domain == ""
        assert input_data.requirements == []
        assert input_data.constraints == ""


class TestAPIValidation:
    """Test API endpoint validation."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_alpha_api_validation_short_query(self):
        """Test Alpha API rejects queries that are too short."""
        response = self.client.post(
            "/api/v1/agents/alpha/invoke",
            json={
                "payload": {"query": ""},  # Empty query should trigger validation
                "streaming": False
            }
        )
        assert response.status_code == 422

    def test_alpha_api_validation_valid_query(self):
        """Test Alpha API accepts valid queries."""
        response = self.client.post(
            "/api/v1/agents/alpha/invoke",
            json={
                "payload": {"query": "What is machine learning?", "context": "Educational"}, 
                "streaming": False
            }
        )
        # Should not be a validation error (might be 500 if Ollama not running, but not 422)
        assert response.status_code != 422

    def test_beta_api_validation_short_problem(self):
        """Test Beta API rejects problems that are too short."""
        response = self.client.post(
            "/api/v1/agents/beta/invoke",
            json={"problem": "Hi there"}  # Only 2 words
        )
        assert response.status_code == 422
        assert "at least 3 words" in response.json()["detail"]

    def test_beta_api_validation_too_many_requirements(self):
        """Test Beta API rejects too many requirements."""
        response = self.client.post(
            "/api/v1/agents/beta/invoke",
            json={
                "problem": "Analyze this complex problem thoroughly",
                "requirements": [f"Requirement {i}" for i in range(21)]  # 21 requirements
            }
        )
        assert response.status_code == 422
        assert "Maximum 20 requirements" in response.json()["detail"]

    def test_beta_api_validation_valid_problem(self):
        """Test Beta API accepts valid problems."""
        response = self.client.post(
            "/api/v1/agents/beta/invoke",
            json={
                "problem": "Analyze the impact of artificial intelligence",
                "domain": "Technology",
                "requirements": ["Consider benefits", "Consider risks"]
            }
        )
        # Should not be a validation error (might be 500 if Ollama not running, but not 422)
        assert response.status_code != 422

    def test_alpha_capabilities_endpoint(self):
        """Test Alpha capabilities endpoint returns validation info."""
        response = self.client.get("/api/v1/agents/alpha/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "validation_rules" in data
        assert "input_requirements" in data

    def test_beta_capabilities_endpoint(self):
        """Test Beta capabilities endpoint returns validation info."""
        response = self.client.get("/api/v1/agents/beta/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "validation_rules" in data
        assert "input_requirements" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
