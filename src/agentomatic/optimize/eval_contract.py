"""Evaluation contracts for the PromptFitter optimisation loop.

An ``EvalContract`` defines the **expected input/output schema** of an agent
endpoint.  This is Agentomatic's deployment-first equivalent of DSPy
*signatures* — but it is not a programming abstraction.  It is a quality
gate that judges can score against.

Usage
-----
::

    contract = EvalContract(
        name="scoping_response",
        input_fields=["query", "context"],
        output_format="json",
        required_output_fields=["answer", "confidence", "risks", "next_questions"],
        optional_output_fields=["missing_information", "citations"],
        constraints=[
            "confidence must be a float between 0.0 and 1.0",
            "risks must be a non-empty list",
        ],
    )

    # Use in scoring
    score = contract.validate(response_text)  # 0.0–1.0
    metric = contract.as_metric(weight=0.10)  # DeterministicMetric
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalContract:
    """Declarative input/output contract for agent evaluation.

    Instead of DSPy-style ``Signature`` objects that compile into programs,
    an ``EvalContract`` describes **what a correct response looks like** so
    metrics and judges can score it.  Users never need to change agent code.

    Attributes:
        name: Human-readable name for the contract (e.g. ``"scoping_response"``).
        input_fields: Expected input field names (``["query", "context"]``).
        output_format: ``"json"`` | ``"text"`` | ``"markdown"`` — expected format.
        required_output_fields: Fields that MUST be present in JSON output.
        optional_output_fields: Fields that MAY be present.
        constraints: Free-text constraints for LLM judges to evaluate against.
        max_output_length: Maximum allowed response length in characters.
        min_output_length: Minimum allowed response length in characters.

    Examples::

        contract = EvalContract(
            name="support_response",
            input_fields=["query"],
            output_format="json",
            required_output_fields=["answer", "confidence"],
            constraints=["confidence must be between 0.0 and 1.0"],
        )

        # Quick validate
        score = contract.validate('{"answer": "...", "confidence": 0.8}')
        assert score == 1.0
    """

    name: str = "default"
    input_fields: list[str] = field(default_factory=lambda: ["query"])
    output_format: str = "text"  # "json", "text", "markdown"
    required_output_fields: list[str] = field(default_factory=list)
    optional_output_fields: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    max_output_length: int | None = None
    min_output_length: int | None = None

    # ── Validation ───────────────────────────────────────────────────

    def validate(self, response: str) -> float:
        """Quick structural validation returning a score 0.0–1.0.

        Checks format, required fields, and length constraints.
        Does NOT evaluate semantic quality (that's the judge's job).

        Args:
            response: The agent's raw response text.

        Returns:
            Score between 0.0 and 1.0 based on structural compliance.
        """
        if not response or not response.strip():
            return 0.0

        checks_passed = 0
        total_checks = 0

        # ── Format check ─────────────────────────────────────────
        if self.output_format == "json":
            total_checks += 1
            try:
                data = json.loads(response)
                checks_passed += 1

                # Required fields
                if self.required_output_fields and isinstance(data, dict):
                    for field_name in self.required_output_fields:
                        total_checks += 1
                        if field_name in data and data[field_name] is not None:
                            checks_passed += 1

            except (json.JSONDecodeError, ValueError):
                # JSON parse failed — still count field checks as failures
                total_checks += len(self.required_output_fields)

        elif self.output_format == "markdown":
            total_checks += 1
            # Check for at least one markdown header
            if re.search(r"^#{1,6}\s", response, re.MULTILINE):
                checks_passed += 1

        # ── Length checks ─────────────────────────────────────────
        if self.max_output_length is not None:
            total_checks += 1
            if len(response) <= self.max_output_length:
                checks_passed += 1

        if self.min_output_length is not None:
            total_checks += 1
            if len(response) >= self.min_output_length:
                checks_passed += 1

        if total_checks == 0:
            return 1.0  # No checks defined → pass by default

        return checks_passed / total_checks

    def validate_details(self, response: str) -> dict[str, Any]:
        """Validate with detailed per-check results.

        Returns:
            Dict with ``score``, ``passed``, ``failed``, and ``checks`` list.
        """
        checks: list[dict[str, Any]] = []

        if not response or not response.strip():
            return {
                "score": 0.0,
                "passed": 0,
                "failed": 1,
                "checks": [{"name": "non_empty", "passed": False}],
            }

        # ── Format ────────────────────────────────────────────────
        if self.output_format == "json":
            try:
                data = json.loads(response)
                checks.append({"name": "json_valid", "passed": True})

                if isinstance(data, dict):
                    for f in self.required_output_fields:
                        present = f in data and data[f] is not None
                        checks.append({"name": f"field:{f}", "passed": present})
            except (json.JSONDecodeError, ValueError):
                checks.append({"name": "json_valid", "passed": False})
                for f in self.required_output_fields:
                    checks.append({"name": f"field:{f}", "passed": False})

        elif self.output_format == "markdown":
            has_header = bool(re.search(r"^#{1,6}\s", response, re.MULTILINE))
            checks.append({"name": "markdown_header", "passed": has_header})

        # ── Length ────────────────────────────────────────────────
        if self.max_output_length is not None:
            checks.append(
                {"name": "max_length", "passed": len(response) <= self.max_output_length}
            )
        if self.min_output_length is not None:
            checks.append(
                {"name": "min_length", "passed": len(response) >= self.min_output_length}
            )

        passed = sum(1 for c in checks if c["passed"])
        failed = len(checks) - passed
        score = passed / len(checks) if checks else 1.0

        return {"score": score, "passed": passed, "failed": failed, "checks": checks}

    # ── Metric conversion ────────────────────────────────────────

    def as_metric(self, weight: float = 0.10) -> Any:
        """Convert this contract into a ``DeterministicMetric``.

        Returns a metric that can be plugged into ``CompositeMetric``.

        Example::

            contract = EvalContract(output_format="json", required_output_fields=["answer"])
            metric = contract.as_metric(weight=0.10)
        """
        from agentomatic.optimize.metrics import DeterministicMetric

        checks: list[dict[str, Any]] = []

        if self.output_format == "json":
            checks.append({"type": "json_valid", "value": True})
            for f in self.required_output_fields:
                checks.append({"type": "contains", "value": f'"{f}"'})

        if self.max_output_length:
            checks.append({"type": "max_length", "value": self.max_output_length})
        if self.min_output_length:
            checks.append({"type": "min_length", "value": self.min_output_length})

        return DeterministicMetric(
            name=f"contract:{self.name}",
            checks=checks,
        )

    def as_judge_criteria(self) -> str:
        """Generate criteria text for an LLM judge based on this contract.

        Useful for feeding into ``LocalJudgeMetric`` as the evaluation criteria.
        """
        parts = [f"Evaluate the response against the '{self.name}' contract."]

        if self.output_format != "text":
            parts.append(f"The response MUST be in {self.output_format} format.")

        if self.required_output_fields:
            fields = ", ".join(self.required_output_fields)
            parts.append(f"Required fields: {fields}.")

        if self.constraints:
            parts.append("Constraints:")
            for c in self.constraints:
                parts.append(f"  - {c}")

        return "\n".join(parts)

    # ── Serialisation ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d: dict[str, Any] = {"name": self.name}
        if self.input_fields:
            d["input_fields"] = self.input_fields
        d["output_format"] = self.output_format
        if self.required_output_fields:
            d["required_output_fields"] = self.required_output_fields
        if self.optional_output_fields:
            d["optional_output_fields"] = self.optional_output_fields
        if self.constraints:
            d["constraints"] = self.constraints
        if self.max_output_length is not None:
            d["max_output_length"] = self.max_output_length
        if self.min_output_length is not None:
            d["min_output_length"] = self.min_output_length
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalContract:
        """Deserialize from dictionary."""
        return cls(
            name=data.get("name", "default"),
            input_fields=data.get("input_fields", ["query"]),
            output_format=data.get("output_format", "text"),
            required_output_fields=data.get("required_output_fields", []),
            optional_output_fields=data.get("optional_output_fields", []),
            constraints=data.get("constraints", []),
            max_output_length=data.get("max_output_length"),
            min_output_length=data.get("min_output_length"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> EvalContract:
        """Load from a YAML file.

        Example YAML::

            name: scoping_response
            input_fields: [query, context]
            output_format: json
            required_output_fields:
              - answer
              - confidence
              - risks
            constraints:
              - confidence must be between 0.0 and 1.0
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML loading: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)
