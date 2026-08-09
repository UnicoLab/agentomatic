"""Multi-strategy JSON extraction from LLM responses.

Extracts JSON objects from noisy LLM outputs using a cascade of
strategies: direct parse, code-block extraction, structural regex,
and JSON repair.  Designed for robustness when models do not emit
perfect JSON.

Example::

    from agentomatic.optimize.json_extractor import JSONExtractor

    extractor = JSONExtractor()
    data = extractor.extract('Sure! Here is your JSON: ```json {"ok": true}```')
    # → {"ok": True}
"""

from __future__ import annotations

import json
import re
from typing import Any


class JSONExtractor:
    """Multi-pass JSON extraction from arbitrary text.

    Tries increasingly aggressive strategies until valid JSON is
    obtained, then validates structural constraints.

    Example::

        extractor = JSONExtractor()
        data = extractor.extract(model_response)
    """

    def __init__(
        self,
        max_attempts: int = 5,
        required_keys: list[str] | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.required_keys = required_keys or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict[str, Any]:
        """Extract a JSON object from *text*.

        Returns an empty dict if no valid JSON is found.
        """
        if not text:
            return {}

        strategies = [
            self._try_parse,
            self._extract_code_block,
            self._find_json_structure,
            self._repair_json,
        ]

        for strategy in strategies:
            try:
                result = strategy(text)
                if isinstance(result, dict) and result:
                    if self._validate(result):
                        return result
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        return {}

    def extract_list(self, text: str) -> list[Any]:
        """Extract a JSON array from *text*.

        Returns an empty list on failure.
        """
        for strategy in [self._try_parse_list, self._extract_code_block_list]:
            try:
                result = strategy(text)
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return []

    # ------------------------------------------------------------------
    # Strategy: direct parse
    # ------------------------------------------------------------------

    def _try_parse(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise

    def _try_parse_list(self, text: str) -> Any:
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        raise ValueError("Not a JSON array")

    # ------------------------------------------------------------------
    # Strategy: code-block extraction
    # ------------------------------------------------------------------

    def _extract_code_block(self, text: str) -> Any:
        # Find ```json ... ``` blocks
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        raise ValueError("No code block found")

    def _extract_code_block_list(self, text: str) -> Any:
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match.strip())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue
        raise ValueError("No list code block found")

    # ------------------------------------------------------------------
    # Strategy: structural regex
    # ------------------------------------------------------------------

    def _find_json_structure(self, text: str) -> Any:
        # Find the outermost { ... } or [ ... ]
        obj_start = text.find("{")
        arr_start = text.find("[")
        if obj_start >= 0 and (arr_start < 0 or obj_start < arr_start):
            end = self._find_matching_brace(text, obj_start, "{", "}")
            if end > obj_start:
                return json.loads(text[obj_start : end + 1])
        if arr_start >= 0:
            end = self._find_matching_brace(text, arr_start, "[", "]")
            if end > arr_start:
                return json.loads(text[arr_start : end + 1])
        raise ValueError("No JSON structure found")

    @staticmethod
    def _find_matching_brace(text: str, start: int, open_c: str, close_c: str) -> int:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    # ------------------------------------------------------------------
    # Strategy: JSON repair
    # ------------------------------------------------------------------

    def _repair_json(self, text: str) -> Any:
        repaired = text.strip()

        # Fix common issues
        repairs: list[tuple[str, str]] = [
            # Trailing commas in objects
            (r",(\s*})", r"\1"),
            # Trailing commas in arrays
            (r",(\s*])", r"\1"),
            # Single quotes → double quotes (keys and simple string values)
            (r"'([^']*)'(\s*):", r'"\1"\2:'),
            # Unquoted keys (word characters only)
            (r"([{,]\s*)(\w+)(\s*):", r'\1"\2"\3:'),
            # Boolean/null lowercase
            ("True", "true"),
            ("False", "false"),
            ("None", "null"),
            # Ellipsis → empty string (escape dots — they are regex wildcards)
            (r"\.\.\.", '""'),
        ]

        for pattern, replacement in repairs:
            repaired = re.sub(pattern, replacement, repaired)

        # Try parsing what's left
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Last resort: extract just the first complete object
            return self._find_json_structure(repaired)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, data: dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        if not self.required_keys:
            return True
        return all(k in data for k in self.required_keys)


# =====================================================================
# Convenience
# =====================================================================

_DEFAULT_EXTRACTOR = JSONExtractor()


def extract_json(text: str, **kwargs: Any) -> dict[str, Any]:
    """Single-call JSON extraction using the default extractor.

    Args:
        text: Raw LLM output text.
        kwargs: Passed to :class:`JSONExtractor`.

    Returns:
        Extracted dict (empty on failure).
    """
    if kwargs:
        return JSONExtractor(**kwargs).extract(text)
    return _DEFAULT_EXTRACTOR.extract(text)
