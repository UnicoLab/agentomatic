"""Non-saturating structured-output metrics for prompt fitting.

These helpers score JSON agent outputs against rich ``expected_output``
labels (string fields, ``must_include`` / ``must_not_include``) so demo
fits do not trivially saturate at 1.0 on schema presence alone.
"""

from __future__ import annotations

import json
from typing import Any


def parse_prediction_json(response: str) -> dict[str, Any]:
    """Extract a JSON object from a raw model response."""
    if not response:
        return {}
    try:
        data = json.loads(response)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        start, end = response.find("{"), response.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            data = json.loads(response[start : end + 1])
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


def parse_expected_spec(expected: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse structured expected_output from a string, quality contract, or dict."""
    if isinstance(expected, dict):
        return dict(expected)
    if not isinstance(expected, str) or not expected.strip():
        return {}
    text = expected.strip()
    marker = "## Expected structured output"
    if marker in text:
        blob = text.split(marker, 1)[1].strip()
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return {"content": text[:500], "must_include": []}


def token_f1(a: str, b: str) -> float:
    """Token-level F1 between two strings (case-insensitive)."""
    ta = {t for t in a.lower().split() if t}
    tb = {t for t in b.lower().split() if t}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    precision = inter / len(tb)
    recall = inter / len(ta)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def coverage(terms: list[str], text: str) -> float:
    """Fraction of required terms found in *text* (substring, case-insensitive)."""
    if not terms:
        return 0.0
    lowered = text.lower()
    return sum(1 for t in terms if t and t.lower() in lowered) / len(terms)


def schema_quality(prediction: dict[str, Any], required_keys: list[str]) -> float:
    """Keys present with non-empty string values (not just JSON validity)."""
    if not isinstance(prediction, dict) or not prediction:
        return 0.0
    scores: list[float] = []
    for key in required_keys:
        val = prediction.get(key)
        if isinstance(val, str) and val.strip():
            scores.append(1.0)
        elif key in prediction:
            scores.append(0.35)
        else:
            scores.append(0.0)
    return sum(scores) / max(len(scores), 1)


def structured_composite_score(
    query: str,
    response: str,
    expected: str | dict[str, Any] | None = None,
    *,
    required_keys: list[str] | None = None,
    context: Any = None,  # noqa: ARG001 — CustomMetric signature
) -> float:
    """Hard fit metric that does **not** saturate on schema-only JSON.

    Weights:
      - schema quality: 0.20
      - content token F1 vs expected content: 0.35
      - must_include coverage: 0.25
      - next_action substance + overlap: 0.20
      - must_not_include penalty (subtract up to 0.30)
    """
    keys = required_keys or ["content", "next_action", "response"]
    # Prefer content/next_action when present; else first two keys.
    primary = [k for k in ("content", "next_action", "response") if k in keys] or keys[:2]

    data = parse_prediction_json(response)
    if not data:
        return 0.0
    exp = parse_expected_spec(expected)
    pred_content = str(data.get("content") or data.get("response") or "")
    pred_next = str(data.get("next_action") or "")
    text = json.dumps(data, ensure_ascii=False)

    schema = schema_quality(data, primary)

    exp_content = exp.get("content") or exp.get("response")
    content_score = (
        token_f1(str(exp_content), pred_content)
        if isinstance(exp_content, str) and exp_content.strip()
        else (0.5 if pred_content.strip() else 0.0)
    )

    must = [str(t) for t in (exp.get("must_include") or []) if str(t).strip()]
    include_score = coverage(must, text) if must else content_score

    exp_next = exp.get("next_action")
    next_overlap = (
        token_f1(str(exp_next), pred_next)
        if isinstance(exp_next, str) and exp_next.strip()
        else 0.0
    )
    next_substance = 1.0 if len(pred_next.split()) >= 4 else (0.4 if pred_next.strip() else 0.0)
    next_score = 0.6 * next_substance + 0.4 * next_overlap
    if "next_action" not in primary:
        next_score = content_score

    forbid = [str(t) for t in (exp.get("must_not_include") or []) if str(t).strip()]
    penalty = 0.3 * coverage(forbid, text) if forbid else 0.0

    score = (
        0.20 * schema + 0.35 * content_score + 0.25 * include_score + 0.20 * next_score - penalty
    )
    if query and pred_content:
        q_terms = [w for w in query.lower().split() if len(w) > 3][:6]
        if q_terms and coverage(q_terms, pred_content.lower()) == 0.0:
            score *= 0.85
    return float(max(0.0, min(1.0, score)))


def make_structured_fit_metric(
    required_keys: list[str] | None = None,
    *,
    name: str = "composite",
) -> Any:
    """Return an optimize ``CustomMetric`` using :func:`structured_composite_score`."""
    from agentomatic.optimize.metrics import CustomMetric

    keys = list(required_keys or ["content", "next_action"])

    def _fn(
        query: str,
        response: str,
        expected: str | None = None,
        context: Any = None,
    ) -> float:
        return structured_composite_score(
            query,
            response,
            expected,
            required_keys=keys,
            context=context,
        )

    return CustomMetric(fn=_fn, name=name)


def agent_keyword_score(example: Any, prediction: dict[str, Any]) -> float:
    """Agent-lifecycle metric: must_include coverage with forbid penalty."""
    expected = dict(getattr(example, "expected_output", None) or {})
    text = json.dumps(prediction, ensure_ascii=False).lower()
    must = [str(t) for t in (expected.get("must_include") or []) if str(t).strip()]
    if not must:
        for key, val in expected.items():
            if key.startswith("must_"):
                continue
            if isinstance(val, str) and val.strip() and val.lower() not in {"true", "false"}:
                must.extend([w for w in val.lower().split() if len(w) > 3][:6])
            elif val is True:
                must.append(str(key))
    if not must:
        return 0.0
    hit = coverage(must, text)
    forbid = [str(t) for t in (expected.get("must_not_include") or []) if str(t).strip()]
    penalty = 0.25 * coverage(forbid, text) if forbid else 0.0
    return max(0.0, hit - penalty)


def agent_field_f1(
    example: Any,
    prediction: dict[str, Any],
    *,
    keys: list[str] | None = None,
) -> float:
    """Token F1 averaged over string expected fields."""
    expected = dict(getattr(example, "expected_output", None) or {})
    field_keys = keys or ["content", "next_action", "response"]
    scores: list[float] = []
    for key in field_keys:
        exp = expected.get(key)
        pred = prediction.get(key, "")
        if not isinstance(exp, str) or not exp.strip():
            continue
        if not isinstance(pred, str) or not pred.strip():
            scores.append(0.0)
            continue
        scores.append(token_f1(exp, pred))
    return sum(scores) / max(len(scores), 1) if scores else 0.0


def agent_schema_quality(
    example: Any,  # noqa: ARG001
    prediction: dict[str, Any],
    *,
    required_keys: list[str] | None = None,
) -> float:
    """Agent-lifecycle schema quality metric."""
    return schema_quality(prediction, required_keys or ["content", "next_action"])
