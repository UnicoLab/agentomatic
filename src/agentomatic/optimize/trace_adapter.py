"""Trace adapters — convert run results / rollouts into critique contexts.

Ports the *idea* of Agent Lightning's ``TraceToMessages`` without OTEL
dependency: build chat-message views and APO critique payloads from
``RunResult`` / ``Rollout`` fields (steps, tools, reasoning, scores).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentomatic.optimize.rollout import Rollout, RolloutSpan
from agentomatic.optimize.runner import RunResult


@dataclass(slots=True)
class CritiqueExperiment:
    """One rollout packaged for textual-gradient / APO critique prompts."""

    status: str
    final_reward: float | None
    messages: list[dict[str, Any]] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    response: str = ""
    expected: str | None = None
    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for LLM prompt injection."""
        return {
            "status": self.status,
            "final_reward": self.final_reward,
            "messages": list(self.messages),
            "spans": list(self.spans),
            "query": self.query,
            "response": self.response,
            "expected": self.expected,
            "feedback": self.feedback,
        }


@dataclass(slots=True)
class TraceToMessages:
    """Adapt rollout / run-result traces to OpenAI-style messages."""

    node_match: str | None = None
    """Optional regex; when set, only matching step/tool names are kept."""

    def _node_allowed(self, name: str) -> bool:
        if not self.node_match:
            return True
        return re.search(self.node_match, name) is not None

    def adapt_run_result(self, rr: RunResult) -> list[dict[str, Any]]:
        """Build messages from a :class:`RunResult`."""
        messages: list[dict[str, Any]] = []
        if rr.query:
            messages.append({"role": "user", "content": rr.query})
        for step in rr.steps_taken or []:
            name = str(step)
            if not self._node_allowed(name):
                continue
            messages.append({"role": "assistant", "content": f"[step:{name}]"})
        for tool in rr.tool_calls or []:
            name = str(tool.get("name") or tool.get("tool") or "tool")
            if not self._node_allowed(name):
                continue
            content = tool.get("output") or tool.get("result") or tool
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": str(content)[:1500],
                }
            )
        if rr.reasoning and (not self.node_match or self._node_allowed("reasoning")):
            messages.append({"role": "assistant", "content": f"[reasoning] {rr.reasoning[:1200]}"})
        if rr.response:
            messages.append({"role": "assistant", "content": rr.response[:2000]})
        return messages

    def adapt_rollout(self, rollout: Rollout) -> list[dict[str, Any]]:
        """Prefer stored messages; otherwise rebuild from spans/steps."""
        if rollout.messages:
            if not self.node_match:
                return list(rollout.messages)
            filtered: list[dict[str, Any]] = []
            for msg in rollout.messages:
                name = str(msg.get("name") or msg.get("role") or "")
                content = str(msg.get("content") or "")
                if name and not self._node_allowed(name) and "[step:" in content:
                    continue
                if name and msg.get("role") == "tool" and not self._node_allowed(name):
                    continue
                filtered.append(msg)
            return filtered or list(rollout.messages)

        synthetic = RunResult(
            query=rollout.query,
            response=rollout.response,
            expected=rollout.expected,
            tool_calls=list(rollout.tool_calls),
            steps_taken=list(rollout.steps_taken),
            reasoning=rollout.reasoning,
            retrieval_context=list(rollout.retrieval_context),
        )
        return self.adapt_run_result(synthetic)

    def adapt_spans(self, spans: list[RolloutSpan]) -> list[dict[str, Any]]:
        """Filter spans by ``node_match`` and dump as dicts."""
        out: list[dict[str, Any]] = []
        for span in spans:
            if not self._node_allowed(span.name):
                continue
            out.append(span.to_dict())
        return out


@dataclass(slots=True)
class TraceToCritiqueContext:
    """Build APO-style critique experiment payloads from eval details."""

    adapter: TraceToMessages = field(default_factory=TraceToMessages)

    def from_eval_detail(self, detail: dict[str, Any]) -> CritiqueExperiment:
        """Convert a fitter ``eval_details`` row into a critique experiment."""
        rr = RunResult(
            query=str(detail.get("query") or ""),
            response=str(detail.get("response") or ""),
            expected=detail.get("expected"),
            tool_calls=list(detail.get("tool_calls") or []),
            steps_taken=list(detail.get("steps_taken") or []),
            reasoning=str(detail.get("reasoning") or ""),
            retrieval_context=list(detail.get("retrieval_context") or []),
            error=detail.get("error"),
        )
        reward = detail.get("avg_score", detail.get("score"))
        try:
            final_reward = float(reward) if reward is not None else None
        except (TypeError, ValueError):
            final_reward = None
        status = "failed" if detail.get("error") else "succeeded"
        messages = detail.get("messages")
        if not isinstance(messages, list):
            messages = self.adapter.adapt_run_result(rr)
        return CritiqueExperiment(
            status=status,
            final_reward=final_reward,
            messages=list(messages),
            spans=list(detail.get("spans") or []),
            query=rr.query,
            response=rr.response,
            expected=rr.expected,
            feedback=str(detail.get("feedback") or detail.get("reason") or ""),
        )

    def from_rollout(self, rollout: Rollout) -> CritiqueExperiment:
        """Convert a stored :class:`Rollout` into a critique experiment."""
        reward = rollout.reward.value if rollout.reward else None
        return CritiqueExperiment(
            status=rollout.status,
            final_reward=reward,
            messages=self.adapter.adapt_rollout(rollout),
            spans=self.adapter.adapt_spans(rollout.spans),
            query=rollout.query,
            response=rollout.response,
            expected=rollout.expected,
            feedback=(rollout.reward.reason if rollout.reward else "") or "",
        )

    def from_eval_details(
        self,
        details: list[dict[str, Any]],
        *,
        max_items: int = 8,
    ) -> list[CritiqueExperiment]:
        """Convert many eval rows, preferring failures / low scores first."""
        ranked = sorted(
            details,
            key=lambda d: float(d.get("avg_score", d.get("score", 1.0)) or 0.0),
        )
        return [self.from_eval_detail(d) for d in ranked[:max_items]]
