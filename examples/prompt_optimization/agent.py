"""Support agent shared by every prompt-optimization example.

A small but realistic RAG class agent: it answers from policy snippets —
the documents sent with the request (``context.documents``) when there are
any, otherwise the ones it retrieves from an in-memory knowledge base — and
takes the customer's plan (``customer_plan``) into account. It lists what it
answered from under ``sources``, so judges can check groundedness and the
optimizer sees what each answer was built from.

What the optimizer tunes is the **system prompt** (and, optionally, few-shot
examples and model parameters). The agent reads it through
:meth:`~agentomatic.agents.BaseGraphAgent.resolve_system_prompt`, which is the
hook every optimization path uses to inject a candidate prompt:

1. ``system_prompt_override`` on the request (how candidates are scored),
2. ``compiled_config["system_prompt"]`` (what ``fit()`` keeps as the best),
3. the ``system_prompt`` class attribute below (the baseline).

An agent that hardcodes its prompt inside a node cannot be optimized — always
go through ``resolve_system_prompt``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentomatic.agents import BaseGraphAgent

#: The facts the agent may use. Answers are only "right" if grounded here, so
#: a metric can check them deterministically (see ``datasets/support.jsonl``).
KNOWLEDGE_BASE: dict[str, str] = {
    "refund": "Refunds are available within 30 days of purchase for annual plans only.",
    "password": "Reset a password from Settings > Security > Reset password; links expire "
    "after 15 minutes.",
    "outage": "Service status and incident updates are published at status.acme.example.",
    "export": "Workspace owners can export all data as CSV or JSON from Settings > Data.",
    "seats": "Seats can be added any time; they are billed pro rata until the renewal date.",
    "sso": "Single sign-on (SAML) is available on the Enterprise plan only.",
    "support hours": "Human support is available Monday to Friday, 08:00-18:00 CET.",
    "delete": "Deleted workspaces can be restored by support for 14 days, then are purged.",
    "api": "The public API allows 600 requests per minute per workspace.",
    "invoice": "Invoices are emailed to the billing contact and listed under Billing > Invoices.",
}

#: Keyword patterns that route a question to a topic (a stand-in for a real
#: retriever — swap in a vector store via ``connections.py`` in a project).
TOPIC_PATTERNS: dict[str, str] = {
    "refund": r"\brefund|money back|\bcancel",
    "password": r"\bpassword",
    "outage": r"\boutage|\bdown\b|\bstatus\b|\bincident",
    "export": r"\bexport|\bdownload|\bbackup|\bcsv\b|\bjson\b",
    "seats": r"\bseats?\b|\badd\b.*\busers?\b",
    "sso": r"\bsso\b|\bsaml\b|single sign-on",
    "support hours": r"support hours|reach support|\bsaturday|\bsunday|\bweekend",
    "delete": r"\bdelet|\brestor|\brecover",
    "api": r"\bapi\b|rate limit|\b429\b",
    "invoice": r"\binvoice|\breceipt",
}


@dataclass
class SupportState:
    """Per-run state of :class:`SupportAgent`."""

    question: str = ""
    #: Documents sent with the request (``context.documents``), if any.
    documents: list[Any] = field(default_factory=list)
    #: The customer's plan, when the caller (or the dataset row) gives it.
    customer_plan: str = ""
    snippets: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)


def documents_from(context: Any) -> list[Any]:
    """Documents from ``{"documents": [...]}``, a list, or one string."""
    if isinstance(context, dict):
        context = context.get("documents")
    if isinstance(context, str):
        context = [context]
    return [doc for doc in context or [] if doc]


def snippet_text(doc: Any) -> str:
    """A document as one snippet line: its text, then its source if it has one."""
    if isinstance(doc, dict):
        text = str(doc.get("content") or doc.get("text") or "")
        return f"{text} (source: {doc['source']})" if doc.get("source") else text
    return str(doc)


class SupportAgent(BaseGraphAgent[SupportState]):
    """Answer customer questions from the knowledge base.

    Args:
        llm: Any LangChain-style chat model (``.invoke(messages)``). ``None``
            makes the agent echo its retrieved snippets, which keeps the
            examples importable without a model server.
    """

    agent_name = "support_agent"
    agent_description = "Answers Acme Cloud customer questions from policy snippets"

    #: Baseline prompt. Deliberately vague so there is something to improve.
    system_prompt = "You are a support assistant for Acme Cloud. Help the customer."

    def __init__(self, *, llm: Any = None) -> None:
        super().__init__()
        self.llm = llm

    # --- graph wiring -----------------------------------------------------

    def build_graph(self) -> Any:
        g = self.new_graph()
        g.add_node("retrieve", self.retrieve)
        g.add_node("answer", self.answer)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "answer")
        g.set_finish_point("answer")
        return g.compile()

    # --- nodes --------------------------------------------------------------

    def retrieve(self, state: SupportState) -> SupportState:
        """Use the supplied documents, else the knowledge-base snippets on topic."""
        if state.documents:
            state.snippets = [snippet_text(doc) for doc in state.documents]
            return state
        question = state.question.lower()
        state.snippets = [
            KNOWLEDGE_BASE[topic]
            for topic, pattern in TOPIC_PATTERNS.items()
            if re.search(pattern, question)
        ]
        return state

    def answer(self, state: SupportState) -> SupportState:
        """Answer with the model, grounded in the retrieved snippets."""
        # The optimizer's candidate prompt arrives through this call.
        prompt = self.resolve_system_prompt(default=self.system_prompt)
        context = "\n".join(f"- {s}" for s in state.snippets) or "- (no matching policy)"
        plan = f"Customer plan: {state.customer_plan}\n\n" if state.customer_plan else ""
        if self.llm is None:
            text = " ".join(state.snippets) or "I could not find a matching policy."
        else:
            reply = self.llm.invoke(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Policy snippets:\n{context}\n\n{plan}Question: {state.question}"
                        ),
                    },
                ]
            )
            text = str(getattr(reply, "content", reply)).strip()
        state.output = {"response": text, "sources": list(state.snippets)}
        return state

    # --- I/O contract -------------------------------------------------------

    def input_to_state(self, data: dict[str, Any]) -> SupportState:
        # REST sends ``query``; the platform normalises it to ``current_query``.
        # Datasets may use ``question`` — accept all three.
        question = data.get("current_query") or data.get("question") or data.get("query") or ""
        # ``context.documents`` arrives nested (and flattened to ``documents``
        # over REST). Every other dataset input — here ``customer_plan`` —
        # arrives exactly as written in the row.
        documents = documents_from(data.get("context")) or documents_from(data.get("documents"))
        return SupportState(
            question=str(question),
            documents=documents,
            customer_plan=str(data.get("customer_plan") or ""),
        )

    def state_to_output(self, state: SupportState) -> dict[str, Any]:
        return state.output
