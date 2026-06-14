"""Conversation memory manager for agentomatic.

Provides session memory (within a thread) and long-term memory
(automatic summarization of older messages) for LangGraph-based agents.

Usage::

    manager = ConversationMemoryManager(store, max_messages=50, summarize_after=30)

    # Before agent invocation — load history into state
    history = await manager.load_history(thread_id, "Hello!")
    state["messages"] = history

    # After agent invocation — persist the turn
    await manager.save_turn(thread_id, "Hello!", "Hi! How can I help?",
                            agent_name="my-agent", metadata={})
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from loguru import logger

from agentomatic.storage.base import BaseStore

# Default summary system prompt
_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Condense the following conversation "
    "into a concise summary that preserves all key facts, decisions, user preferences, "
    "and important context. Write in third person. Be thorough but concise."
)


class ConversationMemoryManager:
    """Manages conversation history with automatic windowing and summarization.

    Features:
        - **Session memory**: Loads prior messages from the store into agent state
        - **Windowing**: Keeps only the last ``max_messages`` in context
        - **Auto-summarization**: When messages exceed ``summarize_after``,
          older messages are compressed into a summary using the LLM
        - **Persistence**: Saves user and assistant messages after each turn

    Args:
        store: Storage backend implementing :class:`BaseStore`.
        max_messages: Maximum messages to keep in the context window.
        summarize_after: Number of messages that triggers summarization.
            When the total message count exceeds this threshold, older
            messages are replaced with a summary. Set to ``0`` to disable.
        summary_token_target: Approximate target length for summaries (in words).
    """

    def __init__(
        self,
        store: BaseStore,
        *,
        max_messages: int = 50,
        summarize_after: int = 30,
        summary_token_target: int = 200,
    ) -> None:
        self.store = store
        self.max_messages = max(1, max_messages)
        self.summarize_after = max(0, summarize_after)
        self.summary_token_target = max(50, summary_token_target)

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    async def get_or_create_thread(
        self,
        thread_id: str | None,
        user_id: str,
        agent_name: str,
        *,
        title: str | None = None,
    ) -> str:
        """Get an existing thread or auto-create one.

        Args:
            thread_id: Existing thread ID, or ``None`` to create a new one.
            user_id: The user who owns the thread.
            agent_name: Agent associated with this thread.
            title: Optional title for new threads.

        Returns:
            The thread ID (existing or newly created).
        """
        if thread_id:
            existing = await self.store.get_thread(thread_id)
            if existing:
                return thread_id
            # Thread ID provided but doesn't exist — create it
            logger.debug(f"Thread {thread_id} not found, creating new one")

        new_id = thread_id or f"thread_{uuid.uuid4().hex[:12]}"
        await self.store.create_thread(
            thread_id=new_id,
            user_id=user_id,
            agent_name=agent_name,
            title=title,
        )
        return new_id

    # ------------------------------------------------------------------
    # History loading
    # ------------------------------------------------------------------

    async def load_history(
        self,
        thread_id: str,
        current_message: str,
        *,
        max_messages: int | None = None,
        include_summary: bool = True,
    ) -> list[BaseMessage]:
        """Load conversation history as LangChain messages.

        Retrieves stored messages, applies windowing, and optionally
        prepends a summary of older messages.

        Args:
            thread_id: Thread to load history from.
            current_message: The current user message (appended at the end).
            max_messages: Override for ``self.max_messages``.
            include_summary: Whether to include a summary of older messages.

        Returns:
            List of :class:`BaseMessage` ready for LangGraph state.
        """
        limit = max_messages or self.max_messages
        result: list[BaseMessage] = []

        try:
            # Load all messages to check if summarization is needed
            all_messages = await self.store.get_messages(thread_id, limit=limit + 50)

            if not all_messages:
                # No history — just the current message
                return [HumanMessage(content=current_message)]

            total_count = len(all_messages)

            # Check if we need summarization
            if include_summary and self.summarize_after > 0 and total_count > self.summarize_after:
                # Split: older messages to summarize, recent to keep
                cutoff = total_count - limit
                older = all_messages[:cutoff] if cutoff > 0 else []
                recent = all_messages[cutoff:] if cutoff > 0 else all_messages

                if older:
                    summary = await self._get_or_generate_summary(thread_id, older)
                    if summary:
                        result.append(SystemMessage(content=f"[Conversation Summary]\n{summary}"))

                # Add recent messages
                result.extend(self._convert_to_langchain_messages(recent))
            else:
                # No summarization needed — just window
                windowed = all_messages[-limit:] if len(all_messages) > limit else all_messages
                result.extend(self._convert_to_langchain_messages(windowed))

            # Append the current user message
            result.append(HumanMessage(content=current_message))

            logger.debug(
                f"Loaded {len(result)} messages for thread {thread_id} "
                f"(total stored: {total_count}, windowed: {limit})"
            )

        except Exception as exc:
            logger.warning(f"Failed to load history for thread {thread_id}: {exc}")
            result = [HumanMessage(content=current_message)]

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save_turn(
        self,
        thread_id: str,
        user_message: str,
        assistant_response: str,
        *,
        agent_name: str = "",
        user_metadata: dict[str, Any] | None = None,
        assistant_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a user/assistant message pair to the store.

        Args:
            thread_id: Thread to save messages to.
            user_message: The user's message text.
            assistant_response: The assistant's response text.
            agent_name: Agent that produced the response.
            user_metadata: Optional metadata for the user message.
            assistant_metadata: Optional metadata for the assistant message.

        Returns:
            Dict with ``user_message_id`` and ``assistant_message_id``.
        """
        try:
            user_msg = await self.store.add_message(
                thread_id=thread_id,
                role="user",
                content=user_message,
                metadata=user_metadata or {},
            )

            asst_meta = {**(assistant_metadata or {})}
            if agent_name:
                asst_meta["agent_name"] = agent_name

            assistant_msg = await self.store.add_message(
                thread_id=thread_id,
                role="assistant",
                content=assistant_response,
                metadata=asst_meta,
            )

            return {
                "user_message_id": user_msg.get("id"),
                "assistant_message_id": assistant_msg.get("id"),
            }
        except Exception as exc:
            logger.warning(f"Failed to save turn for thread {thread_id}: {exc}")
            return {"user_message_id": None, "assistant_message_id": None}

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    async def summarize_messages(self, messages: list[dict[str, Any]]) -> str:
        """Compress a list of messages into a concise summary.

        Uses the configured LLM via ``get_llm()``. Falls back to a
        simple concatenation if LLM is unavailable.

        Args:
            messages: List of message dicts with ``role`` and ``content``.

        Returns:
            Summary text.
        """
        # Build conversation text
        conversation_lines = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            conversation_lines.append(f"{role}: {content}")

        conversation_text = "\n".join(conversation_lines)

        try:
            from agentomatic.providers.llm import get_llm

            llm = get_llm()
            summary_prompt = [
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Summarize this conversation in about "
                        f"{self.summary_token_target} words:\n\n{conversation_text}"
                    )
                ),
            ]
            result = await llm.ainvoke(summary_prompt)
            summary = result.content if hasattr(result, "content") else str(result)
            logger.debug(f"Generated summary ({len(summary)} chars) from {len(messages)} messages")
            return summary

        except Exception as exc:
            logger.warning(f"LLM summarization failed: {exc}. Using truncated fallback.")
            # Fallback: just take the last N lines as a crude summary
            return self._fallback_summary(messages)

    async def get_conversation_summary(self, thread_id: str) -> str:
        """Generate a summary for an entire thread.

        Args:
            thread_id: Thread to summarize.

        Returns:
            Summary text.
        """
        messages = await self.store.get_messages(thread_id, limit=500)
        if not messages:
            return "No messages in this thread."
        return await self.summarize_messages(messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_or_generate_summary(
        self, thread_id: str, messages: list[dict[str, Any]]
    ) -> str | None:
        """Get cached summary or generate a new one.

        Currently always generates fresh. A future version could cache
        summaries in thread metadata.
        """
        if not messages:
            return None
        return await self.summarize_messages(messages)

    @staticmethod
    def _convert_to_langchain_messages(
        messages: list[dict[str, Any]],
    ) -> list[BaseMessage]:
        """Convert stored message dicts to LangChain BaseMessage objects."""
        result: list[BaseMessage] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))
            else:
                # Default to HumanMessage for unknown roles
                result.append(HumanMessage(content=content))
        return result

    @staticmethod
    def _fallback_summary(messages: list[dict[str, Any]], max_chars: int = 500) -> str:
        """Create a simple summary without LLM by concatenating key points."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            lines.append(f"- {role}: {content}")

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return f"[Previous conversation summary ({len(messages)} messages)]\n{text}"
