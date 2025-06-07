"""Common node patterns for agent workflows."""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.base import BaseLanguageModel

from ..prompt_manager import PromptManager


class BaseAgentNode(ABC):
    """Base class for all agent nodes."""

    def __init__(self, agent_name: str, llm: BaseLanguageModel, prompt_manager: PromptManager):
        self.agent_name = agent_name
        self.llm = llm
        self.prompt_manager = prompt_manager

    @abstractmethod
    async def __call__(self, state: Dict[str, Any]) -> Command:
        """Execute the node logic."""
        pass

    async def invoke_llm(self, prompt_version: str, context: Dict[str, Any]) -> str:
        """Invoke LLM with prompt template."""
        try:
            prompt = self.prompt_manager.get_prompt(prompt_version, **context)
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {str(e)}")

    def create_command(self, goto: str, update: Dict[str, Any]) -> Command:
        """Create a command for graph navigation."""
        return Command(goto=goto, update=update)

    def create_ai_message(self, content: str) -> AIMessage:
        """Create an AI message."""
        return AIMessage(content=content)

    def create_human_message(self, content: str) -> HumanMessage:
        """Create a human message."""
        return HumanMessage(content=content)


class SimpleProcessingNode(BaseAgentNode):
    """Simple node that applies a processing function."""

    def __init__(self, agent_name: str, llm: BaseLanguageModel, prompt_manager: PromptManager,
                 processor: Callable[[str], str], next_node: str = "END"):
        super().__init__(agent_name, llm, prompt_manager)
        self.processor = processor
        self.next_node = next_node

    async def __call__(self, state: Dict[str, Any]) -> Command:
        """Process input using the provided function."""
        try:
            messages = state.get("messages", [])
            if not messages:
                return self.create_command(
                    goto="END",
                    update={"error": "No input to process"}
                )

            latest_message = messages[-1]
            input_text = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

            # Apply processing function
            result = self.processor(input_text)

            # Create response message
            ai_message = self.create_ai_message(result)

            return self.create_command(
                goto=self.next_node,
                update={
                    "messages": messages + [ai_message],
                    "processed_result": result
                }
            )

        except Exception as e:
            return self.create_command(
                goto="END",
                update={"error": f"Processing failed: {str(e)}"}
            )


class LLMNode(BaseAgentNode):
    """Node that uses LLM for processing."""

    def __init__(self, agent_name: str, llm: BaseLanguageModel, prompt_manager: PromptManager,
                 prompt_version: str, next_node: str = "END",
                 state_key: str = "llm_result"):
        super().__init__(agent_name, llm, prompt_manager)
        self.prompt_version = prompt_version
        self.next_node = next_node
        self.state_key = state_key

    async def __call__(self, state: Dict[str, Any]) -> Command:
        """Process input using LLM."""
        try:
            messages = state.get("messages", [])
            if not messages:
                return self.create_command(
                    goto="END",
                    update={"error": "No input to process"}
                )

            latest_message = messages[-1]
            input_text = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

            # Prepare context for prompt
            context = {
                "input": input_text,
                "state": state,
                **state.get("metadata", {})
            }

            # Use LLM
            result = await self.invoke_llm(self.prompt_version, context)

            # Create response message
            ai_message = self.create_ai_message(result)

            return self.create_command(
                goto=self.next_node,
                update={
                    "messages": messages + [ai_message],
                    self.state_key: result
                }
            )

        except Exception as e:
            return self.create_command(
                goto="END",
                update={"error": f"LLM processing failed: {str(e)}"}
            )


class ConditionalNode(BaseAgentNode):
    """Node that routes based on a condition."""

    def __init__(self, agent_name: str, llm: BaseLanguageModel, prompt_manager: PromptManager,
                 condition: Callable[[Dict[str, Any]], bool],
                 true_node: str, false_node: str):
        super().__init__(agent_name, llm, prompt_manager)
        self.condition = condition
        self.true_node = true_node
        self.false_node = false_node

    async def __call__(self, state: Dict[str, Any]) -> Command:
        """Route based on condition."""
        try:
            if self.condition(state):
                next_node = self.true_node
            else:
                next_node = self.false_node

            return self.create_command(
                goto=next_node,
                update={"routing_decision": next_node}
            )

        except Exception as e:
            return self.create_command(
                goto="END",
                update={"error": f"Conditional routing failed: {str(e)}"}
            )


class ValidationNode(BaseAgentNode):
    """Node that validates input or state."""

    def __init__(self, agent_name: str, llm: BaseLanguageModel, prompt_manager: PromptManager,
                 validator: Callable[[Dict[str, Any]], tuple[bool, str]],
                 success_node: str, failure_node: str = "END"):
        super().__init__(agent_name, llm, prompt_manager)
        self.validator = validator
        self.success_node = success_node
        self.failure_node = failure_node

    async def __call__(self, state: Dict[str, Any]) -> Command:
        """Validate state and route accordingly."""
        try:
            is_valid, message = self.validator(state)

            if is_valid:
                return self.create_command(
                    goto=self.success_node,
                    update={"validation_result": "passed", "validation_message": message}
                )
            else:
                return self.create_command(
                    goto=self.failure_node,
                    update={"error": f"Validation failed: {message}"}
                )

        except Exception as e:
            return self.create_command(
                goto=self.failure_node,
                update={"error": f"Validation error: {str(e)}"}
            )


# Utility functions for creating common node patterns

def create_simple_agent_node(agent_name: str, llm: BaseLanguageModel,
                           prompt_manager: PromptManager, prompt_version: str,
                           next_node: str = "END") -> LLMNode:
    """Create a simple LLM-based node."""
    return LLMNode(agent_name, llm, prompt_manager, prompt_version, next_node)


def create_processing_node(agent_name: str, llm: BaseLanguageModel,
                         prompt_manager: PromptManager,
                         processor: Callable[[str], str],
                         next_node: str = "END") -> SimpleProcessingNode:
    """Create a simple processing node."""
    return SimpleProcessingNode(agent_name, llm, prompt_manager, processor, next_node)


def create_conditional_node(agent_name: str, llm: BaseLanguageModel,
                          prompt_manager: PromptManager,
                          condition: Callable[[Dict[str, Any]], bool],
                          true_node: str, false_node: str) -> ConditionalNode:
    """Create a conditional routing node."""
    return ConditionalNode(agent_name, llm, prompt_manager, condition, true_node, false_node)


def create_validation_node(agent_name: str, llm: BaseLanguageModel,
                         prompt_manager: PromptManager,
                         validator: Callable[[Dict[str, Any]], tuple[bool, str]],
                         success_node: str, failure_node: str = "END") -> ValidationNode:
    """Create a validation node."""
    return ValidationNode(agent_name, llm, prompt_manager, validator, success_node, failure_node)