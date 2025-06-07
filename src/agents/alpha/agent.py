"""Alpha Agent - Simplified implementation."""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from ...common.base_agent import BaseAgent
from ...common.llm_factory import LLMFactory, LLMConfig, LLMProvider
from ...common.agent_state import AgentState
from ...common.prompt_manager import PromptManager
from .config import AlphaConfig
from .schemas import AlphaInput, AlphaOutput


class AlphaAgent(BaseAgent):
    """Alpha Agent - General purpose analysis and response generation."""

    def __init__(self):
        # Create LLM config from agent config
        agent_config = AlphaConfig()
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name=agent_config.model_name,
            temperature=agent_config.temperature,
            max_tokens=agent_config.max_tokens,
            timeout=agent_config.timeout,
            base_url="http://localhost:11434"
        )

        # Create LLM and prompt manager
        llm = LLMFactory.create_llm_sync(llm_config)
        prompt_manager = PromptManager("alpha")

        super().__init__(
            name="alpha",
            llm=llm,
            prompt_manager=prompt_manager
        )

        self.config = agent_config

    def build_graph(self) -> StateGraph:
        """Build the Alpha agent workflow graph."""
        graph = StateGraph(AgentState)

        # Add processing node
        graph.add_node("process", self._process_node)

        # Set entry point and end
        graph.set_entry_point("process")
        graph.add_edge("process", END)

        return graph.compile()

    async def _process_node(self, state: Dict[str, Any]) -> Command:
        """Main processing node for Alpha agent."""
        try:
            messages = state.get("messages", [])
            if not messages:
                return Command(
                    update={"error": "No input provided"},
                    goto=END
                )

            user_input = messages[-1].content

            # Get prompt and generate response
            prompt = self.format_prompt("v1", query=user_input, context=state.get("context", ""))
            if not prompt:
                # Fallback if prompt not found
                prompt = f"You are Alpha, a helpful AI assistant. Please respond to: {user_input}"

            response = await self.generate_response(prompt)

            return Command(
                update={
                    "output": response,
                    "completed": True
                },
                goto=END
            )

        except Exception as e:
            return Command(
                update={"error": str(e)},
                goto=END
            )

    async def run(self, input_data, streaming: bool = False):
        """Run the Alpha agent."""
        # Convert input to proper format
        if hasattr(input_data, 'query'):
            query = input_data.query
            context = getattr(input_data, 'context', "")
        else:
            query = str(input_data)
            context = ""

        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            context=context,
            agent_name="alpha",
            classification=None,
            output=None,
            completed=False,
            error=None,
            metadata={}
        )

        if streaming:
            return self._stream_response(initial_state)

        try:
            # Run synchronously
            result = await self.graph.ainvoke(initial_state)

            # Handle case where result might be None or missing output
            output = result.get("output") if result else None
            if output is None:
                output = "Error: No response generated"

            # Return properly formatted output
            return AlphaOutput(
                response=output,
                confidence=0.8,  # Default confidence
                tokens_used=len(output) if output else 0,
                processing_time=0.0,  # Could be tracked
                prompt_version="v1"
            )
        except Exception as e:
            # Return error response
            error_msg = f"Agent execution failed: {str(e)}"
            return AlphaOutput(
                response=error_msg,
                confidence=0.0,
                tokens_used=len(error_msg),
                processing_time=0.0,
                prompt_version="v1"
            )

    async def _stream_response(self, initial_state):
        """Stream response chunks."""
        async for chunk in self.graph.astream(initial_state):
            if "output" in chunk:
                yield f"data: {chunk['output']}\n\n"
            if chunk.get("error"):
                yield f"data: [ERROR] {chunk['error']}\n\n"
                return
        yield "data: [DONE]\n\n"


# Export for auto-discovery
agent = AlphaAgent()