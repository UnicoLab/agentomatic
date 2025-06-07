"""Alpha Agent - Example implementation using the new architecture."""

from typing import Dict, Any, AsyncGenerator
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from ...common.base_agent import BaseAgent
from ...common.llm_factory import LLMConfig
from ...common.nodes import BaseAgentNode
from .state import AlphaAgentState


class AlphaAgentNode(BaseAgentNode):
    """Specialized node for Alpha Agent operations."""

    async def process_input(self, state: AlphaAgentState) -> Command:
        """Process initial user input."""
        try:
            # Get the latest message
            messages = state.get("messages", [])
            if not messages:
                return self.create_command(
                    update={"error": "No input message provided"},
                    goto=END
                )

            last_message = messages[-1]
            user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)

            # Use the prompt manager to format the input processing prompt
            response = await self.invoke_llm(
                prompt_version="input_processing",
                context={
                    "user_input": user_input,
                    "agent_context": state.get("context", "")
                }
            )

            # Create AI response
            ai_message = self.create_ai_message(f"Processed input: {response}")

            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "processed_input": response,
                    "current_step": "analysis"
                },
                goto="analyze"
            )

        except Exception as e:
            return self.create_command(
                update={"error": f"Input processing failed: {str(e)}"},
                goto=END
            )

    async def analyze(self, state: AlphaAgentState) -> Command:
        """Analyze the processed input."""
        try:
            processed_input = state.get("processed_input", "")

            response = await self.invoke_llm(
                prompt_version="analysis",
                context={
                    "processed_input": processed_input,
                    "context": state.get("context", "")
                }
            )

            messages = state.get("messages", [])
            ai_message = self.create_ai_message(f"Analysis: {response}")

            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "analysis_result": response,
                    "current_step": "response_generation"
                },
                goto="generate_response"
            )

        except Exception as e:
            return self.create_command(
                update={"error": f"Analysis failed: {str(e)}"},
                goto=END
            )

    async def generate_response(self, state: AlphaAgentState) -> Command:
        """Generate the final response."""
        try:
            analysis_result = state.get("analysis_result", "")
            processed_input = state.get("processed_input", "")

            response = await self.invoke_llm(
                prompt_version="response_generation",
                context={
                    "analysis_result": analysis_result,
                    "processed_input": processed_input,
                    "context": state.get("context", "")
                }
            )

            messages = state.get("messages", [])
            ai_message = self.create_ai_message(response)

            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "final_response": response,
                    "current_step": "completed",
                    "completed": True
                },
                goto=END
            )

        except Exception as e:
            return self.create_command(
                update={"error": f"Response generation failed: {str(e)}"},
                goto=END
            )


class AlphaAgent(BaseAgent):
    """Alpha Agent with multi-step processing workflow."""

    def __init__(self, agent_name: str, llm_config: LLMConfig):
        super().__init__(agent_name, llm_config)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for Alpha Agent."""
        # Create the specialized node instance
        node = AlphaAgentNode(self.agent_name, self.llm, self.prompt_manager)

        # Create the graph
        graph = StateGraph(AlphaAgentState)

        # Add nodes
        graph.add_node("process_input", node.process_input)
        graph.add_node("analyze", node.analyze)
        graph.add_node("generate_response", node.generate_response)

        # Set entry point
        graph.set_entry_point("process_input")

        # Add edges - the Command pattern handles routing automatically
        graph.add_edge("process_input", "analyze")
        graph.add_edge("analyze", "generate_response")
        graph.add_edge("generate_response", END)

        return graph.compile()

    async def run(self, request, streaming: bool = False) -> Any:
        """Execute the Alpha Agent workflow."""
        try:
            # Initialize state
            initial_state = AlphaAgentState(
                messages=[HumanMessage(content=request.input)],
                context=request.context or "",
                current_step="input_processing",
                completed=False
            )

            if streaming:
                return self._run_streaming(initial_state)
            else:
                # Run the graph
                result = await self.graph.ainvoke(initial_state)
                return {
                    "output": result.get("final_response", "No response generated"),
                    "agent": self.agent_name,
                    "steps": result.get("current_step", "unknown"),
                    "completed": result.get("completed", False),
                    "error": result.get("error")
                }

        except Exception as e:
            return {
                "output": f"Agent execution failed: {str(e)}",
                "agent": self.agent_name,
                "error": str(e),
                "completed": False
            }

    async def _run_streaming(self, initial_state: AlphaAgentState) -> AsyncGenerator[str, None]:
        """Run the agent with streaming output."""
        try:
            async for chunk in self.graph.astream(initial_state):
                # Extract the step name and state from the chunk
                for step_name, step_state in chunk.items():
                    if "messages" in step_state and step_state["messages"]:
                        last_message = step_state["messages"][-1]
                        if hasattr(last_message, 'content'):
                            yield f"data: {last_message.content}\n\n"

                    # Yield step completion info
                    if "current_step" in step_state:
                        yield f"data: [STEP] {step_state['current_step']}\n\n"

                    # Yield errors
                    if "error" in step_state:
                        yield f"data: [ERROR] {step_state['error']}\n\n"
                        return

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] Streaming failed: {str(e)}\n\n"