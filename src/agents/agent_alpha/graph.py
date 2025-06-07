"""Updated Alpha Agent with new architecture and node pattern."""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage

from ...common.base_agent import BaseAgent
from ...common.nodes import BaseAgentNode, create_simple_agent_node
from ...common.llm_factory import LLMFactory
from ...common.prompt_manager import PromptManager
from .schemas import AlphaAgentState, AlphaRequest, AlphaResponse


class AlphaAnalysisNode(BaseAgentNode):
    """Node for initial analysis phase."""

    async def __call__(self, state: AlphaAgentState) -> Command:
        """Perform initial analysis of the input."""
        try:
            # Get the latest message
            messages = state.get("messages", [])
            if not messages:
                return self.create_command(
                    goto=END,
                    update={"error": "No input provided"}
                )

            latest_message = messages[-1]
            user_input = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

            # Analyze using LLM
            analysis_result = await self.invoke_llm(
                prompt_version="analysis_v1",
                context={
                    "input": user_input,
                    "context": state.get("context", ""),
                    "analysis_type": "comprehensive"
                }
            )

            # Create response message
            ai_message = self.create_ai_message(f"Analysis: {analysis_result}")

            return self.create_command(
                goto="synthesis",
                update={
                    "messages": messages + [ai_message],
                    "analysis_result": analysis_result,
                    "current_step": "analysis_complete"
                }
            )

        except Exception as e:
            return self.create_command(
                goto=END,
                update={"error": f"Analysis failed: {str(e)}"}
            )


class AlphaSynthesisNode(BaseAgentNode):
    """Node for synthesis phase."""

    async def __call__(self, state: AlphaAgentState) -> Command:
        """Synthesize analysis results into final response."""
        try:
            analysis_result = state.get("analysis_result", "")
            if not analysis_result:
                return self.create_command(
                    goto=END,
                    update={"error": "No analysis result to synthesize"}
                )

            # Synthesize using LLM
            synthesis_result = await self.invoke_llm(
                prompt_version="synthesis_v1",
                context={
                    "analysis": analysis_result,
                    "context": state.get("context", ""),
                    "synthesis_type": "structured"
                }
            )

            # Create final response message
            ai_message = self.create_ai_message(f"Synthesis: {synthesis_result}")
            messages = state.get("messages", [])

            return self.create_command(
                goto=END,
                update={
                    "messages": messages + [ai_message],
                    "final_result": synthesis_result,
                    "current_step": "synthesis_complete"
                }
            )

        except Exception as e:
            return self.create_command(
                goto=END,
                update={"error": f"Synthesis failed: {str(e)}"}
            )


class AlphaAgent(BaseAgent):
    """Alpha Agent with improved node-based architecture."""

    def __init__(self):
        # Initialize LLM and prompt manager
        llm_factory = LLMFactory()
        llm = llm_factory.create_llm()  # Uses default provider from config
        prompt_manager = PromptManager("alpha")

        super().__init__(
            name="alpha",
            llm=llm,
            prompt_manager=prompt_manager
        )

        # Initialize nodes
        self.analysis_node = AlphaAnalysisNode("alpha", llm, prompt_manager)
        self.synthesis_node = AlphaSynthesisNode("alpha", llm, prompt_manager)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(AlphaAgentState)

        # Add nodes
        workflow.add_node("analysis", self.analysis_node)
        workflow.add_node("synthesis", self.synthesis_node)

        # Add edges
        workflow.set_entry_point("analysis")
        workflow.add_edge("analysis", "synthesis")
        workflow.add_edge("synthesis", END)

        return workflow.compile()

    async def run(self, request: AlphaRequest, streaming: bool = False) -> AlphaResponse:
        """Run the agent workflow."""
        try:
            # Prepare initial state
            initial_state = AlphaAgentState(
                messages=[HumanMessage(content=request.input)],
                context=request.context or "",
                current_step="starting",
                metadata=request.metadata or {}
            )

            if streaming:
                # Return async generator for streaming
                return self._stream_workflow(initial_state)
            else:
                # Run workflow to completion
                final_state = await self.graph.ainvoke(initial_state)

                # Extract final result
                final_result = final_state.get("final_result", "")
                if not final_result and final_state.get("messages"):
                    # Fallback to last message
                    last_message = final_state["messages"][-1]
                    final_result = last_message.content if hasattr(last_message, 'content') else str(last_message)

                return AlphaResponse(
                    output=final_result,
                    analysis_result=final_state.get("analysis_result", ""),
                    synthesis_result=final_state.get("final_result", ""),
                    metadata=final_state.get("metadata", {}),
                    error=final_state.get("error")
                )

        except Exception as e:
            return AlphaResponse(
                output="",
                error=f"Agent execution failed: {str(e)}"
            )

    async def _stream_workflow(self, initial_state: AlphaAgentState):
        """Stream workflow execution."""
        async for state in self.graph.astream(initial_state):
            # Yield intermediate results
            current_step = state.get("current_step", "unknown")
            if "messages" in state and state["messages"]:
                last_message = state["messages"][-1]
                content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                yield f"data: {{\"step\": \"{current_step}\", \"content\": \"{content}\"}}\n\n"

            # Check if workflow is complete
            if state.get("final_result") or state.get("error"):
                break

    async def health_check(self) -> Dict[str, Any]:
        """Check agent health."""
        health_info = await super().health_check()

        # Add alpha-specific health checks
        health_info.update({
            "nodes": {
                "analysis": "healthy",
                "synthesis": "healthy"
            },
            "graph_compiled": bool(self.graph),
            "prompt_versions": self.prompt_manager.list_versions()
        })

        return health_info


# Export the agent for registration
agent = AlphaAgent()