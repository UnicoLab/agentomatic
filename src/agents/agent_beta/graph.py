"""Updated Beta Agent with new architecture and node pattern."""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage

from ...common.base_agent import BaseAgent
from ...common.nodes import BaseAgentNode, create_simple_agent_node, create_conditional_node
from ...common.llm_factory import LLMFactory
from ...common.prompt_manager import PromptManager
from .schemas import BetaAgentState, BetaRequest, BetaResponse


class BetaClassificationNode(BaseAgentNode):
    """Node for classifying input type."""

    async def __call__(self, state: BetaAgentState) -> Command:
        """Classify the input to determine processing path."""
        try:
            messages = state.get("messages", [])
            if not messages:
                return self.create_command(
                    goto=END,
                    update={"error": "No input provided"}
                )

            latest_message = messages[-1]
            user_input = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

            # Classify using LLM
            classification_result = await self.invoke_llm(
                prompt_version="classification_v1",
                context={
                    "input": user_input,
                    "context": state.get("context", ""),
                    "classification_types": ["simple", "complex", "analytical"]
                }
            )

            # Create response message
            ai_message = self.create_ai_message(f"Classification: {classification_result}")

            # Determine next node based on classification
            classification_lower = classification_result.lower()
            if "complex" in classification_lower:
                next_node = "complex_processing"
            elif "analytical" in classification_lower:
                next_node = "analytical_processing"
            else:
                next_node = "simple_processing"

            return self.create_command(
                goto=next_node,
                update={
                    "messages": messages + [ai_message],
                    "classification": classification_result,
                    "current_step": "classification_complete"
                }
            )

        except Exception as e:
            return self.create_command(
                goto=END,
                update={"error": f"Classification failed: {str(e)}"}
            )


class BetaSimpleProcessingNode(BaseAgentNode):
    """Node for simple processing."""

    async def __call__(self, state: BetaAgentState) -> Command:
        """Process simple inputs."""
        try:
            messages = state.get("messages", [])
            classification = state.get("classification", "")

            # Process using LLM
            result = await self.invoke_llm(
                prompt_version="simple_processing_v1",
                context={
                    "classification": classification,
                    "context": state.get("context", ""),
                    "processing_type": "streamlined"
                }
            )

            ai_message = self.create_ai_message(f"Simple Processing: {result}")

            return self.create_command(
                goto="finalization",
                update={
                    "messages": messages + [ai_message],
                    "processing_result": result,
                    "current_step": "simple_processing_complete"
                }
            )

        except Exception as e:
            return self.create_command(
                goto=END,
                update={"error": f"Simple processing failed: {str(e)}"}
            )


class BetaComplexProcessingNode(BaseAgentNode):
    """Node for complex processing."""

    async def __call__(self, state: BetaAgentState) -> Command:
        """Process complex inputs with multi-step analysis."""
        try:
            messages = state.get("messages", [])
            classification = state.get("classification", "")

            # Multi-step complex processing
            step1_result = await self.invoke_llm(
                prompt_version="complex_processing_step1_v1",
                context={
                    "classification": classification,
                    "context": state.get("context", ""),
                    "step": "decomposition"
                }
            )

            step2_result = await self.invoke_llm(
                prompt_version="complex_processing_step2_v1",
                context={
                    "step1_result": step1_result,
                    "classification": classification,
                    "step": "analysis"
                }
            )

            final_result = await self.invoke_llm(
                prompt_version="complex_processing_final_v1",
                context={
                    "step1_result": step1_result,
                    "step2_result": step2_result,
                    "step": "synthesis"
                }
            )

            ai_message = self.create_ai_message(f"Complex Processing: {final_result}")

            return self.create_command(
                goto="finalization",
                update={
                    "messages": messages + [ai_message],
                    "processing_result": final_result,
                    "processing_steps": {
                        "step1": step1_result,
                        "step2": step2_result,
                        "final": final_result
                    },
                    "current_step": "complex_processing_complete"
                }
            )

        except Exception as e:
            return self.create_command(
                goto=END,
                update={"error": f"Complex processing failed: {str(e)}"}
            )


class BetaAnalyticalProcessingNode(BaseAgentNode):
    """Node for analytical processing."""

    async def __call__(self, state: BetaAgentState) -> Command:
        """Process analytical inputs with detailed analysis."""
        try:
            messages = state.get("messages", [])
            classification = state.get("classification", "")

            # Analytical processing
            result = await self.invoke_llm(
                prompt_version="analytical_processing_v1",
                context={
                    "classification": classification,
                    "context": state.get("context", ""),
                    "analysis_depth": "comprehensive",
                    "include_metrics": True
                }
            )

            ai_message = self.create_ai_message(f"Analytical Processing: {result}")

            return self.create_command(
                goto="finalization",
                update={
                    "messages": messages + [ai_message],
                    "processing_result": result,
                    "current_step": "analytical_processing_complete"
                }
            )

        except Exception as e:
            return self.create_command(
                goto=END,
                update={"error": f"Analytical processing failed: {str(e)}"}
            )


class BetaFinalizationNode(BaseAgentNode):
    """Node for finalizing results."""

    async def __call__(self, state: BetaAgentState) -> Command:
        """Finalize and format the results."""
        try:
            processing_result = state.get("processing_result", "")
            classification = state.get("classification", "")

            if not processing_result:
                return self.create_command(
                    goto=END,
                    update={"error": "No processing result to finalize"}
                )

            # Finalize using LLM
            final_result = await self.invoke_llm(
                prompt_version="finalization_v1",
                context={
                    "processing_result": processing_result,
                    "classification": classification,
                    "context": state.get("context", ""),
                    "format": "structured"
                }
            )

            ai_message = self.create_ai_message(f"Final Result: {final_result}")
            messages = state.get("messages", [])

            return self.create_command(
                goto=END,
                update={
                    "messages": messages + [ai_message],
                    "final_result": final_result,
                    "current_step": "finalization_complete"
                }
            )

        except Exception as e:
            return self.create_command(
                goto=END,
                update={"error": f"Finalization failed: {str(e)}"}
            )


class BetaAgent(BaseAgent):
    """Beta Agent with improved node-based architecture and dynamic routing."""

    def __init__(self):
        # Initialize LLM and prompt manager
        llm_factory = LLMFactory()
        llm = llm_factory.create_llm()
        prompt_manager = PromptManager("beta")

        super().__init__(
            name="beta",
            llm=llm,
            prompt_manager=prompt_manager
        )

        # Initialize nodes
        self.classification_node = BetaClassificationNode("beta", llm, prompt_manager)
        self.simple_processing_node = BetaSimpleProcessingNode("beta", llm, prompt_manager)
        self.complex_processing_node = BetaComplexProcessingNode("beta", llm, prompt_manager)
        self.analytical_processing_node = BetaAnalyticalProcessingNode("beta", llm, prompt_manager)
        self.finalization_node = BetaFinalizationNode("beta", llm, prompt_manager)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with dynamic routing."""
        # Create the graph
        workflow = StateGraph(BetaAgentState)

        # Add nodes
        workflow.add_node("classification", self.classification_node)
        workflow.add_node("simple_processing", self.simple_processing_node)
        workflow.add_node("complex_processing", self.complex_processing_node)
        workflow.add_node("analytical_processing", self.analytical_processing_node)
        workflow.add_node("finalization", self.finalization_node)

        # Set entry point
        workflow.set_entry_point("classification")

        # Add conditional edges from classification
        workflow.add_conditional_edges(
            "classification",
            self._route_from_classification,
            {
                "simple_processing": "simple_processing",
                "complex_processing": "complex_processing",
                "analytical_processing": "analytical_processing",
                "end": END
            }
        )

        # Add edges to finalization
        workflow.add_edge("simple_processing", "finalization")
        workflow.add_edge("complex_processing", "finalization")
        workflow.add_edge("analytical_processing", "finalization")
        workflow.add_edge("finalization", END)

        return workflow.compile()

    def _route_from_classification(self, state: BetaAgentState) -> str:
        """Route based on classification result."""
        classification = state.get("classification", "").lower()

        if "complex" in classification:
            return "complex_processing"
        elif "analytical" in classification:
            return "analytical_processing"
        elif "simple" in classification:
            return "simple_processing"
        else:
            # Default fallback
            return "simple_processing"

    async def run(self, request: BetaRequest, streaming: bool = False) -> BetaResponse:
        """Run the agent workflow."""
        try:
            # Prepare initial state
            initial_state = BetaAgentState(
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

                # Extract results
                final_result = final_state.get("final_result", "")
                if not final_result and final_state.get("messages"):
                    last_message = final_state["messages"][-1]
                    final_result = last_message.content if hasattr(last_message, 'content') else str(last_message)

                return BetaResponse(
                    output=final_result,
                    classification=final_state.get("classification", ""),
                    processing_result=final_state.get("processing_result", ""),
                    processing_steps=final_state.get("processing_steps", {}),
                    metadata=final_state.get("metadata", {}),
                    error=final_state.get("error")
                )

        except Exception as e:
            return BetaResponse(
                output="",
                error=f"Agent execution failed: {str(e)}"
            )

    async def _stream_workflow(self, initial_state: BetaAgentState):
        """Stream workflow execution."""
        async for state in self.graph.astream(initial_state):
            current_step = state.get("current_step", "unknown")
            if "messages" in state and state["messages"]:
                last_message = state["messages"][-1]
                content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                yield f"data: {{\"step\": \"{current_step}\", \"content\": \"{content}\"}}\n\n"

            if state.get("final_result") or state.get("error"):
                break

    async def health_check(self) -> Dict[str, Any]:
        """Check agent health."""
        health_info = await super().health_check()

        # Add beta-specific health checks
        health_info.update({
            "nodes": {
                "classification": "healthy",
                "simple_processing": "healthy",
                "complex_processing": "healthy",
                "analytical_processing": "healthy",
                "finalization": "healthy"
            },
            "routing": "conditional",
            "graph_compiled": bool(self.graph),
            "prompt_versions": self.prompt_manager.list_versions()
        })

        return health_info


# Export the agent for registration
agent = BetaAgent()