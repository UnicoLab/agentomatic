"""Beta Agent - Simplified implementation with reasoning focus."""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from ...common.base_agent import BaseAgent
from ...common.llm_factory import LLMFactory, LLMConfig, LLMProvider
from ...common.agent_state import AgentState
from ...common.prompt_manager import PromptManager
from .config import BetaConfig
from .schemas import BetaInput, BetaOutput


class BetaAgent(BaseAgent):
    """Beta Agent - Specialized in reasoning and analysis."""

    def __init__(self):
        # Create LLM config from agent config
        agent_config = BetaConfig()
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
        prompt_manager = PromptManager("beta")

        super().__init__(
            name="beta",
            llm=llm,
            prompt_manager=prompt_manager
        )

        self.config = agent_config

    def build_graph(self) -> StateGraph:
        """Build the Beta agent workflow graph with classification."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("classify", self._classify_node)
        graph.add_node("process_simple", self._process_simple_node)
        graph.add_node("process_complex", self._process_complex_node)

        # Set entry point
        graph.set_entry_point("classify")

        # Add conditional routing
        graph.add_conditional_edges(
            "classify",
            self._route_processing,
            {
                "simple": "process_simple",
                "complex": "process_complex"
            }
        )

        # Connect to end
        graph.add_edge("process_simple", END)
        graph.add_edge("process_complex", END)

        return graph.compile()

    async def _classify_node(self, state: Dict[str, Any]) -> Command:
        """Classify input complexity."""
        try:
            messages = state.get("messages", [])
            if not messages:
                return Command(update={"error": "No input provided"}, goto=END)

            user_input = messages[-1].content

            # Simple classification logic
            complexity = "complex" if len(user_input.split()) > 20 or "analyze" in user_input.lower() else "simple"

            return Command(
                update={"classification": complexity},
                goto=complexity
            )

        except Exception as e:
            return Command(update={"error": str(e)}, goto=END)

    def _route_processing(self, state: Dict[str, Any]) -> str:
        """Route based on classification."""
        return state.get("classification", "simple")

    async def _process_simple_node(self, state: Dict[str, Any]) -> Command:
        """Process simple inputs."""
        try:
            messages = state.get("messages", [])
            user_input = messages[-1].content

            prompt = self.format_prompt("simple_processing", input=user_input, context=state.get("context", ""))
            if not prompt:
                prompt = f"You are Beta, an analytical AI assistant. Process this simple request: {user_input}"

            response = await self.generate_response(prompt)

            return Command(
                update={"output": response, "completed": True},
                goto=END
            )

        except Exception as e:
            return Command(update={"error": str(e)}, goto=END)

    async def _process_complex_node(self, state: Dict[str, Any]) -> Command:
        """Process complex inputs with detailed analysis."""
        try:
            messages = state.get("messages", [])
            user_input = messages[-1].content

            # Multi-step processing for complex inputs
            analysis_prompt = self.format_prompt("analysis", problem=user_input, context=state.get("context", ""))
            if not analysis_prompt:
                analysis_prompt = f"Analyze this complex problem step by step: {user_input}"

            analysis = await self.generate_response(analysis_prompt)

            solution_prompt = self.format_prompt("solution", analysis=analysis, problem=user_input)
            if not solution_prompt:
                solution_prompt = f"Based on this analysis: {analysis}\n\nProvide a practical solution for: {user_input}"

            solution = await self.generate_response(solution_prompt)

            final_response = f"Analysis: {analysis}\n\nSolution: {solution}"

            return Command(
                update={"output": final_response, "completed": True},
                goto=END
            )

        except Exception as e:
            return Command(update={"error": str(e)}, goto=END)

    async def run(self, input_data, streaming: bool = False):
        """Run the Beta agent."""
        # Convert input to proper format
        if hasattr(input_data, 'problem'):
            problem = input_data.problem
            context = f"Domain: {getattr(input_data, 'domain', '')}"
        else:
            problem = str(input_data)
            context = ""
        
        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=problem)],
            context=context,
            agent_name="beta",
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
                output = "Error: No analysis generated"
            
            # Return properly formatted output
            return BetaOutput(
                analysis=output,
                reasoning_steps=["Classification", "Processing", "Analysis"],
                solution_approach="Systematic problem solving",
                risk_assessment="Low risk",
                confidence=0.85,
                tokens_used=len(output) if output else 0,
                processing_time=0.0
            )
        except Exception as e:
            # Return error response
            error_msg = f"Agent execution failed: {str(e)}"
            return BetaOutput(
                analysis=error_msg,
                reasoning_steps=["Error handling"],
                solution_approach="Error recovery",
                risk_assessment="High risk - execution failed",
                confidence=0.0,
                tokens_used=len(error_msg),
                processing_time=0.0
            )

    async def _stream_response(self, initial_state):
        """Stream response chunks."""
        async for chunk in self.graph.astream(initial_state):
            if "output" in chunk:
                yield f"data: {chunk['output']}\n\n"
            if "classification" in chunk:
                yield f"data: [CLASSIFICATION] {chunk['classification']}\n\n"
            if chunk.get("error"):
                yield f"data: [ERROR] {chunk['error']}\n\n"
                return
        yield "data: [DONE]\n\n"


# Export for auto-discovery
agent = BetaAgent()