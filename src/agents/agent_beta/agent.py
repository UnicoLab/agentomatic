"""Beta Agent - Advanced implementation with decision-making capabilities."""

from typing import Dict, Any, AsyncGenerator, List
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from ...common.base_agent import BaseAgent
from ...common.llm_factory import LLMConfig
from ...common.nodes import BaseAgentNode
from .state import BetaAgentState


class BetaAgentNode(BaseAgentNode):
    """Specialized node for Beta Agent operations with decision-making."""
    
    async def classify_input(self, state: BetaAgentState) -> Command:
        """Classify the type of input to determine processing path."""
        try:
            messages = state.get("messages", [])
            if not messages:
                return self.create_command(
                    update={"error": "No input message provided"},
                    goto=END
                )
            
            last_message = messages[-1]
            user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # Classify the input type
            classification = await self.invoke_llm(
                prompt_version="input_classification",
                context={
                    "user_input": user_input,
                    "available_types": ["question", "task", "analysis", "creative"]
                }
            )
            
            ai_message = self.create_ai_message(f"Input classified as: {classification}")
            
            # Determine next node based on classification
            next_node = self._determine_processing_path(classification)
            
            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "input_type": classification,
                    "current_step": "processing"
                },
                goto=next_node
            )
            
        except Exception as e:
            return self.create_command(
                update={"error": f"Classification failed: {str(e)}"},
                goto=END
            )
    
    def _determine_processing_path(self, classification: str) -> str:
        """Determine the next processing node based on input classification."""
        classification_lower = classification.lower()
        
        if "question" in classification_lower:
            return "process_question"
        elif "task" in classification_lower:
            return "process_task"
        elif "analysis" in classification_lower:
            return "process_analysis"
        elif "creative" in classification_lower:
            return "process_creative"
        else:
            return "process_general"
    
    async def process_question(self, state: BetaAgentState) -> Command:
        """Process question-type inputs."""
        try:
            messages = state.get("messages", [])
            user_input = messages[0].content if messages else ""
            
            response = await self.invoke_llm(
                prompt_version="question_processing",
                context={
                    "question": user_input,
                    "context": state.get("context", "")
                }
            )
            
            ai_message = self.create_ai_message(response)
            
            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "final_response": response,
                    "processing_type": "question",
                    "completed": True
                },
                goto=END
            )
            
        except Exception as e:
            return self.create_command(
                update={"error": f"Question processing failed: {str(e)}"},
                goto=END
            )
    
    async def process_task(self, state: BetaAgentState) -> Command:
        """Process task-type inputs with multi-step planning."""
        try:
            messages = state.get("messages", [])
            user_input = messages[0].content if messages else ""
            
            # First, create a task plan
            plan = await self.invoke_llm(
                prompt_version="task_planning",
                context={
                    "task": user_input,
                    "context": state.get("context", "")
                }
            )
            
            # Then execute the plan
            execution = await self.invoke_llm(
                prompt_version="task_execution",
                context={
                    "task": user_input,
                    "plan": plan,
                    "context": state.get("context", "")
                }
            )
            
            ai_message = self.create_ai_message(f"Task Plan: {plan}\n\nExecution: {execution}")
            
            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "task_plan": plan,
                    "final_response": execution,
                    "processing_type": "task",
                    "completed": True
                },
                goto=END
            )
            
        except Exception as e:
            return self.create_command(
                update={"error": f"Task processing failed: {str(e)}"},
                goto=END
            )
    
    async def process_analysis(self, state: BetaAgentState) -> Command:
        """Process analysis-type inputs with detailed examination."""
        try:
            messages = state.get("messages", [])
            user_input = messages[0].content if messages else ""
            
            response = await self.invoke_llm(
                prompt_version="analysis_processing",
                context={
                    "subject": user_input,
                    "context": state.get("context", "")
                }
            )
            
            ai_message = self.create_ai_message(response)
            
            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "final_response": response,
                    "processing_type": "analysis",
                    "completed": True
                },
                goto=END
            )
            
        except Exception as e:
            return self.create_command(
                update={"error": f"Analysis processing failed: {str(e)}"},
                goto=END
            )
    
    async def process_creative(self, state: BetaAgentState) -> Command:
        """Process creative-type inputs with imaginative responses."""
        try:
            messages = state.get("messages", [])
            user_input = messages[0].content if messages else ""
            
            response = await self.invoke_llm(
                prompt_version="creative_processing",
                context={
                    "prompt": user_input,
                    "context": state.get("context", "")
                }
            )
            
            ai_message = self.create_ai_message(response)
            
            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "final_response": response,
                    "processing_type": "creative",
                    "completed": True
                },
                goto=END
            )
            
        except Exception as e:
            return self.create_command(
                update={"error": f"Creative processing failed: {str(e)}"},
                goto=END
            )
    
    async def process_general(self, state: BetaAgentState) -> Command:
        """Process general inputs that don't fit other categories."""
        try:
            messages = state.get("messages", [])
            user_input = messages[0].content if messages else ""
            
            response = await self.invoke_llm(
                prompt_version="general_processing",
                context={
                    "input": user_input,
                    "context": state.get("context", "")
                }
            )
            
            ai_message = self.create_ai_message(response)
            
            return self.create_command(
                update={
                    "messages": messages + [ai_message],
                    "final_response": response,
                    "processing_type": "general",
                    "completed": True
                },
                goto=END
            )
            
        except Exception as e:
            return self.create_command(
                update={"error": f"General processing failed: {str(e)}"},
                goto=END
            )


class BetaAgent(BaseAgent):
    """Beta Agent with dynamic routing and specialized processing paths."""
    
    def __init__(self, agent_name: str, llm_config: LLMConfig):
        super().__init__(agent_name, llm_config)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for Beta Agent."""
        # Create the specialized node instance
        node = BetaAgentNode(self.agent_name, self.llm, self.prompt_manager)
        
        # Create the graph
        graph = StateGraph(BetaAgentState)
        
        # Add nodes
        graph.add_node("classify_input", node.classify_input)
        graph.add_node("process_question", node.process_question)
        graph.add_node("process_task", node.process_task)
        graph.add_node("process_analysis", node.process_analysis)
        graph.add_node("process_creative", node.process_creative)
        graph.add_node("process_general", node.process_general)
        
        # Set entry point
        graph.set_entry_point("classify_input")
        
        # Add conditional edges based on classification
        graph.add_conditional_edges(
            "classify_input",
            lambda state: self._route_based_on_classification(state),
            {
                "process_question": "process_question",
                "process_task": "process_task",
                "process_analysis": "process_analysis",
                "process_creative": "process_creative",
                "process_general": "process_general"
            }
        )
        
        # All processing nodes end the workflow
        for node_name in ["process_question", "process_task", "process_analysis", "process_creative", "process_general"]:
            graph.add_edge(node_name, END)
        
        return graph.compile()
    
    def _route_based_on_classification(self, state: BetaAgentState) -> str:
        """Route to appropriate processing node based on input classification."""
        input_type = state.get("input_type", "general").lower()
        
        if "question" in input_type:
            return "process_question"
        elif "task" in input_type:
            return "process_task"
        elif "analysis" in input_type:
            return "process_analysis"
        elif "creative" in input_type:
            return "process_creative"
        else:
            return "process_general"
    
    async def run(self, request, streaming: bool = False) -> Any:
        """Execute the Beta Agent workflow."""
        try:
            # Initialize state
            initial_state = BetaAgentState(
                messages=[HumanMessage(content=request.input)],
                context=request.context or "",
                current_step="classification",
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
                    "input_type": result.get("input_type", "unknown"),
                    "processing_type": result.get("processing_type", "unknown"),
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
    
    async def _run_streaming(self, initial_state: BetaAgentState) -> AsyncGenerator[str, None]:
        """Run the agent with streaming output."""
        try:
            async for chunk in self.graph.astream(initial_state):
                # Extract the step name and state from the chunk
                for step_name, step_state in chunk.items():
                    if "messages" in step_state and step_state["messages"]:
                        last_message = step_state["messages"][-1]
                        if hasattr(last_message, 'content'):
                            yield f"data: {last_message.content}\n\n"
                    
                    # Yield classification info
                    if "input_type" in step_state:
                        yield f"data: [CLASSIFICATION] {step_state['input_type']}\n\n"
                    
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