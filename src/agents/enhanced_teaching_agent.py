"""
Enhanced Teaching Agent with LangGraph Integration

This module provides an advanced teaching agent that combines paper retrieval
and LeetCode problem fetching capabilities using LangGraph for conversation flow.
"""

import os
from typing import Any, TypedDict, Optional

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySupaver
from langgraph.prebuilt import ToolNode
from langmem.short_term import SummarizationNode

from .leetcode_tools import get_problem
from ..retrieval.enhanced_paper_retriever import paper_retriever


class AgentState(MessagesState):
    """State management for the enhanced teaching agent."""
    context: dict[str, Any]


class LLMInputState(TypedDict):
    """Input state for LLM processing."""
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]


def route_after_tools(state: dict[str, Any]) -> str:
    """Route after tool execution based on the last tool used."""
    messages = state["messages"]
    if not messages:
        return "agent"
    
    last_message = messages[-1]
    if isinstance(last_message, ToolMessage) and last_message.name == "paper_retriever":
        return "end"
    return "agent"


def route_after_agent(state: dict[str, Any]) -> str:
    """Route after agent response based on tool calls or recent tool usage."""
    last_message = state["messages"][-1]
    
    # Check if the last LeetCode problem was fetched recently
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and message.name == "get_problem":
            return "end"
    
    # Check if agent wants to call tools
    if last_message.additional_kwargs.get("tool_calls"):
        return "tools"
    
    return "end"


def agent_node(state: MessagesState) -> dict[str, Any]:
    """
    Main agent node that processes messages and generates responses.
    
    This node takes prior messages in state["messages"], sends them 
    to the LLM, and returns the new AIMessage. If the LLM suggests 
    a tool_call, that will be in ai_message.tool_calls.
    """
    last_human_content = ""
    last_tool_content = ""
    
    # Extract the last human message
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human_content = message.content
            break
    
    # Extract the last tool message
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            last_tool_content = message.content
            break

    # Get model with tools bound
    model = init_chat_model("openai:gpt-4o").bind_tools([paper_retriever, get_problem])
    
    # Generate response
    response = model.invoke(last_human_content + last_tool_content)
    print(response)
    
    return {"messages": state["messages"] + [response]}


class EnhancedTeachingAgent:
    """
    Enhanced Teaching Agent with integrated paper retrieval and LeetCode capabilities.
    
    This agent combines:
    - Intelligent paper retrieval from arXiv
    - LeetCode problem fetching for coding practice
    - Conversation memory and summarization
    - LangGraph-powered conversation flow
    """
    
    def __init__(self):
        """Initialize the enhanced teaching agent."""
        self.model = init_chat_model("openai:gpt-4o")
        self.summarization_model = self.model.bind(max_tokens=128)
        
        # Setup summarization node
        self.summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=self.summarization_model,
            max_tokens=256,
            max_tokens_before_summary=256,
            max_summary_tokens=128,
        )
        
        # Setup tool node
        self.tool_node = ToolNode([paper_retriever, get_problem])
        
        # Build the graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph conversation flow."""
        checkpointer = InMemorySupaver()
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("summarize", self.summarization_node)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", self.tool_node)

        # Add edges
        builder.add_edge(START, "summarize")
        builder.add_edge("summarize", "agent")

        # Add conditional edges
        builder.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "tools": "tools",
                "end": "END"
            }
        )

        builder.add_conditional_edges(
            "tools",
            route_after_tools,
            {
                "agent": "agent",
                "end": "END"
            }
        )

        self.graph = builder.compile(checkpointer=checkpointer)
    
    def invoke(self, input_data: dict, config: Optional[dict] = None) -> dict:
        """
        Invoke the enhanced teaching agent.
        
        Args:
            input_data: Dictionary containing messages and context
            config: Optional configuration including thread_id
            
        Returns:
            Dictionary with response messages
        """
        return self.graph.invoke(input_data, config or {})
    
    def visualize_graph(self):
        """Visualize the agent's conversation graph (for debugging)."""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            pass
