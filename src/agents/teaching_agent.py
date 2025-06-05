#!/usr/bin/env python
# coding: utf-8

"""Teaching agent implementation with LangGraph."""

import re
from typing import Dict, Any, List
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode

from .state_management import State, LLMInputState
from ..retrieval.paper_retriever import paper_retriever


class TeachingAgent:
    """LLM Teaching Assistant with paper retrieval capabilities."""
    
    def __init__(self):
        """Initialize the teaching agent."""
        self.model = init_chat_model("openai:gpt-4o")
        self.summarization_model = self.model.bind(max_tokens=128)
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the agent graph with summarization and reaction nodes."""
        # Create summarization node
        summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=self.summarization_model,
            max_tokens=256,
            max_tokens_before_summary=256,
            max_summary_tokens=128,
        )
        
        # Build graph
        builder = StateGraph(State)
        builder.add_node("summarize", summarization_node)
        builder.add_node("react_with_tools", self._react_with_tools)
        builder.add_edge(START, "summarize")
        builder.add_edge("summarize", "react_with_tools")
        
        return builder.compile(checkpointer=self.checkpointer)
    
    def _react_with_tools(self, state: LLMInputState) -> Dict[str, Any]:
        """React with available tools (paper retrieval)."""
        ctx = state.get("context", {})
        chat_history: List[AnyMessage] = state["summarized_messages"]
        running_summary = state["context"].get("running_summary", "")
        
        # Get last user message
        last_user_msg = ""
        for msg in reversed(chat_history):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        tools_block = "paper_retriever"

        prompt_text = f"""
            You are an expert systems engineer and coding coach.
            When teaching, use intuitive explanations and step-by-step math if needed.
            Call paper_retriever when you need content from a paper, then return the retrieved content directly to the user.
            You are a great AI-Assistant that has access to the following tool:
    
            {tools_block}
    
            To use a tool, please use the following format:
    
            '''
            Thought: Do I need to use a tool? Yes
            Action: paper_retriever
            Action Input: <your paper query here>
            Observation: <tool output>
            ... (this Thought/Action/Action Input/Observation can repeat)
            '''
    
            When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
            '''
            Thought: Do I need to use a tool? No
            Final Answer: [your response here]
            '''
    
            Conversation summary so far:
            {running_summary}
    
            Human: {last_user_msg}
        """
        response = self.model.invoke([{"role": "user", "content": prompt_text}])
        raw_text: str = response.content

        # Look for tool usage pattern
        action_pattern = r"Action:\s*paper_retriever\s*[\r\n]+Action Input:\s*(.+)"
        m = re.search(action_pattern, raw_text)
        if m:
            paper_query = m.group(1).strip()
            tool_output = paper_retriever(paper_query)
            final_message = AIMessage(content=tool_output)
        else:
            final_message = response

        return {
            "messages": [final_message],
            "context": ctx
        }
    
    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the teaching agent."""
        return self.graph.invoke(input_data, config)
