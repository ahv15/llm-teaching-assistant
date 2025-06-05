#!/usr/bin/env python
# coding: utf-8

"""Example usage of the LLM Teaching Assistant."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.teaching_agent import TeachingAgent


def main():
    """Demonstrate the teaching assistant functionality."""
    # Initialize the teaching agent
    agent = TeachingAgent()
    
    # Configuration for the conversation
    config = {"configurable": {"thread_id": "1"}}
    
    # Example query
    query = "Can you teach me about some new LLM Training optimizations?"
    
    print(f"Query: {query}")
    print("\n" + "="*50)
    
    # Invoke the agent
    result = agent.invoke(
        {
            "messages": query,
            "context": {}
        },
        config
    )
    
    # Display the result
    if "messages" in result and result["messages"]:
        response = result["messages"][-1]
        if hasattr(response, 'content'):
            print(response.content)
        else:
            print(response)
    else:
        print("No response generated.")


if __name__ == "__main__":
    main()
