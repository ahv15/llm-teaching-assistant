#!/usr/bin/env python3
"""
Example Usage for LLM Teaching Assistant

This script demonstrates how to use the teaching assistant with
both paper retrieval and LeetCode problem fetching capabilities.
"""

import os
from src.agents.teaching_agent import TeachingAgent


def main():
    """Main function demonstrating various usage patterns."""
    
    # Initialize the teaching agent
    print("🚀 Initializing Teaching Assistant...")
    agent = TeachingAgent()
    
    # Example 1: Paper-based learning
    print("\n📚 Example 1: Learning about transformers")
    print("=" * 50)
    
    paper_query = {
        "messages": [
            {"role": "human", "content": "Can you teach me about attention mechanisms in transformers?"}
        ],
        "context": {}
    }
    
    result = agent.invoke(paper_query, {"configurable": {"thread_id": "paper_session"}})
    print(result["messages"][-1].content[:500] + "...")
    
    # Example 2: LeetCode problem practice
    print("\n💻 Example 2: Coding practice")
    print("=" * 50)
    
    coding_query = {
        "messages": [
            {"role": "human", "content": "Give me a coding problem to practice for interviews"}
        ],
        "context": {}
    }
    
    result = agent.invoke(coding_query, {"configurable": {"thread_id": "coding_session"}})
    print(result["messages"][-1].content[:500] + "...")
    
    # Example 3: Mixed conversation
    print("\n🔄 Example 3: Mixed learning session")
    print("=" * 50)
    
    mixed_query = {
        "messages": [
            {"role": "human", "content": "I want to learn about machine learning optimization techniques and then practice some coding"}
        ],
        "context": {"learning_goal": "comprehensive_ai_prep"}
    }
    
    result = agent.invoke(mixed_query, {"configurable": {"thread_id": "mixed_session"}})
    print(result["messages"][-1].content[:500] + "...")
    
    print("\n✅ Teaching Assistant demo completed!")


def demonstrate_individual_tools():
    """Demonstrate individual tool usage."""
    print("\n🔧 Individual Tool Demonstrations")
    print("=" * 50)
    
    # LeetCode tool
    from src.data_fetching.leetcode_fetcher import get_problem
    print("\n📝 Fetching random LeetCode problem...")
    problem = get_problem.invoke({})
    print(f"Problem: {problem.get('title', 'N/A')}")
    print(f"Difficulty: {problem.get('difficulty', 'N/A')}")
    print(f"Statement preview: {problem.get('statement', 'N/A')[:200]}...")
    
    # Paper retrieval tool  
    from src.retrieval.paper_retriever import paper_retriever
    print("\n📄 Retrieving paper on neural networks...")
    lesson = paper_retriever.invoke({"query": "neural network optimization"})
    print(f"Generated lesson preview: {lesson[:300]}...")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    try:
        main()
        demonstrate_individual_tools()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that GROBID service is running on http://localhost:8070")
        print("3. Verify your OpenAI API key is valid")
        print("4. Make sure FAISS index files (summary.faiss, urls.json) exist")
