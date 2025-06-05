#!/usr/bin/env python
# coding: utf-8

"""State management for the teaching agent system."""

from typing import Any, TypedDict, List
from langraph.graph import MessagesState
from langchain_core.messages import AnyMessage


class State(MessagesState):
    """Extended state for the teaching agent."""
    context: dict[str, Any]


class LLMInputState(TypedDict):
    """Input state structure for LLM operations."""
    summarized_messages: List[AnyMessage]
    context: dict[str, Any]
