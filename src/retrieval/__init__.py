"""
Retrieval Package

This package contains paper retrieval and search functionality for the
LLM Teaching Assistant system.
"""

from .paper_retriever import (
    paper_retriever, 
    PaperProcessor,
    generate_section_lesson,
    fetch_and_grobid_sections
)

__all__ = [
    'paper_retriever',
    'PaperProcessor', 
    'generate_section_lesson',
    'fetch_and_grobid_sections'
]
