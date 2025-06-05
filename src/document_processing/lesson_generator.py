#!/usr/bin/env python
# coding: utf-8

"""Lesson generation from research paper sections."""

from openai import OpenAI
from typing import Optional


def generate_section_lesson(
    section_name: str,
    section_text: str,
    next_section_name: Optional[str] = None,
) -> str:
    """
    Generate a beginner-friendly lesson from a research paper section.
    
    Args:
        section_name (str): Name of the section
        section_text (str): Content of the section
        next_section_name (Optional[str]): Name of the next section for transitions
        
    Returns:
        str: Generated lesson content
    """
    # Base instruction
    prompt = f"""
    You're an expert teaching a research paper.
    Please turn the following section "{section_name}" into a beginner-friendly lesson fragment:

    {section_text}

    Your fragment should:
      - Explain every key idea clearly with any math worked out step by step if needed.
      - Use examples wherever helpful.
    """

    # If we know the next section, ask for a transition
    if next_section_name:
        prompt += (
            f'\nAt the end, include one sentence that smoothly '
            f'transitions the learner into the next section, "{next_section_name}".'
        )

    prompt += "\n\nLesson fragment:\n"
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()
