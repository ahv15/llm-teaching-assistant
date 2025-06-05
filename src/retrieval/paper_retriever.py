#!/usr/bin/env python
# coding: utf-8

"""Paper retrieval system using vector similarity search."""

import json
import numpy as np
import faiss
from langchain.tools import tool
from openai import OpenAI

from ..embeddings.vector_store import EmbeddingProcessor
from ..document_processing.pdf_processor import PDFProcessor
from ..document_processing.lesson_generator import generate_section_lesson


@tool(description="Return the teaching content of the paper closest to the query.", return_direct=True)
def paper_retriever(query: str) -> str:
    """
    Given a query string, searches a FAISS index over arXiv paper embeddings
    and returns the teaching content of the best-matched paper.

    Parameters
    ----------
    query : str
        A natural-language query to find the most relevant paper.

    Returns
    -------
    str
        The teaching content of the retrieved arXiv paper.
    """
    # Initialize components
    embedding_processor = EmbeddingProcessor()
    pdf_processor = PDFProcessor()
    
    # Embed the query and retrieve best URL
    resp = OpenAI().embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    q_emb = np.array([resp.data[0].embedding], dtype='float32')
    
    # Load index and search
    embedding_processor.load_faiss_index('summary.faiss')
    _, I = embedding_processor.search_similar(q_emb, k=1)
    best_idx = int(I[0, 0])

    # Load URLs and get best match
    urls = embedding_processor.load_urls('urls.json')
    best_url = urls[best_idx]

    # Fetch, section, and generate lessons
    sections = pdf_processor.fetch_and_extract_sections(best_url)
    full_lessons = []
    names = list(sections.keys())
    for i, sec in enumerate(names):
        text = sections[sec]
        nxt = names[i+1] if i+1 < len(names) else None
        frag = generate_section_lesson(sec, text, next_section_name=nxt)
        full_lessons.append(f"## {sec.title()}\n\n{frag}")

    complete_course = "\n\n".join(full_lessons)
    return complete_course
