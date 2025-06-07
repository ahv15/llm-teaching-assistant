"""
Enhanced Paper Retrieval System

This module provides advanced paper retrieval capabilities with integrated
GROBID processing and lesson generation.
"""

import json
import numpy as np
import faiss
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, Optional

from langchain.tools import tool
from openai import OpenAI


def generate_section_lesson(
    section_name: str,
    section_text: str,
    next_section_name: Optional[str] = None,
) -> str:
    """
    Convert a research paper section into a beginner-friendly lesson fragment.
    
    Args:
        section_name: Name of the current section
        section_text: Text content of the section
        next_section_name: Name of the next section for smooth transitions
        
    Returns:
        Generated lesson fragment text
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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return response.choices[0].message.content.strip()


def fetch_and_grobid_sections(
    arxiv_url: str,
    grobid_url: str = 'http://localhost:8070'
) -> Dict[str, str]:
    """
    Fetch a paper from arXiv and process it with GROBID to extract sections.
    
    Args:
        arxiv_url: arXiv URL (abstract or PDF)
        grobid_url: GROBID service URL
        
    Returns:
        Dictionary mapping section names to their text content
    """
    # Convert arXiv abstract URL to PDF URL if needed
    if '/abs/' in arxiv_url:
        paper_id = arxiv_url.rstrip('/').split('/')[-1]
        pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
    else:
        pdf_url = arxiv_url

    # Fetch PDF
    response = requests.get(pdf_url)
    response.raise_for_status()
    pdf_bytes = response.content

    # Send to GROBID for processing
    files = {'input': ('paper.pdf', BytesIO(pdf_bytes), 'application/pdf')}
    response = requests.post(f'{grobid_url}/api/processFulltextDocument', files=files)
    response.raise_for_status()
    tei_xml = response.text

    # Parse TEI XML
    TEI_NS = 'http://www.tei-c.org/ns/1.0'
    ET.register_namespace('tei', TEI_NS)
    root = ET.fromstring(tei_xml)

    sections: Dict[str, str] = {}
    
    for div in root.findall(f'.//{{{TEI_NS}}}div'):
        # Skip body-level divs
        if div.attrib.get('type') == 'body':
            continue

        # Extract section identifier
        section_key = (
            div.attrib.get('type')
            or div.attrib.get('subtype')
            or next((h.text for h in div.findall(f'./{{{TEI_NS}}}head') if h.text), None)
        )
        
        if not section_key:
            continue

        # Extract text content
        text_parts: list[str] = []
        for element in div.iter():
            if element.text and element.tag != f'{{{TEI_NS}}}head':
                text_parts.append(element.text.strip())
            if element.tail:
                text_parts.append(element.tail.strip())

        sections[section_key.lower()] = ' '.join(part for part in text_parts if part)

    return sections


@tool(description="Return the teaching content of the paper closest to the query.", return_direct=True)
def paper_retriever(query: str) -> str:
    """
    Advanced paper retrieval tool with lesson generation.
    
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
    # Generate query embedding
    client = OpenAI()
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = np.array([response.data[0].embedding], dtype='float32')
    
    # Search FAISS index
    index = faiss.read_index('summary.faiss')
    _, indices = index.search(query_embedding, k=1)
    best_index = int(indices[0, 0])

    # Load paper URLs
    with open('urls.json', 'r', encoding='utf-8') as file:
        urls = json.load(file)
    best_url = urls[best_index]

    # Fetch and process paper
    sections = fetch_and_grobid_sections(best_url)
    
    # Generate lessons for each section
    full_lessons = []
    section_names = list(sections.keys())
    
    for i, section_name in enumerate(section_names):
        text = sections[section_name]
        next_section = section_names[i+1] if i+1 < len(section_names) else None
        
        lesson_fragment = generate_section_lesson(
            section_name, 
            text, 
            next_section_name=next_section
        )
        
        full_lessons.append(f"## {section_name.title()}\n\n{lesson_fragment}")

    complete_course = "\n\n".join(full_lessons)
    return complete_course


class EnhancedPaperProcessor:
    """
    Enhanced paper processing with caching and batch operations.
    
    This class provides advanced features for paper processing including:
    - Caching of processed papers
    - Batch processing capabilities
    - Error handling and retry logic
    """
    
    def __init__(self, grobid_url: str = 'http://localhost:8070'):
        """
        Initialize the enhanced paper processor.
        
        Args:
            grobid_url: URL of the GROBID service
        """
        self.grobid_url = grobid_url
        self.cache: Dict[str, Dict[str, str]] = {}
    
    def process_paper(self, arxiv_url: str, use_cache: bool = True) -> Dict[str, str]:
        """
        Process a single paper with caching support.
        
        Args:
            arxiv_url: arXiv URL to process
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary of section name to content mappings
        """
        if use_cache and arxiv_url in self.cache:
            return self.cache[arxiv_url]
        
        sections = fetch_and_grobid_sections(arxiv_url, self.grobid_url)
        
        if use_cache:
            self.cache[arxiv_url] = sections
            
        return sections
    
    def batch_process_papers(self, arxiv_urls: list[str]) -> Dict[str, Dict[str, str]]:
        """
        Process multiple papers in batch.
        
        Args:
            arxiv_urls: List of arXiv URLs to process
            
        Returns:
            Dictionary mapping URLs to their section content
        """
        results = {}
        
        for url in arxiv_urls:
            try:
                results[url] = self.process_paper(url)
            except Exception as error:
                print(f"Error processing {url}: {error}")
                results[url] = {}
        
        return results
