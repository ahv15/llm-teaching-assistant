#!/usr/bin/env python
# coding: utf-8

"""Setup script to initialize the LLM Teaching Assistant environment."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_fetching.paper_fetcher import fetch_llm_sys_papers, fetch_arxiv_abstract
from embeddings.vector_store import EmbeddingProcessor


def main():
    """Initialize the teaching assistant environment."""
    print("Setting up LLM Teaching Assistant environment...")
    
    # Step 1: Fetch papers
    print("1. Fetching LLM papers metadata...")
    df = fetch_llm_sys_papers()
    arxiv_urls = df.loc[df['url'].str.contains(r'arxiv\.org/(abs|pdf)', na=False), 'url'].tolist()
    print(f"Found {len(arxiv_urls)} arXiv papers")
    
    # Step 2: Fetch abstracts
    print("2. Fetching abstracts from arXiv...")
    summaries = fetch_arxiv_abstract(arxiv_urls)
    print(f"Retrieved {len(summaries)} abstracts")
    
    # Step 3: Create embeddings and FAISS index
    print("3. Creating embeddings and building FAISS index...")
    embedding_processor = EmbeddingProcessor()
    embeddings = embedding_processor.create_embeddings(summaries)
    embedding_processor.build_faiss_index(embeddings)
    
    # Step 4: Save URLs for retrieval
    print("4. Saving URLs for later retrieval...")
    embedding_processor.save_urls(arxiv_urls)
    
    print("Setup complete! You can now use the teaching assistant.")


if __name__ == "__main__":
    main()
