#!/usr/bin/env python
# coding: utf-8

"""Embedding processing and vector store management."""

import json
import numpy as np
import faiss
import openai
from typing import List


class EmbeddingProcessor:
    """Handles embedding creation and FAISS vector store operations."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the embedding processor.
        
        Args:
            model_name (str): OpenAI embedding model name
        """
        self.model_name = model_name
        self.index = None
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        embeddings = []
        for text in texts:
            resp = openai.embeddings.create(input=text, model=self.model_name)
            embeddings.append(resp.data[0].embedding)
        
        return np.array(embeddings, dtype="float32")
    
    def build_faiss_index(self, embeddings: np.ndarray, index_path: str = "summary.faiss"):
        """
        Build and save FAISS index from embeddings.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            index_path (str): Path to save the FAISS index
        """
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        self.index = index
    
    def load_faiss_index(self, index_path: str = "summary.faiss"):
        """
        Load FAISS index from file.
        
        Args:
            index_path (str): Path to the FAISS index file
        """
        self.index = faiss.read_index(index_path)
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 1) -> tuple:
        """
        Search for similar embeddings in the FAISS index.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            k (int): Number of nearest neighbors to return
            
        Returns:
            tuple: (distances, indices) of nearest neighbors
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_faiss_index() first.")
        
        return self.index.search(query_embedding, k=k)
    
    def save_urls(self, urls: List[str], filepath: str = "urls.json"):
        """
        Save URLs to JSON file for later retrieval.
        
        Args:
            urls (List[str]): List of URLs
            filepath (str): Path to save the JSON file
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(urls, f, ensure_ascii=False, indent=2)
    
    def load_urls(self, filepath: str = "urls.json") -> List[str]:
        """
        Load URLs from JSON file.
        
        Args:
            filepath (str): Path to the JSON file
            
        Returns:
            List[str]: List of URLs
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
