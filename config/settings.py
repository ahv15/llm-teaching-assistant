#!/usr/bin/env python
# coding: utf-8

"""Configuration settings for the LLM Teaching Assistant."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Configuration settings for the application."""
    
    # OpenAI settings
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    
    # GROBID settings
    grobid_url: str = "http://localhost:8070"
    
    # File paths
    faiss_index_path: str = "summary.faiss"
    urls_json_path: str = "urls.json"
    
    # LangGraph settings
    max_tokens: int = 256
    max_tokens_before_summary: int = 256
    max_summary_tokens: int = 128
    
    def __post_init__(self):
        """Load settings from environment variables if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings instance from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            grobid_url=os.getenv("GROBID_URL", "http://localhost:8070"),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "summary.faiss"),
            urls_json_path=os.getenv("URLS_JSON_PATH", "urls.json"),
        )
