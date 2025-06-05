# LLM Teaching Assistant

A generative AI-powered teaching assistant that retrieves and explains research papers from arXiv, converting complex academic content into beginner-friendly lessons.

## Features

- **Intelligent Paper Retrieval**: Searches through a curated collection of LLM and AI systems papers
- **Semantic Understanding**: Uses vector embeddings and FAISS for similarity-based paper matching
- **Automated Lesson Generation**: Converts research paper sections into beginner-friendly explanations
- **PDF Processing**: Integrates with GROBID for structured document parsing
- **Conversational Interface**: LangGraph-powered agent with memory and summarization

## Project Structure

```
llm-teaching-assistant/
├── src/                           # Main source code
│   ├── data_fetching/            # Paper metadata and abstract retrieval
│   │   ├── __init__.py
│   │   └── paper_fetcher.py
│   ├── embeddings/               # Vector embeddings and FAISS operations
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── document_processing/      # PDF parsing and lesson generation
│   │   ├── __init__.py
│   │   ├── pdf_processor.py
│   │   └── lesson_generator.py
│   ├── retrieval/               # Paper retrieval and search
│   │   ├── __init__.py
│   │   └── paper_retriever.py
│   ├── agents/                  # LangGraph agent system
│   │   ├── __init__.py
│   │   ├── state_management.py
│   │   └── teaching_agent.py
│   └── __init__.py
├── config/                      # Configuration management
│   ├── __init__.py
│   └── settings.py
├── scripts/                     # Setup and example scripts
│   ├── setup_environment.py
│   └── example_usage.py
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ahv15/llm-teaching-assistant.git
   cd llm-teaching-assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GROBID_URL="http://localhost:8070"  # Optional, defaults to localhost
   ```

4. **Install and start GROBID** (for PDF processing):
   ```bash
   # Download GROBID from: https://github.com/kermitt2/grobid
   # Follow their installation instructions
   # Start GROBID service on port 8070
   ```

## Quick Start

1. **Initialize the environment**:
   ```bash
   python scripts/setup_environment.py
   ```
   This will:
   - Fetch paper metadata from the LLMSys repository
   - Download abstracts from arXiv
   - Create vector embeddings
   - Build the FAISS index

2. **Run the example**:
   ```bash
   python scripts/example_usage.py
   ```

3. **Use in your code**:
   ```python
   from src.agents.teaching_agent import TeachingAgent
   
   # Initialize the agent
   agent = TeachingAgent()
   
   # Ask a question
   result = agent.invoke({
       "messages": "Can you teach me about transformer optimizations?",
       "context": {}
   }, {"configurable": {"thread_id": "1"}})
   
   print(result["messages"][-1].content)
   ```

## Core Components

### Data Fetching
- **`paper_fetcher.py`**: Retrieves paper metadata from LLMSys repository and abstracts from arXiv

### Embeddings
- **`vector_store.py`**: Manages OpenAI embeddings and FAISS vector operations

### Document Processing
- **`pdf_processor.py`**: Interfaces with GROBID for PDF parsing and section extraction
- **`lesson_generator.py`**: Converts academic sections into beginner-friendly lessons

### Retrieval
- **`paper_retriever.py`**: Main retrieval tool that finds relevant papers and generates teaching content

### Agents
- **`teaching_agent.py`**: LangGraph-powered conversational agent
- **`state_management.py`**: State definitions for the agent system

## Configuration

The system can be configured via environment variables or the `config/settings.py` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `EMBEDDING_MODEL`: Embedding model (default: "text-embedding-3-small")
- `CHAT_MODEL`: Chat model (default: "gpt-4o-mini")
- `GROBID_URL`: GROBID service URL (default: "http://localhost:8070")
- `FAISS_INDEX_PATH`: Path to FAISS index file (default: "summary.faiss")
- `URLS_JSON_PATH`: Path to URLs JSON file (default: "urls.json")

## Usage Examples

### Basic Paper Search
```python
from src.retrieval.paper_retriever import paper_retriever

# Find and explain a paper
result = paper_retriever("attention mechanisms in transformers")
print(result)
```

### Manual Component Usage
```python
from src.data_fetching.paper_fetcher import fetch_llm_sys_papers
from src.embeddings.vector_store import EmbeddingProcessor

# Fetch papers
papers = fetch_llm_sys_papers()

# Create embeddings
processor = EmbeddingProcessor()
embeddings = processor.create_embeddings(["sample text"])
```

## Requirements

- Python 3.8+
- OpenAI API key
- GROBID service (for PDF processing)
- Required Python packages (see `requirements.txt`)
