# LLM Teaching Assistant

A generative AI-powered teaching assistant that retrieves and explains research papers from arXiv, converts complex academic content into beginner-friendly lessons, and provides coding practice through LeetCode integration.

## 🚀 Features

- **Intelligent Paper Retrieval**: Searches through a curated collection of LLM and AI systems papers
- **Semantic Understanding**: Uses vector embeddings and FAISS for similarity-based paper matching  
- **Automated Lesson Generation**: Converts research paper sections into beginner-friendly explanations
- **PDF Processing**: Integrates with GROBID for structured document parsing
- **LeetCode Integration**: Fetches random coding problems for interview practice
- **Conversational Interface**: LangGraph-powered agent with memory and summarization
- **Enhanced Agent System**: Advanced conversation flow with tool integration

## 📁 Project Structure

```
llm-teaching-assistant/
├── src/                                    # Main source code
│   ├── agents/                             # AI agent implementations
│   │   ├── __init__.py
│   │   ├── enhanced_teaching_agent.py      # LangGraph-powered enhanced agent
│   │   ├── leetcode_tools.py               # LeetCode problem fetching tools
│   │   ├── state_management.py             # Agent state definitions
│   │   └── teaching_agent.py               # Basic teaching agent
│   ├── data_fetching/                      # Paper metadata and abstract retrieval
│   │   ├── __init__.py
│   │   └── paper_fetcher.py
│   ├── embeddings/                         # Vector embeddings and FAISS operations
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── document_processing/                # PDF parsing and lesson generation
│   │   ├── __init__.py
│   │   ├── pdf_processor.py
│   │   └── lesson_generator.py
│   ├── retrieval/                          # Paper retrieval and search
│   │   ├── __init__.py
│   │   ├── paper_retriever.py              # Basic paper retrieval
│   │   └── enhanced_paper_retriever.py     # Advanced retrieval with GROBID
│   └── __init__.py
├── config/                                 # Configuration management
│   ├── __init__.py
│   └── settings.py
├── scripts/                                # Setup and example scripts
│   ├── setup_environment.py
│   ├── example_usage.py                    # Basic usage examples
│   └── enhanced_example_usage.py           # Enhanced agent examples
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

## 🔧 Installation

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

## 🚀 Quick Start

1. **Initialize the environment**:
   ```bash
   python scripts/setup_environment.py
   ```
   This will:
   - Fetch paper metadata from the LLMSys repository
   - Download abstracts from arXiv
   - Create vector embeddings
   - Build the FAISS index

2. **Run the enhanced example**:
   ```bash
   python scripts/enhanced_example_usage.py
   ```

3. **Use the Enhanced Teaching Agent**:
   ```python
   from src.agents.enhanced_teaching_agent import EnhancedTeachingAgent
   
   # Initialize the enhanced agent
   agent = EnhancedTeachingAgent()
   
   # Ask about research topics
   result = agent.invoke({
       "messages": [
           {"role": "human", "content": "Teach me about transformer optimization techniques"}
       ],
       "context": {}
   }, {"configurable": {"thread_id": "session_1"}})
   
   print(result["messages"][-1].content)
   
   # Get coding practice
   coding_result = agent.invoke({
       "messages": [
           {"role": "human", "content": "Give me a LeetCode problem to practice"}
       ],
       "context": {}
   }, {"configurable": {"thread_id": "session_2"}})
   
   print(coding_result["messages"][-1].content)
   ```

## 📚 Core Components

### Enhanced Teaching Agent
- **`enhanced_teaching_agent.py`**: LangGraph-powered conversational agent with integrated tools
- **`leetcode_tools.py`**: LeetCode problem fetching and processing
- **`state_management.py`**: State definitions for the agent system

### Advanced Paper Processing
- **`enhanced_paper_retriever.py`**: Advanced retrieval with GROBID integration and lesson generation
- **`pdf_processor.py`**: Interfaces with GROBID for PDF parsing and section extraction
- **`lesson_generator.py`**: Converts academic sections into beginner-friendly lessons

### Data Management
- **`paper_fetcher.py`**: Retrieves paper metadata from LLMSys repository and abstracts from arXiv
- **`vector_store.py`**: Manages OpenAI embeddings and FAISS vector operations

### Traditional Components
- **`teaching_agent.py`**: Basic conversational agent
- **`paper_retriever.py`**: Simple paper retrieval functionality

## ⚙️ Configuration

The system can be configured via environment variables or the `config/settings.py` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `EMBEDDING_MODEL`: Embedding model (default: "text-embedding-3-small")
- `CHAT_MODEL`: Chat model (default: "gpt-4o")
- `GROBID_URL`: GROBID service URL (default: "http://localhost:8070")
- `FAISS_INDEX_PATH`: Path to FAISS index file (default: "summary.faiss")
- `URLS_JSON_PATH`: Path to URLs JSON file (default: "urls.json")

## 📖 Usage Examples

### Enhanced Agent with Multiple Capabilities
```python
from src.agents.enhanced_teaching_agent import EnhancedTeachingAgent

# Create agent instance
agent = EnhancedTeachingAgent()

# Research paper learning
result = agent.invoke({
    "messages": [{"role": "human", "content": "Explain BERT architecture"}],
    "context": {"topic": "nlp"}
})

# Coding practice
result = agent.invoke({
    "messages": [{"role": "human", "content": "Give me a medium difficulty coding problem"}],
    "context": {"skill_level": "intermediate"}
})
```

### Individual Tool Usage
```python
# LeetCode problem fetching
from src.agents.leetcode_tools import get_problem

problem = get_problem.invoke({})
print(f"Problem: {problem['title']}")
print(f"Difficulty: {problem['difficulty']}")
print(f"Statement: {problem['statement']}")

# Advanced paper retrieval
from src.retrieval.enhanced_paper_retriever import paper_retriever

lesson = paper_retriever.invoke({"query": "attention mechanisms in transformers"})
print(lesson)
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

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key
- GROBID service (for PDF processing)
- Required Python packages (see `requirements.txt`)

## 🆕 What's New in Enhanced Version

- **LangGraph Integration**: Advanced conversation flow management
- **LeetCode Tools**: Automated coding problem fetching for interview practice
- **Enhanced Paper Processing**: Improved GROBID integration with better error handling
- **Modular Architecture**: Clean separation of concerns with proper package structure
- **Advanced Agent System**: Memory management and conversation summarization
- **Tool Integration**: Seamless switching between paper learning and coding practice

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the teaching assistant!

## 📄 License

This project is open source and available under the MIT License.
