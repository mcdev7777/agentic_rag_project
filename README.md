
# NTT RAG Project

A modern Agentic RAG (Retrieval-Augmented Generation) system built with Pydantic AI, FastAPI, and PostgreSQL (pgvector). This project provides a scalable, modular, and production-ready foundation for document-based AI applications.
## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [License](#license)

## ✨ Features

- 🧠 Pydantic AI-based intelligent agent system
- 🔍 Multiple search strategies: Vector Search and Hybrid Search
- 📄 Advanced PDF processing: Table and image extraction with Docling
- 💾 Database: PostgreSQL + pgvector extension
- 🌊 Real-time streaming: Live responses via Server-Sent Events
- 🎯 Session management: Conversation history and context retention
- 🐳 Docker containerization: Easy deployment
- 🔧 Type safety: Reliable data handling with Pydantic models
- ⚡ Instant ingestion: Upload documents and query immediately

## 🛠️ Tech Stack

### Backend

- [Pydantic AI](https://ai.pydantic.dev/) - AI Agent Framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [LangChain](https://langchain.readthedocs.io/) - Document processing and embeddings
- [Docling](https://github.com/DS4SD/docling) - PDF extraction and analysis
- [AsyncPG](https://magicstack.github.io/asyncpg/) - PostgreSQL async client
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search

### Frontend

- [Streamlit](https://streamlit.io/) - Interactive web interface

### Infrastructure

- [PostgreSQL 17](https://www.postgresql.org/) - Primary database
- [Docker Compose](https://docs.docker.com/compose/) - Multi-container deployment
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

## 🚀 Installation

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/serkanyasr/ntt_rag_project.git
cd ntt_rag_project
```

### 2. Set Up Environment Variables

Create a `.env` file:

```bash
# API Configuration
APP_ENV=development
LOG_LEVEL=INFO
APP_HOST=0.0.0.0
APP_PORT=8058
API_URL=http://api:8058

# Streamlit Configuration
SERVER_PORT=8501
SERVER_HOST=0.0.0.0

# PostgreSQL Vector DB Configuration
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=postgres
DB_PORT=5432
DB_NAME=vector_db

# OpenAI API Configuration (required)
OPENAI_API_KEY=your_openai_api_key
LLM_CHOICE=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Start with Docker

```bash
docker-compose up -d
docker-compose logs -f
```

## 📚 Usage

### Web Interface

Access the Streamlit UI: <http://localhost:8501>

The web interface now includes:

- **Interactive Chat**: Ask questions about your uploaded documents
- **Health Monitoring**: Check API connection status
- **Session Management**: Persistent conversation history

### API Usage

FastAPI documentation: <http://localhost:8058/docs>

#### Chat Endpoint Example

```python
import requests

response = requests.post("http://localhost:8058/chat", json={
    "message": "Hello, how can I help you?",
    "session_id": "optional-session-id",
    "user_id": "user-123",
    "search_type": "hybrid"
})
print(response.json())
```

#### Streaming Chat Example

```python
import requests
import json

response = requests.post(
    "http://localhost:8058/chat/stream",
    json={
        "message": "Give a long explanation",
        "search_type": "hybrid"
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        if data.get("type") == "text":
            print(data.get("content"), end="")
```

### Document Ingestion

#### Command Line Ingestion

```bash
# Place your PDF documents in the documents/ folder
cp your_document.pdf documents/

# Run the ingestion script
python -m ingestion.ingest --documents documents/
```

## 📖 API Reference

### Endpoints

#### Chat Endpoints

- `POST /chat` - Single chat message
- `POST /chat/stream` - Streaming chat
- `GET /chat/sessions/{session_id}` - Session history

#### Search Endpoints

- `POST /search/vector` - Vector search
- `POST /search/hybrid` - Hybrid search

#### Health Check

- `GET /health` - System status

### Request/Response Models

#### Chat Request

```json
{
  "message": "Your question",
  "session_id": "optional-session-id",
  "user_id": "user-id",
  "search_type": "hybrid",
  "metadata": {}
}
```


#### Search Request

```json
{
  "query": "Search query",
  "search_type": "vector",
  "limit": 10,
  "filters": {}
}
```

## 🏗️ Project Structure

```text
ntt_rag_project/
├── agent/                  # AI Agent and business logic
│   ├── agent.py           # Main Pydantic AI agent
│   ├── api.py             # FastAPI endpoints
│   ├── db_utils.py        # Database operations
│   ├── models.py          # Pydantic models
│   ├── prompts.py         # System prompts
│   ├── providers.py       # LLM and embedding providers
│   └── tools.py           # Agent tools
├── ingestion/             # Document processing
│   ├── chunker.py         # Text chunking
│   ├── extract_files.py   # PDF extraction
│   └── ingest.py          # Main ingestion pipeline
├── ui/                    # Streamlit UI
│   └── app.py
├── sql/                   # Database schema
│   └── schema.sql
├── tests/                 # Test files
├── documents/             # PDF documents
├── docker-compose.yml     # Container orchestration
├── Dockerfile
└── pyproject.toml         # Python dependencies
```

### Main Components

#### Agent (`/agent/`)

- **agent.py**: Pydantic AI agent definition and tool registrations
- **api.py**: FastAPI web server and endpoints
- **tools.py**: Vector search, hybrid search, document retrieval tools
- **db_utils.py**: PostgreSQL operations and connection management
- **models.py**: Pydantic data models and validation

#### Ingestion (`/ingestion/`)

- **ingest.py**: Main document processing pipeline
- **extract_files.py**: PDF text, table, and image extraction
- **chunker.py**: Intelligent text chunking strategies

## 🧪 Testing

### Running Tests

```bash
pytest
pytest tests/agent/test_models.py
pytest --cov=agent --cov=ingestion
```

### Test Categories

- **Model Tests**: Pydantic model validation
- **Agent Tests**: AI agent functionality
- **Database Tests**: PostgreSQL operations
- **Ingestion Tests**: Document processing

## ⚙️ Development

### Development Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install uv
uv pip install -r pyproject.toml
pre-commit install
```


### Logging

```bash
export LOG_LEVEL=DEBUG
docker-compose logs -f api
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_NAME` | PostgreSQL database name | `rag_db` |
| `DB_USER` | PostgreSQL user | `rag_user` |
| `DB_PASSWORD` | PostgreSQL password | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `APP_PORT` | FastAPI port | `8058` |
| `SERVER_PORT` | Streamlit port | `8501` |
| `LLM_MODEL` | LLM model | `gpt-4` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |

### Docker Compose Overrides

You can create a `docker-compose.override.yml` for custom configurations.

## 🐛 Troubleshooting

### Common Issues

#### Database Connection Error

```bash
docker-compose ps postgres
docker-compose logs postgres
```

#### OpenAI API Error

```bash
echo $OPENAI_API_KEY
```

#### Port Conflicts

```bash
netstat -an | grep :8058
netstat -an | grep :8501
```

## 📄 License

MIT License - See `LICENSE` for details.

---
