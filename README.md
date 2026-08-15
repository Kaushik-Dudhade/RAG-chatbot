# RAG Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** application built with Python and FastAPI. The project explores document ingestion, chunking, embedding generation, vector storage, semantic retrieval, reranking, and LLM-based response generation.

## Overview

The system allows users to:

* Index raw text through an API
* Upload and index **TXT, PDF, and CSV** documents
* Split documents into retrieval-friendly chunks
* Generate vector embeddings using `sentence-transformers`
* Store embeddings and document metadata in persistent **ChromaDB**
* Retrieve relevant document chunks using semantic similarity
* Rerank retrieved results using cosine similarity
* Generate context-grounded responses using an optional local **Ollama** LLM
* Access the system through a FastAPI REST API and automatically generated OpenAPI documentation

## Architecture

```text
                    ┌──────────────────┐
                    │   User / Client  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    FastAPI API   │
                    └───────┬─────┬──────┘
                            │     │
                  Index     │     │     Query
                            │     │
                            ▼     ▼
                    ┌──────────┐  ┌─────────────┐
                    │ Ingestion│  │  Retriever  │
                    │ & Chunking│  │             │
                    └────┬─────┘  └──────┬──────┘
                         │               │
                         ▼               ▼
                    ┌──────────┐   ┌──────────────┐
                    │ Embeddings│   │   ChromaDB   │
                    │ Sentence  │   │ Vector Store │
                    │Transformers│  └──────┬───────┘
                    └────┬─────┘          │
                         │                │
                         └───────┬────────┘
                                 ▼
                         ┌────────────────┐
                         │ Cosine Similarity│
                         │    Reranking   │
                         └───────┬────────┘
                                 │
                                 ▼
                         ┌────────────────┐
                         │ RAG Generation │
                         │    Ollama      │
                         └────────────────┘
```

## Tech Stack

* **Python**
* **FastAPI**
* **ChromaDB**
* **Sentence Transformers**
* **Hugging Face Transformers**
* **Ollama** (optional local LLM)
* **Pandas**
* **PyPDF**
* **NLTK**
* **NumPy**
* **Pytest**

## Key Components

### Document Ingestion

The API supports indexing raw text as well as uploading:

* `.txt`
* `.pdf`
* `.csv`

Uploaded content is extracted, processed, and indexed into the persistent vector store.

### Chunking

The project includes sentence-aware and token-aware chunking approaches designed to create retrieval-friendly document segments while supporting configurable overlap.

### Embeddings

Text is converted into vector representations using a local `sentence-transformers` model.

The default embedding model is:

```text
all-MiniLM-L6-v2
```

### Vector Retrieval

Document embeddings are stored in a persistent **ChromaDB** collection. At query time, the user's query is embedded and the most relevant chunks are retrieved using vector similarity.

### Reranking

Retrieved results can be reranked using **cosine similarity** between the query embedding and retrieved document embeddings before being passed to the generation stage.

### RAG Generation

The retrieved document context is assembled into a prompt and passed to an optional local Ollama model for context-grounded response generation.

If Ollama is unavailable, the API falls back to returning the retrieved context rather than failing completely.

## API Endpoints

| Method | Endpoint  | Purpose                            |
| ------ | --------- | ---------------------------------- |
| GET    | `/health` | Health check                       |
| POST   | `/index`  | Index raw text                     |
| POST   | `/upload` | Upload and index TXT/PDF/CSV files |
| POST   | `/query`  | Query the RAG system               |
| GET    | `/docs`   | FastAPI OpenAPI documentation      |

### Example Query

```json
POST /query

{
  "query": "What is Retrieval-Augmented Generation?"
}
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAG-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Optional: Enable Ollama

The RAG generation layer can use a locally running Ollama model. Configure the relevant environment variables as needed.

Without Ollama, the application can still perform document indexing and retrieval.

## Testing

The repository includes tests covering indexing and retrieval.

Run:

```bash
PYTHONPATH=. pytest -q
```

## Project Structure

```text
RAG-chatbot/
├── app/
│   ├── config.py          # Environment and application settings
│   ├── embedder.py        # Embedding generation
│   ├── indexer.py         # Document chunking and indexing
│   ├── retriever.py       # Vector retrieval and reranking
│   ├── rag_pipeline.py    # Retrieval + generation pipeline
│   └── main.py            # FastAPI application and endpoints
│
├── tests/
│   ├── test_indexer.py
│   └── test_retriever.py
│
├── requirements.txt
├── .env.example
└── README.md
```

## Learning Focus

This project was built to gain hands-on experience with the core components of a RAG system, including:

* Document ingestion
* Chunking strategies
* Embedding generation
* Vector databases
* Semantic retrieval
* Similarity-based reranking
* Context-grounded generation
* REST API development
* Testing retrieval pipelines
