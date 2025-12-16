# RAG Chatbot

This repository contains a small Retrieval-Augmented Generation (RAG) demo using:

- FastAPI for the HTTP API
- ChromaDB for vector storage
- sentence-transformers for embeddings
- Optional Ollama integration for LLM responses

Features
- Index text or upload files (txt, pdf, csv) via API
- Token-aware and sentence-aware chunking
- Reranking of retrieved chunks by cosine similarity
- Persisted ChromaDB index so restarts keep data

Quick start
1. Create and activate a virtualenv in the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the API:

```bash
python -m uvicorn app.main:app --reload
```

3. API endpoints:
- GET /health
- POST /index           : JSON {"text":"...","source":"..."}
- POST /upload          : multipart file upload (txt/pdf/csv)
- POST /query           : JSON {"query":"..."}
- GET /docs             : OpenAPI UI

Testing

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

Notes
- The `chroma/` directory stores persisted vectors. Keep it safe when moving the project.
- Ollama usage in `rag_pipeline.py` is optional; the pipeline will fall back if Ollama isn't installed.
