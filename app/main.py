# app/main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from app.indexer import Indexer
from app.rag_pipeline import RagPipeline
from app.config import settings

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

import pandas as pd


class IndexRequest(BaseModel):
    text: Optional[str] = None
    source: Optional[str] = "api"


class QueryRequest(BaseModel):
    query: str


app = FastAPI(title="RAG Chatbot API")

# Lazy-initialized globals set at startup
indexer: Optional[Indexer] = None
rag_pipeline: Optional[RagPipeline] = None

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def extract_text_from_file(file_path: str) -> str:
    lower = file_path.lower()
    if lower.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if lower.endswith(".pdf") and PdfReader is not None:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    if lower.endswith(".csv"):
        try:
            df = pd.read_csv(file_path)
            return "\n".join(df.astype(str).apply(lambda row: " ".join(row.values), axis=1).tolist())
        except Exception:
            return ""

    return ""


@app.on_event("startup")
def startup_event():
    global indexer, rag_pipeline
    indexer = Indexer(persist_dir=settings.chroma_persist_dir)
    rag_pipeline = RagPipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/index")
async def index_endpoint(request: IndexRequest):
    if not indexer:
        raise HTTPException(status_code=500, detail="Indexer not initialized")

    if not request.text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request")

    added = indexer.index_text(request.text, source=request.source or "api")
    return {"message": "Document indexed", "chunks_added": added}


@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    if not indexer:
        raise HTTPException(status_code=500, detail="Indexer not initialized")

    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_file(file_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in uploaded file")

    added = indexer.index_text(text, source=file.filename)
    return {"message": "File indexed", "file": file.filename, "chunks_added": added}


@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")

    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    answer = rag_pipeline.run(payload.query)
    return {"answer": answer}


@app.exception_handler(Exception)
def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})
