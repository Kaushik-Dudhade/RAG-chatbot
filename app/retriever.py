# app/retriever.py
import os
from app.embedder import Embedder
from app.config import settings
import chromadb
from chromadb.config import Settings
import numpy as np

class Retriever:
    def __init__(self, persist_dir: str | None = None, top_k=3):
        print(">>> retriever.py started")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir = os.path.join(project_root, persist_dir)
        # use the current Settings API: enable persistence and set directory
        self.client = chromadb.Client(
            Settings(is_persistent=True, persist_directory=self.persist_dir, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name="rag_documents")
        self.embedder = Embedder()
        self.top_k = top_k
        print("Total documents in collection:", self.collection.count())

    def retrieve(self, query: str):
        query_embedding = self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        # Optionally rerank by cosine similarity between query and returned embeddings if available
        try:
            embeddings = results.get("embeddings", [[]])[0]
            docs = results.get("documents", [[]])[0]
            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            if embeddings:
                q = np.array(query_embedding)
                scores = [float(np.dot(q, np.array(e)) / (np.linalg.norm(q) * np.linalg.norm(np.array(e)) + 1e-10)) for e in embeddings]
                # sort by score desc
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                reranked = {
                    "documents": [[docs[i] for i in order]],
                    "scores": [[scores[i] for i in order]],
                    "ids": [[ids[i] for i in order]],
                    "metadatas": [[metadatas[i] for i in order]]
                }
                return reranked
        except Exception:
            pass

        return results

if __name__ == "__main__":
    retriever = Retriever()
    query = "What is Retrieval Augmented Generation?"
    results = retriever.retrieve(query)
    print("\nRetrieved documents:\n")
    for doc in results["documents"][0]:
        print("-", doc)
