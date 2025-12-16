# app/rag_pipeline.py
from app.retriever import Retriever

try:
    import ollama
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False


class RagPipeline:
    def __init__(self):
        # Retriever is relatively cheap; instantiate here
        self.retriever = Retriever()

    def run(self, query: str) -> str:
        results = self.retriever.retrieve(query)

        # results['documents'] is a list-of-lists per query
        context_docs = results.get("documents", [[]])[0]
        context = "\n\n".join(context_docs)

        prompt = f"""
You are a helpful assistant.
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        if _HAS_OLLAMA:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            # Ollama returns a nested structure; be defensive
            return response.get("message", {}).get("content", "")

        # Fallback: return context + a suggestion so API still responds
        return f"[OLLAMA not installed] Context: {context}\n\nQuestion: {query}"
