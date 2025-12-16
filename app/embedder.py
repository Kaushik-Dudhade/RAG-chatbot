"""
embedder.py
Responsible for converting text into vector embeddings.
Uses a local sentence-transformers model (no API keys).
"""

print(">>> embedder.py started")


from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around sentence-transformers embedding model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load embedding model.
        This downloads the model once and caches it locally.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        """
        Embed a single piece of text.
        """
        vector = self.model.encode(text)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts at once (much faster).
        """
        vectors = self.model.encode(texts)
        return [v.tolist() for v in vectors]

    def dimension(self) -> int:
        """
        Return embedding vector size.
        """
        test_vector = self.model.encode("test")
        return len(test_vector)


if __name__ == "__main__":
    # Simple test
    embedder = Embedder()

    print("Model:", embedder.model_name)
    print("Embedding dimension:", embedder.dimension())

    text = "RAG systems combine retrieval and generation"
    vec = embedder.embed(text)

    print("Vector length:", len(vec))
    print("First 10 values:", vec[:10])
