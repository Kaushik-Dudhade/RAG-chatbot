# app/indexer.py
import os
import uuid
from typing import List, Optional
from app.embedder import Embedder
from app.config import settings
import chromadb
from chromadb.config import Settings
import nltk

# tokenization for semantic chunking
from transformers import AutoTokenizer




class Indexer:
    def __init__(self, persist_dir: str | None = None):
        print(">>> indexer.py started")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # prefer provided persist_dir, then env config
        persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir = os.path.join(project_root, persist_dir)

        # Ensure NLTK punkt tokenizer is available (quietly)
        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            nltk.download("punkt", quiet=True)

        # Use current Settings API: enable persistence
        self.client = chromadb.Client(
            Settings(is_persistent=True, persist_directory=self.persist_dir, anonymized_telemetry=False)
        )

        # create or open the collection
        self.collection = self.client.get_or_create_collection(name="rag_documents")

        self.embedder = Embedder()
        # tokenizer for token-aware chunking — use same base model as embedder
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        except Exception:
            # fallback to a small sensible tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def chunk_text(self, text: str, chunk_size=200, overlap=40):
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def chunk_text_by_tokens(self, text: str, target_tokens: int = 200, overlap_tokens: int = 40) -> List[str]:
        """
        Build chunks by combining sentences until token count reaches target_tokens.
        Uses the HuggingFace tokenizer to estimate token counts.
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [text]

        sentence_tokens = [len(self.tokenizer.encode(s, add_special_tokens=False)) for s in sentences]

        chunks: List[str] = []
        i = 0
        n = len(sentences)

        while i < n:
            cur_tokens = 0
            j = i
            buf = []
            while j < n and cur_tokens + sentence_tokens[j] <= target_tokens:
                buf.append(sentences[j])
                cur_tokens += sentence_tokens[j]
                j += 1

            # if no sentence was added because a single sentence exceeds target, add it alone
            if not buf:
                buf.append(sentences[i])
                j = i + 1

            chunks.append(" ".join(buf))

            # step forward with overlap in tokens -> find new i such that overlap_tokens are kept
            if overlap_tokens > 0 and j < n:
                # move back to include overlap_tokens worth of sentences
                k = j - 1
                overlap_count = 0
                while k >= i and overlap_count < overlap_tokens:
                    overlap_count += sentence_tokens[k]
                    k -= 1
                # start next chunk from k+1
                i = max(i + 1, k + 1)
            else:
                i = j

        return chunks


    def index_text(self, text: str, source: str = "sample_doc"):
        # Tokenize into sentences and build sentence-based chunks.
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            # NLTK data may not be installed in some environments (CI/tests).
            # Fall back to a simple splitter on newlines and punctuation.
            import re

            s = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
            sentences = [seg.strip() for seg in s if seg.strip()]

        # Sentence-based chunking parameters (can be tuned)
        sentences_per_chunk = 5
        overlap = 1
        chunks: List[str] = []

        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)
            i += sentences_per_chunk - overlap

        if not chunks:
            return 0

        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in chunks]

        # Add metadata for each chunk
        metadatas = [
            {"source": source, "chunk_index": idx + 1, "total_chunks": len(chunks)}
            for idx in range(len(chunks))
        ]

        # Embed chunks
        embeddings = self.embedder.embed_batch(chunks)

        # Add to ChromaDB collection
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        # Persist to disk so retriever/processes can read
        try:
            self.client.persist()
        except Exception:
            # older/newer chromadb clients may not need explicit persist
            pass

        print(f"Indexed {len(chunks)} sentence-aware chunks from {source}")

        # return number of chunks added for API feedback
        return len(chunks)

if __name__ == "__main__":
    indexer = Indexer()

    sample_text = """
    Retrieval-Augmented Generation (RAG) combines retrieval and LLMs.
    Documents are embedded and stored in a vector database.
    At query time, relevant documents are retrieved as context for the LLM.
    This is paragraph two.
    This is paragraph three.
    This is paragraph four.
    """

    indexer.index_text(sample_text, source="sample_doc")
