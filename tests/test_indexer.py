import os
from app.indexer import Indexer


def test_chunk_by_tokens_small():
    indexer = Indexer(persist_dir="chroma_test")
    text = "This is a short document. It has a few sentences. This should be chunked."
    chunks = indexer.chunk_text_by_tokens(text, target_tokens=50)
    assert len(chunks) >= 1


def test_index_text_returns_count():
    indexer = Indexer(persist_dir="chroma_test")
    count = indexer.index_text("Hello world. " * 10, source="test")
    assert isinstance(count, int)
    assert count > 0
