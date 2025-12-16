from app.indexer import Indexer
from app.retriever import Retriever


def test_index_and_retrieve_cycle():
    idx = Indexer(persist_dir="chroma_test")
    text = "RAG testing document about retrieval and generation."
    idx.index_text(text, source="test_cycle")

    r = Retriever(persist_dir="chroma_test", top_k=2)
    res = r.retrieve("What is retrieval?")
    # Should at least return documents list
    assert "documents" in res
    assert isinstance(res["documents"], list)
