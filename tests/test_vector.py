"""Tests for vector retrieval."""

import pytest

from rag_lab.chunking.chunker import Chunk
from rag_lab.retrieval.vector import VectorRetriever


class TestVectorRetriever:
    def test_index_and_search(self):
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", text="RAG systems retrieve documents", start_char=0, end_char=30),
            Chunk(chunk_id="c2", doc_id="d1", text="Machine learning models learn patterns", start_char=31, end_char=70),
            Chunk(chunk_id="c3", doc_id="d2", text="Python is a programming language", start_char=0, end_char=35),
        ]
        retriever = VectorRetriever()
        retriever.index(chunks)
        results = retriever.search("retrieve documents", top_k=2)
        assert len(results) == 2
        assert results[0].metadata["retriever"] == "vector"
        assert "model" in results[0].metadata

    def test_search_before_index_raises(self):
        retriever = VectorRetriever()
        with pytest.raises(RuntimeError, match="index"):
            retriever.search("query")

    def test_empty_chunks(self):
        retriever = VectorRetriever()
        retriever.index([])
        results = retriever.search("query", top_k=5)
        assert results == []
