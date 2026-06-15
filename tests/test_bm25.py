"""Tests for BM25 retrieval."""

import pytest

from rag_lab.chunking.chunker import Chunk
from rag_lab.retrieval.bm25 import BM25Retriever, SearchResult


class TestBM25Retriever:
    def test_index_and_search(self):
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", text="RAG systems retrieve documents to generate answers", start_char=0, end_char=50),
            Chunk(chunk_id="c2", doc_id="d1", text="BM25 is a classic lexical retrieval algorithm", start_char=51, end_char=100),
            Chunk(chunk_id="c3", doc_id="d2", text="Vector search uses embeddings for semantic similarity", start_char=0, end_char=60),
        ]
        retriever = BM25Retriever()
        retriever.index(chunks)
        results = retriever.search("lexical retrieval algorithm", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].chunk_id == "c2"  # BM25 chunk should rank highest
        assert results[0].metadata["retriever"] == "bm25"

    def test_search_before_index_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="index"):
            retriever.search("query")

    def test_empty_chunks(self):
        retriever = BM25Retriever()
        retriever.index([])
        results = retriever.search("query", top_k=5)
        assert results == []
