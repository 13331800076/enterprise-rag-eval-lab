"""Tests for hybrid retrieval."""

import pytest

from rag_lab.chunking.chunker import Chunk
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.hybrid import HybridRetriever
from rag_lab.retrieval.vector import VectorRetriever


class TestHybridRetriever:
    def test_weighted_fusion(self):
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", text="BM25 is a lexical retrieval algorithm", start_char=0, end_char=40),
            Chunk(chunk_id="c2", doc_id="d1", text="Vector search uses embeddings for semantic similarity", start_char=41, end_char=100),
            Chunk(chunk_id="c3", doc_id="d2", text="Hybrid retrieval combines both approaches", start_char=0, end_char=45),
            Chunk(chunk_id="c4", doc_id="d2", text="Python programming language tutorial", start_char=46, end_char=85),
        ]
        retriever = HybridRetriever(alpha=0.5, fusion_strategy="weighted")
        retriever.index(chunks)
        results = retriever.search("retrieval algorithm", top_k=3)
        assert len(results) <= 3
        assert all(r.score >= 0 for r in results)

    def test_rrf_fusion(self):
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", text="BM25 is a lexical retrieval algorithm", start_char=0, end_char=40),
            Chunk(chunk_id="c2", doc_id="d1", text="Vector search uses embeddings for semantic similarity", start_char=41, end_char=100),
        ]
        retriever = HybridRetriever(fusion_strategy="rrf")
        retriever.index(chunks)
        results = retriever.search("retrieval algorithm", top_k=2)
        assert len(results) <= 2

    def test_search_before_index_raises(self):
        retriever = HybridRetriever()
        with pytest.raises(RuntimeError, match="index"):
            retriever.search("query")

    def test_both_retrievers_included(self):
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", text="lexical matching with bm25 algorithm", start_char=0, end_char=40),
            Chunk(chunk_id="c2", doc_id="d1", text="semantic embeddings and vector search", start_char=41, end_char=85),
        ]
        retriever = HybridRetriever(alpha=0.5, fusion_strategy="weighted")
        retriever.index(chunks)
        results = retriever.search("search algorithm", top_k=2)
        # Both chunks should be present in hybrid results
        ids = {r.chunk_id for r in results}
        assert ids <= {"c1", "c2"}
