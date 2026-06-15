"""Tests for reranking."""

import pytest
from rag_lab.retrieval.bm25 import SearchResult
from rag_lab.reranking.reranker import NoOpReranker, KeywordOverlapReranker, CrossEncoderReranker


class TestNoOpReranker:
    def test_pass_through(self):
        results = [
            SearchResult(chunk_id="c1", text="hello world", score=1.0, metadata={}),
            SearchResult(chunk_id="c2", text="foo bar", score=0.5, metadata={}),
        ]
        reranker = NoOpReranker()
        reranked = reranker.rerank("hello", results)
        assert reranked == results


class TestKeywordOverlapReranker:
    def test_changes_order(self):
        results = [
            SearchResult(chunk_id="c1", text="machine learning models", score=1.0, metadata={}),
            SearchResult(chunk_id="c2", text="machine learning algorithms", score=0.5, metadata={}),
            SearchResult(chunk_id="c3", text="deep learning neural networks", score=0.8, metadata={}),
        ]
        reranker = KeywordOverlapReranker()
        reranked = reranker.rerank("learning algorithms", results)
        # c2 has both "learning" and "algorithms" -> overlap=2, score=0.5+2=2.5
        # c1 has "learning" only -> overlap=1, score=1.0+1=2.0
        # c3 has "learning" only -> overlap=1, score=0.8+1=1.8
        assert reranked[0].chunk_id == "c2"
        assert reranked[1].chunk_id == "c1"
        assert reranked[2].chunk_id == "c3"

    def test_overlap_metadata(self):
        results = [
            SearchResult(chunk_id="c1", text="hello world", score=1.0, metadata={}),
        ]
        reranker = KeywordOverlapReranker()
        reranked = reranker.rerank("hello", results)
        assert reranked[0].metadata["reranker"] == "keyword_overlap"
        assert reranked[0].metadata["overlap"] == 1


@pytest.mark.skip(reason="Cross-encoder model download requires network and takes time")
class TestCrossEncoderReranker:
    def test_rerank_changes_order(self):
        results = [
            SearchResult(chunk_id="c1", text="Python is a programming language", score=1.0, metadata={}),
            SearchResult(chunk_id="c2", text="Machine learning with Python", score=0.5, metadata={}),
            SearchResult(chunk_id="c3", text="Deep learning frameworks", score=0.8, metadata={}),
        ]
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("Python programming", results)
        assert len(reranked) == 3
        assert all(r.metadata["reranker"] == "cross_encoder" for r in reranked)
        assert reranked[0].metadata["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_empty_results(self):
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("query", [])
        assert reranked == []
