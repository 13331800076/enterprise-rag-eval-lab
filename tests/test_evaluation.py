"""Tests for evaluation metrics."""

import math

from rag_lab.evaluation.evaluator import (
    QAExample,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    dcg_at_k,
    evaluate_single,
    aggregate_metrics,
)
from rag_lab.chunking.chunker import Chunk
from rag_lab.retrieval.bm25 import BM25Retriever


class TestMetrics:
    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=2) == 0.5
        assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=1) == 0.5
        assert recall_at_k(["a", "b", "c"], {"d", "e"}, k=3) == 0.0

    def test_mrr_at_k(self):
        assert mrr_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0
        assert mrr_at_k(["a", "b", "c"], {"b"}, k=3) == 0.5
        assert mrr_at_k(["a", "b", "c"], {"d"}, k=3) == 0.0

    def test_dcg_at_k(self):
        rels = [1.0, 0.0, 1.0]
        dcg = dcg_at_k(rels, k=3)
        expected = 1.0 / math.log2(2) + 0.0 / math.log2(3) + 1.0 / math.log2(4)
        assert abs(dcg - expected) < 1e-6

    def test_ndcg_at_k(self):
        assert ndcg_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0
        assert ndcg_at_k(["b", "a", "c"], {"a"}, k=3) < 1.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


class TestEvaluateSingle:
    def test_evaluate_single(self):
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", text="RAG systems retrieve documents", start_char=0, end_char=30),
            Chunk(chunk_id="c2", doc_id="d1", text="BM25 is a lexical retrieval algorithm", start_char=31, end_char=70),
        ]
        retriever = BM25Retriever()
        retriever.index(chunks)

        example = QAExample(
            question="lexical retrieval",
            relevant_chunk_ids={"c2"},
        )
        result = evaluate_single(example, retriever, "BM25", k_values=[1, 2])
        assert result.retriever_name == "BM25"
        assert result.metrics["recall@1"] == 0.0 or result.metrics["recall@1"] == 1.0
        assert result.metrics["recall@2"] == 1.0
        assert result.metrics["mrr@2"] == 0.5

    def test_aggregate(self):
        from dataclasses import dataclass, field

        @dataclass
        class MockResult:
            metrics: dict

        results = [
            MockResult(metrics={"recall@1": 1.0, "mrr@1": 1.0}),
            MockResult(metrics={"recall@1": 0.0, "mrr@1": 0.0}),
        ]
        agg = aggregate_metrics(results)
        assert agg["recall@1"] == 0.5
        assert agg["mrr@1"] == 0.5
