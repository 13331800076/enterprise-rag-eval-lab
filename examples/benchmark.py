"""Benchmark runner comparing all retrieval strategies including reranking."""

import json
from pathlib import Path

from rag_lab.ingestion.ingestion import IngestionPipeline
from rag_lab.chunking.chunker import FixedSizeChunker, HeadingBasedChunker
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.vector import VectorRetriever
from rag_lab.retrieval.hybrid import HybridRetriever
from rag_lab.reranking.reranker import NoOpReranker, KeywordOverlapReranker
from rag_lab.evaluation.evaluator import (
    QAExample,
    evaluate_single,
    aggregate_metrics,
)


def run_full_benchmark():
    # Load documents
    pipeline = IngestionPipeline()
    docs = pipeline.ingest_directory("tests/fixtures")

    # Chunk with mixed strategy
    chunks = []
    for doc in docs:
        if doc.metadata.get("format") == "markdown":
            chunker = HeadingBasedChunker()
        else:
            chunker = FixedSizeChunker(chunk_size=300, overlap=0)
        chunks.extend(chunker.chunk(doc))

    # Load eval dataset
    eval_path = Path("examples/eval_dataset.json")
    with open(eval_path) as f:
        raw = json.load(f)

    examples = [
        QAExample(
            question=e["question"],
            relevant_chunk_ids=set(e.get("relevant_chunk_ids", [])),
            answer=e.get("answer", ""),
        )
        for e in raw
    ]

    k_values = [5]

    # Define all strategies
    strategies = [
        ("BM25", BM25Retriever(), NoOpReranker()),
        ("BM25 + KeywordRerank", BM25Retriever(), KeywordOverlapReranker()),
        ("Vector", VectorRetriever(), NoOpReranker()),
        ("Vector + KeywordRerank", VectorRetriever(), KeywordOverlapReranker()),
        ("Hybrid", HybridRetriever(), NoOpReranker()),
        ("Hybrid + KeywordRerank", HybridRetriever(), KeywordOverlapReranker()),
    ]

    results = {}
    for name, retriever, reranker in strategies:
        retriever.index(chunks)
        eval_results = []
        for ex in examples:
            raw_results = retriever.search(ex.question, top_k=max(k_values))
            reranked = reranker.rerank(ex.question, raw_results)
            eval_results.append(
                evaluate_single(ex, type("_", (), {"search": lambda self, q, k=None: reranked})(), name, k_values)
            )
        results[name] = aggregate_metrics(eval_results)

    # Print leaderboard
    print("\n## RAG Retrieval Leaderboard\n")
    print(f"{'Rank':<6} {'Strategy':<25} {'Recall@5':>10} {'MRR@5':>10} {'nDCG@5':>10}")
    print("-" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1].get("recall@5", 0), reverse=True)
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        print(f"{rank:<6} {name:<25} {metrics.get('recall@5', 0):>10.2f} {metrics.get('mrr@5', 0):>10.2f} {metrics.get('ndcg@5', 0):>10.2f}")

    # Save
    output_path = Path("examples/benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(dict(sorted_results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_full_benchmark()
