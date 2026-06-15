#!/usr/bin/env python3
"""Enhanced evaluation with reranker comparison and leaderboard generation."""

import json
from pathlib import Path

from rag_lab.chunking.chunker import FixedSizeChunker, HeadingBasedChunker
from rag_lab.evaluation.evaluator import (
    QAExample,
    evaluate_single,
    aggregate_metrics,
)
from rag_lab.ingestion.ingestion import IngestionPipeline
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.vector import VectorRetriever
from rag_lab.retrieval.hybrid import HybridRetriever
from rag_lab.reranking.reranker import NoOpReranker, KeywordOverlapReranker


def run_full_evaluation():
    # Load documents
    pipeline = IngestionPipeline()
    docs = pipeline.ingest_directory("tests/fixtures")

    # Chunk
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
        raw_examples = json.load(f)

    examples = [
        QAExample(
            question=e["question"],
            relevant_chunk_ids=set(e.get("relevant_chunk_ids", [])),
            answer=e.get("answer", ""),
        )
        for e in raw_examples
    ]

    k_values = [1, 3, 5, 10]

    # Define all strategy combinations
    strategies = []

    # Base retrievers
    for name, retriever_class in [
        ("BM25", BM25Retriever),
        ("Vector", VectorRetriever),
        ("Hybrid", HybridRetriever),
    ]:
        strategies.append((name, retriever_class(), None))

    # With reranker
    for base_name, retriever_class in [
        ("BM25", BM25Retriever),
        ("Vector", VectorRetriever),
        ("Hybrid", HybridRetriever),
    ]:
        strategies.append(
            (f"{base_name} + Keyword Rerank", retriever_class(), KeywordOverlapReranker())
        )

    # Run evaluation
    all_results = {}
    for name, retriever, reranker in strategies:
        retriever.index(chunks)
        results = []
        for example in examples:
            raw = retriever.search(example.question, top_k=max(k_values))
            if reranker:
                raw = reranker.rerank(example.question, raw)
            # Re-evaluate with reranked results
            eval_result = evaluate_single(
                QAExample(
                    question=example.question,
                    relevant_chunk_ids=example.relevant_chunk_ids,
                ),
                retriever,  # dummy, we'll override
                name,
                k_values,
            )
            # Override with actual reranked results
            retrieved_ids = [r.chunk_id for r in raw[:max(k_values)]]
            from rag_lab.evaluation.evaluator import recall_at_k, mrr_at_k, ndcg_at_k
            metrics = {}
            for k in k_values:
                metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, example.relevant_chunk_ids, k)
                metrics[f"mrr@{k}"] = mrr_at_k(retrieved_ids, example.relevant_chunk_ids, k)
                metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, example.relevant_chunk_ids, k)
            eval_result.metrics = metrics
            results.append(eval_result)

        all_results[name] = aggregate_metrics(results)

    # Generate leaderboard
    print("\n## RAG Retrieval Leaderboard\n")
    print("| Rank | Strategy | Recall@5 | MRR@5 | nDCG@5 |")
    print("|------|----------|----------|-------|--------|")

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("ndcg@5", 0),
        reverse=True,
    )

    for rank, (name, metrics) in enumerate(sorted_results, 1):
        r5 = metrics.get("recall@5", 0)
        m5 = metrics.get("mrr@5", 0)
        n5 = metrics.get("ndcg@5", 0)
        print(f"| {rank} | {name} | {r5:.2f} | {m5:.2f} | {n5:.2f} |")

    # Save JSON
    output_path = Path("examples/leaderboard_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_full_evaluation()
