#!/usr/bin/env python3
"""Enhanced evaluation with reranker comparison and leaderboard generation."""

import json
from pathlib import Path

from rag_lab.ingestion.ingestion import IngestionPipeline
from rag_lab.chunking.chunker import FixedSizeChunker, HeadingBasedChunker
from rag_lab.evaluation.evaluator import (
    QAExample,
    evaluate_single,
    aggregate_metrics,
)
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

    k_values = [1, 3, 5]

    # Define all strategy combinations
    retrievers = {
        "BM25": BM25Retriever(),
        "Vector": VectorRetriever(),
        "Hybrid": HybridRetriever(),
    }

    rerankers = {
        "": NoOpReranker(),
        "+KeywordRerank": KeywordOverlapReranker(),
    }

    # Index all retrievers
    for r in retrievers.values():
        r.index(chunks)

    # Evaluate all combinations
    all_results = {}
    for r_name, retriever in retrievers.items():
        for rank_name, reranker in rerankers.items():
            strategy = f"{r_name}{rank_name}"
            res = []
            for ex in examples:
                raw = retriever.search(ex.question, top_k=max(k_values))
                ranked = reranker.rerank(ex.question, raw)
                # Evaluate using reranked results
                result = evaluate_single(ex, retriever, strategy, k_values)
                # Override with reranked results
                result.retrieved_ids = [r.chunk_id for r in ranked[:max(k_values)]]
                # Recalculate metrics
                from rag_lab.evaluation.evaluator import recall_at_k, mrr_at_k, ndcg_at_k
                for k in k_values:
                    result.metrics[f"recall@{k}"] = recall_at_k(result.retrieved_ids, ex.relevant_chunk_ids, k)
                    result.metrics[f"mrr@{k}"] = mrr_at_k(result.retrieved_ids, ex.relevant_chunk_ids, k)
                    result.metrics[f"ndcg@{k}"] = ndcg_at_k(result.retrieved_ids, ex.relevant_chunk_ids, k)
                res.append(result)
            all_results[strategy] = aggregate_metrics(res)

    # Save results
    output_path = Path("examples/eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print leaderboard
    print("\n## RAG Strategy Leaderboard\n")
    print("| Strategy | Recall@1 | Recall@5 | MRR@5 | nDCG@5 |")
    print("|---|---|---|---|---|")
    for name, metrics in sorted(all_results.items(), key=lambda x: x[1].get("ndcg@5", 0), reverse=True):
        print(f"| {name} | {metrics.get('recall@1', 0):.2f} | {metrics.get('recall@5', 0):.2f} | {metrics.get('mrr@5', 0):.2f} | {metrics.get('ndcg@5', 0):.2f} |")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_full_evaluation()
