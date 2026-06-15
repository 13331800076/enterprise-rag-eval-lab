#!/usr/bin/env python3
"""Enhanced evaluation benchmark with reranker comparison."""
import json
from pathlib import Path

from rag_lab.chunking.chunker import FixedSizeChunker, HeadingBasedChunker
from rag_lab.evaluation.evaluator import (
    QAExample,
    evaluate_single,
    aggregate_metrics,
    format_markdown_table,
)
from rag_lab.ingestion.ingestion import IngestionPipeline
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.vector import VectorRetriever
from rag_lab.retrieval.hybrid import HybridRetriever
from rag_lab.reranking.reranker import NoOpReranker, KeywordOverlapReranker


def main():
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
    if eval_path.exists():
        with open(eval_path) as f:
            raw_examples = json.load(f)
    else:
        raw_examples = []

    examples = [
        QAExample(
            question=e["question"],
            relevant_chunk_ids=set(e.get("relevant_chunk_ids", [])),
            answer=e.get("answer", ""),
        )
        for e in raw_examples
    ]

    if not examples:
        print("No evaluation examples found.")
        return

    k_values = [5]

    # Base retrievers
    base_retrievers = {
        "BM25": BM25Retriever(),
        "Vector": VectorRetriever(),
        "Hybrid": HybridRetriever(),
    }

    # Rerankers
    rerankers = {
        "No Rerank": NoOpReranker(),
        "KeywordOverlap": KeywordOverlapReranker(),
    }

    all_results = {}

    for base_name, base_retriever in base_retrievers.items():
        base_retriever.index(chunks)

        for rerank_name, reranker in rerankers.items():
            # Evaluate with reranker applied
            results = []
            for ex in examples:
                raw = base_retriever.search(ex.question, top_k=max(k_values) * 2)
                reranked = reranker.rerank(ex.question, raw)
                # Truncate to top_k for metric calculation
                from rag_lab.evaluation.evaluator import EvalResult
                retrieved_ids = [r.chunk_id for r in reranked[:max(k_values)]]
                metrics = {}
                from rag_lab.evaluation.evaluator import recall_at_k, mrr_at_k, ndcg_at_k
                for k in k_values:
                    metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, ex.relevant_chunk_ids, k)
                    metrics[f"mrr@{k}"] = mrr_at_k(retrieved_ids, ex.relevant_chunk_ids, k)
                    metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, ex.relevant_chunk_ids, k)
                results.append(
                    EvalResult(
                        question=ex.question,
                        retriever_name=f"{base_name}+{rerank_name}",
                        retrieved_ids=retrieved_ids,
                        metrics=metrics,
                    )
                )
            all_results[f"{base_name}+{rerank_name}"] = aggregate_metrics(results)

    # Print leaderboard
    print("\n## RAG Retrieval Leaderboard\n")
    print(format_markdown_table(all_results, k_values))

    # Save
    output_path = Path("examples/eval_results_full.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}\n")


if __name__ == "__main__":
    main()
