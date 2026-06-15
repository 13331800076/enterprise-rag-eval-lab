#!/usr/bin/env python3
"""Run evaluation comparing BM25, Vector, and Hybrid retrieval."""

import json
from pathlib import Path

from rag_lab.chunking.chunker import FixedSizeChunker
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


def main():
    # Load documents
    pipeline = IngestionPipeline()
    docs = pipeline.ingest_directory("tests/fixtures")

    # Chunk
    chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    chunks = []
    for doc in docs:
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
            relevant_chunk_ids=set(e["relevant_chunk_ids"]),
            answer=e.get("answer", ""),
        )
        for e in raw_examples
    ]

    if not examples:
        print("No evaluation examples found. Please create examples/eval_dataset.json")
        return

    # Initialize retrievers
    bm25 = BM25Retriever()
    bm25.index(chunks)

    vector = VectorRetriever()
    vector.index(chunks)

    hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)
    hybrid.index(chunks)

    retrievers = {
        "BM25": bm25,
        "Vector": vector,
        "Hybrid": hybrid,
    }

    # Run evaluation
    k_values = [5]
    all_results = {}
    for name, retriever in retrievers.items():
        results = []
        for example in examples:
            results.append(evaluate_single(example, retriever, name, k_values))
        all_results[name] = results

    # Aggregate
    aggregated = {name: aggregate_metrics(results) for name, results in all_results.items()}

    # Print Markdown table
    print("\n## Benchmark Results\n")
    print(format_markdown_table(aggregated, k_values))

    # Save JSON
    output_path = Path("examples/eval_results.json")
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nResults saved to {output_path}\n")


if __name__ == "__main__":
    main()
