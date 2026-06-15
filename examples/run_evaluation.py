#!/usr/bin/env python3
"""Run evaluation benchmark across BM25, Vector, and Hybrid retrievers."""

import json
from pathlib import Path

from rag_lab.chunking.chunker import Chunk
from rag_lab.evaluation.evaluator import (
    QAExample,
    evaluate_single,
    aggregate_metrics,
    format_markdown_table,
)
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.vector import VectorRetriever
from rag_lab.retrieval.hybrid import HybridRetriever


def load_eval_dataset(path: str) -> list[QAExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        QAExample(
            question=item["question"],
            relevant_chunk_ids=set(item["relevant_chunk_ids"]),
            answer=item.get("answer", ""),
        )
        for item in data
    ]


def main():
    # Load chunks from demo dataset
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="Enterprise RAG systems combine retrieval with generation", start_char=0, end_char=60),
        Chunk(chunk_id="c2", doc_id="d1", text="BM25 is a probabilistic ranking function used in search engines for lexical retrieval", start_char=61, end_char=140),
        Chunk(chunk_id="c3", doc_id="d2", text="Dense vector retrieval uses embeddings to find semantically similar text chunks", start_char=0, end_char=80),
        Chunk(chunk_id="c4", doc_id="d2", text="Hybrid retrieval combines lexical BM25 scores and semantic vector scores for better results", start_char=81, end_char=170),
        Chunk(chunk_id="c5", doc_id="d3", text="Reranking reorders retrieved documents to improve precision at the top of the result list", start_char=0, end_char=90),
    ]

    # Initialize retrievers
    bm25 = BM25Retriever()
    vector = VectorRetriever()
    hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    bm25.index(chunks)
    vector.index(chunks)
    hybrid.index(chunks)

    # Load eval dataset
    eval_path = Path(__file__).parent / "eval_dataset.json"
    if not eval_path.exists():
        print(f"Eval dataset not found: {eval_path}")
        return

    examples = load_eval_dataset(str(eval_path))

    # Evaluate
    retrievers = {"BM25": bm25, "Vector": vector, "Hybrid": hybrid}
    k_values = [5]

    all_results = {}
    for name, retriever in retrievers.items():
        results = [evaluate_single(ex, retriever, name, k_values) for ex in examples]
        all_results[name] = aggregate_metrics(results)

    # Print table
    print("\n=== Retrieval Benchmark Results ===\n")
    print(format_markdown_table(all_results, k_values))
    print()


if __name__ == "__main__":
    main()
