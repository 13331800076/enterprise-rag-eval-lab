"""Evaluation module for retrieval metrics."""

from dataclasses import dataclass, field
from typing import Any

import math

from rag_lab.retrieval.bm25 import BaseRetriever, SearchResult


@dataclass
class QAExample:
    """A question-answer pair with relevant chunk IDs for evaluation."""
    question: str
    relevant_chunk_ids: set[str]
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result for a single example."""
    question: str
    retriever_name: str
    retrieved_ids: list[str]
    metrics: dict[str, float]


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k: proportion of relevant items retrieved in top-k."""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & relevant_ids) / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Mean Reciprocal Rank@k: reciprocal of the rank of the first relevant item."""
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    return sum(
        rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized DCG@k."""
    if not relevant_ids:
        return 0.0
    relevances = [1.0 if cid in relevant_ids else 0.0 for cid in retrieved_ids[:k]]
    ideal_relevances = [1.0] * min(len(relevant_ids), k)
    dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def evaluate_single(
    example: QAExample,
    retriever: BaseRetriever,
    retriever_name: str,
    k_values: list[int] = None,
) -> EvalResult:
    """Evaluate a single QA example against a retriever."""
    k_values = k_values or [1, 3, 5]
    results = retriever.search(example.question, top_k=max(k_values))
    retrieved_ids = [r.chunk_id for r in results]

    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, example.relevant_chunk_ids, k)
        metrics[f"mrr@{k}"] = mrr_at_k(retrieved_ids, example.relevant_chunk_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, example.relevant_chunk_ids, k)

    return EvalResult(
        question=example.question,
        retriever_name=retriever_name,
        retrieved_ids=retrieved_ids,
        metrics=metrics,
    )


def run_evaluation(
    examples: list[QAExample],
    retrievers: dict[str, BaseRetriever],
    k_values: list[int] = None,
) -> dict[str, list[EvalResult]]:
    """Run evaluation across multiple retrievers and examples."""
    results: dict[str, list[EvalResult]] = {}
    for name, retriever in retrievers.items():
        results[name] = []
        for example in examples:
            results[name].append(evaluate_single(example, retriever, name, k_values))
    return results


def aggregate_metrics(results: list[EvalResult]) -> dict[str, float]:
    """Average metrics across all examples."""
    if not results:
        return {}
    metric_names = list(results[0].metrics.keys())
    return {
        name: sum(r.metrics[name] for r in results) / len(results)
        for name in metric_names
    }


def format_markdown_table(
    aggregated: dict[str, dict[str, float]],
    k_values: list[int] = None,
) -> str:
    """Format aggregated results as a Markdown table."""
    k_values = k_values or [5]
    headers = ["Retriever"] + [f"Recall@{k}" for k in k_values] + [f"MRR@{k}" for k in k_values] + [f"nDCG@{k}" for k in k_values]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for name, metrics in aggregated.items():
        row = [name]
        for k in k_values:
            row.append(f"{metrics.get(f'recall@{k}', 0):.2f}")
        for k in k_values:
            row.append(f"{metrics.get(f'mrr@{k}', 0):.2f}")
        for k in k_values:
            row.append(f"{metrics.get(f'ndcg@{k}', 0):.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
