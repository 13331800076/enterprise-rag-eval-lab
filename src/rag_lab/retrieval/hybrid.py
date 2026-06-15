"""Hybrid retrieval combining BM25 and vector retrievers."""

from typing import Literal

from rag_lab.chunking.chunker import Chunk
from rag_lab.retrieval.bm25 import BaseRetriever, SearchResult
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.vector import VectorRetriever


def _normalize_scores(results: list[SearchResult]) -> list[SearchResult]:
    """Min-max normalize scores to [0, 1]."""
    if not results:
        return []
    scores = [r.score for r in results]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [SearchResult(chunk_id=r.chunk_id, text=r.text, score=1.0, metadata=r.metadata) for r in results]
    return [
        SearchResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=(r.score - min_score) / (max_score - min_score),
            metadata=r.metadata,
        )
        for r in results
    ]


def _reciprocal_rank_fusion(
    bm25_results: list[SearchResult],
    vector_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of two result lists."""
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    metas: dict[str, dict] = {}

    for rank, result in enumerate(bm25_results, start=1):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1 / (k + rank)
        texts[result.chunk_id] = result.text
        metas[result.chunk_id] = {**result.metadata, "rrf_source": "bm25"}

    for rank, result in enumerate(vector_results, start=1):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1 / (k + rank)
        texts[result.chunk_id] = result.text
        # If already seen, merge sources
        existing_source = metas.get(result.chunk_id, {}).get("rrf_source", "")
        if existing_source:
            metas[result.chunk_id] = {**result.metadata, "rrf_source": f"{existing_source}+vector"}
        else:
            metas[result.chunk_id] = {**result.metadata, "rrf_source": "vector"}

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [
        SearchResult(
            chunk_id=cid,
            text=texts[cid],
            score=scores[cid],
            metadata=metas[cid],
        )
        for cid in sorted_ids
    ]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 and dense vector retrieval."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        alpha: float = 0.5,
        fusion_strategy: Literal["weighted", "rrf"] = "weighted",
    ):
        self.bm25 = bm25_retriever or BM25Retriever()
        self.vector = vector_retriever or VectorRetriever()
        self.alpha = alpha
        self.fusion_strategy = fusion_strategy
        self._indexed = False

    def index(self, chunks: list[Chunk]) -> None:
        self.bm25.index(chunks)
        self.vector.index(chunks)
        self._indexed = True

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._indexed:
            raise RuntimeError("Retriever not indexed. Call index() first.")

        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        vector_results = self.vector.search(query, top_k=top_k * 2)

        if self.fusion_strategy == "rrf":
            return _reciprocal_rank_fusion(bm25_results, vector_results)[:top_k]

        # Weighted fusion
        bm25_norm = _normalize_scores(bm25_results)
        vector_norm = _normalize_scores(vector_results)

        combined: dict[str, float] = {}
        texts: dict[str, str] = {}
        metas: dict[str, dict] = {}

        for r in bm25_norm:
            combined[r.chunk_id] = combined.get(r.chunk_id, 0) + (1 - self.alpha) * r.score
            texts[r.chunk_id] = r.text
            metas[r.chunk_id] = {**r.metadata, "fusion_weight": 1 - self.alpha, "source": "bm25"}

        for r in vector_norm:
            combined[r.chunk_id] = combined.get(r.chunk_id, 0) + self.alpha * r.score
            texts[r.chunk_id] = r.text
            existing = metas.get(r.chunk_id, {})
            if existing.get("source") == "bm25":
                metas[r.chunk_id] = {
                    **r.metadata,
                    "fusion_weight": self.alpha,
                    "source": "hybrid",
                    "hybrid_score": combined[r.chunk_id],
                }
            else:
                metas[r.chunk_id] = {**r.metadata, "fusion_weight": self.alpha, "source": "vector"}

        sorted_ids = sorted(combined.keys(), key=lambda cid: combined[cid], reverse=True)
        return [
            SearchResult(
                chunk_id=cid,
                text=texts[cid],
                score=combined[cid],
                metadata=metas[cid],
            )
            for cid in sorted_ids[:top_k]
        ]
