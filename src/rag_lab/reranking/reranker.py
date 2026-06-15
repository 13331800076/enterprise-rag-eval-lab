"""Reranking module."""

from rag_lab.retrieval.bm25 import SearchResult


class BaseReranker:
    """Base interface for rerankers."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    """Pass-through reranker that does nothing."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results


class KeywordOverlapReranker(BaseReranker):
    """Lightweight local reranker based on keyword overlap.

    Re-scores results by counting how many query words appear in the result text.
    This is a simple baseline suitable for local/edge deployments.
    """

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_words = set(query.lower().split())
        reranked = []
        for result in results:
            text_words = set(result.text.lower().split())
            overlap = len(query_words & text_words)
            new_score = result.score + overlap  # Boost original score by overlap count
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=new_score,
                    metadata={**result.metadata, "reranker": "keyword_overlap", "overlap": overlap},
                )
            )
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
