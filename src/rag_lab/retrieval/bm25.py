"""BM25 retrieval using rank-bm25."""

from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from rag_lab.chunking.chunker import Chunk


@dataclass
class SearchResult:
    """Represents a search result."""
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class BaseRetriever:
    """Base interface for retrievers."""

    def index(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    """BM25 retriever using rank-bm25."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self._indexed = False

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._indexed = True
        if not chunks:
            self.bm25 = None
            return
        tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._indexed:
            raise RuntimeError("Retriever not indexed. Call index() first.")
        if self.bm25 is None or not self.chunks:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    metadata={"retriever": "bm25", **chunk.metadata},
                )
            )
        return results
