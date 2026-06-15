"""Vector retrieval using sentence-transformers and FAISS."""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from rag_lab.chunking.chunker import Chunk
from rag_lab.retrieval.bm25 import BaseRetriever, SearchResult


class VectorRetriever(BaseRetriever):
    """Dense vector retriever using FAISS."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.faiss_index: faiss.Index | None = None
        self._dimension: int = self.model.get_embedding_dimension()
        self._indexed = False

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self.faiss_index = None
            self._indexed = True
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        self.faiss_index = faiss.IndexFlatIP(self._dimension)
        self.faiss_index.add(embeddings)
        self._indexed = True

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._indexed:
            raise RuntimeError("Retriever not indexed. Call index() first.")
        if self.faiss_index is None or not self.chunks:
            return []

        query_embedding = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    metadata={"retriever": "vector", "model": self.model_name, **chunk.metadata},
                )
            )
        return results
