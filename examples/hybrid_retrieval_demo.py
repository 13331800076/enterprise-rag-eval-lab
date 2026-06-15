"""Hybrid retrieval demo."""

from rag_lab.chunking.chunker import Chunk
from rag_lab.retrieval.hybrid import HybridRetriever


def main():
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="Enterprise RAG systems combine retrieval with generation", start_char=0, end_char=60),
        Chunk(chunk_id="c2", doc_id="d1", text="BM25 is a probabilistic ranking function used in search engines", start_char=61, end_char=120),
        Chunk(chunk_id="c3", doc_id="d2", text="Dense retrieval uses vector embeddings to find semantically similar text", start_char=0, end_char=75),
        Chunk(chunk_id="c4", doc_id="d2", text="Hybrid retrieval combines lexical and semantic scores for better results", start_char=76, end_char=150),
    ]

    # Weighted fusion
    hybrid = HybridRetriever(alpha=0.6, fusion_strategy="weighted")
    hybrid.index(chunks)

    query = "semantic document retrieval"
    results = hybrid.search(query, top_k=3)

    print(f"Query: {query}")
    print("Fusion strategy: Weighted (alpha=0.6)\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.chunk_id}] score={result.score:.4f} source={result.metadata.get('source', 'unknown')}")
        print(f"   {result.text[:80]}...\n")

    # RRF fusion
    hybrid_rrf = HybridRetriever(fusion_strategy="rrf")
    hybrid_rrf.index(chunks)
    results_rrf = hybrid_rrf.search(query, top_k=3)

    print("Fusion strategy: RRF\n")
    for i, result in enumerate(results_rrf, 1):
        print(f"{i}. [{result.chunk_id}] score={result.score:.4f} rrf_source={result.metadata.get('rrf_source', 'unknown')}")
        print(f"   {result.text[:80]}...\n")


if __name__ == "__main__":
    main()
