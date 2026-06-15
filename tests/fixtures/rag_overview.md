# RAG Overview

## What is RAG

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to ground the LLM's responses in factual information.

## Why RAG Matters

Large language models can hallucinate facts. RAG reduces hallucination by providing external evidence. The retrieved passages act as context windows for the generation model.

## Key Components

A complete RAG pipeline includes document ingestion, text chunking, vector embeddings, retrieval algorithms, and answer generation.

## Evaluation

Common metrics for RAG retrieval include Recall@k, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG). These metrics help compare different retrieval strategies.

## Advanced Techniques

### Reranking

After initial retrieval, a reranker can reorder results to improve precision. Cross-encoder models like BAAI/bge-reranker and cross-encoder/ms-marco-MiniLM evaluate query-passage pairs jointly, producing more accurate relevance scores than bi-encoders.

### Hybrid Fusion

Reciprocal Rank Fusion (RRF) and weighted score fusion combine lexical and semantic retrieval. RRF uses the formula: score = sum(1 / (k + rank)) for each retriever. Weighted fusion uses normalized scores: final = alpha * vector + (1 - alpha) * bm25.

### Multi-hop Retrieval

Some questions require connecting information across multiple documents. Multi-hop retrieval iteratively retrieves and reasons over multiple passages to answer complex queries.

## Deployment Considerations

Enterprise RAG requires robust document parsing, chunking strategies, and hybrid retrieval methods. Docker containers and FastAPI services provide scalable deployment options. FAISS and Milvus offer vector store backends for different scale requirements.
