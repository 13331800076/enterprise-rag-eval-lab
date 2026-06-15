# Enterprise RAG Evaluation Lab

## Goal
Build a production-style RAG pipeline for enterprise documents, including document parsing, chunking, hybrid retrieval, reranking, answer generation, and retrieval evaluation.

## MVP Scope
- Support Markdown, TXT, and simple HTML document ingestion.
- Implement chunking strategies: fixed-size chunking and heading-based chunking.
- Implement BM25 retrieval.
- Implement vector retrieval using sentence-transformers and FAISS.
- Implement hybrid retrieval by combining BM25 and vector scores.
- Implement reranking interface, with a simple placeholder reranker first.
- Implement answer generation interface, initially using a mock LLM or configurable OpenAI-compatible API.
- Implement evaluation metrics: Recall@k, MRR, nDCG.
- Provide FastAPI endpoints.
- Provide a CLI demo.
- Provide Dockerfile and docker-compose.yml.
- Provide tests for core modules.

## Tech Stack
Python, FastAPI, FAISS, rank-bm25, sentence-transformers, pytest, Docker.

## Non-goals
- Do not use company data.
- Do not build a complex frontend in v1.
- Do not depend on paid APIs for the basic demo.
