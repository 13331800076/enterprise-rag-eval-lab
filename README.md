# Enterprise RAG Evaluation Lab

**One-liner:** A production-style RAG pipeline for enterprise documents with hybrid retrieval, reranking, and systematic evaluation.

## Overview

This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline designed for enterprise document use cases. It includes:

- Document ingestion (TXT, Markdown, HTML)
- Multiple chunking strategies
- Hybrid retrieval (BM25 + Dense vector)
- Reranking
- Answer generation
- Evaluation metrics (Recall@k, MRR, nDCG)

## Quick Start (Placeholder)

```bash
pip install -e ".[dev]"
pytest
rag-lab hello
```

## Architecture (Coming Soon)

## Tech Stack

- Python 3.10+
- FastAPI
- sentence-transformers
- FAISS
- rank-bm25
- pytest

## Project Structure

```
├── src/rag_lab/
│   ├── ingestion/
│   ├── chunking/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── evaluation/
│   ├── api/
│   └── cli.py
├── tests/
├── examples/
├── docs/
├── Dockerfile
└── docker-compose.yml
```
