# Enterprise RAG Evaluation Lab

**One-liner:** A production-style RAG pipeline for enterprise documents with hybrid retrieval, reranking, and systematic evaluation.

## Problem Statement

Enterprise RAG systems face a unique challenge: they must handle diverse document formats (TXT, Markdown, HTML), chunk text intelligently, retrieve relevant passages using both lexical and semantic signals, and provide measurable evidence that the retrieval pipeline works. Most public demos skip the evaluation step, leaving no way to prove the system actually retrieves the right content.

This project closes that gap by building a complete, testable, and benchmarked RAG pipeline from ingestion to evaluation.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Document       │────▶│  Text        │────▶│  BM25        │
│  Ingestion      │     │  Chunking    │     │  Retriever   │
│  (txt/md/html)  │     │  (fixed/     │     │              │
│                 │     │   heading)    │     │              │
└─────────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
                              ┌─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Hybrid         │
                    │  Fusion         │◄────┌──────────────┐
                    │  (weighted/RRF) │     │  Vector      │
                    └────────┬────────┘     │  Retriever   │
                             │              │  (FAISS)     │
                             │              └──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Reranking      │
                    │  (keyword/     │
                    │   cross-enc)    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Answer         │
                    │  Generation     │
                    │  (mock/LLM)    │
                    └─────────────────┘
```

## Key Features

- **Document Ingestion**: Supports TXT, Markdown, and HTML with automatic script/style stripping for HTML
- **Chunking Strategies**: Fixed-size chunking with overlap, and heading-based chunking for Markdown
- **Hybrid Retrieval**: Combines BM25 lexical search with dense vector search (FAISS + sentence-transformers)
- **Reranking**: Pluggable reranker interface with keyword overlap baseline
- **Evaluation**: Recall@k, MRR, nDCG metrics with automated benchmark comparison across retrievers
- **FastAPI Service**: RESTful endpoints for ingestion, search, and answer generation
- **Docker Support**: Ready-to-run with `docker compose up`

## Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- sentence-transformers (default: all-MiniLM-L6-v2)
- FAISS (CPU)
- rank-bm25
- pytest
- Docker + Docker Compose

## Quick Start

### Local Development

```bash
pip install -e ".[dev]"
pytest
rag-lab hello
```

### Run FastAPI Server

```bash
uvicorn rag_lab.api.app:app --reload
```

Test endpoints:

```bash
curl http://localhost:8000/health
```

### Docker

```bash
docker compose up
```

## API Usage

### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "examples/docs"}'
```

### Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "enterprise RAG", "top_k": 5, "retriever_type": "hybrid"}'
```

Retriever types: `bm25`, `vector`, `hybrid` (default).

### Answer with Evidence

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid retrieval?", "top_k": 3, "retriever_type": "hybrid"}'
```

## Evaluation Methodology

The evaluation framework compares three retrieval strategies using a manually curated QA dataset (`examples/eval_dataset.json`):

1. **BM25-only**: Lexical baseline
2. **Vector-only**: Dense semantic baseline
3. **Hybrid**: Weighted fusion of BM25 + Vector scores

Metrics computed per example:
- **Recall@k**: Proportion of relevant chunks found in top-k
- **MRR@k**: Mean Reciprocal Rank of the first relevant chunk
- **nDCG@k**: Normalized Discounted Cumulative Gain

## Benchmark Results

> **Note:** These numbers are placeholders. Run the evaluation script to generate real results.

| Retriever | Recall@5 | MRR@5 | nDCG@5 |
|-----------|---------:|------:|-------:|
| BM25      | 0.72     | 0.61  | 0.66   |
| Vector    | 0.76     | 0.64  | 0.69   |
| Hybrid    | 0.84     | 0.72  | 0.77   |

Run evaluation:

```bash
python -m examples.run_evaluation
```

## Project Structure

```
enterprise-rag-eval-lab/
├── README.md
├── SPEC.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── examples/
│   ├── docs/
│   ├── eval_dataset.json
│   ├── bm25_demo.py
│   ├── vector_demo.py
│   └── hybrid_retrieval_demo.py
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
│   ├── fixtures/
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_bm25.py
│   ├── test_vector.py
│   ├── test_hybrid.py
│   ├── test_evaluation.py
│   └── test_api.py
└── docs/
    ├── architecture.md
    └── design-decisions.md
```

## Roadmap

- [x] Document ingestion (TXT, Markdown, HTML)
- [x] Text chunking (fixed-size, heading-based)
- [x] BM25 retrieval
- [x] Vector retrieval (FAISS + sentence-transformers)
- [x] Hybrid retrieval (weighted fusion + RRF)
- [x] Reranking interface (keyword overlap)
- [x] Evaluation metrics (Recall@k, MRR, nDCG)
- [x] FastAPI endpoints
- [x] Docker support
- [ ] Cross-encoder reranker (bge-reranker, ms-marco)
- [ ] OpenAI-compatible LLM generation endpoint
- [ ] Milvus / Neo4j backend support
- [ ] Streamlit frontend
- [ ] Production deployment guide (Kubernetes, etc.)

## What This Project Demonstrates

- **End-to-end RAG system design**: From raw documents to evaluated answers
- **Hybrid retrieval expertise**: Combining lexical and semantic signals with proven fusion strategies
- **Engineering rigor**: Every module has unit tests; the API is lazily initialized to avoid heavy model loading at startup
- **Evaluation discipline**: Metrics-driven development rather than "it looks good"
- **Production readiness**: Dockerized, FastAPI-based, with clear extension points for LLM and cross-encoder rerankers

---

Built for demonstration and learning. Not for production use without additional hardening (authentication, rate limiting, persistent storage, etc.).
