# Enterprise RAG Evaluation Lab

**One-liner:** A production-style RAG pipeline for enterprise documents, featuring hybrid retrieval, reranking, and systematic evaluation — built to demonstrate real-world ML engineering skills.

---

## Problem Statement

Most RAG demos online are toy projects: they ingest a few paragraphs, use a single OpenAI call, and skip evaluation. In production, enterprises need:

- Robust document parsing (PDF, Markdown, HTML, TXT)
- Multiple chunking strategies with metadata preservation
- Hybrid retrieval (lexical + dense) with score fusion
- Reranking to improve precision at the top-k
- Quantitative evaluation before deployment
- FastAPI service and Docker packaging

This project demonstrates every layer of a production RAG pipeline, from ingestion to evaluation, with clean architecture and comprehensive tests.

---

## Architecture

```
┌─────────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Documents      │────▶│  Ingestion  │────▶│  Chunking       │
│  (.md/.txt/.html)│     │  (Parser)   │     │  Fixed / Heading│
└─────────────────┘     └─────────────┘     └─────────────────┘
                                                       │
                       ┌───────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │  BM25 Retriever │
              │  (rank-bm25)    │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
    ┌─────────────────┐  ┌─────────────────┐
    │  Vector         │  │  Hybrid Fusion  │
    │  (FAISS +       │  │  (Weighted /    │
    │  sentence-      │  │  RRF)           │
    │  transformers)  │  └────────┬────────┘
    └────────┬────────┘           │
             │                    │
             └────────┬───────────┘
                      ▼
             ┌─────────────────┐
             │  Reranker       │
             │  (Keyword /     │
             │  Cross-Encoder) │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │  Answer Gen     │
             │  (Mock LLM /    │
             │  OpenAI API)    │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │  Evaluation     │
             │  Recall@k / MRR │
             │  nDCG@k         │
             └─────────────────┘
```

---

## Key Features

| Layer | Feature | Status |
|---|---|---|
| Ingestion | TXT, Markdown, HTML parsing | ✅ |
| Chunking | Fixed-size + Heading-based | ✅ |
| Retrieval | BM25 lexical search | ✅ |
| Retrieval | Dense vector search (FAISS) | ✅ |
| Retrieval | Hybrid fusion (Weighted + RRF) | ✅ |
| Reranking | NoOp / KeywordOverlap | ✅ |
| Generation | Mock LLM answer generation | ✅ |
| Evaluation | Recall@k, MRR, nDCG | ✅ |
| API | FastAPI REST endpoints | ✅ |
| DevOps | Docker + docker-compose | ✅ |
| Tests | pytest for all modules | ✅ |

---

## Tech Stack

- **Python 3.10+**
- **FastAPI** — REST API framework
- **sentence-transformers** — Dense embeddings (`all-MiniLM-L6-v2` by default)
- **FAISS** — Local vector index (CPU)
- **rank-bm25** — Lexical retrieval
- **pytest** — Unit testing
- **Docker** — Containerization

---

## Quick Start

### Local Development

```bash
# Clone and setup
pip install -e ".[dev]"

# Run tests
pytest

# Run CLI demo
rag-lab hello

# Run FastAPI server
uvicorn rag_lab.api.app:app --reload
```

### Docker

```bash
docker compose up
```

The API will be available at `http://localhost:8000`.

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

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
  -d '{"query": "RAG retrieval", "top_k": 5, "retriever_type": "hybrid"}'
```

### Answer (RAG Pipeline)

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid retrieval?", "top_k": 5, "retriever_type": "hybrid"}'
```

---

## Evaluation Methodology

We use a small benchmark dataset (`examples/eval_dataset.json`) to compare retrieval strategies across three metrics:

- **Recall@k**: Proportion of relevant chunks retrieved in top-k
- **MRR@k**: Mean Reciprocal Rank of the first relevant chunk
- **nDCG@k**: Normalized Discounted Cumulative Gain (accounts for ranking position)

### RAG Strategy Leaderboard

Run the full leaderboard evaluation:

```bash
python -m examples.run_leaderboard
```

| Rank | Strategy | Recall@1 | Recall@5 | MRR@5 | nDCG@5 |
|------|---|---|---|---|---|
| 🥇 | Vector | 0.80 | 0.90 | 1.00 | 0.91 |
| 🥈 | Hybrid + Keyword Rerank | 0.70 | 0.90 | 0.84 | 0.85 |
| 🥉 | Vector + Keyword Rerank | 0.70 | 0.90 | 0.85 | 0.84 |
| 4 | Hybrid | 0.60 | 0.90 | 0.75 | 0.79 |
| 5 | BM25 | 0.60 | 0.80 | 0.72 | 0.71 |
| 6 | BM25 + Keyword Rerank | 0.60 | 0.80 | 0.71 | 0.71 |

> **Note:** Results from the built-in leaderboard benchmark (`examples/run_leaderboard.py`) on the fixture dataset. Rerank strategies show different trade-offs across metrics. Your numbers will vary with different documents and embedding models.

---

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
├── src/
│   └── rag_lab/
│       ├── models.py          # Core data models
│       ├── cli.py             # CLI entry point
│       ├── ingestion/         # Document parsing
│       ├── chunking/          # Text chunking
│       ├── retrieval/         # BM25, Vector, Hybrid
│       ├── reranking/         # Rerankers
│       ├── generation/        # Answer generation (placeholder)
│       ├── evaluation/        # Metrics & benchmark
│       └── api/               # FastAPI app
├── tests/
│   ├── fixtures/
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_bm25.py
│   ├── test_vector.py
│   ├── test_hybrid.py
│   ├── test_reranking.py
│   ├── test_evaluation.py
│   └── test_api.py
└── docs/
    ├── architecture.md
    └── design-decisions.md
```

---

## Roadmap

- [x] v1.0: Document ingestion, chunking, BM25, vector, hybrid, reranking, evaluation, FastAPI, Docker
- [x] v1.1: Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker`)
- [x] v1.2: Streamlit UI for interactive demos (Phase 1)
- [x] v1.3: PDF ingestion with layout-aware chunking (Phase 2)
- [x] v1.4: Leaderboard evaluation with standard benchmarks (Phase 3)
- [ ] v1.5: OpenAI-compatible LLM generation endpoint

---

## What This Project Demonstrates

This project is designed to show recruiters and hiring managers that I can:

1. **Build production-grade ML pipelines** — not just Jupyter notebooks
2. **Design clean, modular architectures** — each layer is swappable and tested
3. **Implement and evaluate retrieval algorithms** — BM25, dense vectors, hybrid fusion, RRF
4. **Write production API code** — FastAPI with Pydantic models and error handling
5. **Engineer for deployment** — Docker, lazy loading, environment-aware configuration
6. **Test systematically** — pytest coverage across all core modules

---

## License

MIT
