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

### Benchmark Results

Run the evaluation:

```bash
python -m examples.run_evaluation
```

| Retriever | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| BM25 | 0.70 | 0.37 | 0.43 |
| Vector | 0.90 | 0.67 | 0.69 |
| Hybrid | 0.80 | 0.62 | 0.60 |

> **Note:** Results generated on the included fixture dataset using `sentence-transformers/all-MiniLM-L6-v2`. Your numbers will vary with different documents and embedding models.

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
- [ ] v1.2: OpenAI-compatible LLM generation endpoint
- [ ] v1.3: Milvus / Neo4j vector store backends
- [ ] v1.4: Streamlit UI for interactive demos
- [ ] v1.5: PDF ingestion with layout-aware chunking

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
