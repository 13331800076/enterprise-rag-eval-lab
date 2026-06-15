"""FastAPI application for RAG Lab."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

from rag_lab.ingestion.ingestion import IngestionPipeline
from rag_lab.chunking.chunker import FixedSizeChunker, HeadingBasedChunker
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.vector import VectorRetriever
from rag_lab.retrieval.hybrid import HybridRetriever
from rag_lab.reranking.reranker import NoOpReranker, KeywordOverlapReranker

app = FastAPI(title="RAG Lab API", version="0.1.0")

# In-memory state (lazy-initialized to avoid loading models at import time)
pipeline = IngestionPipeline()
chunks = []
bm25: BM25Retriever | None = None
vector: VectorRetriever | None = None
hybrid: HybridRetriever | None = None

# Default chunker and reranker
chunker = FixedSizeChunker(chunk_size=500, overlap=50)
reranker = NoOpReranker()


class IngestRequest(BaseModel):
    path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    retriever_type: Literal["bm25", "vector", "hybrid"] = "hybrid"


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5
    retriever_type: Literal["bm25", "vector", "hybrid"] = "hybrid"


class SearchResponse(BaseModel):
    results: list[dict]


class AnswerResponse(BaseModel):
    query: str
    evidence: list[dict]
    answer: str


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest")
async def ingest(request: IngestRequest):
    global chunks, bm25
    try:
        documents = pipeline.ingest_directory(request.path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    bm25 = BM25Retriever()
    bm25.index(chunks)

    return {"documents": len(documents), "chunks": len(chunks)}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    global vector, hybrid
    if not chunks or bm25 is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Call /ingest first.")

    if request.retriever_type == "bm25":
        retriever = bm25
    elif request.retriever_type == "vector":
        if vector is None:
            vector = VectorRetriever()
            vector.index(chunks)
        retriever = vector
    else:
        if hybrid is None:
            if vector is None:
                vector = VectorRetriever()
                vector.index(chunks)
            hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)
            hybrid.index(chunks)
        retriever = hybrid

    results = retriever.search(request.query, top_k=request.top_k)
    reranked = reranker.rerank(request.query, results)

    return {"results": [
        {
            "chunk_id": r.chunk_id,
            "text": r.text,
            "score": r.score,
            "metadata": r.metadata,
        }
        for r in reranked
    ]}


@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest):
    global vector, hybrid
    if not chunks or bm25 is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Call /ingest first.")

    if request.retriever_type == "bm25":
        retriever = bm25
    elif request.retriever_type == "vector":
        if vector is None:
            vector = VectorRetriever()
            vector.index(chunks)
        retriever = vector
    else:
        if hybrid is None:
            if vector is None:
                vector = VectorRetriever()
                vector.index(chunks)
            hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)
            hybrid.index(chunks)
        retriever = hybrid

    results = retriever.search(request.query, top_k=request.top_k)
    reranked = reranker.rerank(request.query, results)

    # Mock LLM answer generation
    evidence_texts = [r.text for r in reranked]
    if evidence_texts:
        answer = f"Based on the retrieved evidence, here is the answer: {evidence_texts[0][:200]}..."
    else:
        answer = "No relevant evidence found."

    return {
        "query": request.query,
        "evidence": [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in reranked
        ],
        "answer": answer,
    }
