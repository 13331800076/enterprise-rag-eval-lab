"""Tests for FastAPI endpoints."""
from fastapi.testclient import TestClient

from rag_lab.api.app import app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestSearchWithoutIngest:
    def test_search_before_ingest(self):
        response = client.post("/search", json={"query": "test", "top_k": 3, "retriever_type": "bm25"})
        assert response.status_code == 400
        assert "No documents indexed" in response.json()["detail"]

    def test_answer_before_ingest(self):
        response = client.post("/answer", json={"query": "test", "top_k": 3, "retriever_type": "bm25"})
        assert response.status_code == 400


class TestIngestAndSearch:
    def test_ingest_and_search(self):
        # Ingest the fixtures directory
        response = client.post("/ingest", json={"path": "tests/fixtures"})
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] > 0
        assert data["chunks"] > 0

        # Search BM25 only (avoids loading sentence-transformers model in tests)
        response = client.post("/search", json={"query": "RAG", "top_k": 3, "retriever_type": "bm25"})
        assert response.status_code == 200
        bm25_results = response.json()["results"]
        assert len(bm25_results) <= 3

        # Answer endpoint with BM25
        response = client.post("/answer", json={"query": "RAG", "top_k": 3, "retriever_type": "bm25"})
        assert response.status_code == 200
        answer_data = response.json()
        assert "answer" in answer_data
        assert "evidence" in answer_data
        assert len(answer_data["evidence"]) <= 3

    def test_ingest_bad_path(self):
        response = client.post("/ingest", json={"path": "/nonexistent/path"})
        assert response.status_code == 400
