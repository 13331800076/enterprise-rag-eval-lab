"""Basic tests to ensure pytest can run."""
import pytest


def test_pytest_runs():
    assert True


def test_import_rag_lab():
    import rag_lab
    assert rag_lab.__version__ == "0.1.0"


def test_import_cli():
    from rag_lab.cli import main
    assert main is not None


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from rag_lab.api.app import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
