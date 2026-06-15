"""Tests for PDF parsing."""
from pathlib import Path

import pytest

from rag_lab.ingestion.ingestion import IngestionPipeline
from rag_lab.models import Document

FIXTURES = Path(__file__).parent / "fixtures"


class TestPdfIngestion:
    def test_ingest_pdf(self):
        pipeline = IngestionPipeline()
        doc = pipeline.ingest_file(FIXTURES / "sample.pdf")
        assert isinstance(doc, Document)
        assert doc.title == "RAG Technology Overview"
        assert "Retrieval-Augmented Generation" in doc.text
        assert doc.metadata["format"] == "pdf"
        assert doc.metadata["pages"] == 2

    def test_pdf_page_markers(self):
        pipeline = IngestionPipeline()
        doc = pipeline.ingest_file(FIXTURES / "sample.pdf")
        assert "[Page 1]" in doc.text
        assert "[Page 2]" in doc.text

    def test_unsupported_format(self):
        with pytest.raises(FileNotFoundError):
            pipeline = IngestionPipeline()
            pipeline.ingest_file("nonexistent.pdf")
