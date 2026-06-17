"""Tests for document ingestion."""

from pathlib import Path

import pytest

from rag_lab.ingestion.ingestion import IngestionPipeline
from rag_lab.ingestion.parser import parse_file
from rag_lab.models import Document

FIXTURES = Path(__file__).parent / "fixtures"


class TestIngestionPipeline:
    def test_ingest_txt(self):
        pipeline = IngestionPipeline()
        doc = pipeline.ingest_file(FIXTURES / "bm25_guide.txt")
        assert isinstance(doc, Document)
        assert doc.title == "bm25_guide"
        assert "BM25" in doc.text

    def test_ingest_markdown(self):
        pipeline = IngestionPipeline()
        doc = pipeline.ingest_file(FIXTURES / "rag_overview.md")
        assert isinstance(doc, Document)
        assert doc.title == "RAG Overview"
        assert "RAG" in doc.text
        assert "Evaluation" in doc.text
        assert doc.metadata["format"] == "markdown"
        assert len(doc.metadata["headings"]) > 0

    def test_ingest_html(self):
        pipeline = IngestionPipeline()
        doc = pipeline.ingest_file(FIXTURES / "vector_search.html")
        assert isinstance(doc, Document)
        assert doc.title == "Vector Search Guide"
        assert "script" not in doc.text.lower() or "hidden" not in doc.text.lower()
        assert "Vector search" in doc.text
        assert doc.metadata["format"] == "html"

    def test_ingest_directory(self):
        pipeline = IngestionPipeline()
        docs = pipeline.ingest_directory(FIXTURES, pattern="*")
        assert len(docs) == 4  # md, txt, html, pdf
        formats = {d.metadata["format"] for d in docs}
        assert formats == {"txt", "markdown", "html", "pdf"}

    def test_file_not_found(self):
        pipeline = IngestionPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.ingest_file("nonexistent.txt")

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_file(Path("sample.docx"))
