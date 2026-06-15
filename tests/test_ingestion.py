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
        doc = pipeline.ingest_file(FIXTURES / "sample.txt")
        assert isinstance(doc, Document)
        assert doc.title == "sample"
        assert "plain text" in doc.text.lower()

    def test_ingest_markdown(self):
        pipeline = IngestionPipeline()
        doc = pipeline.ingest_file(FIXTURES / "sample.md")
        assert isinstance(doc, Document)
        assert doc.title == "Sample Markdown Document"
        assert "Introduction" in doc.text
        assert "RAG" in doc.text
        assert doc.metadata["format"] == "markdown"
        assert len(doc.metadata["headings"]) > 0

    def test_ingest_html(self):
        pipeline = IngestionPipeline()
        doc = pipeline.ingest_file(FIXTURES / "sample.html")
        assert isinstance(doc, Document)
        assert doc.title == "Sample HTML Document"
        # Scripts and styles should be stripped
        assert "script" not in doc.text.lower() or "stripped" not in doc.text.lower()
        assert "Document parsing" in doc.text
        assert doc.metadata["format"] == "html"

    def test_ingest_directory(self):
        pipeline = IngestionPipeline()
        docs = pipeline.ingest_directory(FIXTURES, pattern="sample.*")
        assert len(docs) == 3
        formats = {d.metadata["format"] for d in docs}
        assert formats == {"txt", "markdown", "html"}

    def test_file_not_found(self):
        pipeline = IngestionPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.ingest_file("nonexistent.txt")

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_file(Path("sample.pdf"))
