"""Tests for document chunking."""

import pytest

from rag_lab.chunking.chunker import Chunk, FixedSizeChunker, HeadingBasedChunker
from rag_lab.models import Document


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        doc = Document(doc_id="doc1", source_path="", title="", text="Hello world! This is a sample document for testing chunking.")
        chunker = FixedSizeChunker(chunk_size=20, overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.doc_id == "doc1" for c in chunks)

    def test_overlap(self):
        doc = Document(doc_id="doc2", source_path="", title="", text="abcdefghij" * 10)
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        # Overlap means next chunk starts before previous ends
        assert chunks[1].start_char < chunks[0].end_char

    def test_boundary_metadata(self):
        doc = Document(doc_id="doc3", source_path="", title="", text="short")
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len("short")
        assert chunks[0].text == "short"

    def test_overlap_too_large_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=10)


class TestHeadingBasedChunker:
    def test_heading_chunks(self):
        text = "# Introduction\nThis is the intro.\n\n# Main Content\nThis is the main part.\n\n# Conclusion\nThis is the end."
        doc = Document(doc_id="doc4", source_path="", title="", text=text)
        chunker = HeadingBasedChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3
        assert "# Introduction" in chunks[0].text
        assert chunks[0].metadata["heading"] == "# Introduction"
        assert chunks[1].metadata["heading"] == "# Main Content"

    def test_fallback_when_no_headings(self):
        doc = Document(doc_id="doc5", source_path="", title="", text="Just plain text without any headings at all.")
        chunker = HeadingBasedChunker(max_chunk_size=20)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        # Should fallback to fixed-size chunking
        assert chunks[0].metadata["strategy"] == "fixed_size"

    def test_subheading_support(self):
        text = "# Title\nContent\n\n## Sub A\nMore content\n\n## Sub B\nEven more"
        doc = Document(doc_id="doc6", source_path="", title="", text=text)
        chunker = HeadingBasedChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3
        headings = [c.metadata["heading"] for c in chunks]
        assert "# Title" in headings
        assert "## Sub A" in headings
        assert "## Sub B" in headings
