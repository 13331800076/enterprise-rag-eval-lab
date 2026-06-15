"""Document chunking strategies."""

from dataclasses import dataclass, field
from typing import Any

from rag_lab.models import Document


@dataclass
class Chunk:
    """Represents a text chunk."""
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker:
    """Base interface for chunkers."""

    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    """Chunk text into fixed-size blocks with optional overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        chunks = []
        step = self.chunk_size - self.overlap
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{idx}",
                    doc_id=document.doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={"strategy": "fixed_size", "chunk_size": self.chunk_size, "overlap": self.overlap},
                )
            )
            start += step
            idx += 1
            # Prevent infinite loop on final small chunk
            if start >= len(text):
                break
        return chunks


class HeadingBasedChunker(BaseChunker):
    """Chunk Markdown-style text by headings."""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        import re

        text = document.text
        # Match heading lines (e.g., # Heading, ## Subheading)
        heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            # Fallback to fixed-size chunking if no headings found
            fallback = FixedSizeChunker(chunk_size=self.max_chunk_size, overlap=0)
            return fallback.chunk(document)

        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            heading = match.group(1).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{i}",
                    doc_id=document.doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "strategy": "heading_based",
                        "heading": heading,
                        "max_chunk_size": self.max_chunk_size,
                    },
                )
            )
        return chunks
