"""Core data models for RAG Lab."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Represents an ingested document."""
    doc_id: str
    source_path: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
