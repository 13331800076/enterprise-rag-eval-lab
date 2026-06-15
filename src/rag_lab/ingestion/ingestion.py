"""Document ingestion module."""

from pathlib import Path
from typing import Any

from rag_lab.models import Document

from .parser import parse_file


class IngestionPipeline:
    """Pipeline for ingesting documents from various formats."""

    def ingest_file(self, file_path: str | Path) -> Document:
        """Ingest a single file and return a Document."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return parse_file(path)

    def ingest_directory(self, directory: str | Path, pattern: str = "*") -> list[Document]:
        """Ingest all matching files from a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        documents = []
        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file():
                try:
                    documents.append(self.ingest_file(file_path))
                except Exception as e:
                    # Log and skip unsupported or broken files
                    print(f"Skipping {file_path}: {e}")
        return documents
