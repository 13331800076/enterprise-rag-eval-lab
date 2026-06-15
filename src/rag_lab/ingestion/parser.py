"""File parsers for ingestion."""

from pathlib import Path
from typing import Protocol

from rag_lab.models import Document


class Parser(Protocol):
    """Protocol for document parsers."""

    def parse(self, path: Path) -> Document: ...


class TxtParser:
    """Parser for plain text files."""

    def parse(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        return Document(
            doc_id=path.stem,
            source_path=str(path),
            title=path.stem,
            text=text,
            metadata={"format": "txt", "size": len(text)},
        )


class MarkdownParser:
    """Parser for Markdown files."""

    def parse(self, path: Path) -> Document:
        import markdown
        from bs4 import BeautifulSoup

        raw_text = path.read_text(encoding="utf-8")
        # Convert markdown to HTML, then strip tags to get plain text while preserving headings
        html = markdown.markdown(raw_text)
        soup = BeautifulSoup(html, "html.parser")
        # Get text with newlines preserved between block elements
        text = soup.get_text(separator="\n", strip=True)

        # Extract title from first heading if available
        title = path.stem
        first_heading = soup.find(["h1", "h2", "h3"])
        if first_heading:
            title = first_heading.get_text(strip=True)

        return Document(
            doc_id=path.stem,
            source_path=str(path),
            title=title,
            text=text,
            metadata={"format": "markdown", "size": len(text), "headings": self._extract_headings(soup)},
        )

    def _extract_headings(self, soup) -> list[str]:
        return [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])]


class HtmlParser:
    """Parser for HTML files."""

    def parse(self, path: Path) -> Document:
        from bs4 import BeautifulSoup

        raw_html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Extract title
        title = path.stem
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Get readable text
        text = soup.get_text(separator="\n", strip=True)

        return Document(
            doc_id=path.stem,
            source_path=str(path),
            title=title,
            text=text,
            metadata={"format": "html", "size": len(text)},
        )


PARSERS: dict[str, Parser] = {
    ".txt": TxtParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".html": HtmlParser(),
    ".htm": HtmlParser(),
}


def parse_file(path: Path) -> Document:
    """Parse a file based on its extension."""
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return parser.parse(path)
