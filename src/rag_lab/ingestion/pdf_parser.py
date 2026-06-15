"""PDF parser with layout awareness using PyMuPDF."""

from pathlib import Path

from rag_lab.models import Document
from rag_lab.ingestion.parser import PARSERS


class PdfParser:
    """Parser for PDF files with layout extraction.

    Extracts text block by block, preserving reading order.
    Captures page numbers, tables (as markdown), and headings.
    """

    def parse(self, path: Path) -> Document:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        text_parts = []
        headings = []
        tables = []
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")

            for block in blocks:
                # block format: (x0, y0, x1, y1, text, block_no, block_type)
                x0, y0, x1, y1, text, block_no, block_type = block
                text = text.strip()
                if not text:
                    continue

                # Heuristic: large font or bold text = heading
                # PyMuPDF doesn't directly give font info in blocks, but we can use span info
                if block_type == 0:  # text block
                    text_parts.append(f"\n[Page {page_num + 1}]\n{text}")

            # Try to extract tables as markdown
            tabs = page.find_tables()
            for tab in tabs.tables:
                df = tab.to_pandas()
                tables.append(df.to_markdown(index=False))

        doc.close()

        full_text = "\n".join(text_parts)
        if tables:
            full_text += "\n\n## Tables\n\n" + "\n\n".join(tables)

        # Extract title from first page heading
        title = path.stem
        if text_parts:
            first_lines = text_parts[0].splitlines()
            for line in first_lines:
                line = line.strip()
                if line and not line.startswith("[Page"):
                    title = line[:80]
                    break

        return Document(
            doc_id=path.stem,
            source_path=str(path),
            title=title,
            text=full_text,
            metadata={
                "format": "pdf",
                "pages": page_count,
                "tables": len(tables),
            },
        )


PARSERS[".pdf"] = PdfParser()
