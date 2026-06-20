from pathlib import Path

from pypdf import PdfReader


class PdfLoader:
    def load_text(self, path: Path) -> str:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(page for page in pages if page.strip())
