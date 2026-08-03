from dataclasses import dataclass


@dataclass(slots=True)
class DiscoveredPaper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    published_year: str
    categories: list[str]
    doi: str | None = None
