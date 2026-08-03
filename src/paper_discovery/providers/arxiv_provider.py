from src.paper_discovery.interfaces.paper_provider import PaperProvider
from src.paper_discovery.models import DiscoveredPaper
from src.crawler.arxiv_client import ArxivClient


class ArxivProvider(PaperProvider):
    def __init__(self, timeout: float = 30.0) -> None:
        self.client = ArxivClient(timeout=timeout)

    def search(self, query: str, limit: int = 10) -> list[DiscoveredPaper]:
        papers = self.client.search_ir_papers(query, max_results=limit)
        
        discovered = []
        for p in papers:
            year = "N/A"
            if p.published_at:
                year = p.published_at.split("-")[0]
                
            pdf_url = p.metadata.get("pdf_url") or ""
            categories = p.metadata.get("categories", "").split(",")
            categories = [c.strip() for c in categories if c.strip()]
            doi = p.metadata.get("doi")

            discovered.append(
                DiscoveredPaper(
                    paper_id=p.paper_id,
                    title=p.title,
                    authors=p.authors,
                    abstract=p.abstract,
                    pdf_url=pdf_url,
                    published_year=year,
                    categories=categories,
                    doi=doi,
                )
            )
        return discovered
