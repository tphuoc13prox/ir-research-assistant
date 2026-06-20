from src.crawler.arxiv_client import ArxivClient
from src.domain.paper import Paper


class PaperFinder:
    def __init__(self, client: ArxivClient | None = None) -> None:
        self.client = client or ArxivClient()

    def find(self, query: str, max_results: int = 10) -> list[Paper]:
        return self.client.search_ir_papers(query, max_results=max_results)
