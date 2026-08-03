import httpx
from src.paper_discovery.interfaces.paper_provider import PaperProvider
from src.paper_discovery.models import DiscoveredPaper


class OpenAlexProvider(PaperProvider):
    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 10) -> list[DiscoveredPaper]:
        url = "https://api.openalex.org/works"
        params = {
            "filter": "locations.source.id:S4306400194",  # Source ID for arXiv in OpenAlex
            "search": query,
            "per_page": limit,
            "mailto": "research@example.com"
        }
        
        try:
            resp = httpx.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"OpenAlex query failed: {exc}") from exc

        discovered = []
        for item in data.get("results", []):
            arxiv_id = self._extract_arxiv_id(item)
            if not arxiv_id:
                continue
            
            title = item.get("title") or "Untitled Paper"
            
            authorships = item.get("authorships", [])
            authors = [a.get("author", {}).get("display_name", "") for a in authorships]
            authors = [name for name in authors if name]
            if not authors:
                authors = ["Unknown Author"]

            inv_index = item.get("abstract_inverted_index")
            if inv_index:
                words = {}
                for word, positions in inv_index.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words[pos] for pos in sorted(words.keys()))
            else:
                abstract = "No abstract available."

            published_year = str(item.get("publication_year") or "N/A")
            
            concepts = item.get("concepts", [])
            categories = [c.get("display_name", "") for c in concepts[:3]]
            categories = [name for name in categories if name]
            if not categories:
                categories = ["cs.IR"]

            # Construct clean, direct PDF download URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            discovered.append(
                DiscoveredPaper(
                    paper_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    pdf_url=pdf_url,
                    published_year=published_year,
                    categories=categories,
                    doi=item.get("doi"),
                )
            )
        return discovered

    @staticmethod
    def _extract_arxiv_id(item: dict) -> str | None:
        # 1. Check ids.arxiv
        arxiv_url = item.get("ids", {}).get("arxiv")
        if arxiv_url:
            return arxiv_url.split("/abs/")[-1].split("/pdf/")[-1].strip()

        # 2. Iterate locations for landing page or pdf containing arxiv.org
        for loc in item.get("locations", []):
            for key in ["landing_page_url", "pdf_url"]:
                url = loc.get(key)
                if url and "arxiv.org" in url:
                    parts = url.split("/abs/")[-1].split("/pdf/")[-1].strip()
                    if parts.endswith(".pdf"):
                        parts = parts[:-4]
                    return parts

        # 3. Check doi containing 'arxiv'
        doi = item.get("doi")
        if doi and "arxiv." in doi:
            return doi.split("arxiv.")[-1].strip()

        return None
