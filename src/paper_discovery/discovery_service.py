import logging
import re

from src.paper_discovery.interfaces.paper_provider import PaperProvider
from src.paper_discovery.models import DiscoveredPaper
from src.paper_discovery.providers.arxiv_provider import ArxivProvider
from src.paper_discovery.providers.openalex_provider import OpenAlexProvider

logger = logging.getLogger(__name__)


class DiscoveryService:
    def __init__(self, provider: PaperProvider | None = None) -> None:
        self.provider = provider or ArxivProvider()
        self.fallback_provider = OpenAlexProvider()

    def search_papers(self, query: str, limit: int = 10) -> list[DiscoveredPaper]:
        cleaned_query, year_filter = self._parse_year_constraint(query)
        cleaned_query = self._clean_academic_query(cleaned_query)
        
        has_constraint = (cleaned_query != query)
        fetch_limit = max(50, limit * 3) if has_constraint else limit

        try:
            results = self.provider.search(cleaned_query, limit=fetch_limit)
        except Exception as exc:
            logger.warning(
                f"Active provider search failed ({exc}). Falling back to OpenAlex provider..."
            )
            try:
                results = self.fallback_provider.search(cleaned_query, limit=fetch_limit)
            except Exception as fallback_exc:
                logger.error(f"Fallback provider query also failed: {fallback_exc}")
                raise RuntimeError(
                    f"Search failed. Primary error: {exc}. Fallback error: {fallback_exc}"
                ) from fallback_exc

        filtered_results = [r for r in results if year_filter(r.published_year)]
        return filtered_results[:limit]

    @staticmethod
    def _clean_academic_query(query: str) -> str:
        cleaned = query.lower().strip()
        noise = [
            "articles", "article", "papers", "paper", "studies", "study", 
            "research", "methods", "method", "algorithms", "algorithm",
            "published", "publish", "written", "authored", "author", "authors"
        ]
        for term in noise:
            cleaned = re.sub(rf'\b{term}\b', '', cleaned)
        return " ".join(cleaned.split())

    @staticmethod
    def _parse_year_constraint(query: str) -> tuple[str, callable]:
        cleaned = query
        filter_fn = lambda year: True

        # Match "after YYYY" or "published after YYYY"
        m = re.search(r'\b(?:published\s+)?after\s+(\d{4})\b', query, re.IGNORECASE)
        if m:
            target_year = int(m.group(1))
            filter_fn = lambda year: year != "N/A" and year.isdigit() and int(year) > target_year
            cleaned = re.sub(r'\b(?:published\s+)?after\s+\d{4}\b', '', cleaned, flags=re.IGNORECASE)
            return " ".join(cleaned.split()), filter_fn

        # Match "since YYYY", "published since YYYY", "from YYYY"
        m = re.search(r'\b(?:published\s+)?since\s+(\d{4})\b', query, re.IGNORECASE) or re.search(r'\bfrom\s+(\d{4})\b', query, re.IGNORECASE)
        if m:
            target_year = int(m.group(1))
            filter_fn = lambda year: year != "N/A" and year.isdigit() and int(year) >= target_year
            cleaned = re.sub(r'\b(?:published\s+)?since\s+\d{4}\b', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\bfrom\s+\d{4}\b', '', cleaned, flags=re.IGNORECASE)
            return " ".join(cleaned.split()), filter_fn

        # Match "before YYYY", "published before YYYY"
        m = re.search(r'\b(?:published\s+)?before\s+(\d{4})\b', query, re.IGNORECASE)
        if m:
            target_year = int(m.group(1))
            filter_fn = lambda year: year != "N/A" and year.isdigit() and int(year) < target_year
            cleaned = re.sub(r'\b(?:published\s+)?before\s+\d{4}\b', '', cleaned, flags=re.IGNORECASE)
            return " ".join(cleaned.split()), filter_fn

        # Match "in YYYY", "published in YYYY"
        m = re.search(r'\b(?:published\s+)?in\s+(\d{4})\b', query, re.IGNORECASE)
        if m:
            target_year = int(m.group(1))
            filter_fn = lambda year: year != "N/A" and year.isdigit() and int(year) == target_year
            cleaned = re.sub(r'\b(?:published\s+)?in\s+\d{4}\b', '', cleaned, flags=re.IGNORECASE)
            return " ".join(cleaned.split()), filter_fn

        return cleaned, filter_fn
