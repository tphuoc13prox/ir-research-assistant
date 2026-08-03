import logging

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
        try:
            return self.provider.search(query, limit=limit)
        except Exception as exc:
            logger.warning(
                f"Active provider search failed ({exc}). Falling back to OpenAlex provider..."
            )
            try:
                return self.fallback_provider.search(query, limit=limit)
            except Exception as fallback_exc:
                logger.error(f"Fallback provider query also failed: {fallback_exc}")
                raise RuntimeError(
                    f"Search failed. Primary error: {exc}. Fallback error: {fallback_exc}"
                ) from fallback_exc
