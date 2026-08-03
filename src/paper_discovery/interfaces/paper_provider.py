from abc import ABC, abstractmethod
from src.paper_discovery.models import DiscoveredPaper


class PaperProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[DiscoveredPaper]:
        """Search papers on the provider repository by query."""
        pass
