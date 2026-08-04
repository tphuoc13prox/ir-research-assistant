from abc import ABC, abstractmethod
from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult

class BaseRanker(ABC):
    @abstractmethod
    def rank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Rank or adjust scores for a list of retrieval results."""
        pass
