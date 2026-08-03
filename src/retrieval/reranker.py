from abc import ABC, abstractmethod
from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult


class BaseReRanker(ABC):
    @abstractmethod
    def rerank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        pass


class IdentityReRanker(BaseReRanker):
    def rerank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results


class Reranker(IdentityReRanker):
    """Legacy Reranker class maintaining 100% backward compatibility."""
    pass
