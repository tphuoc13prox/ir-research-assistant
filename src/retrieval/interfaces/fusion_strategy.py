from abc import ABC, abstractmethod
from src.domain.retrieval_result import RetrievalResult
from src.domain.query import Query


class FusionStrategy(ABC):
    @abstractmethod
    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        query: Query | None = None,
    ) -> list[RetrievalResult]:
        """Fuse dense and sparse retrieval results into a single ranked list."""
        pass
