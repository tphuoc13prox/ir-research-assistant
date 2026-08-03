from abc import ABC, abstractmethod
from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: Query) -> list[RetrievalResult]:
        """Retrieve relevant passages for a given query."""
        pass
