from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult


class Reranker:
    def rerank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return results
