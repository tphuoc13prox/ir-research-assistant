from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        *,
        dense_weight: float = 0.7,
    ) -> None:
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0.0 and 1.0")
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight

    def retrieve(self, query: Query) -> list[RetrievalResult]:
        dense_results = self.dense_retriever.retrieve(query)
        sparse_results = self.sparse_retriever.retrieve(query)
        dense_scores = self._normalize(dense_results)
        sparse_scores = self._normalize(sparse_results)

        merged: dict[str, RetrievalResult] = {}
        for result in [*dense_results, *sparse_results]:
            merged.setdefault(result.chunk_id, result)

        scored: list[RetrievalResult] = []
        for chunk_id, result in merged.items():
            score = (
                self.dense_weight * dense_scores.get(chunk_id, 0.0)
                + self.sparse_weight * sparse_scores.get(chunk_id, 0.0)
            )
            scored.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    paper_id=result.paper_id,
                    score=score,
                    text=result.text,
                    metadata=result.metadata,
                )
            )

        return sorted(scored, key=lambda result: result.score, reverse=True)[: query.top_k]

    @staticmethod
    def _normalize(results: list[RetrievalResult]) -> dict[str, float]:
        if not results:
            return {}
        scores = [result.score for result in results]
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return {result.chunk_id: 1.0 for result in results}
        return {
            result.chunk_id: (result.score - minimum) / (maximum - minimum)
            for result in results
        }
