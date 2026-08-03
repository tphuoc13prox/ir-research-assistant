from src.domain.retrieval_result import RetrievalResult
from src.retrieval.interfaces.fusion_strategy import FusionStrategy


class ScoreNormalizationFusion(FusionStrategy):
    def __init__(self, dense_weight: float = 0.7) -> None:
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
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
                    page=result.page,
                    section=result.section,
                    metadata=result.metadata,
                )
            )

        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

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
