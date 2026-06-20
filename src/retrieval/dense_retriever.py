from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult
from src.embeddings.embedder import Embedder
from src.indexing.faiss_index import FaissIndex


class DenseRetriever:
    def __init__(
        self,
        embedder: Embedder,
        index: FaissIndex,
        chunks: dict[str, RetrievalResult] | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks or {}

    def retrieve(self, query: Query) -> list[RetrievalResult]:
        if hasattr(self.embedder, "encode_query"):
            query_vector = self.embedder.encode_query(query.text)  # type: ignore[attr-defined]
        else:
            query_vector = self.embedder.encode([query.text])[0]

        matches = self.index.search(query_vector, query.top_k)
        results: list[RetrievalResult] = []
        for chunk_id, score in matches:
            stored = self.chunks.get(chunk_id)
            if stored is None:
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        paper_id="",
                        score=score,
                        text="",
                    )
                )
                continue

            results.append(
                RetrievalResult(
                    chunk_id=stored.chunk_id,
                    paper_id=stored.paper_id,
                    score=score,
                    text=stored.text,
                    metadata=stored.metadata,
                )
            )
        return self._apply_filters(results, query.filters)

    @staticmethod
    def _apply_filters(
        results: list[RetrievalResult],
        filters: dict[str, str],
    ) -> list[RetrievalResult]:
        if not filters:
            return results
        return [
            result
            for result in results
            if all(result.metadata.get(key) == value for key, value in filters.items())
        ]
