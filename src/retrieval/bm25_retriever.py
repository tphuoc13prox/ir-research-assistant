from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult


class BM25Retriever:
    def __init__(self, chunks: list[RetrievalResult] | dict[str, RetrievalResult]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for BM25Retriever. Install project dependencies "
                "with `pip install -r requirements.txt`."
            ) from exc

        self.chunks = list(chunks.values()) if isinstance(chunks, dict) else chunks
        self._tokenized = [self._tokenize(chunk.text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def retrieve(self, query: Query) -> list[RetrievalResult]:
        if self._bm25 is None or query.top_k <= 0:
            return []

        scores = self._bm25.get_scores(self._tokenize(query.text))
        ranked_positions = sorted(
            range(len(scores)),
            key=lambda position: scores[position],
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for position in ranked_positions:
            chunk = self.chunks[position]
            if query.filters and not all(
                chunk.metadata.get(key) == value for key, value in query.filters.items()
            ):
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=float(scores[position]),
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
            if len(results) >= query.top_k:
                break
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
