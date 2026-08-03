from typing import Any

from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult
from src.retrieval.interfaces.base_retriever import BaseRetriever
from src.retrieval.cache import RetrievalCache


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        chunks: list[RetrievalResult] | dict[str, RetrievalResult],
        bm25_instance: Any | None = None,
        cache: RetrievalCache | None = None,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for BM25Retriever. Install project dependencies "
                "with `pip install -r requirements.txt`."
            ) from exc

        self.chunks_dict = chunks if isinstance(chunks, dict) else {c.chunk_id: c for c in chunks}
        self.chunks_list = list(self.chunks_dict.values())
        self.cache = cache or RetrievalCache()

        if bm25_instance is not None:
            self._bm25 = bm25_instance
        else:
            self._tokenized = [self._tokenize(chunk.text) for chunk in self.chunks_list]
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def retrieve(self, query: Query) -> list[RetrievalResult]:
        cached = self.cache.get(
            query.text,
            query.top_k,
            query.filters,
            query.paper_id,
            query.section,
        )
        if cached is not None:
            return cached

        if self._bm25 is None or query.top_k <= 0:
            return []

        has_filters = bool(query.filters or query.paper_id or query.section)
        search_k = max(100, query.top_k * 5) if has_filters else query.top_k

        scores = self._bm25.get_scores(self._tokenize(query.text))
        ranked_positions = sorted(
            range(len(scores)),
            key=lambda position: scores[position],
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for position in ranked_positions:
            chunk = self.chunks_list[position]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=float(scores[position]),
                    text=chunk.text,
                    page=chunk.page,
                    section=chunk.section,
                    metadata=chunk.metadata,
                )
            )

        filtered = self._apply_filters(results, query)
        final_results = filtered[:query.top_k]

        self.cache.set(
            query.text,
            query.top_k,
            query.filters,
            query.paper_id,
            query.section,
            final_results,
        )
        return final_results

    @staticmethod
    def _apply_filters(results: list[RetrievalResult], query: Query) -> list[RetrievalResult]:
        filtered = results
        
        if query.paper_id:
            filtered = [r for r in filtered if r.paper_id == query.paper_id]
            
        if query.section:
            filtered = [r for r in filtered if r.section == query.section or r.metadata.get("section") == query.section]
            
        year_filter = query.filters.get("year")
        if year_filter:
            filtered = [
                r for r in filtered
                if str(r.metadata.get("year")) == str(year_filter) or
                (r.metadata.get("published_at") and str(r.metadata.get("published_at")).startswith(str(year_filter)))
            ]

        for key, value in query.filters.items():
            if key == "year":
                continue
            filtered = [r for r in filtered if r.metadata.get(key) == value]
            
        return filtered

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
