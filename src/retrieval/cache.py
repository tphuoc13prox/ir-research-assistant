import threading
from src.domain.retrieval_result import RetrievalResult


class RetrievalCache:
    def __init__(self, max_size: int = 1000) -> None:
        self._cache: dict[tuple, list[RetrievalResult]] = {}
        self._lock = threading.Lock()
        self.max_size = max_size
        self._keys_fifo: list[tuple] = []

    def get(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str],
        paper_id: str | None = None,
        section: str | None = None,
    ) -> list[RetrievalResult] | None:
        key = self._make_key(query, top_k, filters, paper_id, section)
        with self._lock:
            results = self._cache.get(key)
            if results is not None:
                return [
                    RetrievalResult(
                        chunk_id=r.chunk_id,
                        paper_id=r.paper_id,
                        score=r.score,
                        text=r.text,
                        page=r.page,
                        section=r.section,
                        dense_rank=r.dense_rank,
                        sparse_rank=r.sparse_rank,
                        fusion_score=r.fusion_score,
                        metadata=dict(r.metadata),
                    )
                    for r in results
                ]
            return None

    def set(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str],
        paper_id: str | None,
        section: str | None,
        results: list[RetrievalResult],
    ) -> None:
        key = self._make_key(query, top_k, filters, paper_id, section)
        cached_results = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                score=r.score,
                text=r.text,
                page=r.page,
                section=r.section,
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
                fusion_score=r.fusion_score,
                metadata=dict(r.metadata),
            )
            for r in results
        ]
        with self._lock:
            if key not in self._cache:
                if len(self._cache) >= self.max_size:
                    oldest_key = self._keys_fifo.pop(0)
                    self._cache.pop(oldest_key, None)
                self._keys_fifo.append(key)
            self._cache[key] = cached_results

    def _make_key(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str],
        paper_id: str | None,
        section: str | None,
    ) -> tuple:
        filters_tuple = tuple(sorted(filters.items())) if filters else ()
        return (query, top_k, filters_tuple, paper_id, section)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._keys_fifo.clear()
