from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult
from src.embeddings.interfaces.embedding_service import EmbeddingService
from src.indexing.faiss_index import FaissIndex
from src.retrieval.interfaces.base_retriever import BaseRetriever
from src.retrieval.cache import RetrievalCache


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: EmbeddingService,
        index: FaissIndex,
        chunks: dict[str, RetrievalResult] | None = None,
        cache: RetrievalCache | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks or {}
        self.cache = cache or RetrievalCache()

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

        has_filters = bool(query.filters or query.paper_id or query.section)
        search_k = max(100, query.top_k * 5) if has_filters else query.top_k

        if hasattr(self.embedder, "encode_query"):
            query_vector = self.embedder.encode_query(query.text)
        else:
            query_vector = self.embedder.encode([query.text])[0]

        matches = self.index.search(query_vector, search_k)
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
                    page=stored.page,
                    section=stored.section,
                    metadata=stored.metadata,
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
