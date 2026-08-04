import time
from concurrent.futures import ThreadPoolExecutor

from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult
from src.retrieval.interfaces.base_retriever import BaseRetriever
from src.retrieval.interfaces.fusion_strategy import FusionStrategy
from src.retrieval.fusion.rrf import ReciprocalRankFusion
from src.retrieval.fusion.score_normalization import ScoreNormalizationFusion


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        *,
        fusion_strategy: FusionStrategy | None = None,
        dense_weight: float | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        if fusion_strategy is not None:
            self.fusion_strategy = fusion_strategy
        elif dense_weight is not None:
            self.fusion_strategy = ScoreNormalizationFusion(dense_weight=dense_weight)
        else:
            self.fusion_strategy = ReciprocalRankFusion(k=rrf_k)

        self.last_dense_time = 0.0
        self.last_sparse_time = 0.0
        self.last_fusion_time = 0.0
        self.last_dense_results: list[RetrievalResult] = []
        self.last_sparse_results: list[RetrievalResult] = []
        self.last_fused_results: list[RetrievalResult] = []

    def retrieve(self, query: Query) -> list[RetrievalResult]:
        candidate_k = max(100, query.top_k * 5)
        hybrid_query = Query(
            text=query.text,
            top_k=candidate_k,
            filters=query.filters,
            paper_id=query.paper_id,
            section=query.section,
        )

        def run_dense():
            t0 = time.time()
            res = self.dense_retriever.retrieve(hybrid_query)
            return res, (time.time() - t0) * 1000

        def run_sparse():
            t0 = time.time()
            res = self.sparse_retriever.retrieve(hybrid_query)
            return res, (time.time() - t0) * 1000

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(run_dense)
            sparse_future = executor.submit(run_sparse)
            
            dense_results, dense_time = dense_future.result()
            sparse_results, sparse_time = sparse_future.result()

        self.last_dense_time = dense_time
        self.last_dense_results = dense_results
        self.last_sparse_time = sparse_time
        self.last_sparse_results = sparse_results

        fusion_start = time.time()
        fused_results = self.fusion_strategy.fuse(dense_results, sparse_results, query.top_k, query)
        self.last_fusion_time = (time.time() - fusion_start) * 1000
        self.last_fused_results = fused_results

        return fused_results
