from src.domain.retrieval_result import RetrievalResult
from src.retrieval.interfaces.fusion_strategy import FusionStrategy


class ReciprocalRankFusion(FusionStrategy):
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        dense_ranks = {res.chunk_id: rank for rank, res in enumerate(dense_results, start=1)}
        sparse_ranks = {res.chunk_id: rank for rank, res in enumerate(sparse_results, start=1)}
        
        unique_chunks = {}
        for res in dense_results:
            unique_chunks[res.chunk_id] = res
        for res in sparse_results:
            unique_chunks[res.chunk_id] = res

        fused_results = []
        for chunk_id, res in unique_chunks.items():
            dense_rank = dense_ranks.get(chunk_id)
            sparse_rank = sparse_ranks.get(chunk_id)
            
            score = 0.0
            if dense_rank is not None:
                score += 1.0 / (self.k + dense_rank)
            if sparse_rank is not None:
                score += 1.0 / (self.k + sparse_rank)
                
            fused_results.append(
                RetrievalResult(
                    chunk_id=res.chunk_id,
                    paper_id=res.paper_id,
                    score=score,
                    text=res.text,
                    page=res.page,
                    section=res.section,
                    dense_rank=dense_rank,
                    sparse_rank=sparse_rank,
                    fusion_score=score,
                    metadata=res.metadata,
                )
            )

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:top_k]
