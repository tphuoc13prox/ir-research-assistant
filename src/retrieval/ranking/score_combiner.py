import logging
from src.domain.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)

class ScoreCombiner:
    def __init__(self, rrf_weight: float = 0.7, field_weight: float = 0.3) -> None:
        self.rrf_weight = rrf_weight
        self.field_weight = field_weight

    def combine_list(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        # 1. Collect score components
        rrf_scores = [r.ranking_features.rrf_score for r in results if r.ranking_features]
        if not rrf_scores:
            return results

        min_rrf, max_rrf = min(rrf_scores), max(rrf_scores)
        rrf_range = max_rrf - min_rrf

        field_scores = []
        for r in results:
            f = r.ranking_features
            if f:
                f_score = (
                    f.title_score + 
                    f.keyword_score + 
                    f.abstract_score + 
                    f.body_score + 
                    f.phrase_score +
                    getattr(f, "intent_score", 0.0)
                )
            else:
                f_score = 0.0
            field_scores.append(f_score)

        min_field, max_field = min(field_scores), max(field_scores)
        field_range = max_field - min_field

        # 2. Normalize and combine
        combined_results = []
        for res, f_score in zip(results, field_scores):
            f = res.ranking_features
            if not f:
                combined_results.append(res)
                continue

            # Min-Max Normalization to [0, 1]
            norm_rrf = (f.rrf_score - min_rrf) / rrf_range if rrf_range > 0 else 1.0
            norm_field = (f_score - min_field) / field_range if field_range > 0 else 1.0

            # Final Score = α × Normalized RRF + β × Normalized Field Score
            final_score = self.rrf_weight * norm_rrf + self.field_weight * norm_field
            f.final_score = final_score
            res.score = final_score
            if res.fusion_score is not None:
                res.fusion_score = final_score

            # Developer logging as requested
            title = (res.metadata or {}).get("title", "Unknown")
            logger.info(
                f"Paper: {title}\n"
                f"  Dense Rank: {res.dense_rank}\n"
                f"  Sparse Rank: {res.sparse_rank}\n"
                f"  Original RRF Score: {f.rrf_score:.4f}\n"
                f"  Title Score: {f.title_score:.4f}\n"
                f"  Keyword Score: {f.keyword_score:.4f}\n"
                f"  Abstract Score: {f.abstract_score:.4f}\n"
                f"  Body Score: {f.body_score:.4f}\n"
                f"  Phrase Score: {f.phrase_score:.4f}\n"
                f"  Intent Score: {getattr(f, 'intent_score', 0.0):.4f}\n"
                f"  Final Score: {final_score:.4f}"
            )
            combined_results.append(res)

        # 3. Sort by final score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results
