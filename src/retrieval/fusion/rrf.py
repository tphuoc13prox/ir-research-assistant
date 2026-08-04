from src.domain.retrieval_result import RetrievalResult
from src.retrieval.interfaces.fusion_strategy import FusionStrategy


from src.domain.query import Query

class ReciprocalRankFusion(FusionStrategy):
    def __init__(self, k: int = 60, weights: tuple[float, float] = (1.0, 1.0)) -> None:
        self.k = k
        self.weights = weights

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        query: Query | None = None,
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
                score += self.weights[0] * (1.0 / (self.k + dense_rank))
            if sparse_rank is not None:
                score += self.weights[1] * (1.0 / (self.k + sparse_rank))
                
            # Apply field boosting if query context is available
            boost_multiplier = 1.0
            if query and query.text:
                q_text = query.text.lower().strip()
                title = res.metadata.get("title", "").lower() if res.metadata else ""
                abstract = res.metadata.get("abstract", "").lower() if res.metadata else ""
                
                # 1. Exact phrase match on raw query
                if q_text in title:
                    boost_multiplier = 2.5
                elif q_text in abstract:
                    boost_multiplier = 1.5
                else:
                    # 2. Extract core words by filtering out year/date parameters, academic noise, and stopwords
                    stopwords = {
                        "the", "and", "for", "with", "from", "that", "this", "about", "using", 
                        "paper", "papers", "article", "articles", "study", "studies", "research", 
                        "published", "after", "before", "since", "in", "year", "years", "on", "of",
                        "to", "a", "an", "is", "by", "as", "at", "or", "how", "what", "which", "where"
                    }
                    q_words = [w for w in q_text.split() if len(w) > 2 and w not in stopwords and not w.isdigit()]
                    if q_words:
                        core_phrase = " ".join(q_words)
                        if core_phrase in title:
                            boost_multiplier = max(boost_multiplier, 2.5)
                        elif core_phrase in abstract:
                            boost_multiplier = max(boost_multiplier, 1.5)
                        else:
                            title_matches = sum(1 for w in q_words if w in title)
                            if title_matches == len(q_words):
                                boost_multiplier = max(boost_multiplier, 2.0)
                            elif title_matches > 0:
                                boost_multiplier = max(boost_multiplier, 1.2 + 0.3 * (title_matches / len(q_words)))
                            
                            abstract_matches = sum(1 for w in q_words if w in abstract)
                            if abstract_matches == len(q_words):
                                boost_multiplier = max(boost_multiplier, 1.4)
                            elif abstract_matches > 0:
                                boost_multiplier = max(boost_multiplier, 1.05 + 0.15 * (abstract_matches / len(q_words)))

            score *= boost_multiplier

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
