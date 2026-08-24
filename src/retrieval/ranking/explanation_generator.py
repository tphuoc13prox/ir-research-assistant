from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult

def generate_explanation(result: RetrievalResult, query: Query) -> dict:
    features = result.ranking_features
    if features is None:
        return {}

    # Determine topical relevance classification
    if features.intent_score >= 5.0:
        topical_relevance = "High"
    elif features.intent_score >= 2.0:
        topical_relevance = "Medium"
    else:
        topical_relevance = "Low"

    # Capture any applied query constraints in explainability dictionary
    applied_filters = []
    if query.metadata_constraints.get("year"):
        applied_filters.append(f"year_constraint: {query.metadata_constraints['year']}")
    if query.intent != "general":
        applied_filters.append(f"intent_filter: {query.intent}")
    if query.exclude_terms:
        applied_filters.append(f"exclude_terms: {query.exclude_terms}")

    return {
        "topical_relevance": topical_relevance,
        "matched_title": features.title_score > 0.0,
        "matched_abstract": features.abstract_score > 0.0,
        "matched_keywords": features.keyword_score > 0.0,
        "matched_body": features.body_score > 0.0,
        "matched_phrase": features.phrase_score > 0.0,
        "applied_filters": applied_filters,
        "dense_rank": result.dense_rank,
        "sparse_rank": result.sparse_rank,
        "rrf_score": features.rrf_score,
        "final_score": result.score,
    }
