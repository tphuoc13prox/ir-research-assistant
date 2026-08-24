from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult, RankingFeatures
from src.retrieval.ranking.base_ranker import BaseRanker

INTENT_KEYWORDS_MAP = {
    "survey": {"survey", "review", "overview", "tutorial", "advances", "recent progress"},
    "comparison": {"vs", "versus", "comparison", "compare", "alternative", "differential"},
    "dataset": {"dataset", "benchmark", "corpus", "datasets", "data set"}
}

class IntentRanker(BaseRanker):
    def rank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        intent = query.intent
        boost_terms = query.boost_terms

        for r in results:
            if r.ranking_features is None:
                r.ranking_features = RankingFeatures()

            features = r.ranking_features
            intent_score = 0.0

            # 1. Topic Match Relevance (evaluated from field scores)
            # High topical relevance: Match in Title or Keywords
            if features.title_score > 0.0 or features.keyword_score > 0.0:
                intent_score += 5.0
            # Medium topical relevance: Match in Abstract
            elif features.abstract_score > 0.0:
                intent_score += 2.0
            # Low topical relevance: Match in Body text only
            elif features.body_score > 0.0:
                intent_score += 0.5

            # 2. Intent-Specific Metadata Boost
            if intent in INTENT_KEYWORDS_MAP:
                target_kws = INTENT_KEYWORDS_MAP[intent]
                title = r.metadata.get("title", "").lower()
                abstract = r.metadata.get("abstract", "").lower()

                kws = r.metadata.get("keywords", [])
                if isinstance(kws, str):
                    kws = [k.strip().lower() for k in kws.split(",") if k.strip()]
                elif isinstance(kws, list):
                    kws = [str(k).lower() for k in kws]
                else:
                    kws = []

                has_intent_kw = (
                    any(kw in title for kw in target_kws) or
                    any(kw in abstract for kw in target_kws) or
                    any(kw in kws for kw in target_kws)
                )
                if has_intent_kw:
                    intent_score += 3.0

            # 3. Boost Terms Match (e.g. quoted terms)
            if boost_terms:
                text_lower = r.text.lower()
                title_lower = r.metadata.get("title", "").lower()
                abstract_lower = r.metadata.get("abstract", "").lower()
                
                for term in boost_terms:
                    if term in text_lower or term in title_lower or term in abstract_lower:
                        intent_score += 2.0

            features.intent_score = intent_score

        return results
