import re
from dataclasses import dataclass
from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult, RankingFeatures
from src.retrieval.ranking.base_ranker import BaseRanker
from src.retrieval.ranking.ranking_weights import RankingConfig

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "about", "using", 
    "paper", "papers", "article", "articles", "study", "studies", "research", 
    "on", "of", "to", "in", "a", "an", "is", "by", "as", "at", "or", "how", 
    "what", "which", "where"
}

@dataclass(frozen=True)
class ProcessedQuery:
    raw_text: str
    lower_text: str
    terms: list[str]

class FieldAwareRanker(BaseRanker):
    def __init__(self, config: RankingConfig | None = None) -> None:
        self.config = config or RankingConfig()

    def rank(self, query: Query, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not self.config.enabled or not results:
            # If disabled or empty, return as is (but we can initialize empty ranking features)
            for res in results:
                if res.ranking_features is None:
                    res.ranking_features = RankingFeatures(
                        rrf_score=res.fusion_score if res.fusion_score is not None else res.score,
                        final_score=res.fusion_score if res.fusion_score is not None else res.score,
                    )
            return results

        processed = self._process_query(query)
        weights = self.config.weights

        for res in results:
            metadata = res.metadata or {}
            title = str(metadata.get("title", "")).lower()
            abstract = str(metadata.get("abstract", "")).lower()
            
            keywords_raw = metadata.get("keywords", [])
            if not keywords_raw:
                keywords_raw = metadata.get("concepts", [])
            if not keywords_raw:
                keywords_raw = metadata.get("categories", [])
                
            if isinstance(keywords_raw, list):
                keywords_list = [str(k).lower() for k in keywords_raw if k]
            else:
                keywords_list = [str(keywords_raw).lower()]
            keywords_str = " ".join(keywords_list)
            
            body = (res.text or "").lower()

            # 1. Term counts
            title_terms = sum(title.count(t) for t in processed.terms)
            keyword_terms = sum(keywords_str.count(t) for t in processed.terms)
            abstract_terms = sum(abstract.count(t) for t in processed.terms)
            body_terms = sum(body.count(t) for t in processed.terms)

            # 2. Saturation-Based Scoring
            title_score = weights.get("title", 4.0) * (title_terms / (title_terms + 1.0)) if title_terms > 0 else 0.0
            keyword_score = weights.get("keywords", 3.0) * (keyword_terms / (keyword_terms + 1.0)) if keyword_terms > 0 else 0.0
            abstract_score = weights.get("abstract", 2.0) * (abstract_terms / (abstract_terms + 2.0)) if abstract_terms > 0 else 0.0
            body_score = weights.get("body", 1.0) * (body_terms / (body_terms + 20.0)) if body_terms > 0 else 0.0

            # 3. Exact Phrase Matching (case-insensitive)
            phrase_weight = weights.get("phrase", 2.0)
            phrase_title = title.count(processed.lower_text)
            phrase_abstract = abstract.count(processed.lower_text)
            phrase_keywords = keywords_str.count(processed.lower_text)
            phrase_body = body.count(processed.lower_text)

            phrase_score = phrase_weight * (
                (phrase_title / (phrase_title + 1.0) if phrase_title > 0 else 0.0) +
                (phrase_keywords / (phrase_keywords + 1.0) if phrase_keywords > 0 else 0.0) +
                (phrase_abstract / (phrase_abstract + 1.0) if phrase_abstract > 0 else 0.0) +
                (phrase_body / (phrase_body + 5.0) if phrase_body > 0 else 0.0)
            )

            rrf = res.fusion_score if res.fusion_score is not None else res.score
            
            # Store ranking features
            res.ranking_features = RankingFeatures(
                rrf_score=rrf,
                title_score=title_score,
                keyword_score=keyword_score,
                abstract_score=abstract_score,
                body_score=body_score,
                phrase_score=phrase_score,
            )

        return results

    def _process_query(self, query: Query) -> ProcessedQuery:
        lower_text = query.text.lower().strip()
        terms = [
            w for w in re.findall(r"\b\w+\b", lower_text)
            if w not in STOPWORDS and len(w) > 1 and not w.isdigit()
        ]
        return ProcessedQuery(
            raw_text=query.text,
            lower_text=lower_text,
            terms=terms,
        )
