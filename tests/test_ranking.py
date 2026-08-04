import pytest
from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult, RankingFeatures
from src.retrieval.ranking.field_aware_ranker import FieldAwareRanker
from src.retrieval.ranking.score_combiner import ScoreCombiner
from src.retrieval.ranking.ranking_weights import RankingConfig

def test_title_match_only() -> None:
    # 1. Query term appears only in title
    res1 = RetrievalResult("c1", "p1", 0.5, "c1 text without keyword", metadata={"title": "Naive Bayes Method"})
    res2 = RetrievalResult("c2", "p2", 0.5, "c2 text without keyword", metadata={"title": "Other Paper"})
    
    query = Query("Naive Bayes", top_k=2)
    rank_config = RankingConfig()
    ranker = FieldAwareRanker(rank_config)
    results = ranker.rank(query, [res1, res2])
    
    combiner = ScoreCombiner(rrf_weight=0.7, field_weight=0.3)
    ranked = combiner.combine_list(results)
    
    # Confirm title match ranks first
    assert ranked[0].chunk_id == "c1"
    assert ranked[0].ranking_features.title_score > 0.0
    assert ranked[1].ranking_features.title_score == 0.0

def test_abstract_match_only() -> None:
    # 2. Query term appears only in abstract
    res1 = RetrievalResult("c1", "p1", 0.5, "c1 text", metadata={"title": "Paper 1", "abstract": "We analyze Naive Bayes algorithm"})
    res2 = RetrievalResult("c2", "p2", 0.5, "c2 text", metadata={"title": "Paper 2", "abstract": "Other text"})
    
    query = Query("Naive Bayes", top_k=2)
    rank_config = RankingConfig()
    ranker = FieldAwareRanker(rank_config)
    results = ranker.rank(query, [res1, res2])
    
    combiner = ScoreCombiner(rrf_weight=0.7, field_weight=0.3)
    ranked = combiner.combine_list(results)
    
    assert ranked[0].chunk_id == "c1"
    assert ranked[0].ranking_features.abstract_score > 0.0
    assert ranked[1].ranking_features.abstract_score == 0.0

def test_body_match_only() -> None:
    # 3. Query term appears only in body
    res1 = RetrievalResult("c1", "p1", 0.5, "This study implements Naive Bayes models.", metadata={"title": "Paper 1"})
    res2 = RetrievalResult("c2", "p2", 0.5, "Other random text", metadata={"title": "Paper 2"})
    
    query = Query("Naive Bayes", top_k=2)
    rank_config = RankingConfig()
    ranker = FieldAwareRanker(rank_config)
    results = ranker.rank(query, [res1, res2])
    
    combiner = ScoreCombiner(rrf_weight=0.7, field_weight=0.3)
    ranked = combiner.combine_list(results)
    
    assert ranked[0].chunk_id == "c1"
    assert ranked[0].ranking_features.body_score > 0.0
    assert ranked[1].ranking_features.body_score == 0.0

def test_multi_field_match() -> None:
    # 4. Query term appears in multiple fields
    res1 = RetrievalResult(
        "c1", "p1", 0.5, "Implementing Naive Bayes classifier", 
        metadata={"title": "Naive Bayes Classifier", "abstract": "Naive Bayes is fast", "keywords": ["Naive Bayes"]}
    )
    
    query = Query("Naive Bayes", top_k=1)
    rank_config = RankingConfig()
    ranker = FieldAwareRanker(rank_config)
    results = ranker.rank(query, [res1])
    
    combiner = ScoreCombiner(rrf_weight=0.7, field_weight=0.3)
    ranked = combiner.combine_list(results)
    
    f = ranked[0].ranking_features
    assert f.title_score > 0.0
    assert f.keyword_score > 0.0
    assert f.abstract_score > 0.0
    assert f.body_score > 0.0
    assert f.phrase_score > 0.0

def test_no_body_dominance() -> None:
    # 5. Multiple occurrences in body should not outweigh a title match
    # res1: Match in Title once, no matches in body.
    res1 = RetrievalResult("c1", "p1", 0.5, "no matches here", metadata={"title": "Naive Bayes Classifier"})
    
    # res2: 100 matches in body, but no match in title.
    body_text_with_repeats = "Naive Bayes " * 100
    res2 = RetrievalResult("c2", "p2", 0.5, body_text_with_repeats, metadata={"title": "Other Topic"})
    
    query = Query("Naive Bayes", top_k=2)
    rank_config = RankingConfig()
    ranker = FieldAwareRanker(rank_config)
    results = ranker.rank(query, [res1, res2])
    
    combiner = ScoreCombiner(rrf_weight=0.7, field_weight=0.3)
    ranked = combiner.combine_list(results)
    
    # Verify res1 (title match) ranks HIGHER than res2 (repeated body matches)
    assert ranked[0].chunk_id == "c1"
    assert ranked[0].ranking_features.title_score > ranked[1].ranking_features.body_score

def test_disabled_ranking_preserves_order() -> None:
    # 6. Disabled ranking should preserve original RRF order
    res1 = RetrievalResult("c1", "p1", 0.9, "Naive Bayes here", metadata={"title": "Other Title"})
    res2 = RetrievalResult("c2", "p2", 0.95, "other text", metadata={"title": "Naive Bayes Classifier"})
    
    # Original RRF order: c2 (0.95) then c1 (0.9)
    # If ranking is enabled: c1 would match in body, c2 matches in title, so c2 stays top.
    # If we reverse the RRF scores: c1 (0.95), c2 (0.9). Enabled would put c2 first (title boost).
    # Disabled must preserve c1 first!
    res1.score = 0.95
    res1.fusion_score = 0.95
    res2.score = 0.90
    res2.fusion_score = 0.90
    
    query = Query("Naive Bayes", top_k=2)
    
    # Disabled ranker
    rank_config = RankingConfig()
    rank_config.enabled = False
    
    ranker = FieldAwareRanker(rank_config)
    results = ranker.rank(query, [res1, res2])
    
    # Should not re-sort because ranking is disabled, preserving original RRF order
    assert results[0].chunk_id == "c1"
