from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult, RankingFeatures
from src.retrieval.ranking.intent_ranker import IntentRanker
from src.retrieval.ranking.explanation_generator import generate_explanation

def test_intent_ranker_boosts():
    q = Query(text="survey on CNN", intent="survey")
    
    r1 = RetrievalResult(
        chunk_id="c1", 
        paper_id="p1", 
        score=1.0, 
        text="text", 
        metadata={"title": "A Survey of CNN architectures"},
        ranking_features=RankingFeatures(title_score=4.0)
    )
    r2 = RetrievalResult(
        chunk_id="c2", 
        paper_id="p2", 
        score=1.0, 
        text="text", 
        metadata={"title": "CNN performance details"},
        ranking_features=RankingFeatures(title_score=4.0)
    )
    
    ranker = IntentRanker()
    ranked = ranker.rank(q, [r1, r2])
    
    # r1: 5.0 (title_score > 0) + 3.0 (intent survey in title) = 8.0
    assert ranked[0].ranking_features.intent_score == 8.0
    # r2: 5.0 (title_score > 0) = 5.0
    assert ranked[1].ranking_features.intent_score == 5.0

def test_explanation_generator():
    q = Query(text="survey on CNN", intent="survey", metadata_constraints={"year": {">": 2023}})
    r = RetrievalResult(
        chunk_id="c1", 
        paper_id="p1", 
        score=0.9, 
        text="text", 
        metadata={"title": "Survey on CNN"},
        ranking_features=RankingFeatures(
            rrf_score=0.7,
            title_score=4.0,
            intent_score=8.0
        )
    )
    
    exp = generate_explanation(r, q)
    assert exp["topical_relevance"] == "High"
    assert exp["matched_title"] is True
    assert exp["matched_body"] is False
    assert any("year_constraint" in f for f in exp["applied_filters"])
    assert exp["final_score"] == 0.9
