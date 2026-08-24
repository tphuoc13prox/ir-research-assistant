from src.domain.query import Query
from src.query_understanding.query_parser import parse_query

def test_query_parser_year_after():
    q = Query(text="BM25 papers after 2023")
    parsed = parse_query(q)
    assert parsed.metadata_constraints.get("year") == {">": 2023}
    assert parsed.intent == "general"
    assert "bm25" in parsed.topic

def test_query_parser_year_before():
    q = Query(text="Transformer papers before 2025")
    parsed = parse_query(q)
    assert parsed.metadata_constraints.get("year") == {"<": 2025}
    assert "transformer" in parsed.topic

def test_query_parser_year_between():
    q = Query(text="naive bayes between 2020 and 2025")
    parsed = parse_query(q)
    assert parsed.metadata_constraints.get("year") == {">=": 2020, "<=": 2025}
    assert "naive bayes" in parsed.topic

def test_query_parser_recent():
    q = Query(text="recent survey on CNN")
    parsed = parse_query(q)
    assert parsed.metadata_constraints.get("year") == {">=": 2024}
    assert parsed.intent == "survey"
    assert "cnn" in parsed.topic

def test_query_parser_survey_intent():
    q = Query(text="Transformer survey papers")
    parsed = parse_query(q)
    assert parsed.intent == "survey"
    assert "transformer" in parsed.topic

def test_query_parser_comparison_intent():
    q = Query(text="CNN vs Vision Transformer")
    parsed = parse_query(q)
    assert parsed.intent == "comparison"
    assert "cnn" in parsed.topic
    assert "vision transformer" in parsed.topic

def test_query_parser_dataset_intent():
    q = Query(text="NLP datasets for training")
    parsed = parse_query(q)
    assert parsed.intent == "dataset"
    assert "nlp" in parsed.topic
    assert "training" in parsed.topic

def test_query_parser_boost_and_exclude():
    q = Query(text='RAG papers "vector search" -neural not dense')
    parsed = parse_query(q)
    assert "vector search" in parsed.boost_terms
    assert "neural" in parsed.exclude_terms
    assert "dense" in parsed.exclude_terms
