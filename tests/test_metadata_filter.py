from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult
from src.retrieval.filtering.metadata_filter import filter_results

def test_metadata_filter_year():
    q = Query(text="papers after 2023", metadata_constraints={"year": {">": 2023}})
    
    r1 = RetrievalResult(chunk_id="c1", paper_id="p1", score=1.0, text="text", metadata={"year": "2024"})
    r2 = RetrievalResult(chunk_id="c2", paper_id="p2", score=1.0, text="text", metadata={"year": "2022"})
    r3 = RetrievalResult(chunk_id="c3", paper_id="p3", score=1.0, text="text", metadata={"year": "2023"})
    
    filtered = filter_results([r1, r2, r3], q)
    assert len(filtered) == 1
    assert filtered[0].paper_id == "p1"

def test_metadata_filter_survey():
    q = Query(text="survey on transformers", intent="survey")
    
    r1 = RetrievalResult(chunk_id="c1", paper_id="p1", score=1.0, text="text", metadata={"title": "A Survey of Transformers"})
    r2 = RetrievalResult(chunk_id="c2", paper_id="p2", score=1.0, text="text", metadata={"title": "Attention is all you need"})
    r3 = RetrievalResult(chunk_id="c3", paper_id="p3", score=1.0, text="text", metadata={"abstract": "This review summarizes recent progress..."})
    
    filtered = filter_results([r1, r2, r3], q)
    assert len(filtered) == 2
    assert {f.paper_id for f in filtered} == {"p1", "p3"}

def test_metadata_filter_exclude():
    q = Query(text="RAG -neural", exclude_terms=["neural"])
    
    r1 = RetrievalResult(chunk_id="c1", paper_id="p1", score=1.0, text="RAG with vector databases")
    r2 = RetrievalResult(chunk_id="c2", paper_id="p2", score=1.0, text="Neural network for retrieval")
    
    filtered = filter_results([r1, r2], q)
    assert len(filtered) == 1
    assert filtered[0].paper_id == "p1"
