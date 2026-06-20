import pytest

from src.domain.query import Query
from src.domain.retrieval_result import RetrievalResult
from src.indexing.faiss_index import FaissIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["neural ranking"]
        return [[1.0, 0.0]]


def test_faiss_index_ranks_by_cosine_similarity() -> None:
    pytest.importorskip("faiss")
    index = FaissIndex(dimension=2)
    index.add([[10.0, 0.0], [0.0, 1.0], [0.8, 0.2]], ["exact", "orthogonal", "close"])

    results = index.search([1.0, 0.0], top_k=3)

    assert [chunk_id for chunk_id, _ in results] == ["exact", "close", "orthogonal"]
    assert results[0][1] == pytest.approx(1.0)


def test_dense_retriever_returns_hydrated_filtered_results() -> None:
    pytest.importorskip("faiss")
    index = FaissIndex(dimension=2)
    index.add([[1.0, 0.0], [0.0, 1.0]], ["c1", "c2"])
    chunks = {
        "c1": RetrievalResult(
            chunk_id="c1",
            paper_id="p1",
            score=0.0,
            text="Dense retrieval with contrastive encoders.",
            metadata={"year": "2024"},
        ),
        "c2": RetrievalResult(
            chunk_id="c2",
            paper_id="p2",
            score=0.0,
            text="Unrelated topic.",
            metadata={"year": "2023"},
        ),
    }
    retriever = DenseRetriever(FakeEmbedder(), index, chunks)

    results = retriever.retrieve(Query("neural ranking", top_k=2, filters={"year": "2024"}))

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].paper_id == "p1"
    assert results[0].score == pytest.approx(1.0)


def test_bm25_retriever_returns_sparse_matches() -> None:
    chunks = [
        RetrievalResult("c1", "p1", 0.0, "neural neural retrieval ranking"),
        RetrievalResult("c2", "p2", 0.0, "language modeling pretraining"),
    ]
    retriever = BM25Retriever(chunks)

    results = retriever.retrieve(Query("neural retrieval", top_k=1))

    assert results[0].chunk_id == "c1"


class StaticRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: Query) -> list[RetrievalResult]:
        return self.results[: query.top_k]


def test_hybrid_retriever_combines_normalized_scores() -> None:
    dense = StaticRetriever(
        [
            RetrievalResult("c1", "p1", 0.9, "dense winner"),
            RetrievalResult("c2", "p2", 0.1, "sparse winner"),
        ]
    )
    sparse = StaticRetriever(
        [
            RetrievalResult("c1", "p1", 0.1, "dense winner"),
            RetrievalResult("c2", "p2", 0.9, "sparse winner"),
        ]
    )
    retriever = HybridRetriever(dense, sparse, dense_weight=0.75)

    results = retriever.retrieve(Query("test", top_k=2))

    assert [result.chunk_id for result in results] == ["c1", "c2"]
