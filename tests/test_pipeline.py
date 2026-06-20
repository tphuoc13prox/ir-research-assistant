import pytest

from src.domain.chunk import Chunk
from src.indexing.index_manager import IndexManager


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "neural retrieval": [1.0, 0.0],
            "language modeling": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def test_index_manager_builds_and_loads_dense_index(tmp_path) -> None:
    pytest.importorskip("faiss")
    manager = IndexManager(
        index_path=tmp_path / "faiss.index",
        ids_path=tmp_path / "ids.txt",
        chunks_path=tmp_path / "chunks.jsonl",
    )
    chunks = [
        Chunk(chunk_id="c1", paper_id="p1", text="neural retrieval"),
        Chunk(chunk_id="c2", paper_id="p2", text="language modeling"),
    ]

    manager.build(chunks, FakeEmbedder())
    index, stored_chunks = manager.load()

    assert index.search([1.0, 0.0], top_k=1)[0][0] == "c1"
    assert stored_chunks["c1"].text == "neural retrieval"
