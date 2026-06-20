from src.chunking.chunker import Chunker


def test_chunker_splits_text() -> None:
    chunks = Chunker().split("paper-1", "abcdef", chunk_size=2)

    assert [chunk.text for chunk in chunks] == ["ab", "cd", "ef"]
