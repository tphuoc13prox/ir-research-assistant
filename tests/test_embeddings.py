from src.embeddings.embedder import E5Embedder


class FakeE5(E5Embedder):
    def __init__(self) -> None:
        pass

    def _encode_raw(self, texts: list[str]) -> list[list[float]]:
        self.last_texts = texts
        return [[1.0, 0.0] for _ in texts]


def test_e5_query_prefix() -> None:
    embedder = FakeE5()

    vector = embedder.encode_query("neural ranking")

    assert vector == [1.0, 0.0]
    assert embedder.last_texts == ["query: neural ranking"]
