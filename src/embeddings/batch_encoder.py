from src.embeddings.embedder import Embedder


class BatchEncoder:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def encode_batches(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self.embedder.encode(texts[start : start + batch_size]))
        return vectors
