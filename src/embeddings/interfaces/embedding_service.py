from typing import Protocol


class EmbeddingService(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into dense vectors."""
        ...
