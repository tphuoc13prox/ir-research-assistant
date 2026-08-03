from __future__ import annotations

from typing import Protocol
from src.embeddings.interfaces.embedding_service import EmbeddingService
from src.embeddings.providers.sentence_transformer_service import SentenceTransformerEmbeddingService


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into dense vectors."""
        ...


class SentenceTransformerEmbedder(SentenceTransformerEmbeddingService):
    """Sentence-transformers wrapper for IR embedding models such as BGE/E5.
    
    Subclasses SentenceTransformerEmbeddingService to maintain 100% backward compatibility.
    """
    
    def _encode_raw(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts)
 
    @staticmethod
    def _clean_texts(texts: list[str]) -> list[str]:
        return SentenceTransformerEmbeddingService._clean_texts(texts)


class E5Embedder(SentenceTransformerEmbedder):
    """E5 model wrapper that applies the recommended query/passage prefixes."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        *,
        normalize: bool = True,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            normalize=normalize,
            batch_size=batch_size,
            device=device,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        passages = [
            text if text.startswith(("query: ", "passage: ")) else f"passage: {text}"
            for text in self._clean_texts(texts)
        ]
        return self._encode_raw(passages)

    def encode_query(self, text: str) -> list[float]:
        return self._encode_raw([f"query: {text}"])[0]
