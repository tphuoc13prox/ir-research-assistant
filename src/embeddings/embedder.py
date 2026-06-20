from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into dense vectors."""
        ...


class SentenceTransformerEmbedder:
    """Sentence-transformers wrapper for IR embedding models such as BGE/E5."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        normalize: bool = True,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._encode_raw(texts)

    def _encode_raw(self, texts: list[Any]) -> list[list[float]]:
        clean_texts = self._clean_texts(texts)
        if not clean_texts:
            return []

        vectors = self.model.encode(
            clean_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    @staticmethod
    def _clean_texts(texts: list[Any]) -> list[str]:
        clean_texts: list[str] = []
        for text in texts:
            if text is None:
                continue
            value = text if isinstance(text, str) else str(text)
            value = " ".join(value.split())
            if value:
                clean_texts.append(value)
        return clean_texts


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
