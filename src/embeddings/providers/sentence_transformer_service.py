import numpy as np
from src.embeddings.interfaces.embedding_service import EmbeddingService


class SentenceTransformerEmbeddingService(EmbeddingService):
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
                "sentence-transformers is required for SentenceTransformerEmbeddingService. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self._cache: dict[str, list[float]] = {}

    def encode(self, texts: list[str]) -> list[list[float]]:
        clean_texts = self._clean_texts(texts)
        if not clean_texts:
            return []

        uncached_texts = list(set([t for t in clean_texts if t not in self._cache]))
        
        if uncached_texts:
            vectors = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            vector_list = np.asarray(vectors, dtype=np.float32).tolist()
            for text, vec in zip(uncached_texts, vector_list):
                self._cache[text] = vec

        return [self._cache[t] for t in clean_texts]

    @staticmethod
    def _clean_texts(texts: list[str]) -> list[str]:
        clean_texts: list[str] = []
        for text in texts:
            if text is None:
                continue
            value = text if isinstance(text, str) else str(text)
            value = " ".join(value.split())
            if value:
                clean_texts.append(value)
        return clean_texts
