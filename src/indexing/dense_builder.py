from pathlib import Path
from typing import Callable

from src.domain.chunk import Chunk
from src.embeddings.interfaces.embedding_service import EmbeddingService
from src.indexing.faiss_index import FaissIndex


class DenseIndexBuilder:
    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def build(
        self,
        chunks: list[Chunk],
        embedding_service: EmbeddingService,
        index_path: Path,
        ids_path: Path,
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> FaissIndex:
        clean_chunks = [c for c in chunks if isinstance(c.text, str) and c.text.strip()]
        texts = [c.text for c in clean_chunks]
        
        total_chunks = len(texts)
        if progress_callback:
            progress_callback(0, total_chunks)
            
        vectors = []
        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            vectors.extend(embedding_service.encode(texts[start:end]))
            if progress_callback:
                progress_callback(end, total_chunks)

        if len(vectors) != len(clean_chunks):
            raise ValueError(
                f"Embedding mismatch: encoded {len(vectors)} vectors for {len(clean_chunks)} chunks"
            )

        index = FaissIndex(normalize=self.normalize)
        index.add(vectors, [c.chunk_id for c in clean_chunks])
        index.save(index_path, ids_path)
        return index
