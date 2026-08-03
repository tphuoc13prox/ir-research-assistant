from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable

from src.domain.chunk import Chunk
from src.domain.retrieval_result import RetrievalResult
from src.embeddings.interfaces.embedding_service import EmbeddingService
from src.indexing.faiss_index import FaissIndex
from src.indexing.dense_builder import DenseIndexBuilder
from src.indexing.sparse_builder import SparseIndexBuilder


class IndexManager:
    def __init__(
        self,
        *,
        index_path: Path = Path("data/index/faiss.index"),
        sparse_index_path: Path = Path("data/index/bm25.pkl"),
        ids_path: Path = Path("data/index/ids.txt"),
        chunks_path: Path = Path("data/index/chunks.jsonl"),
        normalize: bool = True,
    ) -> None:
        self.index_path = index_path
        self.sparse_index_path = sparse_index_path
        self.ids_path = ids_path
        self.chunks_path = chunks_path
        self.normalize = normalize

    def build(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingService,
        *,
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> FaissIndex:
        # Build dense index
        dense_builder = DenseIndexBuilder(normalize=self.normalize)
        index = dense_builder.build(
            chunks,
            embedder,
            self.index_path,
            self.ids_path,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

        # Build sparse index
        sparse_builder = SparseIndexBuilder()
        sparse_builder.build(chunks, self.sparse_index_path)

        # Save chunks metadata
        self._save_chunks(chunks)
        return index

    def load(self) -> tuple[FaissIndex, dict[str, RetrievalResult]]:
        index = FaissIndex.load(
            self.index_path,
            self.ids_path,
            normalize=self.normalize,
        )
        return index, self.load_chunks()

    def load_hybrid(self) -> tuple[FaissIndex, dict[str, RetrievalResult], any]:
        index, stored_chunks = self.load()
        
        # Lazy loading or fallback build of BM25 sparse index
        if self.sparse_index_path.exists():
            try:
                with open(self.sparse_index_path, "rb") as f:
                    bm25_instance = pickle.load(f)
            except Exception:
                bm25_instance = self._rebuild_sparse_index(stored_chunks)
        else:
            bm25_instance = self._rebuild_sparse_index(stored_chunks)
            
        return index, stored_chunks, bm25_instance

    def _rebuild_sparse_index(self, stored_chunks: dict[str, RetrievalResult]) -> any:
        # Backward compatibility fallback
        chunks_list = [
            Chunk(
                chunk_id=c.chunk_id,
                paper_id=c.paper_id,
                text=c.text,
                metadata=c.metadata,
            )
            for c in stored_chunks.values()
        ]
        sparse_builder = SparseIndexBuilder()
        return sparse_builder.build(chunks_list, self.sparse_index_path)

    def load_chunks(self) -> dict[str, RetrievalResult]:
        chunks: dict[str, RetrievalResult] = {}
        if not self.chunks_path.exists():
            return chunks

        for line in self.chunks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            chunks[item["chunk_id"]] = RetrievalResult(
                chunk_id=item["chunk_id"],
                paper_id=item["paper_id"],
                score=0.0,
                text=item["text"],
                page=item.get("page") or item.get("metadata", {}).get("page"),
                section=item.get("section") or item.get("metadata", {}).get("section"),
                metadata=item.get("metadata", {}),
            )
        return chunks

    def _save_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": chunk.metadata,
                    "page": chunk.metadata.get("page"),
                },
                ensure_ascii=True,
            )
            for chunk in chunks
        ]
        self.chunks_path.write_text("\n".join(lines), encoding="utf-8")
