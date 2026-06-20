from __future__ import annotations

import json
from pathlib import Path

from src.domain.chunk import Chunk
from src.domain.retrieval_result import RetrievalResult
from src.embeddings.embedder import Embedder
from src.indexing.faiss_index import FaissIndex


class IndexManager:
    def __init__(
        self,
        *,
        index_path: Path = Path("data/index/faiss.index"),
        ids_path: Path = Path("data/index/ids.txt"),
        chunks_path: Path = Path("data/index/chunks.jsonl"),
        normalize: bool = True,
    ) -> None:
        self.index_path = index_path
        self.ids_path = ids_path
        self.chunks_path = chunks_path
        self.normalize = normalize

    def build(self, chunks: list[Chunk], embedder: Embedder, *, batch_size: int = 32) -> FaissIndex:
        chunks = [chunk for chunk in chunks if isinstance(chunk.text, str) and chunk.text.strip()]
        texts = [chunk.text for chunk in chunks]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(embedder.encode(texts[start : start + batch_size]))

        if len(vectors) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: encoded {len(vectors)} vectors for {len(chunks)} chunks"
            )

        index = FaissIndex(normalize=self.normalize)
        index.add(vectors, [chunk.chunk_id for chunk in chunks])
        index.save(self.index_path, self.ids_path)
        self._save_chunks(chunks)
        return index

    def load(self) -> tuple[FaissIndex, dict[str, RetrievalResult]]:
        index = FaissIndex.load(
            self.index_path,
            self.ids_path,
            normalize=self.normalize,
        )
        return index, self.load_chunks()

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
                },
                ensure_ascii=True,
            )
            for chunk in chunks
        ]
        self.chunks_path.write_text("\n".join(lines), encoding="utf-8")
