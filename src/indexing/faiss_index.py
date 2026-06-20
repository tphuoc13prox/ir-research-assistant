from __future__ import annotations

from pathlib import Path

import numpy as np


class FaissIndex:
    """Cosine-similarity FAISS index backed by normalized inner product search."""

    def __init__(self, dimension: int | None = None, *, normalize: bool = True) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FaissIndex. Install project dependencies "
                "with `pip install -r requirements.txt`."
            ) from exc

        self._faiss = faiss
        self.dimension = dimension
        self.normalize = normalize
        self._index = self._create_index(dimension) if dimension is not None else None
        self._ids: list[str] = []

    def add(self, vectors: list[list[float]], ids: list[str]) -> None:
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        if not vectors:
            return

        matrix = self._as_matrix(vectors)
        if self._index is None:
            self.dimension = int(matrix.shape[1])
            self._index = self._create_index(self.dimension)
        elif matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with dimension {self.dimension}, got {matrix.shape[1]}"
            )

        if self.normalize:
            self._faiss.normalize_L2(matrix)
        self._index.add(matrix)
        self._ids.extend(ids)

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0 or self._index is None or not self._ids:
            return []

        query = self._as_matrix([vector])
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, got {query.shape[1]}"
            )

        if self.normalize:
            self._faiss.normalize_L2(query)

        limit = min(top_k, len(self._ids))
        scores, positions = self._index.search(query, limit)
        results: list[tuple[str, float]] = []
        for position, score in zip(positions[0], scores[0], strict=False):
            if position < 0:
                continue
            results.append((self._ids[int(position)], float(score)))
        return results

    def save(self, index_path: str | Path, ids_path: str | Path) -> None:
        if self._index is None:
            raise ValueError("Cannot save an empty FAISS index")

        index_path = Path(index_path)
        ids_path = Path(ids_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(index_path))
        ids_path.write_text("\n".join(self._ids), encoding="utf-8")

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        ids_path: str | Path,
        *,
        normalize: bool = True,
    ) -> "FaissIndex":
        instance = cls(dimension=None, normalize=normalize)
        instance._index = instance._faiss.read_index(str(index_path))
        instance.dimension = int(instance._index.d)
        instance._ids = Path(ids_path).read_text(encoding="utf-8").splitlines()
        return instance

    def _create_index(self, dimension: int | None):
        if dimension is None:
            return None
        return self._faiss.IndexFlatIP(int(dimension))

    @staticmethod
    def _as_matrix(vectors: list[list[float]]) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D list")
        return np.ascontiguousarray(matrix)
