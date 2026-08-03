import pickle
from pathlib import Path

from src.domain.chunk import Chunk


class SparseIndexBuilder:
    def build(self, chunks: list[Chunk], output_path: Path):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for SparseIndexBuilder. Install project dependencies "
                "with `pip install -r requirements.txt`."
            ) from exc

        clean_chunks = [c for c in chunks if isinstance(c.text, str) and c.text.strip()]
        tokenized_corpus = [self._tokenize(c.text) for c in clean_chunks]
        
        bm25_instance = BM25Okapi(tokenized_corpus)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(bm25_instance, f)
            
        return bm25_instance

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
